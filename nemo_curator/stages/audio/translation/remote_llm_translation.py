# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remote-server variant of :class:`LLMTranslationStage`.

Instead of loading an in-process vLLM engine, this stage sends
OpenAI-compatible chat requests to a shared inference server (e.g. a
``nemo_curator.core.serve.InferenceServer`` running the Dynamo backend, or
any external OpenAI-compatible endpoint). Many such CPU-only client stages
can run concurrently against one GPU-backed server pool, avoiding the
per-stage engine loads and GPU time-sharing churn of the in-process path.

It subclasses :class:`LLMTranslationStage` purely to reuse the field set and
the prompt-resolution / fan-out helpers (``_resolve_prompt``,
``_build_prompt_values``, ``_emit_empty_translations``, ``inputs``,
``outputs``). It NEVER imports vLLM or loads a tokenizer — the server applies
the chat template.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import string
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.audio.translation.llm_translation import (
    _DEFAULT_PROMPT_PATH,
    _DEFAULT_SYSTEM_PROMPT_PATH,
    LLMTranslationStage,
)
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.tasks import AudioTask


@dataclass
class RemoteLLMTranslationStage(LLMTranslationStage):
    """:class:`LLMTranslationStage` that calls a remote OpenAI-compatible server.

    Args (in addition to the inherited :class:`LLMTranslationStage` fields):
        inference_base_url: OpenAI-compatible base URL of the server,
            e.g. ``http://host:8000/v1``. Required.
        inference_api_key: API key sent with each request (servers that do not
            authenticate accept any non-empty value).
        served_model_name: Value passed as ``model=`` in each request.
            Defaults to ``model_id`` when unset.
        max_concurrent_requests: Max in-flight requests per stage actor
            (bounds the async client's semaphore).
        request_timeout: Per-request timeout in seconds.

    GPU-engine fields inherited from :class:`LLMTranslationStage`
    (``max_model_len``, ``kv_cache_dtype``, ``gpu_memory_utilization``,
    ``tensor_parallel_size`` …) are ignored here — those live on the server.
    """

    inference_base_url: str = ""
    inference_api_key: str = "EMPTY"
    served_model_name: str | None = None
    max_concurrent_requests: int = 64
    request_timeout: int = 120

    _client: Any = field(default=None, init=False, repr=False)
    _gen_config: Any = field(default=None, init=False, repr=False)
    _loop_runner: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # NB: do NOT call super().__post_init__() — it calls get_gpu_count()
        # and sets Resources(gpus=tp). Remote mode holds no engine, so we
        # resolve the prompts here and force CPU-only resources instead.
        if not self.model_id:
            msg = "RemoteLLMTranslationStage: model_id is required"
            raise ValueError(msg)
        if not self.inference_base_url:
            msg = "RemoteLLMTranslationStage requires inference_base_url"
            raise ValueError(msg)

        self._translation_prompt = self._resolve_prompt(
            inline=self.translation_prompt,
            file_path=self.translation_prompt_file,
            default_path=_DEFAULT_PROMPT_PATH,
            label="translation",
        )
        self._system_prompt = self._resolve_prompt(
            inline=self.system_prompt,
            file_path=self.system_prompt_file,
            default_path=_DEFAULT_SYSTEM_PROMPT_PATH,
            label="system",
        )
        self._prompt_placeholders = frozenset(
            field_name
            for _, field_name, _, _ in string.Formatter().parse(self._translation_prompt)
            if field_name
        )

        # CPU-only: the GPUs belong to the server. Asking for a GPU here would
        # deadlock waiting for one the server never frees.
        self.resources = Resources(cpus=1.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup_on_node(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._client is not None:
            return

        from nemo_curator.models.client.async_runner import PersistentEventLoop
        from nemo_curator.models.client.llm_client import GenerationConfig
        from nemo_curator.models.client.openai_client import AsyncOpenAIClient

        self._client = AsyncOpenAIClient(
            max_concurrent_requests=self.max_concurrent_requests,
            base_url=self.inference_base_url,
            api_key=self.inference_api_key,
            timeout=self.request_timeout,
        )
        self._client.setup()
        # One event loop for this actor's lifetime so the async client (and its
        # connection pool) stays bound to a single, always-running loop. Driving
        # it via a fresh asyncio.run() per batch can wedge on a primitive bound
        # to a since-closed loop — a silent, timeout-immune hang.
        self._loop_runner = PersistentEventLoop(name=self.name)
        self._loop_runner.start()
        # Forward the in-process sampling params. temperature/top_p/seed/
        # max_tokens are native OpenAI fields; vLLM-only knobs (top_k, min_p,
        # repetition_penalty) and the chat-template flag ride in extra_body, and
        # presence_penalty is added as a native create kwarg via extra_kwargs.
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }
        self._gen_config = GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_output_tokens,
            seed=self.seed,
            extra_kwargs={"extra_body": extra_body, "presence_penalty": self.presence_penalty},
        )
        logger.info(
            "RemoteLLMTranslation: ready (remote={}, model={})",
            self.inference_base_url,
            self.served_model_name or self.model_id,
        )

    def teardown(self) -> None:
        if self._n_processed:
            logger.info("RemoteLLMTranslation: processed {} entries", self._n_processed)
        if self._loop_runner is not None:
            if self._client is not None:
                with contextlib.suppress(Exception):
                    self._loop_runner.run(self._client.client.close(), timeout=30)
            self._loop_runner.close()
            self._loop_runner = None
        self._client = None

    # ------------------------------------------------------------------
    # Prompt / inference
    # ------------------------------------------------------------------

    def _build_messages(self, data: dict, target_lang: str, source_lang: str) -> list[dict[str, str]]:
        """Build chat messages (the server applies the template).

        Mirrors :meth:`LLMTranslationStage._format_prompt` message construction
        but returns the raw message list instead of a tokenized string.
        """
        user_content = self._translation_prompt.format_map(
            self._build_prompt_values(data, target_lang, source_lang)
        )
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _batch_timeout(self, n_requests: int) -> float:
        """Generous wall-clock cap for a whole batch, so a real hang surfaces."""
        max_retries = getattr(self._client, "max_retries", 3)
        per_request = self.request_timeout * (max_retries + 1)
        concurrency = max(1, self.max_concurrent_requests)
        waves = max(1, math.ceil(n_requests / concurrency))
        return per_request * waves + 120

    def _generate_remote(self, messages_list: list[list[dict[str, str]]]) -> list[str]:
        """Fan out one chat request per message list, preserving order."""
        model = self.served_model_name or self.model_id

        async def _one(messages: list[dict[str, str]]) -> str:
            resp = await self._client.query_model(
                messages=messages,
                model=model,
                generation_config=self._gen_config,
            )
            return (resp[0] if resp else "").strip()

        async def _all() -> list[str]:
            return await asyncio.gather(*[_one(m) for m in messages_list])

        return self._loop_runner.run(_all(), timeout=self._batch_timeout(len(messages_list)))

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        # Ray Data may pass tasks as an ndarray, so use len() not `if not tasks`.
        if len(tasks) == 0:
            return []
        if self._client is None:
            msg = "Client not initialised — setup() was not called"
            raise RuntimeError(msg)

        messages_list: list[list[dict[str, str]]] = []
        prompt_owners: list[tuple[int, str]] = []

        for task_idx, task in enumerate(tasks):
            data = task.data

            raw_targets = data.pop(self.target_lang_key, None) or []
            source_lang = data.pop(self.source_lang_key, "")

            if isinstance(raw_targets, str):
                raw_targets = [raw_targets]
            targets = list(dict.fromkeys(raw_targets))

            if not targets:
                continue

            if data.get(self.skip_me_key, ""):
                self._emit_empty_translations(task, targets, "skipped (flagged)")
                continue

            text = data.get(self.text_key, "")
            if not text or not text.strip():
                self._emit_empty_translations(task, targets, "skipped (empty text)")
                continue

            for target_lang in targets:
                messages_list.append(self._build_messages(data, target_lang, source_lang))
                prompt_owners.append((task_idx, target_lang))

        if messages_list:
            results = self._generate_remote(messages_list)

            for seq_idx, (task_idx, target_lang) in enumerate(prompt_owners):
                task = tasks[task_idx]
                translation = results[seq_idx]

                if not translation:
                    logger.warning("RemoteLLMTranslation: empty translation for target={}", target_lang)

                translations = task.data.get(self.translations_key) or {}
                translations[target_lang] = translation
                task.data[self.translations_key] = translations
                self._n_processed += 1

        logger.debug(
            "RemoteLLMTranslation: batch of {} tasks ({} translations)", len(tasks), len(messages_list)
        )
        return tasks
