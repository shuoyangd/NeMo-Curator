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

"""Shared contract helpers for the translation pipeline.

Two things live here so the stages cannot drift apart:

1. The per-direction shard file contract. Every output file is named::

    {output_dir}/{shard_key}_{src}-{tgt}.jsonl[.done]

   where ``shard_key`` may itself contain subdirectories mirrored from the input
   manifest tree (e.g. ``en/m1``). The reader (resume + pre-flight) and the writer
   (recovery + completion) all build and parse this contract.

2. The internal scratch row keys. The reader writes them, the LLM stage consumes
   and pops them, and the expander strips them before output, so they never reach
   the final manifest. They are a fixed internal contract, intentionally not
   user-configurable.
"""

from __future__ import annotations

import os

from nemo_curator.stages.audio.translation.language_map import _normalize_code

JSONL_EXT = ".jsonl"
DONE_EXT = ".jsonl.done"

# Internal scratch keys (never appear in the output manifest).
SOURCE_LANG_NAME_KEY = "source_lang_name"
TRANSLATE_TO_KEY = "translate_to"
# Internal handoff: LLMTranslationStage writes this dict, TranslationExpander
# reads it and strips it before output, so it never reaches the final manifest.
TRANSLATIONS_KEY = "translations"


def direction_key(src_code: str, tgt_code: str) -> str:
    """Build the normalised ``"{src}-{tgt}"`` direction key (e.g. ``"en-de"``)."""
    return f"{_normalize_code(src_code)}-{_normalize_code(tgt_code)}"


def handle_key(shard_key: str, direction: str) -> str:
    """Build the relative handle key ``"{shard_key}_{direction}"`` (no extension)."""
    return f"{shard_key}_{direction}"


def output_paths(output_dir: str, shard_key: str, direction: str) -> tuple[str, str]:
    """Return the ``(jsonl_path, done_path)`` pair for one (shard, direction)."""
    jsonl = os.path.join(output_dir, f"{handle_key(shard_key, direction)}{JSONL_EXT}")
    return jsonl, jsonl + ".done"


def parse_handle_key(relpath_no_ext: str) -> tuple[str, str] | None:
    """Split a ``"{shard_key}_{src}-{tgt}"`` relative key into its parts.

    The last ``_`` separates the (possibly subdirectoried) shard key from the
    direction; the direction must contain ``-``. Returns ``(shard_key, direction)``
    or ``None`` when ``relpath_no_ext`` does not match the contract.
    """
    parts = relpath_no_ext.rsplit("_", 1)
    if len(parts) == 2 and "-" in parts[1]:
        return parts[0], parts[1]
    return None
