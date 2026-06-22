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

"""AudioTask-native bitext filtering stages for the translation pipeline.

These stages mark (never drop) rows, so the row count per ``(shard, direction)``
stays equal to ``direction_counts`` and ``DirectionalShardedWriterStage`` can
still rename each shard's ``.jsonl`` to ``.jsonl.done``. Marking sets the
``_skipme`` boolean gate and records a per-stage reason via ``set_note`` into
``additional_notes`` — the same convention ``LLMTranslationStage`` uses — so
each filter's decision is queryable without string parsing.

Each marker reuses the existing string-level scoring objects
(``DocumentFilter`` / ``BitextFilter`` from ``nemo_curator.stages.text.filters``)
and the QE ``QEModel`` wrappers, applied per row, so there is no
``DocumentBatch`` round-trip and every ``AudioTask`` keeps its ``_metadata``
(``_shard_key`` / ``direction_counts``) intact for the writer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.audio.translation.language_map import _normalize_code
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.stages.text.filters.bitext import BitextFilter
    from nemo_curator.stages.text.filters.doc_filter import DocumentFilter

SKIP_KEY = "_skipme"
NOTES_KEY = "additional_notes"

# Fixed regex cleanup applied to the translated field (ported verbatim from the
# bitext-cleaning recipe). The substitutions run in order and the result is
# stripped.
REGEX_PARAMS_LIST: list[dict[str, str]] = [
    {"pattern": "’", "repl": "'"},
    {"pattern": "‘", "repl": "'"},
    {"pattern": "—", "repl": "-"},
    {"pattern": "–", "repl": "-"},
    {"pattern": "-", "repl": "-"},
    {"pattern": "_", "repl": " "},
    {"pattern": "——", "repl": "-"},
    {"pattern": "Ё", "repl": "Е"},
    {"pattern": "ё", "repl": "е"},
    {"pattern": "♫", "repl": " "},
    {"pattern": "♪", "repl": " "},
    {"pattern": "♬", "repl": " "},
    {"pattern": "♩", "repl": " "},
    {"pattern": "♭", "repl": " "},
    {"pattern": r"\|", "repl": " "},
    {"pattern": ";", "repl": ","},
    {"pattern": r"\[[^\]]*\]", "repl": ""},
    {"pattern": r" ?\([^\)]+\)", "repl": ""},
    {"pattern": r" ?{[^}]+}", "repl": ""},
    {
        "pattern": "[^ !$%',-.0123456789;?ABCDEFGHIJKLMNOPQRSßTUVWXYŸZabcdefghijklmnopqrsẞtuvwxyÿz¡£¿ÀÁÂÃÄÅÆÇÈÉÊÌÍÎÑÒÓÔÕÖØÙÚÜÝàáâãäåæçèéêëìíîïñòóôõöøùúûüýĀāĂăĄąĆćĊċČčĎďĐđĒēĖėĘęĚěĠġĢģĦħĪīĮįĶķĹĺĻļĽľŁłŃńŅņŇňŐőŒœŔŕŘřŚśŠšŤťŪūŮůŰűŲųŹźŻżŽžȘșȚțΆΈΉΌΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩάέήίαβγδεζηθικλμνξοπρστυφχψωϊόύώЁЄІЇАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїҐґ€₴₽/:]",
        "repl": "",
    },
    {"pattern": r"\s+\.", "repl": "."},
    {"pattern": r"\?+", "repl": "?"},
    {"pattern": r"\.+", "repl": "."},
    {"pattern": ",+", "repl": ","},
    {"pattern": "!+", "repl": "!"},
    {"pattern": r"\s+", "repl": " "},
    {"pattern": r" ([.,!?])", "repl": r"\1"},
    {"pattern": " '", "repl": "'"},
    {"pattern": "' ", "repl": "'"},
    {"pattern": "'+", "repl": "'"},
    {"pattern": " - ", "repl": "-"},
]


def _is_skipped(task: AudioTask, skip_key: str = SKIP_KEY) -> bool:
    """True when an upstream filter already marked this row for skipping."""
    return bool(task.data.get(skip_key, 0))


def _mark_skip(task: AudioTask, stage_name: str, detail: str, notes_key: str = NOTES_KEY, skip_key: str = SKIP_KEY) -> None:
    """Set the ``_skipme`` gate and record the per-stage reason in ``additional_notes``."""
    task.data[skip_key] = 1
    set_note(task.data, stage_name, detail, notes_key)


class RegexSubstitutionModifier:
    """Apply a fixed sequence of regex substitutions to one text value."""

    def __init__(self, regex_params_list: list[dict[str, str]]) -> None:
        self._substitutions = [(re.compile(p["pattern"]), p["repl"]) for p in regex_params_list]

    def modify_document(self, text: object) -> str:
        value = "" if text is None else str(text)
        for pattern, repl in self._substitutions:
            value = pattern.sub(repl, value)
        return value.strip()


@dataclass
class AudioTaskFieldMarker(ProcessingStage[AudioTask, AudioTask]):
    """Mark rows by scoring one text field with a single-field ``DocumentFilter``.

    Two modes:
      * fixed: pass ``filter_obj`` — every row scored with the same filter.
      * per-row language dispatch: pass ``filters_by_lang`` (``{lang_code:
        DocumentFilter}``) + ``lang_key``. Each row is scored with the filter for
        its own ``_normalize_code(row[lang_key])``. This keeps language-aware
        filters (word count splitter, histogram, fastText) correct across a
        bidirectional run (en->X and X->en) where the source/target language
        varies per row. Rows whose language has no entry are left unscored.

    Reuses the filter's ``score_document`` / ``keep_document``. Rows already
    marked (``_skipme`` set) are not re-scored. The numeric score is written to
    ``score_key`` for every evaluated row; rejected rows additionally get
    ``_skipme=1`` and an ``additional_notes[name]`` reason.
    """

    filter_obj: DocumentFilter | None = None
    filters_by_lang: dict[str, DocumentFilter] | None = None
    lang_key: str | None = None
    text_key: str = ""
    score_key: str = ""
    name: str = "field_marker"
    skip_key: str = SKIP_KEY
    notes_key: str = NOTES_KEY

    def _all_filters(self) -> list[DocumentFilter]:
        if self.filters_by_lang is not None:
            return list(self.filters_by_lang.values())
        return [self.filter_obj] if self.filter_obj is not None else []

    def _pick_filter(self, task: AudioTask) -> DocumentFilter | None:
        if self.filters_by_lang is not None:
            lang = _normalize_code(str(task.data.get(self.lang_key, "")))
            return self.filters_by_lang.get(lang)
        return self.filter_obj

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.score_key]

    def requires_setup(self) -> bool:
        return any(hasattr(f, "load_model") or hasattr(f, "load_tokenizer") for f in self._all_filters())

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: self.requires_setup()}

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None) -> None:
        for filter_obj in self._all_filters():
            if hasattr(filter_obj, "model_check_or_download"):
                filter_obj.model_check_or_download()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        for filter_obj in self._all_filters():
            if hasattr(filter_obj, "load_model"):
                filter_obj.load_model()
            if hasattr(filter_obj, "load_tokenizer"):
                filter_obj.load_tokenizer()

    def apply_row(self, task: AudioTask) -> bool:
        """Score and (maybe) mark one row. Returns True if the row is now skipped.

        Used both by ``process_batch`` and by ``AudioTaskMarkerChain`` to run
        several markers in one stage. Returns True for an already-marked row so a
        chain short-circuits the remaining markers.
        """
        if _is_skipped(task, self.skip_key):
            return True
        filter_obj = self._pick_filter(task)
        if filter_obj is None:
            return False
        text = str(task.data.get(self.text_key, "") or "")
        score = filter_obj.score_document(text)
        task.data[self.score_key] = score
        if not filter_obj.keep_document(score):
            _mark_skip(task, self.name, f"{self.score_key}={score}", self.notes_key, self.skip_key)
            return True
        task.data.setdefault(self.skip_key, 0)
        return False

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            self.apply_row(task)
        return tasks


@dataclass
class AudioTaskBitextMarker(ProcessingStage[AudioTask, AudioTask]):
    """Mark rows by scoring a source/target pair with a ``BitextFilter``.

    Two modes:
      * fixed: pass ``filter_obj``.
      * per-row language dispatch: pass ``filters_by_pair`` (``{(src_lang,
        tgt_lang): BitextFilter}``) + ``src_lang_key``/``tgt_lang_key``. Each row
        is scored with the filter for its own
        ``(_normalize_code(row[src_lang_key]), _normalize_code(row[tgt_lang_key]))``,
        so length-ratio word splitting stays correct per direction. Rows whose
        language pair has no entry are left unscored.

    Reuses ``score_bitext`` / ``keep_bitext`` per row; same mark-only / skip-
    already-marked semantics as ``AudioTaskFieldMarker``.
    """

    filter_obj: BitextFilter | None = None
    filters_by_pair: dict[tuple[str, str], BitextFilter] | None = None
    src_lang_key: str | None = None
    tgt_lang_key: str | None = None
    src_key: str = ""
    tgt_key: str = ""
    score_key: str = ""
    name: str = "bitext_marker"
    skip_key: str = SKIP_KEY
    notes_key: str = NOTES_KEY

    def _pick_filter(self, task: AudioTask) -> BitextFilter | None:
        if self.filters_by_pair is not None:
            src_lang = _normalize_code(str(task.data.get(self.src_lang_key, "")))
            tgt_lang = _normalize_code(str(task.data.get(self.tgt_lang_key, "")))
            return self.filters_by_pair.get((src_lang, tgt_lang))
        return self.filter_obj

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.src_key, self.tgt_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.score_key]

    def requires_setup(self) -> bool:
        return False  # BitextFilter (length ratio) loads no model

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None) -> None:
        return None

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        return None

    def apply_row(self, task: AudioTask) -> bool:
        """Score and (maybe) mark one row; returns True if the row is now skipped."""
        if _is_skipped(task, self.skip_key):
            return True
        filter_obj = self._pick_filter(task)
        if filter_obj is None:
            return False
        src = str(task.data.get(self.src_key, "") or "")
        tgt = str(task.data.get(self.tgt_key, "") or "")
        score = filter_obj.score_bitext(src, tgt)
        task.data[self.score_key] = score
        if not filter_obj.keep_bitext(score):
            _mark_skip(task, self.name, f"{self.score_key}={score}", self.notes_key, self.skip_key)
            return True
        task.data.setdefault(self.skip_key, 0)
        return False

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            self.apply_row(task)
        return tasks


@dataclass
class AudioTaskMarkerChain(ProcessingStage[AudioTask, AudioTask]):
    """Run several field/bitext markers in a single pass over each row.

    Fuses what would otherwise be separate per-row CPU marker stages into one
    stage, so a row is handed between pipeline operators once instead of N times.
    Each member marker (``AudioTaskFieldMarker`` / ``AudioTaskBitextMarker``)
    still writes its own ``score_key`` and ``additional_notes`` reason; the chain
    applies them in order and short-circuits as soon as one marks the row
    ``_skipme`` (preserving the standalone "first reason wins" behavior). Members
    are held as scoring objects only — their resources/specs are ignored; the
    chain's own ``.with_(...)``/``ray_stage_spec`` govern scheduling.
    """

    markers: list[Any] = field(default_factory=list)
    name: str = "marker_chain"
    skip_key: str = SKIP_KEY

    def inputs(self) -> tuple[list[str], list[str]]:
        keys: list[str] = []
        for marker in self.markers:
            keys.extend(marker.inputs()[1])
        return [], sorted(set(keys))

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], sorted({marker.score_key for marker in self.markers})

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: any(m.requires_setup() for m in self.markers)}

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:
        for marker in self.markers:
            marker.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        for marker in self.markers:
            marker.setup(worker_metadata)

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            if _is_skipped(task, self.skip_key):
                continue
            for marker in self.markers:
                if marker.apply_row(task):
                    break
        return tasks


@dataclass
class AudioTaskQEMarker(ProcessingStage[AudioTask, AudioTask]):
    """Mark rows below a quality-estimation cutoff, batching the model call.

    ``setup()`` loads the QE model once per worker (``COMET`` or ``PyMarian``
    Cometoid). ``process_batch`` gathers every not-yet-skipped row, builds QE
    inputs (per-row direction handling for ``always_en_x`` / ``bidi``), runs a
    single ``model.predict(...)``, writes ``score_key``, and marks rows scoring
    below ``cutoff``. This is the only filter that needs a real batch.
    """

    model_name: str
    cutoff: float
    mode: str = "always_en_x"
    gpu: bool = True
    src_key: str = "pnc_text"
    tgt_key: str = "translation"
    src_lang_key: str = "source_lang"
    tgt_lang_key: str = "target_lang"
    score_key: str = "qe_score"
    name: str = "qe_filter"
    skip_key: str = SKIP_KEY
    notes_key: str = NOTES_KEY
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    model: Any = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.src_key, self.tgt_key, self.src_lang_key, self.tgt_lang_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.score_key]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        from nemo_curator.stages.text.filters.qe import QualityEstimationFilter

        model_cls = QualityEstimationFilter.SUPPORTED_MODELS[self.model_name]
        self.model = model_cls.load_model(self.model_name, gpu=self.gpu, **self.model_kwargs)

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self.model is None:
            msg = "QE model not initialised — setup() was not called"
            raise RuntimeError(msg)

        pending = [task for task in tasks if not _is_skipped(task, self.skip_key)]
        if not pending:
            return tasks

        scores = self._score(pending)
        for task, score in zip(pending, scores, strict=True):
            value = float(score)
            task.data[self.score_key] = value
            if value < self.cutoff:
                _mark_skip(task, self.name, f"{self.score_key}={value:.4f}<{self.cutoff}", self.notes_key, self.skip_key)
            else:
                task.data.setdefault(self.skip_key, 0)
        return tasks

    def _score(self, pending: list[AudioTask]) -> list[float]:
        srcs = [str(task.data.get(self.src_key, "") or "") for task in pending]
        tgts = [str(task.data.get(self.tgt_key, "") or "") for task in pending]

        if self.mode == "simple":
            inputs = [self.model.wrap_qe_input(src, tgt) for src, tgt in zip(srcs, tgts, strict=True)]
            return [float(score) for score in self.model.predict(inputs)]

        if self.mode == "always_en_x":
            src_langs = [str(task.data.get(self.src_lang_key, "")).lower() for task in pending]
            tgt_langs = [str(task.data.get(self.tgt_lang_key, "")).lower() for task in pending]
            inputs = [
                self.model.wrap_qe_input(src, tgt, reverse=(sl != "en" and tl == "en"))
                for src, tgt, sl, tl in zip(srcs, tgts, src_langs, tgt_langs, strict=True)
            ]
            return [float(score) for score in self.model.predict(inputs)]

        # bidi: average forward and reverse scores.
        forward = [self.model.wrap_qe_input(src, tgt) for src, tgt in zip(srcs, tgts, strict=True)]
        reverse = [self.model.wrap_qe_input(src, tgt, reverse=True) for src, tgt in zip(srcs, tgts, strict=True)]
        all_scores = [float(score) for score in self.model.predict(forward + reverse)]
        mid = len(forward)
        return [(fwd + rev) / 2 for fwd, rev in zip(all_scores[:mid], all_scores[mid:], strict=True)]


@dataclass
class AudioTaskRegexModifier(ProcessingStage[AudioTask, AudioTask]):
    """Apply the fixed ``REGEX_PARAMS_LIST`` cleanup to one text field per row."""

    field_key: str = "translation"
    regex_params: list[dict[str, str]] = field(default_factory=lambda: REGEX_PARAMS_LIST)
    name: str = "regex_cleanup"

    def __post_init__(self) -> None:
        self._modifier = RegexSubstitutionModifier(self.regex_params)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.field_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.field_key]

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            task.data[self.field_key] = self._modifier.modify_document(task.data.get(self.field_key, ""))
        return tasks


__all__ = [
    "AudioTaskBitextMarker",
    "AudioTaskFieldMarker",
    "AudioTaskMarkerChain",
    "AudioTaskQEMarker",
    "AudioTaskRegexModifier",
    "REGEX_PARAMS_LIST",
    "RegexSubstitutionModifier",
]
