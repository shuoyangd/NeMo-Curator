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

import argparse
import os
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.translation.language_map import _normalize_code
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage

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


# ----------------------------------------------------------------------------
# Bitext filtering: CLI args + stage builders
#
# These compose the AudioTask-native marker stages from ``bitext_filters`` into
# the pre-translation (source-only) and post-translation (bitext) filter blocks.
# Heavy filter imports are function-local so importing this contract module
# (done early by the reader/writer stages) stays cheap.
# ----------------------------------------------------------------------------

# The translation field written by ``TranslationExpanderStage`` (its
# ``translation_key`` default) and the lang keys the writer/QE read per row.
_TRANSLATION_FIELD = "translation"
_SOURCE_LANG_FIELD = "source_lang"
_TARGET_LANG_FIELD = "target_lang"

QE_SCORE_FIELDS = {
    "comet-qe": "comet_qe_score",
    "cometoid-wmt23": "pymarian_qe_score",
    "cometoid-wmt23-mqm": "pymarian_mqm_qe_score",
}


def qe_cutoff_for_model(model_name: str, args: argparse.Namespace) -> float:
    """Return the keep-cutoff for one QE model from the parsed CLI args."""
    if model_name == "comet-qe":
        return args.qe_comet_cutoff
    if model_name.startswith("cometoid"):
        return args.qe_pymarian_cutoff
    msg = f"Unsupported QE model: {model_name}"
    raise ValueError(msg)


def add_bitext_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add the bitext-filtering CLI args to the translation pipeline parser.

    Args are named ``<filter>_<param>`` and grouped per filter in ``--help``.
    Per-side toggles are ``<filter>_src`` / ``<filter>_tgt`` (``--flag``/``--no-flag``):
    word count (``--wc_src`` / ``--wc_tgt``, default on), histogram (default off),
    langid (default off; needs ``--langid_model_path``). Length ratio
    (``--length_ratio``, default on), QE (``--qe``), and regex cleanup
    (``--regex_cleanup``, default on) are not per-side. All filters are mark-only:
    rows are annotated (``_skipme`` + ``additional_notes``) but never dropped, so the
    directional writer's ``.done`` counting is preserved.
    """
    shared = parser.add_argument_group("bitext filtering: resources")
    shared.add_argument("--cpu_stage_cpus", type=float, default=1.0, help="CPUs per CPU filter stage.")

    wc = parser.add_argument_group("bitext filtering: word count (wc)")
    wc.add_argument(
        "--wc_src", action=argparse.BooleanOptionalAction, default=True,
        help="Word-count filter on the source text, pre-translation (marks short sources skip). Default: on.",
    )
    wc.add_argument(
        "--wc_tgt", action=argparse.BooleanOptionalAction, default=True,
        help="Word-count filter on the translation, post-translation. Default: on.",
    )
    wc.add_argument("--wc_min_words", type=int, default=4, help="Minimum words to keep (src and tgt).")

    lr = parser.add_argument_group("bitext filtering: length ratio")
    lr.add_argument(
        "--length_ratio", action=argparse.BooleanOptionalAction, default=True,
        help="Length-ratio filter on the (source, translation) pair. Default: on.",
    )
    lr.add_argument("--length_ratio_max", type=float, default=9.0, help="Max src/tgt length ratio.")

    hist = parser.add_argument_group("bitext filtering: histogram")
    hist.add_argument(
        "--histogram_src", action=argparse.BooleanOptionalAction, default=False,
        help="NLLB histogram language check on the source, pre-translation. Default: off.",
    )
    hist.add_argument(
        "--histogram_tgt", action=argparse.BooleanOptionalAction, default=False,
        help="NLLB histogram language check on the translation, post-translation. Default: off.",
    )
    hist.add_argument("--histogram_threshold", type=float, default=0.8)
    hist.add_argument("--histogram_cache_dir", type=str, default=None)

    langid = parser.add_argument_group("bitext filtering: language id (fastText)")
    langid.add_argument(
        "--langid_src", action=argparse.BooleanOptionalAction, default=False,
        help="fastText language-id check on the source, pre-translation. Requires --langid_model_path. Default: off.",
    )
    langid.add_argument(
        "--langid_tgt", action=argparse.BooleanOptionalAction, default=False,
        help="fastText language-id check on the translation, post-translation. Requires --langid_model_path. Default: off.",
    )
    langid.add_argument(
        "--langid_model_path", type=str, default=None,
        help="Path to a fastText langid model (required when --langid_src/--langid_tgt is enabled).",
    )
    langid.add_argument("--langid_min_score", type=float, default=0.5)

    qe = parser.add_argument_group("bitext filtering: quality estimation (QE)")
    qe.add_argument(
        "--qe", action=argparse.BooleanOptionalAction, default=False,
        help="Enable COMET / Cometoid QE filtering. Default: off.",
    )
    qe.add_argument("--qe_models", nargs="+", default=["cometoid-wmt23"], choices=sorted(QE_SCORE_FIELDS))
    qe.add_argument("--qe_mode", choices=["simple", "always_en_x", "bidi"], default="always_en_x")
    qe.add_argument("--qe_cpu", action="store_true", help="Run QE on CPU (avoids GPU contention with vLLM).")
    qe.add_argument("--qe_cpus", type=float, default=1.0)
    qe.add_argument("--qe_batch_size", type=int, default=64)
    qe.add_argument("--qe_comet_cutoff", type=float, default=-0.5)
    qe.add_argument("--qe_pymarian_cutoff", type=float, default=0.6)
    qe.add_argument("--qe_pymarian_shard_size", type=int, default=5000)
    qe.add_argument("--qe_pymarian_args", type=str, default=None)

    regex = parser.add_argument_group("bitext filtering: regex cleanup")
    regex.add_argument(
        "--regex_cleanup", action=argparse.BooleanOptionalAction, default=True,
        help="Apply regex cleanup to the output text. Default: on. (Strips non-Latin/Cyrillic/Greek scripts.)",
    )
    regex.add_argument("--regex_cleanup_field", choices=["translation", "source"], default="translation")


def _candidate_langs(args: argparse.Namespace) -> list[str]:
    """Languages a row's source/target may be, for building per-language filters.

    The reader is English-centric and drops any row whose source language is
    neither ``en`` nor a configured target, so every row that reaches the filters
    has a source/target language in ``{"en"} ∪ target_langs``.
    """
    return sorted(lang for lang in ({"en"} | {_normalize_code(c) for c in args.target_langs}) if lang)


def _lang_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Directed (src_lang, tgt_lang) pairs for per-row length-ratio dispatch."""
    cand = _candidate_langs(args)
    return [(src, tgt) for src in cand for tgt in cand if src != tgt]


def _require_fasttext_model(args: argparse.Namespace, flag_name: str, enabled: bool) -> None:
    """A langid stage needs a model path; fail loudly if it was enabled without one."""
    if enabled and not args.langid_model_path:
        msg = f"{flag_name} requires --langid_model_path to be set"
        raise ValueError(msg)


def build_source_prefilter_stages(args: argparse.Namespace) -> list[ProcessingStage]:
    """Pre-translation, source-only markers placed between the reader and the LLM.

    These set ``_skipme`` so ``LLMTranslationStage`` skips vLLM on rejected rows
    (GPU savings). They score the source text under ``args.text_key`` and are
    dispatched per row by the row's ``source_lang`` (so the right word splitter /
    histogram / expected language is used for each direction).
    """
    from nemo_curator.stages.audio.translation.bitext_filters import AudioTaskFieldMarker, AudioTaskMarkerChain
    from nemo_curator.stages.text.filters.heuristic import WordCountFilter

    _require_fasttext_model(args, "--langid_src", args.langid_src)
    cand = _candidate_langs(args)
    markers: list[Any] = []

    if args.wc_src:
        markers.append(
            AudioTaskFieldMarker(
                filters_by_lang={lang: WordCountFilter(min_words=args.wc_min_words, lang=lang) for lang in cand},
                lang_key=_SOURCE_LANG_FIELD,
                text_key=args.text_key,
                score_key="src_word_count",
                name="src_word_count",
            )
        )

    if args.histogram_src:
        from nemo_curator.stages.text.filters.histogram import HistogramFilter

        markers.append(
            AudioTaskFieldMarker(
                filters_by_lang={
                    lang: HistogramFilter(
                        lang=lang, threshold=args.histogram_threshold, cache_dir=args.histogram_cache_dir
                    )
                    for lang in cand
                },
                lang_key=_SOURCE_LANG_FIELD,
                text_key=args.text_key,
                score_key="src_histogram",
                name="src_histogram",
            )
        )

    if args.langid_src:
        from nemo_curator.stages.text.filters.fasttext import FastTextLangId

        # One FastTextLangId per candidate language (each loads the shared model
        # file at setup(); the candidate set is small = en + target langs).
        markers.append(
            AudioTaskFieldMarker(
                filters_by_lang={
                    lang: FastTextLangId(
                        model_path=args.langid_model_path,
                        min_langid_score=args.langid_min_score,
                        expected_lang=lang,
                    )
                    for lang in cand
                },
                lang_key=_SOURCE_LANG_FIELD,
                text_key=args.text_key,
                score_key="src_langid",
                name="src_langid",
            )
        )

    if not markers:
        return []
    # Fuse all source markers into one stage so a row is handed off once, not once per filter.
    return [AudioTaskMarkerChain(markers=markers, name="source_filters").with_(resources=Resources(cpus=args.cpu_stage_cpus))]


def build_bitext_filter_stages(args: argparse.Namespace) -> list[ProcessingStage]:
    """Post-translation bitext markers placed between the expander and the writer.

    Operate on the source text (``args.text_key``) and the translated text
    (``translation``). All mark-only; QE and regex cleanup are opt-in.
    """
    from nemo_curator.stages.audio.translation.bitext_filters import (
        AudioTaskBitextMarker,
        AudioTaskFieldMarker,
        AudioTaskMarkerChain,
        AudioTaskQEMarker,
        AudioTaskRegexModifier,
    )
    from nemo_curator.stages.text.filters.bitext import LengthRatioFilter
    from nemo_curator.stages.text.filters.heuristic import WordCountFilter

    _require_fasttext_model(args, "--langid_tgt", args.langid_tgt)
    cpu = Resources(cpus=args.cpu_stage_cpus)
    cand = _candidate_langs(args)
    pairs = _lang_pairs(args)

    # Per-row mark-only markers, fused into one chain stage (one hand-off per row).
    markers: list[Any] = []

    if args.wc_tgt:
        markers.append(
            AudioTaskFieldMarker(
                filters_by_lang={lang: WordCountFilter(min_words=args.wc_min_words, lang=lang) for lang in cand},
                lang_key=_TARGET_LANG_FIELD,
                text_key=_TRANSLATION_FIELD,
                score_key="tgt_word_count",
                name="tgt_word_count",
            )
        )

    # Length ratio is a bitext (both-sides) filter, not separable into src/tgt.
    if args.length_ratio:
        markers.append(
            AudioTaskBitextMarker(
                filters_by_pair={
                    (src, tgt): LengthRatioFilter(max_ratio=args.length_ratio_max, src_lang=src, tgt_lang=tgt)
                    for src, tgt in pairs
                },
                src_lang_key=_SOURCE_LANG_FIELD,
                tgt_lang_key=_TARGET_LANG_FIELD,
                src_key=args.text_key,
                tgt_key=_TRANSLATION_FIELD,
                score_key="length_ratio",
                name="length_ratio",
            )
        )

    if args.histogram_tgt:
        from nemo_curator.stages.text.filters.histogram import HistogramFilter

        markers.append(
            AudioTaskFieldMarker(
                filters_by_lang={
                    lang: HistogramFilter(
                        lang=lang, threshold=args.histogram_threshold, cache_dir=args.histogram_cache_dir
                    )
                    for lang in cand
                },
                lang_key=_TARGET_LANG_FIELD,
                text_key=_TRANSLATION_FIELD,
                score_key="tgt_histogram",
                name="tgt_histogram",
            )
        )

    if args.langid_tgt:
        from nemo_curator.stages.text.filters.fasttext import FastTextLangId

        markers.append(
            AudioTaskFieldMarker(
                filters_by_lang={
                    lang: FastTextLangId(
                        model_path=args.langid_model_path,
                        min_langid_score=args.langid_min_score,
                        expected_lang=lang,
                    )
                    for lang in cand
                },
                lang_key=_TARGET_LANG_FIELD,
                text_key=_TRANSLATION_FIELD,
                score_key="tgt_langid",
                name="tgt_langid",
            )
        )

    stages: list[ProcessingStage] = [
        AudioTaskMarkerChain(markers=markers, name="target_filters").with_(resources=cpu)
    ]

    # QE is a separate batched (GPU) stage; regex cleanup runs on every row (incl. skipped).
    if args.qe:
        for model_name in args.qe_models:
            model_kwargs: dict[str, Any] = {}
            if model_name.startswith("cometoid"):
                model_kwargs["shard_size"] = args.qe_pymarian_shard_size
                if args.qe_pymarian_args:
                    model_kwargs["marian_args"] = args.qe_pymarian_args
            gpu = not args.qe_cpu
            qe_resources = Resources(cpus=args.qe_cpus, gpus=1.0) if gpu else Resources(cpus=args.qe_cpus)
            stages.append(
                AudioTaskQEMarker(
                    model_name=model_name,
                    cutoff=qe_cutoff_for_model(model_name, args),
                    mode=args.qe_mode,
                    gpu=gpu,
                    src_key=args.text_key,
                    tgt_key=_TRANSLATION_FIELD,
                    src_lang_key=_SOURCE_LANG_FIELD,
                    tgt_lang_key=_TARGET_LANG_FIELD,
                    score_key=QE_SCORE_FIELDS[model_name],
                    name=f"qe:{model_name}",
                    model_kwargs=model_kwargs,
                ).with_(resources=qe_resources, batch_size=args.qe_batch_size)
            )

    if args.regex_cleanup:
        cleanup_field = args.text_key if args.regex_cleanup_field == "source" else _TRANSLATION_FIELD
        stages.append(AudioTaskRegexModifier(field_key=cleanup_field).with_(resources=cpu))

    return stages
