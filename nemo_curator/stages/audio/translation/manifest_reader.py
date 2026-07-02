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

"""Per-file manifest reader for the translation pipeline.

One input manifest = one shard.  Composite stage that wraps
``FilePartitioningStage`` for input discovery (file / dir / glob / list) and
``TranslationManifestReaderStage`` for the actual per-file read and
per-row language enrichment.

Shard identity
--------------
``shard_key`` is the manifest path relative to the inferred input root with
its extension stripped, so input subdirectories are preserved (e.g. ``en/m1``);
it falls back to the bare filename stem for a single file / flat input.  The
writer materialises one output file per ``(shard_key, direction)`` pair at
``{output_dir}/{shard_key}_{src}-{tgt}.jsonl[.done]``.

Resume semantics
----------------
A shard is fully done when every direction in ``direction_counts`` for that
shard has a corresponding ``.jsonl.done`` file in ``output_dir``.  On the
next run the reader, after reading the manifest and computing
``direction_counts``:

  * skips the whole file if all expected ``.done`` files are present, or
  * deletes any partial ``{shard_key}_{src}-{tgt}.jsonl`` (no ``.done``
    sibling) for an expected direction, then re-emits all rows so the
    writer's append mode starts from a clean slate.
"""

from __future__ import annotations

import glob as _glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.translation.language_map import LANGUAGE_MAP, _normalize_code, lang_code_to_name
from nemo_curator.stages.audio.translation.translation_utils import (
    LOW_QUALITY_REASON,
    SOURCE_LANG_CODE_KEY,
    SOURCE_LANG_NAME_KEY,
    TRANSLATE_TO_KEY,
    TRANSLATION_SKIP_KEY,
    output_paths,
)
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, FileGroupTask, _EmptyTask


# ----------------------------------------------------------------------------
# Pre-flight helper
# ----------------------------------------------------------------------------


def _resolve_input_paths(manifest_path: str | list[str]) -> list[str]:
    """Expand a path / list-of-paths / dir / glob into a flat list of files.

    Mirrors what ``FilePartitioningStage`` accepts so the pre-flight check
    sees the same inputs that the pipeline will.  Filters to ``.jsonl`` and
    ``.json`` extensions.
    """
    inputs = manifest_path if isinstance(manifest_path, list) else [manifest_path]
    resolved: list[str] = []
    for p in inputs:
        if not p:
            continue
        if os.path.isfile(p):
            resolved.append(p)
        elif os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in files:
                    if f.endswith((".jsonl", ".json")):
                        resolved.append(os.path.join(root, f))
        elif any(ch in p for ch in "*?["):
            resolved.extend(_glob.glob(p))
        # else: missing path; ignored (pre-flight will fall back to False).
    return resolved


def _derive_input_root(manifest_path: str | list[str]) -> str | None:
    """Infer a common input root directory from the ``--manifest`` argument.

    Used to preserve the input subdirectory hierarchy in the output: each
    shard's relative path is computed against this root. For each input we take
    its directory (a directory as-is, a glob's non-wildcard prefix directory, or
    a file's parent). Returns that single directory, the ``os.path.commonpath``
    of several, or ``None`` when no directory can be determined. A single file
    or a same-directory list therefore yields the file's own directory, so the
    relative shard key collapses to the bare stem (flat output).
    """
    inputs = manifest_path if isinstance(manifest_path, list) else [manifest_path]
    dirs: list[str] = []
    for p in inputs:
        if not p:
            continue
        if os.path.isdir(p):
            dirs.append(os.path.abspath(p))
        elif any(ch in p for ch in "*?["):
            prefix = p
            for ch in "*?[":
                prefix = prefix.split(ch, 1)[0]
            dirs.append(os.path.abspath(os.path.dirname(prefix)))
        else:
            dirs.append(os.path.abspath(os.path.dirname(p)))
    if not dirs:
        return None
    if len(dirs) == 1:
        return dirs[0]
    try:
        return os.path.commonpath(dirs)
    except ValueError:
        return None


def _relative_shard_key(manifest_path: str, input_root: str | None) -> str:
    """Return the shard key (relative path without extension) for a manifest.

    When ``input_root`` is set and the manifest lives under it, the key is the
    relative path with the ``.jsonl`` / ``.json`` extension stripped (preserving
    subdirectories). Otherwise it falls back to the bare filename stem.
    """
    if input_root:
        rel = os.path.relpath(os.path.abspath(manifest_path), os.path.abspath(input_root))
        if not rel.startswith(".."):
            for ext in (".jsonl", ".json"):
                if rel.endswith(ext):
                    return rel[: -len(ext)]
            return rel
    return Path(manifest_path).stem


def _is_low_quality(val: object) -> bool:
    """True only when an input ``high_quality`` value is explicitly false-y.

    Missing / True / anything unrecognised -> keep (not low quality). Accepts bool,
    0/0.0, and the strings ``"false"``/``"0"``/``"no"`` (case-insensitive).
    """
    if isinstance(val, bool):
        return not val
    if isinstance(val, (int, float)):
        return val == 0
    if isinstance(val, str):
        return val.strip().lower() in ("false", "0", "no")
    return False


def _row_target_codes(src_norm: str, target_codes_norm: list[str]) -> list[str]:
    """Target-language codes a source row translates to (En->X / X->En only).

    Mirrors the direction rules that build ``direction_counts``: an English
    source expands to every non-English target; a source that is itself a
    configured target goes to English; anything else yields no direction.
    Shared by :meth:`TranslationManifestReaderStage._row_targets` and the
    :func:`all_shards_done` pre-flight so the expected direction set stays in
    sync between the reader and the resume check.
    """
    if src_norm == "en":
        return [c for c in target_codes_norm if c != "en"]
    if src_norm and src_norm in frozenset(target_codes_norm):
        return ["en"]
    return []


def all_shards_done(
    manifest_path: str | list[str],
    output_dir: str,
    target_lang_codes: list[str] | None = None,
    source_lang_key: str = SOURCE_LANG_CODE_KEY,
) -> bool:
    """Pre-flight check used to skip ``pipeline.run()`` entirely on full resume.

    Returns True only when **every expected ``(shard, direction)`` output has a
    ``.jsonl.done``**. The expected directions per shard are computed exactly
    like the reader's ``direction_counts``: each input manifest is read, and for
    every row the source language is expanded via :func:`_row_target_codes`
    (En->X / X->En). A shard is complete only if *all* of its expected
    directions are ``.done`` — a partial or entirely missing direction leaves
    its ``.done`` absent, so the check returns False and the pipeline runs,
    letting the reader resume per direction.

    This deliberately reads the manifests (only reached when ``output_dir``
    already exists, i.e. a resume) rather than inferring completeness from the
    output tree alone: a direction that produced no file at all — e.g. one whose
    partial ``.jsonl`` the reader deleted before an interrupted rewrite — is
    invisible in the output tree and would otherwise be wrongly counted as done.

    Returns False when the check cannot be made confidently (missing/empty
    inputs, no ``target_lang_codes``, or an unreadable/malformed manifest): the
    pipeline then runs and the reader still skips completed shards correctly.
    """
    if not os.path.isdir(output_dir):
        return False
    if not target_lang_codes:
        # Cannot compute the expected direction set without the targets.
        return False

    paths = _resolve_input_paths(manifest_path)
    if not paths:
        return False

    target_codes_norm = [_normalize_code(c) for c in target_lang_codes]
    input_root = _derive_input_root(manifest_path)

    for path in paths:
        shard_key = _relative_shard_key(path, input_root)
        expected: set[str] = set()
        try:
            fs, resolved = url_to_fs(path)
            with fs.open(resolved, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    if not raw_line.strip():
                        continue
                    src_raw = json.loads(raw_line.strip()).get(source_lang_key, "")
                    src_norm = _normalize_code(src_raw) if src_raw else ""
                    for tgt_norm in _row_target_codes(src_norm, target_codes_norm):
                        expected.add(f"{src_norm}-{tgt_norm}")
        except (OSError, ValueError):
            # Unreadable / malformed manifest: cannot confirm -> run the pipeline.
            return False

        # A manifest with no translatable rows yields no output; the reader
        # skips it too, so an empty expected set does not block completion.
        for direction in expected:
            _, done_path = output_paths(output_dir, shard_key, direction)
            if not os.path.exists(done_path):
                return False

    return True


# ----------------------------------------------------------------------------
# Per-file reader
# ----------------------------------------------------------------------------


@dataclass
class TranslationManifestReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    """Read one manifest file and emit one ``AudioTask`` per non-empty line.

    Per-row enrichment (mirrors what the upstream LLM stage consumes):
      * ``source_lang_name`` — display name resolved from the row's
        ``source_lang`` code via ``LANGUAGE_MAP``.
      * ``translate_to``    — list of target display names per the
        direction rules (En -> X, X -> En only).

    Each emitted ``AudioTask`` carries:
      * ``_metadata["_shard_key"]``       — manifest path relative to the input
        root, extension stripped (subdirectories preserved); the bare stem when
        input is flat.
      * ``_metadata["_shard_total"]``     — total non-empty lines.
      * ``_metadata["direction_counts"]`` — dict ``"{src}-{tgt}" -> int``
        the writer uses to know when each direction is complete.

    Resume:
      * if every expected direction's ``.jsonl.done`` exists, the whole
        shard is skipped;
      * any partial ``.jsonl`` for an expected direction is deleted before
        re-emitting rows (writer appends, so clean slate is required).
    """

    name: str = "TranslationManifestReader"
    output_dir: str = ""
    target_lang_codes: list[str] = field(default_factory=list)
    source_lang_key: str = "source_lang"
    source_lang_name_key: str = SOURCE_LANG_NAME_KEY
    translate_to_key: str = TRANSLATE_TO_KEY
    input_skip_key: str = "_skipme"
    high_quality_key: str = "high_quality"
    input_root: str | None = None

    _target_codes_norm: list[str] = field(default_factory=list, init=False, repr=False)
    _target_set: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)
    _en_to_x_codes: list[str] = field(default_factory=list, init=False, repr=False)
    _en_to_x_names: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.target_lang_codes:
            msg = "TranslationManifestReaderStage: target_lang_codes must be non-empty"
            raise ValueError(msg)

        self._target_codes_norm = [_normalize_code(c) for c in self.target_lang_codes]
        self._target_set = frozenset(self._target_codes_norm)
        self._en_to_x_codes = [c for c in self._target_codes_norm if c != "en"]
        # Fails loudly if a configured target code is not in LANGUAGE_MAP.
        self._en_to_x_names = [LANGUAGE_MAP[c] for c in self._en_to_x_codes]

    def _row_targets(self, src_norm: str) -> list[str]:
        return _row_target_codes(src_norm, self._target_codes_norm)

    def _row_target_names(self, src_norm: str) -> list[str]:
        if src_norm == "en":
            return list(self._en_to_x_names)
        if src_norm and src_norm in self._target_set:
            return [LANGUAGE_MAP["en"]]
        return []

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.source_lang_name_key, self.translate_to_key]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        results: list[AudioTask] = []

        for manifest in task.data:
            shard_key = _relative_shard_key(manifest, self.input_root)
            fs, resolved = url_to_fs(manifest)

            entries: list[dict[str, Any]] = []
            with fs.open(resolved, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    if not raw_line.strip():
                        continue
                    entries.append(json.loads(raw_line.strip()))

            if not entries:
                logger.warning("TranslationManifestReader: empty manifest {}, skipping", manifest)
                continue

            # Per-row enrichment + direction_counts in a single pass. Rows whose
            # source language yields no valid translation direction (neither
            # English nor a configured target) are dropped here: emitting them
            # would push a target-less row through LLMTranslation, which skips it
            # without writing a ``translations`` key and then fails
            # TranslationExpander's input validation.
            direction_counts: dict[str, int] = {}
            kept_entries: list[dict[str, Any]] = []
            for row in entries:
                src_raw = row.get(self.source_lang_key, "")
                src_norm = _normalize_code(src_raw) if src_raw else ""

                target_names = self._row_target_names(src_norm)
                if not target_names:
                    continue

                if src_raw:
                    row[self.source_lang_name_key] = lang_code_to_name(src_raw)
                # Canonicalize the source-lang code into a fixed key so downstream
                # stages (filters, writer) don't depend on the input column name.
                row[SOURCE_LANG_CODE_KEY] = src_raw
                # Seed the working skip flag from the original input flag (left
                # untouched). It is a reason STRING like the input ``_skipme``: copy
                # the original reason when set, else "" (keep). All filters and the LLM
                # gate on translation_skipme; downstream filters overwrite "" with their
                # own reason on the first rejection (first reason wins).
                original_skip = row.get(self.input_skip_key)
                row[TRANSLATION_SKIP_KEY] = str(original_skip) if original_skip else ""
                # A false-y input ``high_quality`` flags the row as low quality: skip it (no
                # translation) with a distinct reason so finalize gives it the minimum QE score
                # and a "skipped: low_quality" note. Only fills an empty gate (input _skipme wins).
                if not row[TRANSLATION_SKIP_KEY] and _is_low_quality(row.get(self.high_quality_key)):
                    row[TRANSLATION_SKIP_KEY] = LOW_QUALITY_REASON
                row[self.translate_to_key] = target_names

                for tgt_norm in self._row_targets(src_norm):
                    key = f"{src_norm}-{tgt_norm}"
                    direction_counts[key] = direction_counts.get(key, 0) + 1

                kept_entries.append(row)

            if not kept_entries:
                logger.info(
                    "TranslationManifestReader: no translatable rows in {} "
                    "(shard_key={}), skipping",
                    manifest,
                    shard_key,
                )
                continue

            # Resume: skip whole shard if every expected direction is .done.
            # Otherwise delete any partial .jsonl (no .done sibling) so the
            # writer's append mode starts from a clean slate.
            if self.output_dir and direction_counts:
                direction_files = {
                    d: output_paths(self.output_dir, shard_key, d) for d in direction_counts
                }
                if all(os.path.exists(done) for _, done in direction_files.values()):
                    logger.info(
                        "TranslationManifestReader: skipping completed shard {} "
                        "({} direction(s) all .done)",
                        shard_key,
                        len(direction_files),
                    )
                    continue

                for partial, done_path in direction_files.values():
                    if os.path.exists(done_path):
                        continue
                    if os.path.exists(partial):
                        try:
                            os.remove(partial)
                            logger.info(
                                "TranslationManifestReader: removed partial {}",
                                partial,
                            )
                        except OSError as exc:
                            logger.warning(
                                "TranslationManifestReader: failed to remove partial {}: {}",
                                partial,
                                exc,
                            )

            shard_total = len(kept_entries)
            metadata_template: dict[str, Any] = {
                **task._metadata,
                "_shard_key": shard_key,
                "_shard_total": shard_total,
                "direction_counts": dict(direction_counts),
            }

            for entry in kept_entries:
                results.append(
                    AudioTask(
                        data=entry,
                        _metadata=dict(metadata_template),
                        _stage_perf=list(task._stage_perf),
                    )
                )

            logger.info(
                "TranslationManifestReader: loaded {} entries from {} "
                "(shard_key={}, direction_counts={})",
                shard_total,
                manifest,
                shard_key,
                direction_counts,
            )

        return results


# ----------------------------------------------------------------------------
# Composite stage
# ----------------------------------------------------------------------------


@dataclass
class TranslationManifestReader(CompositeStage[_EmptyTask, AudioTask]):
    """Composite stage: file discovery + per-file translation reader.

    Decomposes into ``FilePartitioningStage`` followed by
    ``TranslationManifestReaderStage``.

    Args:
        manifest_path:        Single path, list of paths, directory, or
                              glob.  ``FilePartitioningStage`` discovers
                              files matching ``file_extensions``.
        output_dir:           Output directory used by
                              ``DirectionalShardedWriterStage``; required
                              for resume support.
        target_lang_codes:    ISO 639-1 codes of all target languages.
        source_lang_key:      Row key holding the source-language ISO code
                              (default: ``"source_lang"``).
        source_lang_name_key: Row key to write the resolved source display
                              name into (default: ``"source_lang_name"``).
        translate_to_key:     Row key to write the per-row list of target
                              display names into (default: ``"translate_to"``).
        files_per_partition:  Files per ``FilePartitioningStage`` partition
                              (default: ``1`` — one manifest per partition,
                              which makes shard == file).
        file_extensions:      Extensions to filter when discovering
                              (default: ``[".jsonl", ".json"]``).
        storage_options:      fsspec storage options for cloud inputs.
    """

    name: str = "TranslationManifestReader"
    manifest_path: str | list[str] = ""
    output_dir: str = ""
    target_lang_codes: list[str] = field(default_factory=list)
    source_lang_key: str = "source_lang"
    source_lang_name_key: str = SOURCE_LANG_NAME_KEY
    translate_to_key: str = TRANSLATE_TO_KEY
    input_skip_key: str = "_skipme"
    high_quality_key: str = "high_quality"
    files_per_partition: int | None = 1
    file_extensions: list[str] | None = None
    storage_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__init__()
        if not self.manifest_path:
            msg = "TranslationManifestReader: manifest_path is required"
            raise ValueError(msg)
        if not self.target_lang_codes:
            msg = "TranslationManifestReader: target_lang_codes must be non-empty"
            raise ValueError(msg)
        if not self.output_dir:
            msg = "TranslationManifestReader: output_dir is required for resume support"
            raise ValueError(msg)

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.manifest_path,
                files_per_partition=self.files_per_partition,
                file_extensions=self.file_extensions or [".jsonl", ".json"],
                storage_options=self.storage_options,
            ),
            TranslationManifestReaderStage(
                output_dir=self.output_dir,
                target_lang_codes=self.target_lang_codes,
                source_lang_key=self.source_lang_key,
                source_lang_name_key=self.source_lang_name_key,
                translate_to_key=self.translate_to_key,
                input_skip_key=self.input_skip_key,
                high_quality_key=self.high_quality_key,
                input_root=_derive_input_root(self.manifest_path),
            ),
        ]

    def get_description(self) -> str:
        parts = [f"Read translation JSONL manifests from {self.manifest_path}"]
        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        return ", ".join(parts)
