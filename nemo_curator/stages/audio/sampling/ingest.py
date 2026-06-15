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

"""Manifest ingestion for the sampling pipeline.

Reads one or more JSONL manifests, strips all original fields, and returns a
lightweight DataFrame with only the columns needed for bucketing and sampling.
Each row carries a back-reference to its source file and line number so the
translation pipeline can reconstruct the full record without duplicating any
metadata.
"""

from __future__ import annotations

import glob as _glob
import json
import os

import pandas as pd
from loguru import logger

from nemo_curator.stages.audio.translation.manifest_reader import _derive_input_root, _relative_shard_key


def _discover_manifests(manifest: str | list[str]) -> list[str]:
    """Expand a path / directory / glob / list into a flat list of JSONL files."""
    inputs = manifest if isinstance(manifest, list) else [manifest]
    paths: list[str] = []
    for p in inputs:
        if not p:
            continue
        if os.path.isfile(p):
            paths.append(os.path.abspath(p))
        elif os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    if f.endswith((".jsonl", ".json")):
                        paths.append(os.path.abspath(os.path.join(root, f)))
        elif any(ch in p for ch in "*?["):
            paths.extend(sorted(os.path.abspath(x) for x in _glob.glob(p)))
    return paths


def _is_truthy(value: object) -> bool:
    """Return True for values the translation pipeline treats as skip signals."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no")
    return bool(value)


def ingest_manifests(
    manifest: str | list[str],
    text_key: str = "pnc_text",
    source_lang_key: str = "source_lang",
    source_dataset_key: str | None = None,
    skip_me_key: str = "_skipme",
) -> pd.DataFrame:
    """Read JSONL manifests and return a stripped DataFrame.

    All original fields are discarded. Only the back-reference columns and the
    columns required for bucketing and sampling are kept:

    ``_manifest_path``
        Absolute path to the source JSONL file.
    ``_line_index``
        0-based raw line number within that file (counting all lines, including
        blank ones) so callers can reconstruct the exact record without
        ambiguity (e.g. ``open(path).readlines()[line_index]``).
    ``_source_dataset``
        Dataset name: the value of ``source_dataset_key`` when present in the
        row, otherwise the manifest's path relative to the common input root
        (extension stripped, e.g. ``sourceA/manifest_0``). Using the relative
        path keeps same-stem manifests in different subfolders distinct.
    ``source_lang``
        Source language ISO code (value of ``source_lang_key``).
    ``_text``
        Transcript text (value of ``text_key``). Used for word-count bucketing
        only — not written to the final output JSONL.

    Skipped rows:
    * Blank lines.
    * Rows where ``skip_me_key`` is truthy (same semantics as the translation
      pipeline).
    * Rows where the transcript text is empty or whitespace-only.
    * Rows where the source language field is missing or empty.

    Parameters
    ----------
    manifest:
        Single file path, directory, glob pattern, or list thereof.
    text_key:
        JSONL field containing the transcript (default: ``pnc_text``).
    source_lang_key:
        JSONL field containing the ISO language code (default: ``source_lang``).
    source_dataset_key:
        Optional JSONL field to use as the dataset name. When absent or not
        present in a row, the manifest's relative path is used instead.
    skip_me_key:
        JSONL field that marks rows to skip (default: ``_skipme``).

    Returns
    -------
    pd.DataFrame with columns:
        ``_manifest_path``, ``_line_index``, ``_source_dataset``,
        ``source_lang``, ``_text``.
    """
    paths = _discover_manifests(manifest)
    if not paths:
        logger.warning("ingest_manifests: no JSONL files found for manifest={}", manifest)
        return pd.DataFrame(columns=["_manifest_path", "_line_index", "_source_dataset", "source_lang", "_text"])

    # Common input root so each manifest's source label is its relative path
    # (extension stripped), keeping same-stem files in different subfolders distinct.
    input_root = _derive_input_root(manifest)

    records: list[dict] = []
    total_skipped = 0

    for filepath in paths:
        default_dataset = _relative_shard_key(filepath, input_root)
        file_skipped = 0
        file_kept = 0

        with open(filepath, encoding="utf-8") as fh:
            for line_index, raw_line in enumerate(fh):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line.strip())
                except json.JSONDecodeError:
                    logger.warning("ingest_manifests: invalid JSON at {}:{}, skipping", filepath, line_index)
                    file_skipped += 1
                    continue

                # Skip rows flagged by skip_me_key.
                if skip_me_key and _is_truthy(row.get(skip_me_key, False)):
                    file_skipped += 1
                    continue

                text = row.get(text_key, "")
                if not isinstance(text, str) or not text.strip():
                    file_skipped += 1
                    continue

                lang = row.get(source_lang_key, "")
                if not lang:
                    file_skipped += 1
                    continue

                dataset = row.get(source_dataset_key, default_dataset) if source_dataset_key else default_dataset

                records.append(
                    {
                        "_manifest_path": filepath,
                        "_line_index": line_index,
                        "_source_dataset": str(dataset),
                        "source_lang": str(lang),
                        "_text": text.strip(),
                    }
                )
                file_kept += 1

        total_skipped += file_skipped
        logger.info(
            "ingest_manifests: {} -> kept={}, skipped={}",
            os.path.basename(filepath),
            file_kept,
            file_skipped,
        )

    df = pd.DataFrame(records, columns=["_manifest_path", "_line_index", "_source_dataset", "source_lang", "_text"])
    logger.info(
        "ingest_manifests: total rows={}, skipped={}, files={}",
        len(df),
        total_skipped,
        len(paths),
    )
    return df
