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

"""Speech translation data curation — sampling pipeline.

Subsamples a diverse multilingual speech-transcript dataset from one or more
JSONL manifest sources before the expensive LLM translation step.  Diversity
is achieved through structured bucketing and proportional quota sampling — no
embeddings and no LLMs required.

Pipeline
--------
::

    Ingest & tag
        Read JSONL manifests; skip _skipme rows and empty transcripts; strip
        all original fields; retain only a back-reference to each row's source
        file and line number plus the fields needed for bucketing.

    Bucket assignment
        Group rows into (source_lang × source_dataset × length_range) buckets.
        Rows exceeding max_words or not covered by any length range are
        discarded.

    Proportional sampling
        For each language, distribute the per-language quota proportionally
        across its buckets by size.  Buckets smaller than their allocation
        contribute all available rows; the shortfall is redistributed
        proportionally to the remaining buckets.

Output
------
``{output_dir}/{relative_manifest_path}``
    One output JSONL per source manifest, mirroring the input subdirectory
    hierarchy so same-stem files in different subfolders don't collide
    (e.g. ``en/manifest_0.jsonl`` and ``de/manifest_0.jsonl`` stay separate).
    Each line contains the five fields needed by downstream pipelines:
    ``{"_manifest_path": "...", "_line_index": 42, "pnc_text": "...", "length": 11, "source_lang": "en"}``.

``{output_dir}/sampling_stats.json``
    Machine-readable per-bucket and per-language stats:
    ``{lang: {bucket_key: {available, requested, selected}, "_totals": {...}}}``.

Example
-------
::

    python run_sampling_pipeline.py \\
        --manifest /data/manifests \\
        --output_dir /data/sampled \\
        --language_quotas "en:10000,fr:5000,de:8000" \\
        --length_ranges "1:50,51:100,101:150,151:200,201:250,251:300" \\
        --max_chars 300
"""

from __future__ import annotations

import argparse
import json
import os

from loguru import logger

from nemo_curator.stages.audio.sampling import assign_buckets, ingest_manifests, proportional_sample, write_stats
from nemo_curator.stages.audio.sampling.bucketing import parse_length_ranges
from nemo_curator.stages.audio.translation.manifest_reader import _derive_input_root


def _parse_language_quotas(spec: str) -> dict[str, int]:
    """Parse ``"en:10000,fr:5000"`` into ``{"en": 10000, "fr": 5000}``."""
    quotas: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lang, count = part.split(":")
        quotas[lang.strip()] = int(count.strip())
    return quotas


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Speech translation data curation — sampling pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ I/O
    ap.add_argument(
        "--manifest",
        type=str,
        required=True,
        help=(
            "Path to JSONL manifest(s). Accepts a single file, a directory "
            "(scanned recursively for *.jsonl/*.json), or a glob pattern."
        ),
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Output directory. One JSONL per source manifest is written here, "
            "preserving the input subdirectory hierarchy. sampling_stats.json "
            "is also written here."
        ),
    )

    # ------------------------------------------------------------------ Field keys
    ap.add_argument(
        "--text_key",
        type=str,
        default="pnc_text",
        help="JSONL field containing the transcript text.",
    )
    ap.add_argument(
        "--source_lang_key",
        type=str,
        default="source_lang",
        help="JSONL field containing the source language ISO code.",
    )
    ap.add_argument(
        "--source_dataset_key",
        type=str,
        default=None,
        help=(
            "Optional JSONL field to use as the dataset name. "
            "When absent, the manifest filename stem is used."
        ),
    )
    ap.add_argument(
        "--skip_me_key",
        type=str,
        default="_skipme",
        help=(
            "JSONL field that marks rows to skip. When truthy (non-empty string "
            "or boolean True), the row is excluded. Same semantics as the "
            "translation pipeline."
        ),
    )

    # ------------------------------------------------------------------ Bucketing
    ap.add_argument(
        "--length_ranges",
        type=str,
        default="1:50,51:100,101:150,151:200,201:250,251:300",
        help=(
            "Comma-separated character-count brackets as 'min:max' pairs (inclusive). "
            "E.g. '1:50,51:100,101:150'. Rows not covered by any bracket are discarded."
        ),
    )
    ap.add_argument(
        "--max_chars",
        type=int,
        default=300,
        help="Discard transcripts with more than this many characters.",
    )

    # ------------------------------------------------------------------ Sampling
    ap.add_argument(
        "--language_quotas",
        type=str,
        default=None,
        help=(
            "Per-language row quotas as 'lang:count' pairs, e.g. 'en:10000,fr:5000,de:8000'. "
            "Languages absent from this list keep all available rows."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling within each bucket.",
    )

    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    length_ranges = parse_length_ranges(args.length_ranges)
    language_quotas = _parse_language_quotas(args.language_quotas) if args.language_quotas else {}

    logger.info("Sampling pipeline starting.")
    logger.info("  manifest      : {}", args.manifest)
    logger.info("  output_dir    : {}", args.output_dir)
    logger.info("  length_ranges : {}", length_ranges)
    logger.info("  max_chars     : {}", args.max_chars)
    logger.info("  language_quotas: {}", language_quotas)
    logger.info("  seed          : {}", args.seed)

    # Step 1: ingest.
    df = ingest_manifests(
        manifest=args.manifest,
        text_key=args.text_key,
        source_lang_key=args.source_lang_key,
        source_dataset_key=args.source_dataset_key,
        skip_me_key=args.skip_me_key,
    )
    if df.empty:
        logger.error("No rows ingested — check --manifest path and field keys.")
        return

    # Step 2: bucket assignment.
    df = assign_buckets(df, length_ranges=length_ranges, max_chars=args.max_chars)
    if df.empty:
        logger.error("No rows remain after bucketing — check --length_ranges and --max_chars.")
        return

    # Step 3: proportional sampling.
    sampled_df, stats = proportional_sample(df, language_quotas=language_quotas, seed=args.seed)

    # Step 4: write one output JSONL per source manifest, preserving the input
    # subdirectory hierarchy so same-stem files in different subfolders don't collide.
    input_root = _derive_input_root(args.manifest)
    total_written = 0
    for manifest_path, group in sampled_df.groupby("_manifest_path"):
        if input_root:
            rel = os.path.relpath(manifest_path, input_root)
        else:
            rel = os.path.basename(manifest_path)
        output_path = os.path.join(args.output_dir, rel)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for _, row in group.iterrows():
                record = {
                    "_manifest_path": row["_manifest_path"],
                    "_line_index": int(row["_line_index"]),
                    args.text_key: row["_text"],
                    "length": int(row["_char_count"]),
                    "source_lang": row["source_lang"],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("wrote {} rows -> {}", len(group), output_path)
        total_written += len(group)
    logger.info("Total written: {} rows across {} manifests", total_written, sampled_df["_manifest_path"].nunique())

    # Step 5: write stats.
    stats_path = os.path.join(args.output_dir, "sampling_stats.json")
    write_stats(stats, stats_path)


if __name__ == "__main__":
    main()
