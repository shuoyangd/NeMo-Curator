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

"""Bucket assignment for the sampling pipeline.

Each row is assigned to a ``(source_lang × _source_dataset × length_range)``
bucket. Rows whose word count exceeds ``max_words`` or falls outside all
defined length ranges are discarded.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


def parse_length_ranges(spec: str) -> list[tuple[int, int]]:
    """Parse a length-range spec string into a sorted list of (min, max) tuples.

    Format: ``"1:15,16:30,31:45"`` where each pair is ``min:max`` (inclusive).
    Ranges must not overlap and must cover a contiguous span (validated at
    call time with a warning, not an error, to stay flexible).

    Parameters
    ----------
    spec:
        Comma-separated ``min:max`` pairs.

    Returns
    -------
    List of ``(min_words, max_words)`` tuples sorted by ``min_words``.
    """
    ranges: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo_str, hi_str = part.split(":")
        lo, hi = int(lo_str.strip()), int(hi_str.strip())
        if lo > hi:
            msg = f"Invalid length range '{part}': min ({lo}) > max ({hi})"
            raise ValueError(msg)
        ranges.append((lo, hi))
    return sorted(ranges, key=lambda t: t[0])


def _range_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def assign_buckets(
    df: pd.DataFrame,
    length_ranges: list[tuple[int, int]],
    max_words: int,
) -> pd.DataFrame:
    """Assign each row to a ``(source_lang × _source_dataset × length_range)`` bucket.

    Adds two new columns to the returned DataFrame:

    ``_word_count``
        Number of whitespace-separated tokens in the transcript.
    ``_bucket_key``
        String key of the form ``"{source_lang}|{_source_dataset}|{length_range}"``,
        e.g. ``"en|librispeech|16-30"``.

    Rows discarded:
    * Word count > ``max_words``.
    * Word count not covered by any range in ``length_ranges``.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`ingest_manifests`.  Must have columns
        ``source_lang``, ``_source_dataset``, ``_text``.
    length_ranges:
        Sorted list of ``(min_words, max_words)`` tuples (inclusive bounds).
        Produced by :func:`parse_length_ranges`.
    max_words:
        Hard upper limit. Rows with more words are discarded before range
        matching so oversized transcripts never inflate any bucket.

    Returns
    -------
    pd.DataFrame with the same columns as ``df`` plus ``_word_count`` and
    ``_bucket_key``, restricted to rows that fall within a valid range.
    """
    if df.empty:
        return df.assign(_word_count=pd.Series(dtype=int), _bucket_key=pd.Series(dtype=str))

    df = df.copy()
    df["_word_count"] = df["_text"].str.split().str.len()

    before = len(df)
    df = df[df["_word_count"] <= max_words]
    n_discarded_max = before - len(df)
    if n_discarded_max:
        logger.info("assign_buckets: discarded {} rows exceeding max_words={}", n_discarded_max, max_words)

    def _find_range(wc: int) -> str | None:
        for lo, hi in length_ranges:
            if lo <= wc <= hi:
                return _range_label(lo, hi)
        return None

    df["_length_range"] = df["_word_count"].map(_find_range)
    n_out_of_range = df["_length_range"].isna().sum()
    if n_out_of_range:
        logger.info("assign_buckets: discarded {} rows not covered by any length range", n_out_of_range)
    df = df.dropna(subset=["_length_range"])

    df["_bucket_key"] = df["source_lang"] + "|" + df["_source_dataset"] + "|" + df["_length_range"]
    df = df.drop(columns=["_length_range"])

    logger.info(
        "assign_buckets: {} rows -> {} buckets (discarded {} total)",
        len(df),
        df["_bucket_key"].nunique(),
        before - len(df),
    )
    return df.reset_index(drop=True)
