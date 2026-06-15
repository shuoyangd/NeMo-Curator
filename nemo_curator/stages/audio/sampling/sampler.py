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

"""Proportional quota sampling for the sampling pipeline.

For each language the user specifies a total quota.  Within that language the
quota is distributed proportionally across ``(source × length_range)`` buckets
by their cleaned size.  Buckets smaller than their allocation contribute all
available rows and the shortfall is redistributed proportionally to the
remaining buckets.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


def _allocate(bucket_sizes: dict[str, int], quota: int) -> dict[str, int]:
    """Distribute ``quota`` rows proportionally across buckets with redistribution.

    Parameters
    ----------
    bucket_sizes:
        Mapping of bucket key → available row count (all > 0).
    quota:
        Total number of rows to allocate.

    Returns
    -------
    Mapping of bucket key → allocated row count (≤ bucket size).
    The sum may be < ``quota`` when the total available is less than quota.
    """
    total_available = sum(bucket_sizes.values())
    if total_available == 0:
        return {k: 0 for k in bucket_sizes}

    effective_quota = min(quota, total_available)
    remaining_quota = effective_quota
    remaining_sizes = dict(bucket_sizes)
    allocation: dict[str, int] = {}

    # Iteratively cap over-allocated buckets and redistribute.
    while True:
        total_remaining = sum(remaining_sizes.values())
        if total_remaining == 0 or remaining_quota == 0:
            for k in remaining_sizes:
                allocation[k] = 0
            break

        # Compute proportional allocation for uncapped buckets.
        proposed: dict[str, int] = {}
        for k, size in remaining_sizes.items():
            proposed[k] = round(size / total_remaining * remaining_quota)

        # Adjust for rounding: ensure sum == remaining_quota by tweaking the
        # largest bucket.
        diff = remaining_quota - sum(proposed.values())
        if diff != 0 and proposed:
            largest = max(proposed, key=lambda k: proposed[k])
            proposed[largest] = max(0, proposed[largest] + diff)

        # Find over-allocated buckets (allocated > available).
        capped: dict[str, int] = {}
        uncapped: dict[str, int] = {}
        for k, alloc in proposed.items():
            size = remaining_sizes[k]
            if alloc >= size:
                capped[k] = size
            else:
                uncapped[k] = alloc

        allocation.update(capped)
        if not uncapped:
            break
        if not capped:
            # No bucket exceeded its size — proportional split is final.
            allocation.update(uncapped)
            break

        remaining_quota -= sum(capped.values())
        remaining_sizes = {k: remaining_sizes[k] for k in uncapped}

    return allocation


def proportional_sample(
    df: pd.DataFrame,
    language_quotas: dict[str, int],
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Sample rows proportionally from buckets within each language.

    For languages not present in ``language_quotas``, all available rows are
    kept (no quota applied).  A warning is logged for any language that could
    not meet its quota.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`~nemo_curator.stages.audio.sampling.bucketing.assign_buckets`.
        Must have columns ``source_lang`` and ``_bucket_key``.
    language_quotas:
        Mapping of ISO language code → desired row count.
    seed:
        Random seed for reproducible sampling within each bucket.

    Returns
    -------
    sampled_df:
        DataFrame of selected rows (same columns as ``df``).
    stats:
        Nested dict: ``{lang: {bucket_key: {available, requested, selected}}}``.
        Also includes a ``"_totals"`` entry per language with ``{available,
        quota, selected}``.
    """
    stats: dict[str, dict] = {}
    sampled_parts: list[pd.DataFrame] = []

    all_langs = df["source_lang"].unique().tolist()

    for lang in all_langs:
        lang_df = df[df["source_lang"] == lang]
        bucket_sizes = lang_df.groupby("_bucket_key").size().to_dict()
        quota = language_quotas.get(lang)

        lang_stats: dict[str, dict] = {}

        if quota is None:
            # No quota: keep everything.
            for bk, size in bucket_sizes.items():
                lang_stats[bk] = {"available": size, "requested": size, "selected": size}
            sampled_parts.append(lang_df)
            total_selected = len(lang_df)
        else:
            allocation = _allocate(bucket_sizes, quota)
            bucket_samples: list[pd.DataFrame] = []

            for bk, alloc in allocation.items():
                bucket_df = lang_df[lang_df["_bucket_key"] == bk]
                size = len(bucket_df)
                selected = min(alloc, size)
                if selected > 0:
                    bucket_samples.append(bucket_df.sample(n=selected, random_state=seed))
                lang_stats[bk] = {"available": size, "requested": alloc, "selected": selected}

            if bucket_samples:
                sampled_parts.append(pd.concat(bucket_samples, ignore_index=True))

            total_selected = sum(s["selected"] for s in lang_stats.values())
            total_available = sum(bucket_sizes.values())

            if total_selected < quota and total_selected < total_available:
                logger.warning(
                    "proportional_sample: lang={} quota={} but only {} rows available",
                    lang,
                    quota,
                    total_available,
                )
            elif total_selected < quota:
                logger.warning(
                    "proportional_sample: lang={} quota={} not met — selected={} (insufficient data)",
                    lang,
                    quota,
                    total_selected,
                )

        lang_stats["_totals"] = {
            "available": sum(bucket_sizes.values()),
            "quota": quota if quota is not None else sum(bucket_sizes.values()),
            "selected": total_selected,
        }
        stats[lang] = lang_stats

        logger.info(
            "proportional_sample: lang={} quota={} selected={} buckets={}",
            lang,
            quota,
            total_selected,
            len(bucket_sizes),
        )

    sampled_df = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else df.iloc[0:0].copy()
    return sampled_df, stats
