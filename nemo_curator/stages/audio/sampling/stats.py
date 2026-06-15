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

"""Stats reporting for the sampling pipeline."""

from __future__ import annotations

import json
import os

from loguru import logger


def write_stats(stats: dict, output_path: str) -> None:
    """Write sampling stats to a JSON file and print a summary table to stdout.

    Parameters
    ----------
    stats:
        Nested dict returned by :func:`~nemo_curator.stages.audio.sampling.sampler.proportional_sample`:
        ``{lang: {bucket_key: {available, requested, selected}, "_totals": {...}}}``.
    output_path:
        Destination path for the JSON report (e.g. ``{output_dir}/sampling_stats.json``).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("write_stats: stats written to {}", output_path)

    # Print per-language summary table.
    header = f"{'Lang':<6} {'Available':>10} {'Quota':>10} {'Selected':>10} {'Coverage':>10}"
    print("\n" + "=" * len(header))
    print("Sampling summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for lang in sorted(stats):
        totals = stats[lang].get("_totals", {})
        available = totals.get("available", 0)
        quota = totals.get("quota", available)
        selected = totals.get("selected", 0)
        coverage = f"{selected / quota * 100:.1f}%" if quota else "n/a"

        if selected < quota:
            flag = "  [WARNING: quota not met]"
        else:
            flag = ""

        print(f"{lang:<6} {available:>10,} {quota:>10,} {selected:>10,} {coverage:>10}{flag}")

    print("=" * len(header) + "\n")

    # Per-bucket detail.
    print(f"{'Bucket':<50} {'Avail':>7} {'Req':>7} {'Sel':>7}")
    print("-" * 74)
    for lang in sorted(stats):
        for bk, counts in stats[lang].items():
            if bk == "_totals":
                continue
            avail = counts.get("available", 0)
            req = counts.get("requested", 0)
            sel = counts.get("selected", 0)
            short = bk if len(bk) <= 50 else "…" + bk[-49:]
            print(f"{short:<50} {avail:>7,} {req:>7,} {sel:>7,}")
    print()
