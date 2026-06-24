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

"""Lock-free, per-worker translation cache with a driver-side merge.

Speech corpora contain many recurring **short** utterances ("Yes.", "Thank you.", numbers,
boilerplate). Re-translating identical short sources wastes GPU. A previous attempt used a single
shared SQLite (``diskcache``) consulted by every worker; on Lustre its writer lock serialized all
workers and killed parallelism. This module avoids any shared store **during** the run:

* Each worker keeps its translations in an **in-memory** dict (zero contention; disk is touched
  only at ``setup()``/``teardown()``).
* At ``teardown()`` each worker **dumps only the entries it touched this run** to its **own** file
  (no two writers ever touch the same file).
* The driver, after the pipeline finishes, calls :func:`merge_translation_cache` to fold all per-worker dumps and the
  previous canonical cache into one canonical file (union of keys, **summed** occurrence counter),
  written atomically.

So it is a map-reduce: workers map (accumulate locally), the driver reduces (merge -> atomic write).

Value / limitation: this gives a **cross-run warm start** for common short phrases. Within a single
run there is no cross-worker sharing (a string seen on two workers is translated by both) -- the
accepted tradeoff for being lock-free.

Cache key: ``sha256(model_id | src_lang | tgt_lang | text)`` -- the target language matters because
the translation differs per target, and the source language captures direction. With a fixed
seed/prompt the translation is deterministic, so model + langs + source text is enough.

Entry format (JSONL):

* canonical line: ``{"k": <hash>, "src": <source text>, "v": <translation>, "n": <total count>}``
* per-worker dump line: ``{"k": <hash>, "src": <source text>, "v": <translation>, "dn": <count this run>}``

``n`` is an **informational** occurrence counter (how many times the key was requested, summed across
all workers and runs); it is NOT used to evict (no LFU).
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import shutil
import socket

from loguru import logger

CANONICAL_FILENAME = "canonical.jsonl"
DUMPS_DIRNAME = "dumps"
_KEY_SEP = "\x1f"  # unit separator -- unlikely to appear in model ids / language names / text


def cache_key(model_id: str, src_lang: str, tgt_lang: str, text: str) -> str:
    """Stable content hash for one (model, direction, source text) translation request."""
    raw = _KEY_SEP.join((model_id or "", src_lang or "", tgt_lang or "", text or ""))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_cacheable(text: str, max_chars: int) -> bool:
    """Only short, non-empty sources are cached (long texts always translate)."""
    if not text:
        return False
    return 0 < len(text.strip()) <= max_chars


def canonical_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, CANONICAL_FILENAME)


def dump_path(cache_dir: str, run_id: str, host: str, pid: int) -> str:
    return os.path.join(cache_dir, DUMPS_DIRNAME, run_id, f"worker_{host}_{pid}.jsonl")


class WorkerTranslationCache:
    """Per-worker, in-memory translation cache.

    Built in ``LLMTranslationStage.setup()`` and torn down (dumped) in ``teardown()``. Loads the
    canonical cache as a warm start, serves lookups from memory during the run, and records every
    request (for the occurrence counter) so the dump carries per-run deltas the driver can sum.
    """

    def __init__(self, cache_dir: str, run_id: str, max_chars: int = 40) -> None:
        self.cache_dir = cache_dir
        self.run_id = run_id
        self.max_chars = max_chars
        self._entries: dict[str, str] = {}  # canonical warm start: key -> translation
        self._touched: dict[str, int] = {}  # key -> request count THIS run
        self._src: dict[str, str] = {}  # key -> source text (for the dump)
        self._val: dict[str, str] = {}  # key -> translation (hit value or freshly recorded)

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Read the canonical cache into memory. Fail open (empty cache) on any error."""
        path = canonical_path(self.cache_dir)
        try:
            if not os.path.exists(path):
                logger.info("TranslationCache: no canonical cache at {} (cold start)", path)
                return
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    self._entries[entry["k"]] = entry["v"]
            logger.info("TranslationCache: loaded {} entries from {}", len(self._entries), path)
        except Exception as exc:  # noqa: BLE001 - cache is best-effort; never block the run
            logger.warning("TranslationCache: failed to load {}: {}; starting empty", path, exc)
            self._entries = {}

    # ---------------------------------------------------------------- lookup
    def is_cacheable(self, text: str) -> bool:
        return is_cacheable(text, self.max_chars)

    def lookup(
        self, model_id: str, src_lang: str, tgt_lang: str, text: str
    ) -> tuple[str | None, str | None]:
        """Return ``(key, value)``.

        ``(None, None)`` -> not cacheable (caller translates normally, no bookkeeping).
        ``(key, value)`` -> cache hit (use ``value``, skip the LLM).
        ``(key, None)`` -> cacheable miss (caller translates, then calls :meth:`record`).

        Counts the request and records the source text either way (hit or miss).
        """
        if not self.is_cacheable(text):
            return None, None
        key = cache_key(model_id, src_lang, tgt_lang, text)
        self._touched[key] = self._touched.get(key, 0) + 1
        self._src[key] = text
        value = self._entries.get(key)
        if value is not None:
            self._val[key] = value
            return key, value
        return key, None

    def record(self, key: str, translation: str) -> None:
        """Record a freshly translated (miss) result so it lands in this worker's dump."""
        if key and translation:
            self._val[key] = translation

    # ------------------------------------------------------------------ dump
    def dump(self) -> None:
        """Write this worker's touched entries to its own per-run file. Best-effort."""
        if not self._touched:
            return
        path = dump_path(self.cache_dir, self.run_id, socket.gethostname(), os.getpid())
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            written = 0
            with open(path, "w", encoding="utf-8") as f:
                for key, count in self._touched.items():
                    value = self._val.get(key)
                    if not value:  # miss that produced no (non-empty) translation -> skip
                        continue
                    f.write(
                        json.dumps(
                            {"k": key, "src": self._src.get(key, ""), "v": value, "dn": count},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
            logger.info("TranslationCache: dumped {} entries to {}", written, path)
        except Exception as exc:  # noqa: BLE001 - best-effort; a lost dump just means cache misses
            logger.warning("TranslationCache: failed to dump to {}: {}", path, exc)


def merge_translation_cache(cache_dir: str, max_entries: int = 0) -> int:
    """Fold all per-worker dumps + the previous canonical into one canonical file.

    Single-process, run by the driver after the pipeline finishes -- no locks, since every worker
    has already written its own file. Unions keys (deduped by hash), **sums** the occurrence counter
    ``n`` across dumps, writes the result atomically, and deletes the consumed dump dirs. Returns the
    number of entries in the merged canonical. Never raises -- all errors are logged.
    """
    canonical = canonical_path(cache_dir)
    merged: dict[str, dict] = {}

    # 1. previous canonical first (its src/v win on collision; values are deterministic anyway)
    try:
        if os.path.exists(canonical):
            with open(canonical, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    merged[entry["k"]] = {
                        "src": entry.get("src", ""),
                        "v": entry["v"],
                        "n": int(entry.get("n", 0)),
                    }
    except Exception as exc:  # noqa: BLE001
        logger.warning("TranslationCache.merge: failed to read {}: {}", canonical, exc)

    # 2. fold in every per-worker dump (this run + any leftover prior runs)
    dump_files = sorted(glob.glob(os.path.join(cache_dir, DUMPS_DIRNAME, "*", "*.jsonl")))
    consumed_dirs: set[str] = set()
    for path in dump_files:
        consumed_dirs.add(os.path.dirname(path))
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    key = entry["k"]
                    delta = int(entry.get("dn", 0))
                    if key in merged:
                        merged[key]["n"] += delta
                    else:
                        merged[key] = {"src": entry.get("src", ""), "v": entry["v"], "n": delta}
        except Exception as exc:  # noqa: BLE001 - skip a corrupt dump, keep the rest
            logger.warning("TranslationCache.merge: skipping bad dump {}: {}", path, exc)

    # 3. optional crude safety cap (insertion order; the counter is NOT an eviction key)
    if max_entries and len(merged) > max_entries:
        logger.warning(
            "TranslationCache.merge: {} entries > cap {}; keeping first {}",
            len(merged),
            max_entries,
            max_entries,
        )
        merged = dict(itertools.islice(merged.items(), max_entries))

    # 4. atomic publish: write temp then rename over canonical
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = canonical + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for key, entry in merged.items():
                f.write(
                    json.dumps(
                        {"k": key, "src": entry["src"], "v": entry["v"], "n": entry["n"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        os.replace(tmp, canonical)
    except Exception as exc:  # noqa: BLE001 - never fail the run on a cache write
        logger.warning("TranslationCache.merge: failed to write {}: {}", canonical, exc)
        return len(merged)

    # 5. drop the consumed dump dirs so they don't accumulate across runs
    for directory in consumed_dirs:
        shutil.rmtree(directory, ignore_errors=True)

    logger.info("TranslationCache: merged to {} entries at {}", len(merged), canonical)
    return len(merged)


__all__ = [
    "WorkerTranslationCache",
    "cache_key",
    "canonical_path",
    "dump_path",
    "is_cacheable",
    "merge_translation_cache",
]
