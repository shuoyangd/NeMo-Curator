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

"""On-disk LFU cache for short-text translations.

``TranslationCache`` wraps :mod:`diskcache` so ``LLMTranslationStage`` can avoid
re-translating identical short source utterances (very common in speech corpora:
"Yes.", "Thank you.", numbers, boilerplate). Design:

* **One cache directory per language pair** (``{cache_dir}/{src_iso}-{tgt_iso}``),
  each a separate ``diskcache.Cache`` with ``least-frequently-used`` eviction so
  frequently-seen translations stay hot and rare ones are evicted under the size
  limit. ``en->x`` pairs get a larger budget (English source dominates volume).
* **Short texts only** — callers gate on :meth:`is_cacheable`.
* **Key** = ``sha256(model_id | source_text)``; the pair is encoded by the
  directory, so the model + source text fully identify an entry.
* **Fail-open** — if ``diskcache`` is missing, the directory is unwritable, or a
  language name can't be mapped to an ISO code, every operation degrades to a
  miss / no-op and the caller just translates normally.

``diskcache`` is SQLite-backed and multi-process safe, so the same on-disk cache
is shared across Ray actors and across runs without extra coordination.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.audio.translation.language_map import name_to_code

if TYPE_CHECKING:
    from diskcache import FanoutCache

# diskcache uses 1 GiB as its own default; we expose larger, pair-specific budgets.
_GIB = 1 << 30


@dataclass
class TranslationCache:
    """Per-language-pair on-disk LFU cache for short translations (fail-open).

    Each pair uses a :class:`diskcache.FanoutCache` (not a plain ``Cache``): the
    translation actors are separate processes sharing one on-disk cache per pair,
    and a plain ``Cache`` is a single SQLite database with one writer lock. Under
    the ``least-frequently-used`` policy *every* ``get`` is a write (it bumps the
    access count), so all replicas serialize on that one lock and stall each other
    — leaving the GPUs idle. ``FanoutCache`` shards keys across ``shards`` separate
    SQLite databases, so distinct keys never contend; and a short ``timeout`` makes
    the rare same-shard collision fail open to a miss (caught below) instead of
    blocking a GPU actor for the 60 s diskcache default.

    Networked filesystems (Lustre/NFS): diskcache defaults to ``journal_mode=wal``,
    which **does not work on Lustre/NFS** — WAL needs a shared-memory ``-shm`` mmap
    and locking those filesystems don't provide, so SQLite ops hang. We therefore
    force ``journal_mode=delete`` (rollback journal + POSIX locks, networked-FS safe)
    and disable the SQLite mmap. For best performance put ``cache_dir`` on node-local
    disk; on Lustre it still requires the mount to honor file locks (``-o flock``).
    """

    cache_dir: str
    max_chars: int = 40
    size_limit: int = _GIB           # budget per x->en pair
    size_limit_en: int = 5 * _GIB    # budget per en->x pair (English source dominates)
    eviction_policy: str = "least-frequently-used"
    shards: int = 8                  # SQLite DBs per pair; spreads the writer lock
    timeout: float = 0.1             # s; contended op fails open (miss/no-op) vs blocking the GPU
    journal_mode: str = "delete"     # NOT wal: wal hangs on Lustre/NFS (needs -shm mmap)

    _ok: bool = field(default=False, init=False, repr=False)
    _caches: dict[tuple[str, str], "FanoutCache"] = field(default_factory=dict, init=False, repr=False)

    def open(self) -> None:
        """Prepare the cache root and confirm diskcache is importable (fail-open)."""
        try:
            import diskcache  # noqa: F401

            os.makedirs(self.cache_dir, exist_ok=True)
            self._ok = True
            logger.info("TranslationCache enabled at {} (max_chars={})", self.cache_dir, self.max_chars)
        except Exception as e:  # noqa: BLE001 - any failure disables the cache, never the run
            self._ok = False
            logger.warning("TranslationCache disabled (translating without cache): {}", e)

    def is_cacheable(self, text: str) -> bool:
        """True if the cache is live and ``text`` is a non-empty short source."""
        return self._ok and 0 < len(text.strip()) <= self.max_chars

    @staticmethod
    def key(model_id: str | None, text: str) -> str:
        """Stable key for ``(model, source_text)``; the pair is the directory."""
        return hashlib.sha256(f"{model_id}\x1f{text}".encode()).hexdigest()

    def _pair_cache(self, src_iso: str, tgt_iso: str) -> "FanoutCache":
        """Memoized per-pair sharded cache (en-source pairs get the larger budget)."""
        pair = (src_iso, tgt_iso)
        cache = self._caches.get(pair)
        if cache is None:
            from diskcache import FanoutCache

            limit = self.size_limit_en if src_iso == "en" else self.size_limit
            cache = FanoutCache(
                directory=os.path.join(self.cache_dir, f"{src_iso}-{tgt_iso}"),
                shards=self.shards,
                timeout=self.timeout,
                size_limit=limit,
                eviction_policy=self.eviction_policy,
                sqlite_journal_mode=self.journal_mode,  # delete, not wal (Lustre/NFS safe)
                sqlite_mmap_size=0,                      # mmap also unreliable on networked FS
            )
            self._caches[pair] = cache
        return cache

    def _iso_pair(self, src_lang: str, tgt_lang: str) -> tuple[str, str] | None:
        """Map the stage's display names to ISO codes; None if either is unknown."""
        try:
            return name_to_code(src_lang), name_to_code(tgt_lang)
        except KeyError:
            return None

    def get(self, src_lang: str, tgt_lang: str, model_id: str | None, text: str) -> str | None:
        """Return a cached translation (bumping its LFU count) or None on miss."""
        if not self._ok:
            return None
        pair = self._iso_pair(src_lang, tgt_lang)
        if pair is None:
            return None
        try:
            return self._pair_cache(*pair).get(self.key(model_id, text))
        except Exception as e:  # noqa: BLE001 - a cache error must not fail the row
            logger.warning("TranslationCache get failed (treating as miss): {}", e)
            return None

    def set(self, src_lang: str, tgt_lang: str, model_id: str | None, text: str, translation: str) -> None:
        """Store a non-empty translation for the pair (no-op on any failure)."""
        if not self._ok or not translation:
            return
        pair = self._iso_pair(src_lang, tgt_lang)
        if pair is None:
            return
        try:
            self._pair_cache(*pair).set(self.key(model_id, text), translation)
        except Exception as e:  # noqa: BLE001 - a cache error must not fail the row
            logger.warning("TranslationCache set failed (skipping write): {}", e)

    def close(self) -> None:
        """Close all open per-pair cache handles."""
        for cache in self._caches.values():
            try:
                cache.close()
            except Exception:  # noqa: BLE001, S110 - best-effort cleanup
                pass
        self._caches.clear()
