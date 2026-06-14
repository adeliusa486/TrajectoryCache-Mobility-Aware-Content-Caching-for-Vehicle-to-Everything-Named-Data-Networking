"""Content Store (CS) management for TrajectoryCache.

Models the NDN Content Store as an ordered dictionary supporting:
- CS hit/miss tracking
- LRU rank computation
- High/low watermark-based eviction triggering
- Integration with the TrajectoryCache eviction engine
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ContentChunk:
    """A single cached content chunk.

    Attributes:
        name: NDN name prefix, e.g. /v2x/riyadh/seg42/hdmap/chunk003
        size_bytes: Chunk size in bytes
        grz_x: GRZ center x coordinate (meters)
        grz_y: GRZ center y coordinate (meters)
        r_grz: GRZ radius (meters); defaults to RSU coverage radius
        content_type: One of 'hdmap', 'traffic_advisory', 'firmware'
        popularity_rank: Zipf rank (1 = most popular)
        cached_at: Simulation time when chunk was inserted
        last_accessed: Simulation time of most recent CS hit
        access_count: Total number of CS hits for this chunk
    """

    name: str
    size_bytes: int
    grz_x: float
    grz_y: float
    r_grz: float = 300.0
    content_type: str = "hdmap"
    popularity_rank: int = 1
    cached_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0

    def touch(self, current_time: float) -> None:
        """Record a cache hit."""
        self.last_accessed = current_time
        self.access_count += 1

    @property
    def age(self) -> float:
        """Seconds since last access."""
        return self.last_accessed  # Will be compared relatively


@dataclass
class CsStats:
    """Content Store statistics snapshot."""

    total_interests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    eviction_cycles: int = 0
    total_evicted: int = 0
    backhaul_requests: int = 0
    bytes_cached: int = 0
    capacity_bytes: int = 0

    @property
    def miss_rate(self) -> float:
        if self.total_interests == 0:
            return 0.0
        return self.cache_misses / self.total_interests

    @property
    def hit_rate(self) -> float:
        return 1.0 - self.miss_rate

    @property
    def occupancy(self) -> float:
        if self.capacity_bytes == 0:
            return 0.0
        return self.bytes_cached / self.capacity_bytes


class ContentStore:
    """NDN Content Store with LRU tracking and watermark-based eviction.

    Maintains an ordered dict keyed by chunk name. The LRU order is
    preserved via move_to_end() on each access.

    The eviction trigger is external (EvictionEngine) — this class only
    provides the data structure and statistics.
    """

    def __init__(self, capacity_bytes: int, current_time: float = 0.0) -> None:
        """
        Args:
            capacity_bytes: Maximum CS size in bytes (C in the paper).
            current_time: Initial simulation clock value.
        """
        self._store: OrderedDict[str, ContentChunk] = OrderedDict()
        self.capacity_bytes = capacity_bytes
        self.current_time = current_time
        self.stats = CsStats(capacity_bytes=capacity_bytes)

    # ------------------------------------------------------------------
    # Core CS operations
    # ------------------------------------------------------------------

    def lookup(self, name: str) -> Optional[ContentChunk]:
        """Look up a chunk by NDN name. Updates LRU rank on hit.

        Returns:
            ContentChunk on cache hit, None on miss.
        """
        self.stats.total_interests += 1
        chunk = self._store.get(name)
        if chunk is not None:
            chunk.touch(self.current_time)
            self._store.move_to_end(name)  # Most recently used → tail
            self.stats.cache_hits += 1
            return chunk
        else:
            self.stats.cache_misses += 1
            self.stats.backhaul_requests += 1
            return None

    def insert(self, chunk: ContentChunk) -> bool:
        """Insert a chunk into the CS (without triggering eviction).

        The caller is responsible for checking watermarks before insertion.
        Returns True if inserted, False if already present.
        """
        if chunk.name in self._store:
            return False

        if chunk.size_bytes > self.capacity_bytes:
            logger.warning(
                "Chunk %s (%d bytes) exceeds CS capacity (%d bytes); skipping",
                chunk.name,
                chunk.size_bytes,
                self.capacity_bytes,
            )
            return False

        chunk.cached_at = self.current_time
        chunk.last_accessed = self.current_time
        self._store[chunk.name] = chunk
        self.stats.bytes_cached += chunk.size_bytes
        return True

    def evict(self, name: str) -> Optional[ContentChunk]:
        """Remove a specific chunk from the CS.

        Returns the evicted chunk, or None if not found.
        """
        chunk = self._store.pop(name, None)
        if chunk is not None:
            self.stats.bytes_cached -= chunk.size_bytes
            self.stats.total_evicted += 1
        return chunk

    def evict_lru(self) -> Optional[ContentChunk]:
        """Evict the least recently used chunk (head of ordered dict)."""
        if not self._store:
            return None
        name, chunk = next(iter(self._store.items()))
        return self.evict(name)

    # ------------------------------------------------------------------
    # Watermark and occupancy helpers
    # ------------------------------------------------------------------

    @property
    def occupancy(self) -> float:
        """Current CS fill fraction [0, 1]."""
        return self.stats.bytes_cached / self.capacity_bytes

    def above_high_watermark(self, eta_hw: float) -> bool:
        return self.occupancy >= eta_hw

    def above_low_watermark(self, eta_lw: float) -> bool:
        return self.occupancy > eta_lw

    # ------------------------------------------------------------------
    # Introspection helpers for MRS scorer
    # ------------------------------------------------------------------

    def all_chunks(self) -> List[ContentChunk]:
        """Return all cached chunks (order: LRU → MRU)."""
        return list(self._store.values())

    def lru_ranks(self) -> Dict[str, float]:
        """Return min-max normalised LRU rank for each chunk.

        Rank 0 = most recently used (tail), Rank 1 = least recently used (head).
        """
        names = list(self._store.keys())  # head = LRU, tail = MRU
        n = len(names)
        if n == 0:
            return {}
        if n == 1:
            return {names[0]: 0.5}
        return {name: i / (n - 1) for i, name in enumerate(names)}

    def chunk_names(self) -> List[str]:
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def advance_clock(self, t: float) -> None:
        """Update the internal simulation clock."""
        self.current_time = t
