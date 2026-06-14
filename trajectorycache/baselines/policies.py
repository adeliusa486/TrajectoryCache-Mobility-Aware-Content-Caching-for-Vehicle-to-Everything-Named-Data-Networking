"""Baseline cache replacement policies for comparison with TrajectoryCache.

Implements:
  - LRU  : Least Recently Used (ndnSIM default)
  - LFU  : Least Frequently Used (60 s sliding window)
  - ProbCache : Probabilistic caching based on path hop count
  - WAVE : Weighted Age and Velocity Estimation (popularity smoothing)

All baselines expose the same interface as EvictionEngine:
    run_cycle(vehicle_states, current_time) → EvictionCycleResult
    insert_and_maybe_evict(chunk, vehicle_states, current_time) → Optional[result]
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from trajectorycache.core.content_store import ContentChunk, ContentStore
from trajectorycache.core.eviction_engine import EvictionCycleResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LRU Baseline
# ---------------------------------------------------------------------------


class LruEvictionEngine:
    """Standard LRU eviction — evicts the least recently used chunk."""

    def __init__(
        self,
        content_store: ContentStore,
        eta_hw: float = 0.90,
        eta_lw: float = 0.70,
    ) -> None:
        self.cs = content_store
        self.eta_hw = eta_hw
        self.eta_lw = eta_lw
        self.cycle_count = 0
        self.total_evicted = 0

    def run_cycle(self, vehicle_states=None, current_time: float = 0.0) -> EvictionCycleResult:
        result = EvictionCycleResult(triggered=False)
        result.pre_occupancy = self.cs.occupancy
        self.cs.advance_clock(current_time)

        if not self.cs.above_high_watermark(self.eta_hw):
            return result

        result.triggered = True
        self.cycle_count += 1
        evicted = []

        while self.cs.above_low_watermark(self.eta_lw):
            chunk = self.cs.evict_lru()
            if chunk is None:
                break
            evicted.append(chunk.name)
            self.total_evicted += 1

        result.evicted_names = evicted
        result.n_evicted = len(evicted)
        result.post_occupancy = self.cs.occupancy
        self.cs.stats.eviction_cycles += 1
        return result

    def insert_and_maybe_evict(self, chunk, vehicle_states=None, current_time: float = 0.0):
        result = None
        if self.cs.above_high_watermark(self.eta_hw):
            result = self.run_cycle(vehicle_states, current_time)
        self.cs.insert(chunk)
        return result


# ---------------------------------------------------------------------------
# LFU Baseline (60 s sliding window)
# ---------------------------------------------------------------------------


class LfuEvictionEngine:
    """LFU with a 60-second sliding window for frequency counting."""

    WINDOW_S: float = 60.0

    def __init__(
        self,
        content_store: ContentStore,
        eta_hw: float = 0.90,
        eta_lw: float = 0.70,
    ) -> None:
        self.cs = content_store
        self.eta_hw = eta_hw
        self.eta_lw = eta_lw
        self._access_times: Dict[str, deque] = defaultdict(deque)
        self.cycle_count = 0
        self.total_evicted = 0

    def record_access(self, name: str, t: float) -> None:
        q = self._access_times[name]
        q.append(t)
        # Trim old entries
        while q and (t - q[0]) > self.WINDOW_S:
            q.popleft()

    def frequency(self, name: str, current_time: float) -> int:
        q = self._access_times.get(name)
        if q is None:
            return 0
        while q and (current_time - q[0]) > self.WINDOW_S:
            q.popleft()
        return len(q)

    def run_cycle(self, vehicle_states=None, current_time: float = 0.0) -> EvictionCycleResult:
        result = EvictionCycleResult(triggered=False)
        result.pre_occupancy = self.cs.occupancy
        self.cs.advance_clock(current_time)

        if not self.cs.above_high_watermark(self.eta_hw):
            return result

        result.triggered = True
        self.cycle_count += 1

        chunks = self.cs.all_chunks()
        scores = {c.name: self.frequency(c.name, current_time) for c in chunks}
        sorted_chunks = sorted(scores.items(), key=lambda kv: kv[1])  # ascending freq

        evicted = []
        for name, _ in sorted_chunks:
            if not self.cs.above_low_watermark(self.eta_lw):
                break
            chunk = self.cs.evict(name)
            if chunk:
                evicted.append(name)
                self.total_evicted += 1

        result.evicted_names = evicted
        result.n_evicted = len(evicted)
        result.post_occupancy = self.cs.occupancy
        self.cs.stats.eviction_cycles += 1
        return result

    def insert_and_maybe_evict(self, chunk, vehicle_states=None, current_time: float = 0.0):
        result = None
        if self.cs.above_high_watermark(self.eta_hw):
            result = self.run_cycle(vehicle_states, current_time)
        if self.cs.insert(chunk):
            self.record_access(chunk.name, current_time)
        return result


# ---------------------------------------------------------------------------
# ProbCache Baseline
# ---------------------------------------------------------------------------


class ProbCacheEvictionEngine:
    """ProbCache: probabilistic caching with path-length-based probability.

    Caches content with probability 1/n_hops (encourages caching closer
    to consumers). When CS is full, evicts the chunk with the lowest
    cache probability.
    """

    DEFAULT_MAX_HOPS = 8

    def __init__(
        self,
        content_store: ContentStore,
        eta_hw: float = 0.90,
        eta_lw: float = 0.70,
        max_hops: int = DEFAULT_MAX_HOPS,
    ) -> None:
        self.cs = content_store
        self.eta_hw = eta_hw
        self.eta_lw = eta_lw
        self.max_hops = max_hops
        self._cache_prob: Dict[str, float] = {}
        self.cycle_count = 0
        self.total_evicted = 0

    def should_cache(self, n_hops: int = 1) -> bool:
        """Return True with probability 1/n_hops."""
        import random
        prob = 1.0 / max(1, n_hops)
        return random.random() < prob

    def run_cycle(self, vehicle_states=None, current_time: float = 0.0) -> EvictionCycleResult:
        result = EvictionCycleResult(triggered=False)
        result.pre_occupancy = self.cs.occupancy
        self.cs.advance_clock(current_time)

        if not self.cs.above_high_watermark(self.eta_hw):
            return result

        result.triggered = True
        self.cycle_count += 1

        chunks = self.cs.all_chunks()
        # Sort by cache probability ascending (lowest prob evicted first)
        scores = [(c.name, self._cache_prob.get(c.name, 0.5)) for c in chunks]
        scores.sort(key=lambda kv: kv[1])

        evicted = []
        for name, _ in scores:
            if not self.cs.above_low_watermark(self.eta_lw):
                break
            chunk = self.cs.evict(name)
            if chunk:
                evicted.append(name)
                self._cache_prob.pop(name, None)
                self.total_evicted += 1

        result.evicted_names = evicted
        result.n_evicted = len(evicted)
        result.post_occupancy = self.cs.occupancy
        self.cs.stats.eviction_cycles += 1
        return result

    def insert_and_maybe_evict(self, chunk, vehicle_states=None, current_time: float = 0.0):
        result = None
        if self.cs.above_high_watermark(self.eta_hw):
            result = self.run_cycle(vehicle_states, current_time)
        # Assign a random cache probability (proxy for hop count)
        import random
        hops = random.randint(1, self.max_hops)
        self._cache_prob[chunk.name] = 1.0 / hops
        self.cs.insert(chunk)
        return result


# ---------------------------------------------------------------------------
# WAVE Baseline
# ---------------------------------------------------------------------------


class WaveEvictionEngine:
    """WAVE: Weighted Age and Velocity Estimation.

    Uses exponential smoothing of request rates to forecast future demand.
    Evicts chunks with lowest predicted future demand.

    WAVE is the strongest non-mobility-aware baseline in the paper.
    """

    SMOOTHING_ALPHA = 0.3  # EMA smoothing factor

    def __init__(
        self,
        content_store: ContentStore,
        eta_hw: float = 0.90,
        eta_lw: float = 0.70,
    ) -> None:
        self.cs = content_store
        self.eta_hw = eta_hw
        self.eta_lw = eta_lw
        self._request_rate: Dict[str, float] = {}
        self._last_request_t: Dict[str, float] = {}
        self.cycle_count = 0
        self.total_evicted = 0

    def record_request(self, name: str, current_time: float) -> None:
        """Update exponential smoothed request rate for a chunk."""
        last_t = self._last_request_t.get(name)
        if last_t is None:
            self._request_rate[name] = 1.0
        else:
            inter_arrival = max(1e-3, current_time - last_t)
            instant_rate = 1.0 / inter_arrival
            old_rate = self._request_rate.get(name, 0.0)
            self._request_rate[name] = (
                (1 - self.SMOOTHING_ALPHA) * old_rate
                + self.SMOOTHING_ALPHA * instant_rate
            )
        self._last_request_t[name] = current_time

    def predicted_rate(self, name: str) -> float:
        return self._request_rate.get(name, 0.0)

    def run_cycle(self, vehicle_states=None, current_time: float = 0.0) -> EvictionCycleResult:
        result = EvictionCycleResult(triggered=False)
        result.pre_occupancy = self.cs.occupancy
        self.cs.advance_clock(current_time)

        if not self.cs.above_high_watermark(self.eta_hw):
            return result

        result.triggered = True
        self.cycle_count += 1

        chunks = self.cs.all_chunks()
        scores = [(c.name, self.predicted_rate(c.name)) for c in chunks]
        scores.sort(key=lambda kv: kv[1])  # ascending: lowest rate evicted first

        evicted = []
        for name, _ in scores:
            if not self.cs.above_low_watermark(self.eta_lw):
                break
            chunk = self.cs.evict(name)
            if chunk:
                evicted.append(name)
                self.total_evicted += 1

        result.evicted_names = evicted
        result.n_evicted = len(evicted)
        result.post_occupancy = self.cs.occupancy
        self.cs.stats.eviction_cycles += 1
        return result

    def insert_and_maybe_evict(self, chunk, vehicle_states=None, current_time: float = 0.0):
        result = None
        if self.cs.above_high_watermark(self.eta_hw):
            result = self.run_cycle(vehicle_states, current_time)
        self.cs.insert(chunk)
        return result
