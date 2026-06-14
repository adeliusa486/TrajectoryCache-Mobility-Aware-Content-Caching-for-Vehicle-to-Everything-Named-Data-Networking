"""TrajectoryCache Eviction Engine.

Implements the full eviction algorithm from the paper (Algorithm 1):

    S_evict(c) = λ·(1 - MRS̃(c)) + (1-λ)·LRŨ(c)     [Eq. 7]

Where:
    MRS̃  = min-max normalized MRS
    LRŨ  = min-max normalized LRU rank (0 = most recent, 1 = least recent)
    λ    = composite weight (default 0.75)

Eviction procedure:
    1. Check if CS ≥ η_hw · C  → if not, return immediately
    2. Compute trajectories and MRS scores for all CS chunks
    3. Compute composite eviction scores S_evict
    4. Sort ascending; evict until CS ≤ η_lw · C

Formal guarantee (Theorem 1): U_TC ≥ ½ · U*
Empirical: 0.784–0.912 of clairvoyant oracle utility.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from trajectorycache.core.affinity_estimator import AffinityEstimator
from trajectorycache.core.bsm_listener import VehicleState
from trajectorycache.core.content_store import ContentChunk, ContentStore
from trajectorycache.core.ctrv_predictor import CTRVPredictor, compute_dwell_time
from trajectorycache.core.mrs_scorer import MrsScorer

logger = logging.getLogger(__name__)


@dataclass
class EvictionCycleResult:
    """Result of a single eviction cycle."""
    triggered: bool                    # Was eviction actually triggered?
    evicted_names: List[str] = field(default_factory=list)
    n_evicted: int = 0
    cycle_time_ms: float = 0.0
    mrs_time_ms: float = 0.0
    pre_occupancy: float = 0.0
    post_occupancy: float = 0.0
    n_vehicles: int = 0


class EvictionEngine:
    """Composite eviction engine for TrajectoryCache.

    Decoupled from the Interest-forwarding path. Triggered:
    - On demand when CS ≥ η_hw (called from ContentStore.insert path)
    - Periodically via a background cycle timer (every eviction_cycle_s)

    In ndnSIM this overrides EvictEntry() in ndn::cs::Cs.
    In the Python simulation it is called explicitly from the SimulationLoop.
    """

    def __init__(
        self,
        content_store: ContentStore,
        scorer: MrsScorer,
        predictor: CTRVPredictor,
        affinity: AffinityEstimator,
        lambda_weight: float = 0.75,
        eta_hw: float = 0.90,
        eta_lw: float = 0.70,
        rsu_x: float = 0.0,
        rsu_y: float = 0.0,
        coverage_radius: float = 300.0,
    ) -> None:
        self.cs = content_store
        self.scorer = scorer
        self.predictor = predictor
        self.affinity = affinity
        self.lambda_weight = lambda_weight
        self.eta_hw = eta_hw
        self.eta_lw = eta_lw
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.coverage_radius = coverage_radius

        # History
        self.cycle_count: int = 0
        self.total_evicted: int = 0
        self.cycle_history: List[EvictionCycleResult] = []

    def run_cycle(
        self,
        vehicle_states: Dict[str, VehicleState],
        current_time: float,
    ) -> EvictionCycleResult:
        """Execute one eviction cycle.

        Step 1: Check if CS ≥ η_hw · C
        Step 2: Phase 1 — Trajectory prediction + urgency weights
        Step 3: Phase 2 — MRS scoring
        Step 4: Phase 3 — Composite score + batch eviction

        Args:
            vehicle_states: Current neighbor table snapshot.
            current_time: Current simulation time (seconds).

        Returns:
            EvictionCycleResult with cycle statistics.
        """
        t_start = time.perf_counter()
        self.cs.advance_clock(current_time)

        result = EvictionCycleResult(triggered=False)
        result.pre_occupancy = self.cs.occupancy

        # --- Guard: below high-water mark → no action ---
        if not self.cs.above_high_watermark(self.eta_hw):
            return result

        result.triggered = True
        result.n_vehicles = len(vehicle_states)
        self.cycle_count += 1

        chunks = self.cs.all_chunks()
        if not chunks:
            return result

        # --- Phase 1: Trajectory prediction and dwell times ---
        dwell_times: Dict[str, float] = {}
        for vid, state in vehicle_states.items():
            dwell_times[vid] = compute_dwell_time(
                state,
                self.rsu_x,
                self.rsu_y,
                self.coverage_radius,
                self.predictor,
            )

        # --- Phase 2: MRS scoring ---
        self.scorer.build_index(chunks)
        mrs_norm = self.scorer.compute_normalized_mrs(vehicle_states, dwell_times)
        result.mrs_time_ms = self.scorer.last_cycle_ms

        # --- Phase 3: Composite eviction score and batch eviction ---
        lru_ranks = self.cs.lru_ranks()
        lru_norm = _normalize_dict(lru_ranks)

        eviction_scores = self._compute_eviction_scores(
            chunks, mrs_norm, lru_norm
        )

        # Sort ascending by eviction score (highest score = evict first)
        sorted_chunks = sorted(
            eviction_scores.items(), key=lambda item: item[1], reverse=True
        )

        evicted_names: List[str] = []
        for name, score in sorted_chunks:
            if not self.cs.above_low_watermark(self.eta_lw):
                break
            evicted = self.cs.evict(name)
            if evicted is not None:
                self.scorer.remove_chunk(name)
                evicted_names.append(name)
                self.total_evicted += 1

        result.evicted_names = evicted_names
        result.n_evicted = len(evicted_names)
        result.post_occupancy = self.cs.occupancy
        result.cycle_time_ms = (time.perf_counter() - t_start) * 1000.0

        self.cs.stats.eviction_cycles += 1
        self.cycle_history.append(result)

        logger.debug(
            "Eviction cycle %d: evicted=%d, occ=%.2f→%.2f, time=%.2fms",
            self.cycle_count,
            result.n_evicted,
            result.pre_occupancy,
            result.post_occupancy,
            result.cycle_time_ms,
        )

        return result

    def _compute_eviction_scores(
        self,
        chunks: List[ContentChunk],
        mrs_norm: Dict[str, float],
        lru_norm: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute S_evict(c) = λ·(1 - MRS̃(c)) + (1-λ)·LRŨ(c) for all chunks."""
        scores: Dict[str, float] = {}
        lam = self.lambda_weight

        for chunk in chunks:
            name = chunk.name
            mrs_score = mrs_norm.get(name, 0.0)
            lru_score = lru_norm.get(name, 0.5)
            scores[name] = lam * (1.0 - mrs_score) + (1.0 - lam) * lru_score

        return scores

    def insert_and_maybe_evict(
        self,
        chunk: ContentChunk,
        vehicle_states: Dict[str, VehicleState],
        current_time: float,
    ) -> Optional[EvictionCycleResult]:
        """Insert a chunk and trigger eviction if needed.

        Mirrors the ndnSIM Data packet path:
        1. Try to insert chunk
        2. If CS ≥ η_hw, run eviction cycle
        3. Insert chunk (now there is space)

        Returns eviction result if triggered, else None.
        """
        result = None

        # Pre-eviction if needed
        if self.cs.above_high_watermark(self.eta_hw):
            result = self.run_cycle(vehicle_states, current_time)

        inserted = self.cs.insert(chunk)
        if inserted:
            self.scorer.add_chunk(chunk)

        return result

    @property
    def average_cycle_time_ms(self) -> float:
        if not self.cycle_history:
            return 0.0
        triggered = [r for r in self.cycle_history if r.triggered]
        if not triggered:
            return 0.0
        return sum(r.cycle_time_ms for r in triggered) / len(triggered)

    @property
    def average_mrs_time_ms(self) -> float:
        if not self.cycle_history:
            return 0.0
        triggered = [r for r in self.cycle_history if r.triggered]
        if not triggered:
            return 0.0
        return sum(r.mrs_time_ms for r in triggered) / len(triggered)


def _normalize_dict(d: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize a dict of floats to [0, 1]."""
    if not d:
        return {}
    values = list(d.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}
