"""Mobility Relevance Score (MRS) Scorer.

Implements Equation 5 from the paper:

    MRS(c, r, t) = Σ_{v ∈ V_r(t)}  w_v · 𝟙[d̂(v,c,t) ≤ r_grz] · φ(v,c)

Uses an R-tree spatial index over GRZ centers to efficiently retrieve
candidate (vehicle, chunk) pairs without an O(|V| × |CS|) brute-force scan.

Complexity: O(|V_r| · (log|CS_r| + k̄)) per eviction cycle
where k̄ = mean GRZ intersections per trajectory corridor
(4.2 ± 1.1 highway, 6.3 ± 1.8 urban per paper).

Empirical speedup: 12×–18× over naïve approach at |CS| = 5000.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from trajectorycache.core.affinity_estimator import AffinityEstimator
from trajectorycache.core.bsm_listener import VehicleState
from trajectorycache.core.content_store import ContentChunk
from trajectorycache.core.ctrv_predictor import CTRVPredictor, Trajectory

logger = logging.getLogger(__name__)


class SpatialIndex:
    """Lightweight R-tree-like spatial index for GRZ centers.

    In production this wraps libspatialindex (via Rtree Python bindings).
    This implementation uses a numpy-based brute-force fallback that is
    functionally correct and used when rtree is not installed.

    For full performance (12–18× speedup), install:
        pip install rtree
    which links against libspatialindex.
    """

    def __init__(self) -> None:
        self._data: Dict[str, Tuple[float, float, float]] = {}  # name → (x, y, r)
        self._use_rtree = False

        try:
            from rtree import index as rtree_index
            p = rtree_index.Property()
            p.dimension = 2
            self._rtree = rtree_index.Index(properties=p)
            self._rtree_id_map: Dict[int, str] = {}
            self._name_to_id: Dict[str, int] = {}
            self._id_counter = 0
            self._use_rtree = True
            logger.info("Using libspatialindex R-tree for spatial queries")
        except ImportError:
            logger.warning(
                "rtree package not found; using numpy brute-force spatial index. "
                "Install with: pip install rtree"
            )

    def insert(self, name: str, x: float, y: float, r: float) -> None:
        """Insert a GRZ entry."""
        self._data[name] = (x, y, r)

        if self._use_rtree:
            int_id = self._id_counter
            self._id_counter += 1
            self._rtree_id_map[int_id] = name
            self._name_to_id[name] = int_id
            # Bounding box for the GRZ disk: (min_x, min_y, max_x, max_y)
            self._rtree.insert(int_id, (x - r, y - r, x + r, y + r))

    def delete(self, name: str) -> None:
        """Remove a GRZ entry."""
        if name not in self._data:
            return
        x, y, r = self._data.pop(name)

        if self._use_rtree and name in self._name_to_id:
            int_id = self._name_to_id.pop(name)
            del self._rtree_id_map[int_id]
            self._rtree.delete(int_id, (x - r, y - r, x + r, y + r))

    def query_bbox(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> List[str]:
        """Return names of GRZ entries whose bounding box overlaps the query bbox."""
        if self._use_rtree:
            ids = list(self._rtree.intersection((min_x, min_y, max_x, max_y)))
            return [self._rtree_id_map[i] for i in ids if i in self._rtree_id_map]
        else:
            # Brute-force numpy fallback
            return self._query_brute(min_x, min_y, max_x, max_y)

    def _query_brute(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> List[str]:
        results = []
        for name, (x, y, r) in self._data.items():
            if x - r <= max_x and x + r >= min_x and y - r <= max_y and y + r >= min_y:
                results.append(name)
        return results

    def __len__(self) -> int:
        return len(self._data)

    def get_entry(self, name: str) -> Optional[Tuple[float, float, float]]:
        return self._data.get(name)


class MrsScorer:
    """Computes MRS for all chunks in the Content Store.

    Workflow per eviction cycle:
    1. For each vehicle v: compute trajectory corridor bounding box
    2. R-tree query: retrieve candidate chunks whose GRZ overlaps corridor bbox
    3. For each (vehicle, candidate chunk): check exact distance ≤ r_grz
    4. Accumulate: MRS(c) += w_v · φ(v, c) for qualifying pairs
    5. Normalise MRS̃ to [0, 1] via min-max over current CS population
    """

    def __init__(
        self,
        predictor: CTRVPredictor,
        affinity: AffinityEstimator,
        r_grz: float = 300.0,
        alpha: float = 0.5,
        no_prediction: bool = False,
    ) -> None:
        """
        Args:
            predictor: CTRV trajectory predictor.
            affinity: Affinity estimator φ(v, c).
            r_grz: Geographic Relevance Zone radius (meters).
            alpha: Arrival urgency decay coefficient (s⁻¹), Eq. 6.
            no_prediction: If True, disables trajectory prediction.
        """
        self.predictor = predictor
        self.affinity = affinity
        self.r_grz = r_grz
        self.alpha = alpha
        self.no_prediction = no_prediction

        self._spatial_index = SpatialIndex()
        self._indexed_chunks: Dict[str, ContentChunk] = {}

        # Profiling
        self.last_cycle_ms: float = 0.0
        self.last_rtree_queries: int = 0
        self.last_exact_checks: int = 0

    def build_index(self, chunks: List[ContentChunk]) -> None:
        """Build (or rebuild) the spatial index from the current CS contents.

        Call once when CS is initialised or after bulk changes.
        For incremental updates, use add_chunk / remove_chunk.
        """
        # Clear existing index
        for name in list(self._indexed_chunks.keys()):
            self._spatial_index.delete(name)
        self._indexed_chunks.clear()

        for chunk in chunks:
            self._insert_chunk(chunk)

    def add_chunk(self, chunk: ContentChunk) -> None:
        """Add a single chunk to the spatial index."""
        if chunk.name not in self._indexed_chunks:
            self._insert_chunk(chunk)

    def remove_chunk(self, name: str) -> None:
        """Remove a chunk from the spatial index."""
        if name in self._indexed_chunks:
            self._spatial_index.delete(name)
            del self._indexed_chunks[name]

    def _insert_chunk(self, chunk: ContentChunk) -> None:
        r = chunk.r_grz if chunk.r_grz > 0 else self.r_grz
        self._spatial_index.insert(chunk.name, chunk.grz_x, chunk.grz_y, r)
        self._indexed_chunks[chunk.name] = chunk

    def compute_mrs(
        self,
        vehicle_states: Dict[str, VehicleState],
        dwell_times: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute raw MRS for all indexed chunks.

        Args:
            vehicle_states: {vehicle_id: VehicleState} from neighbor table.
            dwell_times: {vehicle_id: T_dwell_s} prediction horizons.

        Returns:
            {chunk_name: MRS_raw} — raw (unnormalized) MRS values.
        """
        import time as _time
        t0 = _time.perf_counter()

        mrs: Dict[str, float] = {name: 0.0 for name in self._indexed_chunks}
        n_queries = 0
        n_exact = 0

        for vid, state in vehicle_states.items():
            horizon = dwell_times.get(vid, 0.0)
            if horizon <= 0:
                continue

            if self.no_prediction:
                horizon = 0.0

            # Arrival urgency weight w_v = 1 / (1 + α·T_arrive)
            w_v = 1.0 / (1.0 + self.alpha * max(0.0, state.t_arrive))

            if w_v < 1e-6:
                continue  # Negligible contribution

            # Predict trajectory
            trajectory = self.predictor.predict(state, horizon)
            if len(trajectory) < 1:
                continue

            # Get bounding box of trajectory corridor
            bbox = self.predictor.compute_corridor_bbox(trajectory, self.r_grz)
            min_x, min_y, max_x, max_y = bbox

            # R-tree range query
            candidates = self._spatial_index.query_bbox(min_x, min_y, max_x, max_y)
            n_queries += 1
            n_exact += len(candidates)

            for chunk_name in candidates:
                chunk = self._indexed_chunks.get(chunk_name)
                if chunk is None:
                    continue

                # Exact distance check: min dist trajectory → GRZ center ≤ r_grz
                min_dist = self.predictor.min_distance_to_point(
                    trajectory, chunk.grz_x, chunk.grz_y
                )
                chunk_r = chunk.r_grz if chunk.r_grz > 0 else self.r_grz
                if min_dist <= chunk_r:
                    phi = self.affinity.get(vid, chunk_name)
                    mrs[chunk_name] = mrs.get(chunk_name, 0.0) + w_v * phi

        self.last_cycle_ms = (_time.perf_counter() - t0) * 1000.0
        self.last_rtree_queries = n_queries
        self.last_exact_checks = n_exact

        return mrs

    def normalize_mrs(self, mrs: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalise MRS scores to [0, 1]."""
        if not mrs:
            return {}
        values = list(mrs.values())
        lo, hi = min(values), max(values)
        if hi == lo:
            return {k: 0.5 for k in mrs}
        return {k: (v - lo) / (hi - lo) for k, v in mrs.items()}

    def compute_normalized_mrs(
        self,
        vehicle_states: Dict[str, VehicleState],
        dwell_times: Dict[str, float],
    ) -> Dict[str, float]:
        """Convenience method: compute and normalize MRS in one call."""
        raw = self.compute_mrs(vehicle_states, dwell_times)
        return self.normalize_mrs(raw)
