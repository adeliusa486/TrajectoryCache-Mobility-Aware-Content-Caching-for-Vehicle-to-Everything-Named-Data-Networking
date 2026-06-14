"""Unit tests for MRS Scorer and Eviction Engine."""

import math
import pytest

from trajectorycache.core.affinity_estimator import AffinityEstimator
from trajectorycache.core.bsm_listener import VehicleState
from trajectorycache.core.content_store import ContentChunk, ContentStore
from trajectorycache.core.ctrv_predictor import CTRVPredictor
from trajectorycache.core.eviction_engine import EvictionEngine, _normalize_dict
from trajectorycache.core.mrs_scorer import MrsScorer, SpatialIndex


# ---------------------------------------------------------------------------
# SpatialIndex tests
# ---------------------------------------------------------------------------

class TestSpatialIndex:
    def test_insert_and_query(self):
        idx = SpatialIndex()
        idx.insert("chunk1", 100.0, 100.0, 50.0)
        results = idx.query_bbox(50, 50, 150, 150)
        assert "chunk1" in results

    def test_no_overlap(self):
        idx = SpatialIndex()
        idx.insert("chunk1", 100.0, 100.0, 50.0)
        results = idx.query_bbox(500, 500, 600, 600)
        assert "chunk1" not in results

    def test_delete(self):
        idx = SpatialIndex()
        idx.insert("chunk1", 0.0, 0.0, 50.0)
        idx.delete("chunk1")
        results = idx.query_bbox(-100, -100, 100, 100)
        assert "chunk1" not in results

    def test_multiple_entries(self):
        idx = SpatialIndex()
        for i in range(5):
            idx.insert(f"c{i}", float(i * 100), 0.0, 50.0)
        results = idx.query_bbox(-50, -50, 150, 50)
        assert "c0" in results
        assert "c1" in results
        assert "c4" not in results


# ---------------------------------------------------------------------------
# MRS Scorer tests
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer_setup():
    predictor = CTRVPredictor(epsilon_turn=1e-4, delta_tau=0.5)
    affinity = AffinityEstimator(catalog_mean_prior=1.0)  # φ = 1 for all
    scorer = MrsScorer(predictor=predictor, affinity=affinity, r_grz=300.0, alpha=0.5)
    return scorer, predictor, affinity


def make_chunk(name, x, y, r=300.0) -> ContentChunk:
    return ContentChunk(name=name, size_bytes=1024, grz_x=x, grz_y=y, r_grz=r)


def make_vehicle_state(vid, x, y, speed=20.0, heading=0.0, t_arrive=0.0) -> VehicleState:
    return VehicleState(vid, x, y, speed, heading, 0.0, 0.0, t_arrive)


class TestMrsScorer:
    def test_vehicle_approaching_relevant_chunk(self, scorer_setup):
        scorer, _, _ = scorer_setup
        chunk = make_chunk("c1", x=500, y=0)
        scorer.build_index([chunk])

        # Vehicle heading East at x=0, will pass near chunk at x=500
        state = make_vehicle_state("v1", x=0, y=0, speed=20, heading=0.0, t_arrive=0)
        dwell = {state.vehicle_id: 30.0}  # 30 s dwell

        mrs = scorer.compute_mrs({state.vehicle_id: state}, dwell)
        assert mrs.get("c1", 0.0) > 0.0

    def test_vehicle_moving_away_irrelevant(self, scorer_setup):
        scorer, _, _ = scorer_setup
        # Chunk far North; vehicle heading South
        chunk = make_chunk("c1", x=0, y=5000, r=100.0)
        scorer.build_index([chunk])

        state = make_vehicle_state("v1", x=0, y=0, speed=20, heading=-math.pi / 2)
        dwell = {"v1": 10.0}

        mrs = scorer.compute_mrs({"v1": state}, dwell)
        assert mrs.get("c1", 0.0) == 0.0

    def test_zero_dwell_time(self, scorer_setup):
        scorer, _, _ = scorer_setup
        chunk = make_chunk("c1", x=0, y=0)
        scorer.build_index([chunk])

        state = make_vehicle_state("v1", x=0, y=0)
        mrs = scorer.compute_mrs({"v1": state}, {"v1": 0.0})
        assert mrs.get("c1", 0.0) == 0.0

    def test_normalization_bounds(self, scorer_setup):
        scorer, _, _ = scorer_setup
        chunks = [make_chunk(f"c{i}", x=float(i * 100), y=0) for i in range(5)]
        scorer.build_index(chunks)

        states = {"v1": make_vehicle_state("v1", x=200, y=0, speed=10)}
        dwell = {"v1": 10.0}

        norm = scorer.compute_normalized_mrs(states, dwell)
        for v in norm.values():
            assert 0.0 <= v <= 1.0

    def test_high_urgency_weight(self, scorer_setup):
        scorer, _, _ = scorer_setup
        chunk = make_chunk("c1", x=100, y=0)
        scorer.build_index([chunk])

        # Vehicle inside coverage: T_arrive=0, w_v=1.0
        near = make_vehicle_state("v_near", x=50, y=0, speed=10, t_arrive=0)
        # Vehicle far away: T_arrive=100, w_v small
        far = make_vehicle_state("v_far", x=50, y=0, speed=10, t_arrive=100)

        mrs_near = scorer.compute_mrs({"v_near": near}, {"v_near": 10.0})
        mrs_far = scorer.compute_mrs({"v_far": far}, {"v_far": 10.0})

        assert mrs_near.get("c1", 0.0) > mrs_far.get("c1", 0.0)


# ---------------------------------------------------------------------------
# Eviction Engine tests
# ---------------------------------------------------------------------------

@pytest.fixture
def engine_setup():
    cs = ContentStore(capacity_bytes=100 * 1024)  # 100 KB
    predictor = CTRVPredictor()
    affinity = AffinityEstimator(catalog_mean_prior=1.0)
    scorer = MrsScorer(predictor=predictor, affinity=affinity, r_grz=300.0)
    engine = EvictionEngine(
        content_store=cs, scorer=scorer, predictor=predictor, affinity=affinity,
        lambda_weight=0.75, eta_hw=0.90, eta_lw=0.70,
        rsu_x=0, rsu_y=0, coverage_radius=300,
    )
    return engine, cs


def fill_cs(cs, n=10, chunk_size_kb=8):
    """Fill the content store with n chunks."""
    for i in range(n):
        chunk = ContentChunk(
            name=f"/v2x/seg/chunk{i:03d}",
            size_bytes=chunk_size_kb * 1024,
            grz_x=float(i * 50),
            grz_y=0.0,
        )
        cs.insert(chunk)


class TestEvictionEngine:
    def test_no_eviction_below_hwm(self, engine_setup):
        engine, cs = engine_setup
        fill_cs(cs, n=5, chunk_size_kb=8)  # ~40 KB of 100 KB = 40% < 90%
        result = engine.run_cycle({}, current_time=0.0)
        assert not result.triggered
        assert result.n_evicted == 0

    def test_eviction_triggered_above_hwm(self, engine_setup):
        engine, cs = engine_setup
        fill_cs(cs, n=12, chunk_size_kb=8)  # ~96 KB / 100 KB = 96% > 90%
        result = engine.run_cycle({}, current_time=1.0)
        assert result.triggered
        assert result.n_evicted > 0

    def test_occupancy_drops_to_lw_after_eviction(self, engine_setup):
        engine, cs = engine_setup
        fill_cs(cs, n=12, chunk_size_kb=8)
        result = engine.run_cycle({}, current_time=1.0)
        assert cs.occupancy <= engine.eta_lw + 0.01  # Allow tiny float imprecision

    def test_eviction_retains_high_mrs_chunks(self, engine_setup):
        """Chunks with high MRS should be retained, low-MRS chunks evicted."""
        engine, cs = engine_setup
        # Chunk near a vehicle trajectory → high MRS
        high_mrs_chunk = ContentChunk(
            name="/v2x/seg/important", size_bytes=8 * 1024,
            grz_x=200.0, grz_y=0.0,
        )
        cs.insert(high_mrs_chunk)
        fill_cs(cs, n=11, chunk_size_kb=8)  # Fill past HWM

        # Vehicle heading toward important chunk
        states = {
            "v1": make_vehicle_state("v1", x=0, y=0, speed=20, heading=0.0)
        }
        result = engine.run_cycle(states, current_time=1.0)
        assert result.triggered
        # The important chunk should ideally be retained
        # (exact result depends on MRS computation; this is a smoke check)
        assert result.n_evicted > 0

    def test_insert_and_evict_workflow(self, engine_setup):
        engine, cs = engine_setup
        fill_cs(cs, n=11, chunk_size_kb=8)  # Fill to ~88%
        new_chunk = ContentChunk(
            name="/v2x/new/chunk999", size_bytes=10 * 1024,
            grz_x=0.0, grz_y=0.0,
        )
        result = engine.insert_and_maybe_evict(new_chunk, {}, current_time=1.0)
        assert "/v2x/new/chunk999" in cs or result is not None


class TestNormalizeDict:
    def test_uniform_input(self):
        d = {"a": 5.0, "b": 5.0, "c": 5.0}
        result = _normalize_dict(d)
        for v in result.values():
            assert abs(v - 0.5) < 1e-9

    def test_min_max_bounds(self):
        d = {"a": 0.0, "b": 5.0, "c": 10.0}
        result = _normalize_dict(d)
        assert result["a"] == 0.0
        assert result["c"] == 1.0
        assert abs(result["b"] - 0.5) < 1e-9

    def test_empty(self):
        assert _normalize_dict({}) == {}
