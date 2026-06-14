#!/usr/bin/env python3
"""Smoke test: verify all core components initialise and run a minimal workflow.

Run with:
    python scripts/smoke_test.py

Exit code 0 = all passed, 1 = failures detected.
"""

import sys
import time
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

results = []


def check(name: str, fn):
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  ✓  {name:55s}  ({elapsed:.1f} ms)")
        results.append((name, True, None))
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  ✗  {name:55s}  FAILED: {e}")
        traceback.print_exc()
        results.append((name, False, str(e)))


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
print("\n── 1. Core Imports ──────────────────────────────────────────────────")

def _import_core():
    import trajectorycache
    from trajectorycache.core.bsm_listener import BsmListener, VehicleState
    from trajectorycache.core.ctrv_predictor import CTRVPredictor
    from trajectorycache.core.content_store import ContentStore, ContentChunk
    from trajectorycache.core.affinity_estimator import AffinityEstimator
    from trajectorycache.core.mrs_scorer import MrsScorer
    from trajectorycache.core.eviction_engine import EvictionEngine

check("Import trajectorycache core modules", _import_core)

def _import_catalog():
    from trajectorycache.catalog.generator import generate_catalog, generate_highway_segments

check("Import catalog generator", _import_catalog)

def _import_baselines():
    from trajectorycache.baselines.policies import (
        LruEvictionEngine, LfuEvictionEngine, ProbCacheEvictionEngine, WaveEvictionEngine
    )

check("Import baseline policies", _import_baselines)

def _import_evaluation():
    from trajectorycache.evaluation.experiments import run_policy, aggregate_results
    from trajectorycache.evaluation.plots import plot_all

check("Import evaluation modules", _import_evaluation)

def _import_api():
    try:
        from api.main import app
    except ImportError as e:
        if "fastapi" in str(e) or "pydantic" in str(e):
            print("    (skipped: fastapi not installed — install with: pip install fastapi uvicorn pydantic)")
            return
        raise

check("Import FastAPI app", _import_api)


# ---------------------------------------------------------------------------
# 2. Config loading
# ---------------------------------------------------------------------------
print("\n── 2. Configuration Loading ─────────────────────────────────────────")

def _load_highway_config():
    from trajectorycache.config import load_config
    cfg = load_config(Path("configs/highway_default.json"))
    assert cfg.tc.lambda_weight == 0.75

check("Load highway_default.json config", _load_highway_config)

def _load_urban_config():
    from trajectorycache.config import load_config
    cfg = load_config(Path("configs/urban_default.json"))
    assert cfg.sim.scenario_name == "urban"

check("Load urban_default.json config", _load_urban_config)

def _default_configs():
    from trajectorycache.config import default_highway_config, default_urban_config
    cfg1 = default_highway_config()
    cfg2 = default_urban_config()
    cfg1.validate()
    cfg2.validate()

check("Instantiate and validate default configs", _default_configs)


# ---------------------------------------------------------------------------
# 3. Component initialisation
# ---------------------------------------------------------------------------
print("\n── 3. Component Initialisation ──────────────────────────────────────")

def _init_bsm():
    from trajectorycache.core.bsm_listener import BsmListener
    bl = BsmListener(rsu_x=0, rsu_y=0, coverage_radius=300)
    state = bl.process_bsm("v1", 10, 5, 15.0, 0.1, 0.5)
    assert state is not None
    assert state.vehicle_id == "v1"

check("BsmListener: process 1 BSM", _init_bsm)

def _init_predictor():
    import math
    from trajectorycache.core.ctrv_predictor import CTRVPredictor
    from trajectorycache.core.bsm_listener import VehicleState
    pred = CTRVPredictor(epsilon_turn=1e-4, delta_tau=0.5)
    state = VehicleState("v1", 0, 0, 20, 0, 0.1, 0, 0)
    traj = pred.predict(state, horizon_s=5.0)
    assert len(traj) >= 2

check("CTRVPredictor: predict 5 s trajectory", _init_predictor)

def _init_content_store():
    from trajectorycache.core.content_store import ContentStore, ContentChunk
    cs = ContentStore(capacity_bytes=10 * 1024 * 1024)
    chunk = ContentChunk("/v2x/test/chunk001", 8192, 100, 100)
    cs.insert(chunk)
    hit = cs.lookup("/v2x/test/chunk001")
    assert hit is not None
    miss = cs.lookup("/v2x/test/nonexistent")
    assert miss is None

check("ContentStore: insert + hit + miss", _init_content_store)

def _init_affinity():
    from trajectorycache.core.affinity_estimator import AffinityEstimator
    af = AffinityEstimator(beta=1/300)
    af.update("v1", "/v2x/chunk1", True)
    phi = af.get("v1", "/v2x/chunk1")
    assert 0.0 < phi <= 1.0

check("AffinityEstimator: update + get", _init_affinity)

def _init_spatial_index():
    from trajectorycache.core.mrs_scorer import SpatialIndex
    idx = SpatialIndex()
    idx.insert("c1", 100, 100, 50)
    results = idx.query_bbox(50, 50, 150, 150)
    assert "c1" in results

check("SpatialIndex: insert + query", _init_spatial_index)

def _init_catalog():
    from trajectorycache.catalog.generator import generate_catalog, generate_highway_segments
    segs = generate_highway_segments(length_m=1000)
    catalog = generate_catalog(segs, random_seed=42)
    assert len(catalog) > 0

check("Catalog generator: generate_catalog()", _init_catalog)


# ---------------------------------------------------------------------------
# 4. Short simulation run
# ---------------------------------------------------------------------------
print("\n── 4. Simulation Pipeline ───────────────────────────────────────────")

def _run_highway_sim():
    from trajectorycache.config import default_highway_config
    from trajectorycache.simulation.sim_loop import TrajectorySimulation
    cfg = default_highway_config()
    cfg.sim.num_vehicles = 15
    cfg.sim.duration_s = 15.0
    cfg.sim.warmup_s = 5.0
    sim = TrajectorySimulation(cfg)
    metrics = sim.run()
    assert 0.0 <= metrics.miss_rate <= 1.0

check("Highway simulation: 15 s, 15 vehicles", _run_highway_sim)

def _run_urban_sim():
    from trajectorycache.config import default_urban_config
    from trajectorycache.simulation.sim_loop import TrajectorySimulation
    cfg = default_urban_config()
    cfg.sim.num_vehicles = 15
    cfg.sim.duration_s = 15.0
    cfg.sim.warmup_s = 5.0
    sim = TrajectorySimulation(cfg)
    metrics = sim.run()
    assert 0.0 <= metrics.miss_rate <= 1.0

check("Urban simulation: 15 s, 15 vehicles", _run_urban_sim)

def _run_all_baselines():
    from trajectorycache.config import default_highway_config
    from trajectorycache.simulation.sim_loop import TrajectorySimulation
    from trajectorycache.baselines.policies import (
        LruEvictionEngine, LfuEvictionEngine, ProbCacheEvictionEngine, WaveEvictionEngine
    )
    for PolicyClass in [LruEvictionEngine, LfuEvictionEngine, ProbCacheEvictionEngine, WaveEvictionEngine]:
        cfg = default_highway_config()
        cfg.sim.num_vehicles = 10
        cfg.sim.duration_s = 10.0
        cfg.sim.warmup_s = 3.0
        sim = TrajectorySimulation(cfg)
        sim.engine = PolicyClass(sim.cs, cfg.tc.eta_hw, cfg.tc.eta_lw)
        m = sim.run()
        assert 0.0 <= m.miss_rate <= 1.0

check("All 4 baselines: 10 s run each", _run_all_baselines)


# ---------------------------------------------------------------------------
# 5. API initialisation
# ---------------------------------------------------------------------------
print("\n── 5. API Initialisation ────────────────────────────────────────────")

def _api_import_and_routes():
    try:
        from api.main import app
    except ImportError as e:
        if "fastapi" in str(e) or "pydantic" in str(e):
            print("    (skipped: fastapi not installed)")
            return
        raise
    route_paths = [r.path for r in app.routes]
    assert "/health" in route_paths
    assert "/metrics" in route_paths
    assert "/config" in route_paths
    assert "/neighbors" in route_paths

check("FastAPI: import and route registration", _api_import_and_routes)


# ---------------------------------------------------------------------------
# 6. Evaluation helpers
# ---------------------------------------------------------------------------
print("\n── 6. Evaluation Helpers ────────────────────────────────────────────")

def _aggregate_results():
    from trajectorycache.evaluation.experiments import RunResult, aggregate_results
    runs = [
        RunResult("tc", "highway", 100, 42, 0.20, 25.0, 0.85, 2.1, 1.5),
        RunResult("tc", "highway", 100, 99, 0.22, 26.0, 0.83, 2.3, 1.6),
        RunResult("lru", "highway", 100, 42, 0.30, 35.0, 0.75, 0.0, 0.0),
    ]
    agg = aggregate_results(runs)
    assert len(agg) == 2

check("Evaluation: aggregate_results()", _aggregate_results)

def _confidence_interval():
    from trajectorycache.evaluation.experiments import confidence_interval_95
    import numpy as np
    vals = [0.20, 0.22, 0.19, 0.21, 0.23]
    mean, ci = confidence_interval_95(vals)
    assert abs(mean - np.mean(vals)) < 1e-9
    assert ci >= 0

check("Evaluation: confidence_interval_95()", _confidence_interval)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "─" * 70)
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"  Results: {n_pass} passed, {n_fail} failed out of {len(results)} checks")

if n_fail > 0:
    print("\n  FAILED checks:")
    for name, ok, err in results:
        if not ok:
            print(f"    ✗  {name}: {err}")
    print()
    sys.exit(1)
else:
    print("\n  All smoke tests passed. ✓\n")
    sys.exit(0)
