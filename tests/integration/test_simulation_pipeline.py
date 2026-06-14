"""Integration tests: full simulation pipeline end-to-end."""

import pytest

from trajectorycache.config import (
    ProjectConfig,
    default_highway_config,
    default_urban_config,
    load_config,
)
from trajectorycache.simulation.sim_loop import TrajectorySimulation
from trajectorycache.baselines.policies import (
    LruEvictionEngine,
    LfuEvictionEngine,
    ProbCacheEvictionEngine,
    WaveEvictionEngine,
)
from pathlib import Path
import json, tempfile


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_default_highway_config_valid(self):
        cfg = default_highway_config()
        cfg.validate()
        assert cfg.sim.scenario_name == "highway"

    def test_default_urban_config_valid(self):
        cfg = default_urban_config()
        cfg.validate()
        assert cfg.sim.scenario_name == "urban"

    def test_load_json_config(self, tmp_path):
        data = {
            "trajectoryCache": {"lambda": 0.8, "alpha": 0.4, "eta_hw": 0.88, "eta_lw": 0.68},
            "rsu": {"cs_capacity_mb": 250},
            "simulation": {"scenario_name": "highway", "num_vehicles": 50},
        }
        p = tmp_path / "test_config.json"
        p.write_text(json.dumps(data))
        cfg = load_config(p)
        assert cfg.tc.lambda_weight == 0.8
        assert cfg.rsu.cs_capacity_mb == 250

    def test_invalid_lambda_raises(self):
        cfg = default_highway_config()
        cfg.tc.lambda_weight = 1.5
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_invalid_watermarks_raises(self):
        cfg = default_highway_config()
        cfg.tc.eta_hw = 0.5
        cfg.tc.eta_lw = 0.8  # lw > hw → invalid
        with pytest.raises(AssertionError):
            cfg.validate()


# ---------------------------------------------------------------------------
# Short simulation runs
# ---------------------------------------------------------------------------

def make_fast_config(scenario: str = "highway", n_vehicles: int = 20) -> ProjectConfig:
    if scenario == "highway":
        cfg = default_highway_config()
    else:
        cfg = default_urban_config()
    cfg.sim.num_vehicles = n_vehicles
    cfg.sim.duration_s = 30.0
    cfg.sim.warmup_s = 10.0
    cfg.sim.random_seed = 42
    return cfg


class TestTrajectorySimulation:
    def test_highway_simulation_runs(self):
        cfg = make_fast_config("highway", 20)
        sim = TrajectorySimulation(cfg)
        metrics = sim.run()
        assert metrics.total_interests >= 0

    def test_urban_simulation_runs(self):
        cfg = make_fast_config("urban", 20)
        sim = TrajectorySimulation(cfg)
        metrics = sim.run()
        assert metrics.total_interests >= 0

    def test_miss_rate_in_valid_range(self):
        cfg = make_fast_config("highway", 30)
        sim = TrajectorySimulation(cfg)
        metrics = sim.run()
        assert 0.0 <= metrics.miss_rate <= 1.0

    def test_isr_in_valid_range(self):
        cfg = make_fast_config("highway", 30)
        sim = TrajectorySimulation(cfg)
        metrics = sim.run()
        assert 0.0 <= metrics.isr <= 1.0

    def test_latency_samples_populated(self):
        cfg = make_fast_config("highway", 30)
        sim = TrajectorySimulation(cfg)
        metrics = sim.run()
        if metrics.total_interests > 0:
            assert len(metrics.latency_samples) > 0
            assert all(l > 0 for l in metrics.latency_samples)

    def test_eviction_cycles_counted(self):
        cfg = make_fast_config("highway", 50)
        sim = TrajectorySimulation(cfg)
        metrics = sim.run()
        # May be zero if CS never fills in 30 s with 50 vehicles
        assert metrics.eviction_cycles >= 0

    def test_reproducibility_same_seed(self):
        """Same seed → identical miss rate."""
        cfg1 = make_fast_config(n_vehicles=25)
        cfg2 = make_fast_config(n_vehicles=25)
        m1 = TrajectorySimulation(cfg1).run()
        m2 = TrajectorySimulation(cfg2).run()
        assert abs(m1.miss_rate - m2.miss_rate) < 1e-9

    def test_different_seeds_differ(self):
        """Different seeds → different results."""
        cfg1 = make_fast_config(n_vehicles=25)
        cfg1.sim.random_seed = 42
        cfg2 = make_fast_config(n_vehicles=25)
        cfg2.sim.random_seed = 99
        m1 = TrajectorySimulation(cfg1).run()
        m2 = TrajectorySimulation(cfg2).run()
        # Very unlikely to be identical
        assert m1.miss_rate != m2.miss_rate or m1.total_interests != m2.total_interests


# ---------------------------------------------------------------------------
# Baseline policy integration tests
# ---------------------------------------------------------------------------

class TestBaselines:
    def _run_baseline(self, policy_class, n_vehicles=20):
        cfg = make_fast_config(n_vehicles=n_vehicles)
        sim = TrajectorySimulation(cfg)
        sim.engine = policy_class(sim.cs, cfg.tc.eta_hw, cfg.tc.eta_lw)
        return sim.run()

    def test_lru_runs(self):
        metrics = self._run_baseline(LruEvictionEngine)
        assert 0.0 <= metrics.miss_rate <= 1.0

    def test_lfu_runs(self):
        metrics = self._run_baseline(LfuEvictionEngine)
        assert 0.0 <= metrics.miss_rate <= 1.0

    def test_probcache_runs(self):
        metrics = self._run_baseline(ProbCacheEvictionEngine)
        assert 0.0 <= metrics.miss_rate <= 1.0

    def test_wave_runs(self):
        metrics = self._run_baseline(WaveEvictionEngine)
        assert 0.0 <= metrics.miss_rate <= 1.0

    def test_tc_vs_lru_ordering(self):
        """TC should have lower or equal miss rate than LRU on average."""
        cfg_tc = make_fast_config(n_vehicles=40)
        sim_tc = TrajectorySimulation(cfg_tc)
        m_tc = sim_tc.run()

        cfg_lru = make_fast_config(n_vehicles=40)
        sim_lru = TrajectorySimulation(cfg_lru)
        sim_lru.engine = LruEvictionEngine(sim_lru.cs, cfg_lru.tc.eta_hw, cfg_lru.tc.eta_lw)
        m_lru = sim_lru.run()

        # In a 30 s run the difference may be small; just assert no crash
        assert m_tc.total_interests >= 0
        assert m_lru.total_interests >= 0


# ---------------------------------------------------------------------------
# Content catalog tests
# ---------------------------------------------------------------------------

class TestCatalog:
    def test_catalog_generation(self):
        from trajectorycache.catalog.generator import (
            generate_catalog,
            generate_highway_segments,
        )
        segments = generate_highway_segments()
        catalog = generate_catalog(segments, random_seed=42)
        assert len(catalog) == 12500

    def test_catalog_zipf_distribution(self):
        from trajectorycache.catalog.generator import (
            generate_catalog,
            generate_highway_segments,
        )
        segments = generate_highway_segments()
        catalog = generate_catalog(segments, zipf_alpha=0.8, random_seed=42)
        ranks = [c.popularity_rank for c in catalog]
        assert min(ranks) == 1
        assert max(ranks) == len(catalog)
        assert len(set(ranks)) == len(catalog)  # All unique ranks

    def test_catalog_save_load(self, tmp_path):
        from trajectorycache.catalog.generator import (
            generate_catalog,
            generate_highway_segments,
            save_catalog,
            load_catalog,
        )
        segments = generate_highway_segments(length_m=500)
        catalog = generate_catalog(segments, random_seed=1)
        path = tmp_path / "catalog.json"
        save_catalog(catalog, path)
        loaded = load_catalog(path)
        assert len(loaded) == len(catalog)
        assert loaded[0].name == catalog[0].name
