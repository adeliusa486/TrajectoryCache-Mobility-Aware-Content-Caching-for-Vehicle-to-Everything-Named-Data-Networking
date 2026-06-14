"""Evaluation metrics, statistical tests, and experiment runner.

Implements:
  - Multi-policy comparison across vehicle densities
  - 95% CI via Student's t-distribution
  - Paired Wilcoxon signed-rank test for significance (p < 0.001)
  - λ sensitivity sweep
  - GPS noise degradation sweep
  - Ablation study variants (TC-NoPred, TC-NoGRZ, TC-NoAff, TC-EqW)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from trajectorycache.config import ProjectConfig, default_highway_config, default_urban_config
from trajectorycache.simulation.sim_loop import SimulationMetrics, TrajectorySimulation
from trajectorycache.baselines.policies import (
    LruEvictionEngine,
    LfuEvictionEngine,
    ProbCacheEvictionEngine,
    WaveEvictionEngine,
)
from trajectorycache.core.affinity_estimator import AffinityEstimator
from trajectorycache.core.content_store import ContentStore
from trajectorycache.core.ctrv_predictor import CTRVPredictor
from trajectorycache.core.eviction_engine import EvictionEngine
from trajectorycache.core.mrs_scorer import MrsScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-run result
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    policy: str
    scenario: str
    n_vehicles: int
    seed: int
    miss_rate: float
    mean_latency_ms: float
    isr: float
    per_interest_overhead_us: float
    mean_cycle_time_ms: float


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------


SEEDS = [42, 137, 271, 314, 512, 613, 718, 828, 919, 1001]


def run_policy(
    policy_name: str,
    config: ProjectConfig,
    seed: int,
) -> RunResult:
    """Run one policy for one seed and return results."""
    config.sim.random_seed = seed
    sim = TrajectorySimulation(config)

    # Override the eviction engine for baseline policies
    if policy_name == "lru":
        sim.engine = LruEvictionEngine(sim.cs, config.tc.eta_hw, config.tc.eta_lw)
    elif policy_name == "lfu":
        sim.engine = LfuEvictionEngine(sim.cs, config.tc.eta_hw, config.tc.eta_lw)
    elif policy_name == "probcache":
        sim.engine = ProbCacheEvictionEngine(sim.cs, config.tc.eta_hw, config.tc.eta_lw)
    elif policy_name == "wave":
        sim.engine = WaveEvictionEngine(sim.cs, config.tc.eta_hw, config.tc.eta_lw)
    elif policy_name == "tc":
        pass  # Default TrajectoryCache
    elif policy_name == "tc_nopred":
        # Ablation: disable trajectory prediction (use current position only)
        sim.predictor = CTRVPredictor(epsilon_turn=1e4)  # Force CV always
    elif policy_name == "tc_noaff":
        # Ablation: disable affinity (φ = 1 always)
        sim.affinity = AffinityEstimator(catalog_mean_prior=1.0)
    elif policy_name == "tc_eqw":
        # Ablation: equal weighting (λ = 0.5)
        config.tc.lambda_weight = 0.5
        sim.engine.lambda_weight = 0.5

    metrics = sim.run()

    return RunResult(
        policy=policy_name,
        scenario=config.sim.scenario_name,
        n_vehicles=config.sim.num_vehicles,
        seed=seed,
        miss_rate=metrics.miss_rate,
        mean_latency_ms=metrics.mean_latency_ms,
        isr=metrics.isr,
        per_interest_overhead_us=metrics.per_interest_overhead_us,
        mean_cycle_time_ms=metrics.mean_cycle_time_ms,
    )


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def confidence_interval_95(values: List[float]) -> Tuple[float, float]:
    """Return (mean, half-width) of 95% CI via Student's t."""
    arr = np.array(values)
    n = len(arr)
    if n < 2:
        return float(np.mean(arr)), 0.0
    se = stats.sem(arr)
    t_val = stats.t.ppf(0.975, df=n - 1)
    return float(np.mean(arr)), float(t_val * se)


def wilcoxon_p(a: List[float], b: List[float]) -> float:
    """Paired Wilcoxon signed-rank test p-value."""
    if len(a) != len(b) or len(a) < 3:
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b)
        return float(p)
    except Exception:
        return 1.0


@dataclass
class AggregatedResult:
    """Summary statistics for one (policy, scenario, n_vehicles) combination."""
    policy: str
    scenario: str
    n_vehicles: int
    miss_rate_mean: float
    miss_rate_ci: float
    latency_mean: float
    latency_ci: float
    isr_mean: float
    isr_ci: float
    overhead_mean_us: float
    n_seeds: int


def aggregate_results(runs: List[RunResult]) -> List[AggregatedResult]:
    """Aggregate multi-seed runs into summary statistics."""
    from itertools import groupby
    from operator import attrgetter

    key_fn = lambda r: (r.policy, r.scenario, r.n_vehicles)
    grouped: Dict[Tuple, List[RunResult]] = {}
    for run in runs:
        k = key_fn(run)
        grouped.setdefault(k, []).append(run)

    results = []
    for (policy, scenario, n_vehicles), group in grouped.items():
        miss_rates = [r.miss_rate for r in group]
        latencies = [r.mean_latency_ms for r in group]
        isrs = [r.isr for r in group]
        overheads = [r.per_interest_overhead_us for r in group]

        mr_mean, mr_ci = confidence_interval_95(miss_rates)
        lat_mean, lat_ci = confidence_interval_95(latencies)
        isr_mean, isr_ci = confidence_interval_95(isrs)

        results.append(AggregatedResult(
            policy=policy,
            scenario=scenario,
            n_vehicles=n_vehicles,
            miss_rate_mean=mr_mean,
            miss_rate_ci=mr_ci,
            latency_mean=lat_mean,
            latency_ci=lat_ci,
            isr_mean=isr_mean,
            isr_ci=isr_ci,
            overhead_mean_us=float(np.mean(overheads)),
            n_seeds=len(group),
        ))

    return results


# ---------------------------------------------------------------------------
# Full experiment sweeps
# ---------------------------------------------------------------------------


def run_density_sweep(
    scenario: str = "highway",
    densities: List[int] = [100, 200, 300, 400, 500],
    policies: List[str] = ["tc", "lru", "lfu", "probcache", "wave"],
    seeds: List[int] = SEEDS,
    fast_mode: bool = True,
) -> List[RunResult]:
    """Run the main density-vs-miss-rate experiment.

    Args:
        scenario: 'highway' or 'urban'
        densities: Vehicle counts to sweep
        policies: Policy names to run
        seeds: Random seeds for replication
        fast_mode: Reduce simulation duration for quick testing (30 s vs 600 s)

    Returns:
        List of RunResult for all combinations.
    """
    all_results: List[RunResult] = []

    for n_veh in densities:
        for policy in policies:
            for seed in seeds:
                if scenario == "highway":
                    cfg = default_highway_config()
                else:
                    cfg = default_urban_config()
                cfg.sim.num_vehicles = n_veh
                cfg.sim.random_seed = seed
                if fast_mode:
                    cfg.sim.duration_s = 150.0
                    cfg.sim.warmup_s = 30.0

                logger.info("Running: policy=%s scenario=%s n=%d seed=%d",
                            policy, scenario, n_veh, seed)
                try:
                    result = run_policy(policy, cfg, seed)
                    all_results.append(result)
                except Exception as e:
                    logger.error("Run failed: %s", e)

    return all_results


def run_lambda_sweep(
    lambda_values: List[float] = [0.0, 0.1, 0.25, 0.5, 0.65, 0.75, 0.85, 0.95, 1.0],
    n_vehicles: int = 300,
    seeds: List[int] = SEEDS[:3],
    fast_mode: bool = True,
) -> List[RunResult]:
    """λ sensitivity sweep (Fig. 7)."""
    results = []
    for lam in lambda_values:
        for seed in seeds:
            cfg = default_highway_config()
            cfg.sim.num_vehicles = n_vehicles
            cfg.tc.lambda_weight = lam
            if fast_mode:
                cfg.sim.duration_s = 150.0
                cfg.sim.warmup_s = 30.0
            policy_name = f"tc_lambda_{lam:.2f}"
            r = run_policy("tc", cfg, seed)
            r.policy = policy_name
            results.append(r)
    return results


def run_gps_noise_sweep(
    sigma_values: List[float] = [0.0, 5.0, 10.0, 15.0, 20.0],
    n_vehicles: int = 300,
    seeds: List[int] = SEEDS[:3],
    fast_mode: bool = True,
) -> List[RunResult]:
    """GPS noise degradation sweep (Fig. 8)."""
    results = []
    for sigma in sigma_values:
        for seed in seeds:
            cfg = default_highway_config()
            cfg.sim.num_vehicles = n_vehicles
            cfg.sim.gps_noise_sigma = sigma
            if fast_mode:
                cfg.sim.duration_s = 150.0
                cfg.sim.warmup_s = 30.0
            policy_name = f"tc_gps_{sigma:.1f}"
            r = run_policy("tc", cfg, seed)
            r.policy = policy_name
            results.append(r)
    return results


def run_ablation_study(
    n_vehicles: int = 300,
    seeds: List[int] = SEEDS[:5],
    fast_mode: bool = True,
) -> List[RunResult]:
    """Ablation study (Fig. 6): TC-Full, TC-NoPred, TC-NoGRZ, TC-NoAff, TC-EqW, LRU."""
    policies = ["tc", "tc_nopred", "tc_noaff", "tc_eqw", "lru"]
    results = []
    for policy in policies:
        for seed in seeds:
            cfg = default_highway_config()
            cfg.sim.num_vehicles = n_vehicles
            if fast_mode:
                cfg.sim.duration_s = 150.0
                cfg.sim.warmup_s = 30.0
            r = run_policy(policy, cfg, seed)
            results.append(r)
    return results


def save_results(results: List[RunResult], path: Path) -> None:
    """Save raw results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {k: v for k, v in vars(r).items()}
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d results to %s", len(results), path)


def load_results(path: Path) -> List[RunResult]:
    """Load raw results from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [RunResult(**item) for item in data]
