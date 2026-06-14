"""Demo entry point: run a quick simulation and print a results table.

Usage:
    python -m trajectorycache.demo --scenario highway --vehicles 100 --duration 60
    python -m trajectorycache.demo --compare all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_demo(scenario: str, n_vehicles: int, duration: float, seed: int) -> None:
    from trajectorycache.config import default_highway_config, default_urban_config
    from trajectorycache.simulation.sim_loop import TrajectorySimulation
    from trajectorycache.baselines.policies import (
        LruEvictionEngine, LfuEvictionEngine, WaveEvictionEngine
    )

    if scenario == "highway":
        cfg = default_highway_config()
    else:
        cfg = default_urban_config()
    cfg.sim.num_vehicles = n_vehicles
    cfg.sim.duration_s = duration
    cfg.sim.warmup_s = min(30.0, duration / 5)
    cfg.sim.random_seed = seed

    policies = {
        "TrajectoryCache": None,
        "LRU": LruEvictionEngine,
        "LFU": LfuEvictionEngine,
        "WAVE": WaveEvictionEngine,
    }

    print(f"\n{'='*72}")
    print(f"  TrajectoryCache Demo — {scenario.capitalize()} | {n_vehicles} vehicles | {duration:.0f}s")
    print(f"{'='*72}")
    print(f"  {'Policy':<20} {'Miss Rate':>10} {'Latency(ms)':>13} {'ISR(50ms)':>10} {'OH(µs)':>8}")
    print(f"  {'-'*64}")

    for name, PolicyClass in policies.items():
        import importlib
        import copy

        cfg2 = copy.deepcopy(cfg)
        sim = TrajectorySimulation(cfg2)
        if PolicyClass is not None:
            sim.engine = PolicyClass(sim.cs, cfg2.tc.eta_hw, cfg2.tc.eta_lw)

        metrics = sim.run()
        print(
            f"  {name:<20} {metrics.miss_rate*100:>9.1f}% "
            f"{metrics.mean_latency_ms:>12.1f} "
            f"{metrics.isr*100:>9.1f}% "
            f"{metrics.per_interest_overhead_us:>7.2f}"
        )

    print(f"{'='*72}\n")


def main():
    parser = argparse.ArgumentParser(description="TrajectoryCache quick demo")
    parser.add_argument("--scenario", choices=["highway", "urban"], default="highway")
    parser.add_argument("--vehicles", type=int, default=100)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_demo(args.scenario, args.vehicles, args.duration, args.seed)


if __name__ == "__main__":
    main()
