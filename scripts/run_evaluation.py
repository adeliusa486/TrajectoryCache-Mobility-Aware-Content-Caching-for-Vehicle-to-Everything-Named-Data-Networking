#!/usr/bin/env python3
"""Run the full experimental evaluation sweep.

Examples:
    # Quick test run (fast_mode=True, 3 seeds)
    python scripts/run_evaluation.py --scenario highway --fast

    # Full paper sweep (10 seeds, 600 s sims) — takes ~2-4 hours
    python scripts/run_evaluation.py --scenario all --full

    # Lambda sweep only
    python scripts/run_evaluation.py --sweep lambda

    # Generate plots from existing results
    python scripts/run_evaluation.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("experiments/results")
FIGURES_DIR = Path("docs/figures")


def main():
    parser = argparse.ArgumentParser(description="TrajectoryCache evaluation sweep")
    parser.add_argument("--scenario", choices=["highway", "urban", "all"], default="highway")
    parser.add_argument("--vehicles", nargs="+", type=int, default=[100, 200, 300, 400, 500])
    parser.add_argument("--fast", action="store_true", help="Fast mode: 150 s sims, 3 seeds")
    parser.add_argument("--full", action="store_true", help="Full mode: 600 s sims, 10 seeds")
    parser.add_argument("--sweep", choices=["density", "lambda", "gps", "ablation", "all"],
                        default="density")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    from trajectorycache.evaluation.experiments import (
        run_density_sweep,
        run_lambda_sweep,
        run_gps_noise_sweep,
        run_ablation_study,
        save_results,
        SEEDS,
    )
    from trajectorycache.evaluation.plots import plot_all

    if args.plot_only:
        logger.info("Generating plots from %s", args.output_dir)
        plot_all(args.output_dir, FIGURES_DIR)
        return

    fast_mode = not args.full
    seeds = SEEDS[:3] if (fast_mode and not args.full) else SEEDS

    scenarios = ["highway", "urban"] if args.scenario == "all" else [args.scenario]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        if args.sweep in ("density", "all"):
            logger.info("Running density sweep: scenario=%s", scenario)
            results = run_density_sweep(
                scenario=scenario,
                densities=args.vehicles,
                fast_mode=fast_mode,
                seeds=seeds,
            )
            path = args.output_dir / f"density_{scenario}.json"
            save_results(results, path)
            logger.info("Saved density results → %s", path)

    if args.sweep in ("lambda", "all"):
        logger.info("Running λ sensitivity sweep")
        results = run_lambda_sweep(fast_mode=fast_mode, seeds=seeds[:3])
        save_results(results, args.output_dir / "lambda_sweep.json")

    if args.sweep in ("gps", "all"):
        logger.info("Running GPS noise sweep")
        results = run_gps_noise_sweep(fast_mode=fast_mode, seeds=seeds[:3])
        save_results(results, args.output_dir / "gps_noise_sweep.json")

    if args.sweep in ("ablation", "all"):
        logger.info("Running ablation study")
        results = run_ablation_study(fast_mode=fast_mode, seeds=seeds[:5])
        save_results(results, args.output_dir / "ablation.json")

    # Generate plots
    logger.info("Generating plots → %s", FIGURES_DIR)
    plot_all(args.output_dir, FIGURES_DIR)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
