"""Plot generation for all paper figures.

Figures produced:
  Fig. 3: Cache miss rate vs vehicle density
  Fig. 4: Content retrieval latency vs vehicle density
  Fig. 6: Ablation study bar chart
  Fig. 7: λ sensitivity curve
  Fig. 8: GPS noise degradation curve

Usage:
    python -m trajectorycache.evaluation.plots --results_dir experiments/results --output_dir docs/figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from trajectorycache.evaluation.experiments import (
    AggregatedResult,
    RunResult,
    aggregate_results,
    load_results,
)

matplotlib.use("Agg")  # Non-interactive backend for server environments

logger = logging.getLogger(__name__)

# Paper-style color palette
COLORS = {
    "tc": "#1f77b4",       # Blue
    "lru": "#d62728",      # Red
    "lfu": "#ff7f0e",      # Orange
    "probcache": "#9467bd", # Purple
    "wave": "#2ca02c",     # Green
}

LABELS = {
    "tc": "TrajectoryCache",
    "lru": "LRU",
    "lfu": "LFU",
    "probcache": "ProbCache",
    "wave": "WAVE",
}

DENSITIES = [100, 200, 300, 400, 500]


def _style_ax(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_miss_rate(
    results: List[RunResult],
    output_path: Path,
    scenario: str = "highway",
) -> None:
    """Fig. 3: Cache miss rate vs vehicle density."""
    agg = aggregate_results([r for r in results if r.scenario == scenario])

    fig, ax = plt.subplots(figsize=(7, 5))
    policies = ["tc", "lru", "lfu", "probcache", "wave"]

    for policy in policies:
        rows = [r for r in agg if r.policy == policy]
        rows.sort(key=lambda r: r.n_vehicles)
        if not rows:
            continue
        xs = [r.n_vehicles for r in rows]
        ys = [r.miss_rate_mean * 100 for r in rows]
        errs = [r.miss_rate_ci * 100 for r in rows]
        ax.errorbar(
            xs, ys, yerr=errs,
            label=LABELS.get(policy, policy),
            color=COLORS.get(policy, "gray"),
            marker="o", linewidth=2, capsize=4,
        )

    _style_ax(ax, "Number of Vehicles", "Cache Miss Rate (%)",
              f"Cache Miss Rate vs Vehicle Density ({scenario.capitalize()})")
    ax.set_xticks(DENSITIES)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved miss rate plot → %s", output_path)


def plot_latency(
    results: List[RunResult],
    output_path: Path,
    scenario: str = "highway",
) -> None:
    """Fig. 4: Retrieval latency vs vehicle density."""
    agg = aggregate_results([r for r in results if r.scenario == scenario])

    fig, ax = plt.subplots(figsize=(7, 5))
    for policy in ["tc", "lru", "wave"]:
        rows = [r for r in agg if r.policy == policy]
        rows.sort(key=lambda r: r.n_vehicles)
        if not rows:
            continue
        xs = [r.n_vehicles for r in rows]
        ys = [r.latency_mean for r in rows]
        errs = [r.latency_ci for r in rows]
        ax.errorbar(
            xs, ys, yerr=errs,
            label=LABELS.get(policy, policy),
            color=COLORS.get(policy, "gray"),
            marker="s", linewidth=2, capsize=4,
        )

    _style_ax(ax, "Number of Vehicles", "Mean Retrieval Latency (ms)",
              f"Content Retrieval Latency ({scenario.capitalize()})")
    ax.set_xticks(DENSITIES)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved latency plot → %s", output_path)


def plot_ablation(
    results: List[RunResult],
    output_path: Path,
) -> None:
    """Fig. 6: Ablation study bar chart."""
    policy_order = ["tc", "tc_nopred", "tc_noaff", "tc_eqw", "lru"]
    labels_abl = {
        "tc": "TC-Full",
        "tc_nopred": "TC-NoPred",
        "tc_noaff": "TC-NoAff",
        "tc_eqw": "TC-EqW",
        "lru": "LRU",
    }
    colors_abl = {
        "tc": "#1f77b4",
        "tc_nopred": "#aec7e8",
        "tc_noaff": "#ffbb78",
        "tc_eqw": "#98df8a",
        "lru": "#d62728",
    }

    # Average across seeds and densities for the ablation
    miss_by_policy: Dict[str, List[float]] = {p: [] for p in policy_order}
    for r in results:
        if r.policy in miss_by_policy:
            miss_by_policy[r.policy].append(r.miss_rate * 100)

    means = [np.mean(miss_by_policy[p]) if miss_by_policy[p] else 0
             for p in policy_order]
    stds = [np.std(miss_by_policy[p]) if len(miss_by_policy[p]) > 1 else 0
            for p in policy_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(policy_order))
    bars = ax.bar(
        x, means, yerr=stds, capsize=5,
        color=[colors_abl[p] for p in policy_order],
        edgecolor="black", linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels_abl[p] for p in policy_order], fontsize=11)
    ax.set_ylabel("Cache Miss Rate (%)", fontsize=12)
    ax.set_title("Ablation Study: Component Contribution", fontsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ablation plot → %s", output_path)


def plot_lambda_sensitivity(
    results: List[RunResult],
    output_path: Path,
) -> None:
    """Fig. 7: Miss rate vs λ."""
    lambda_vals = sorted(set(
        float(r.policy.split("_")[-1])
        for r in results
        if r.policy.startswith("tc_lambda_")
    ))
    if not lambda_vals:
        logger.warning("No lambda sweep results to plot")
        return

    means, cis = [], []
    for lam in lambda_vals:
        key = f"tc_lambda_{lam:.2f}"
        vals = [r.miss_rate * 100 for r in results if r.policy == key]
        if vals:
            m = np.mean(vals)
            ci = 1.96 * np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        else:
            m, ci = 0.0, 0.0
        means.append(m)
        cis.append(ci)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(lambda_vals, means, yerr=cis, color="#1f77b4",
                marker="o", linewidth=2, capsize=4, label="TrajectoryCache")
    ax.axvline(0.75, color="red", linestyle="--", alpha=0.7, label="Default λ=0.75")
    _style_ax(ax, "λ (MRS weight)", "Cache Miss Rate (%)", "λ Sensitivity Analysis")
    ax.set_xlim([-0.05, 1.05])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved lambda sensitivity plot → %s", output_path)


def plot_gps_noise(
    results: List[RunResult],
    output_path: Path,
) -> None:
    """Fig. 8: Miss rate vs GPS noise σ."""
    sigma_vals = sorted(set(
        float(r.policy.split("_")[-1])
        for r in results
        if r.policy.startswith("tc_gps_")
    ))
    if not sigma_vals:
        logger.warning("No GPS noise results to plot")
        return

    tc_means, tc_cis = [], []
    for s in sigma_vals:
        key = f"tc_gps_{s:.1f}"
        vals = [r.miss_rate * 100 for r in results if r.policy == key]
        m = np.mean(vals) if vals else 0.0
        ci = 1.96 * np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        tc_means.append(m)
        tc_cis.append(ci)

    lru_baseline = 28.9  # From paper Table II at n=300 highway

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(sigma_vals, tc_means, yerr=tc_cis, color="#1f77b4",
                marker="o", linewidth=2, capsize=4, label="TrajectoryCache")
    ax.axhline(lru_baseline, color="#d62728", linestyle="--",
               linewidth=1.5, label=f"LRU baseline ({lru_baseline}%)")
    _style_ax(ax, "GPS Noise σ (meters)", "Cache Miss Rate (%)",
              "GPS Noise Sensitivity")
    ax.set_xlim([-0.5, max(sigma_vals) + 0.5])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved GPS noise plot → %s", output_path)


def plot_all(results_dir: Path, output_dir: Path) -> None:
    """Generate all paper figures from saved result files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all result files
    all_results: List[RunResult] = []
    for f in results_dir.glob("*.json"):
        try:
            all_results.extend(load_results(f))
        except Exception as e:
            logger.warning("Could not load %s: %s", f, e)

    if not all_results:
        logger.error("No results found in %s", results_dir)
        return

    density_results = [r for r in all_results if r.policy in ["tc", "lru", "lfu", "probcache", "wave"]]
    if density_results:
        plot_miss_rate(density_results, output_dir / "fig3_miss_rate_highway.png", "highway")
        plot_latency(density_results, output_dir / "fig4_latency_highway.png", "highway")

    ablation_results = [r for r in all_results if r.policy in ["tc", "tc_nopred", "tc_noaff", "tc_eqw", "lru"]]
    if ablation_results:
        plot_ablation(ablation_results, output_dir / "fig6_ablation.png")

    lambda_results = [r for r in all_results if r.policy.startswith("tc_lambda_")]
    if lambda_results:
        plot_lambda_sensitivity(lambda_results, output_dir / "fig7_lambda.png")

    gps_results = [r for r in all_results if r.policy.startswith("tc_gps_")]
    if gps_results:
        plot_gps_noise(gps_results, output_dir / "fig8_gps_noise.png")

    logger.info("All plots saved to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate TrajectoryCache paper figures")
    parser.add_argument("--results_dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--output_dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    plot_all(args.results_dir, args.output_dir)
