# TrajectoryCache

**Mobility-Aware Content Caching for NDN-based V2X Networks**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![CI](https://github.com/trajectorycache/trajectorycache/actions/workflows/ci.yml/badge.svg)](https://github.com/trajectorycache/trajectorycache/actions)
[![arXiv](https://img.shields.io/badge/Paper-IEEE%20TVT%202025-red)](https://ieeexplore.ieee.org)

> TrajectoryCache: Mobility-Aware Content Caching for Named Data Networking in Vehicle-to-Everything (V2X) Environments.

## Overview

TrajectoryCache is a **mobility-aware cache replacement extension** for Named Data Networking (NDN) Roadside Units (RSUs) in V2X environments. It replaces standard LRU eviction with a **Mobility Relevance Score (MRS)** that predicts which cached content chunks will be demanded by approaching vehicles — derived from kinematic trajectory prediction, geographic content zones, and content affinity.

### Key Results

| Metric | Improvement vs LRU |
|---|---|
| Cache miss rate | 18.3–26.1% reduction |
| Content retrieval latency | 15.2–20.4% reduction |
| Backhaul traffic | up to 22.6% reduction |
| Interest Satisfaction Ratio | 0.91 vs 0.73 at 500 vehicles |
| Per-Interest overhead | 2.33 µs (< 3% baseline cost) |

### Architecture

```
Vehicles (BSM @ 10 Hz) → BSM Listener → Trajectory Predictor (CTRV)
                                                   ↓
Content Catalog (GRZ-tagged) → MRS Scorer (R-tree spatial index)
                                                   ↓
                        Eviction Score = λ·(1 - MRS̃) + (1-λ)·LRŨ
                                                   ↓
                           NDN Content Store (managed eviction)
```

---

## Repository Structure

```
trajectorycache/
├── src/core/           # C++ core: BSM listener, CTRV predictor, MRS scorer, eviction engine
├── src/utils/          # OSM segment table, R-tree wrapper, map-matching
├── trajectorycache/    # Python library: simulation of core algorithms
├── scenarios/          # SUMO + NS-3 scenario configs (S1 highway, S2 urban)
├── catalog/            # Content catalog generation (Zipf, GRZ tagging)
├── evaluation/         # Metric collection, stats, plots
├── baselines/          # LRU, LFU, ProbCache, WAVE implementations
├── scripts/            # Run scripts for full experimental sweep
├── tests/              # Unit and integration tests
├── api/                # FastAPI monitoring/control interface
├── docs/               # Full documentation
├── deployment/         # Docker + Kubernetes manifests
├── monitoring/         # Prometheus + Grafana configs
└── configs/            # Scenario and system configuration files
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional) SUMO 1.14 for full co-simulation
- (Optional) Docker for containerized deployment

### 1. Install Python Library

```bash
git clone https://github.com/trajectorycache/trajectorycache.git
cd trajectorycache
pip install -e ".[dev]"
```

### 2. Run the Python Simulation (No SUMO Required)

```bash
# Run a quick demo with synthetic BSM data
python -m trajectorycache.demo --scenario highway --vehicles 100 --duration 60

# Run full evaluation sweep
python scripts/run_evaluation.py --scenario all --vehicles 100,200,300,400,500
```

### 3. Run Unit Tests

```bash
pytest tests/ -v
```

### 4. Start the Monitoring API

```bash
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000/docs
```

### 5. Docker Deployment

```bash
docker-compose up -d
```

---

## Full Co-Simulation Setup (SUMO + NS-3)

See [docs/SETUP.md](docs/SETUP.md) for complete installation instructions including:
- NS-3.36 + ndnSIM 2.8 build
- SUMO 1.14 + TraCI configuration
- C++ core compilation with CMake

---

## Core Concepts

### Mobility Relevance Score (MRS)

```
MRS(c, r, t) = Σ_{v ∈ V_r(t)}  w_v · 𝟙[d̂(v,c,t) ≤ r_grz] · φ(v,c)
```

- **w_v** = 1/(1 + α·T_arrive(v)) — arrival urgency weight
- **𝟙[...]** — geographic relevance indicator (is vehicle's trajectory within GRZ?)
- **φ(v,c)** — content affinity (EMA of vehicle's historical requests)

### Composite Eviction Score

```
S_evict(c) = λ·(1 - MRS̃(c)) + (1-λ)·LRŨ(c)
```

Chunks with **high** MRS (many approaching vehicles want it) and **low** LRU rank (recently accessed) are **retained**. The rest are evicted first.

### CTRV Trajectory Prediction

For |θ̇| ≥ ε:
```
x̂(t+Δt) = x + (s/θ̇)·[sin(θ + θ̇·Δt) - sin(θ)]
ŷ(t+Δt) = y + (s/θ̇)·[cos(θ) - cos(θ + θ̇·Δt)]
```

---

## Configuration

All parameters are configurable via `configs/highway_default.json` or `configs/urban_default.json`:

```json
{
  "trajectoryCache": {
    "lambda": 0.75,
    "alpha": 0.5,
    "eta_hw": 0.90,
    "eta_lw": 0.70,
    "r_grz": 300.0,
    "epsilon_turn": 1e-4,
    "affinity_window_s": 300,
    "eviction_cycle_s": 1.0
  }
}
```

---

## Reproducing Paper Results

```bash
# Full 20-run sweep (10 seeds × 2 scenarios) — requires SUMO + ndnSIM
bash scripts/run_all_scenarios.sh

# Python-only statistical analysis on pre-generated traces
python evaluation/compute_stats.py --results_dir experiments/results/

# Generate all paper figures
python evaluation/plot_all.py --results_dir experiments/results/ --output_dir docs/figures/
```

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for full reproduction guide.

---

## Baselines Implemented

| Policy | Description |
|---|---|
| LRU | Least Recently Used (ndnSIM default) |
| LFU | Least Frequently Used (60 s window) |
| ProbCache | Probabilistic caching based on path length |
| WAVE | Weighted Age and Velocity Estimation |
| TrajectoryCache | This work |

---

## Citation

```bibtex
@article{alrashidi2025trajectorycache,
  title={TrajectoryCache: Mobility-Aware Content Caching for Named Data Networking in Vehicle-to-Everything (V2X) Environments},
  author={Al-Rashidi, Ahmad and Mansour, Layla and Siddiqui, Omar and Al-Zahrawi, Fatima},
  journal={IEEE Transactions on Vehicular Technology},
  year={2025},
  doi={10.1109/TVT.2025.XXXXXXX}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome PRs for:
- Multi-RSU cooperative caching
- EKF trajectory predictor
- Authenticated BSM integration
- Hardware testbed validation

## Acknowledgments

Supported by SDAIA-KAUST Center of Excellence in Data Science and Artificial Intelligence (SDAIA-KAUST-2023-04) and KFUPM Research Institute grant RI-2023-EE-11.
