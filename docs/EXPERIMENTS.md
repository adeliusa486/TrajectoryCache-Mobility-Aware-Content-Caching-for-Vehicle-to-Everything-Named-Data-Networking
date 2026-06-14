# Reproducing Paper Results

## Hardware Used in Paper

- 2× Intel Xeon Gold 6226R (16-core, 2.9 GHz)
- 128 GB DDR4 ECC RAM
- Ubuntu 20.04 LTS
- SUMO 1.14, NS-3.36, ndnSIM 2.8
- Python 3.10, NumPy 1.24, SciPy 1.10

Full paper sweep (20 runs × 5 densities × 2 scenarios × 5 policies) required ~14 hours
on the above hardware.

---

## Quick Python-Only Reproduction (~5–15 min)

```bash
# Fast sweep: 3 seeds, 150 s sims
python scripts/run_evaluation.py --scenario highway --fast --sweep all

# Generate plots
python scripts/run_evaluation.py --plot-only

# View figures
ls docs/figures/
```

**Expected outputs** (Python simulation, fast mode, may differ from paper by ±1–2%
due to simplified mobility model vs SUMO):

| Metric (highway, n=300) | Paper | Python sim |
|---|---|---|
| TC miss rate | 21.4% | ~23–25% |
| LRU miss rate | 28.9% | ~28–31% |
| TC mean latency | 18.2 ms | ~17–20 ms |
| TC ISR (50 ms) | 0.91 | ~0.87–0.92 |

---

## Full Co-Simulation Reproduction (~14 hours)

```bash
# Requires SUMO + ndnSIM (see docs/SETUP.md)
bash scripts/run_all_scenarios.sh

# After completion:
python scripts/run_evaluation.py --plot-only --output-dir experiments/results/cosim
```

---

## Statistical Methodology

- **Replication**: 10 independent seeds (42, 137, 271, 314, 512, 613, 718, 828, 919, 1001)
- **Warmup**: First 120 s discarded from all metrics
- **Measurement window**: 120–600 s (480 s per run)
- **Confidence intervals**: 95% via Student's t (df = 9)
- **Significance testing**: Paired Wilcoxon signed-rank test, p < 0.001

---

## Key Parameter Values (Table I in paper)

| Parameter | Symbol | Value |
|---|---|---|
| CS capacity | C | 500 MB |
| Coverage radius | R | 300 m |
| GRZ radius | r_grz | 300 m |
| MRS weight | λ | 0.75 |
| Urgency decay | α | 0.5 s⁻¹ |
| High watermark | η_hw | 0.90 |
| Low watermark | η_lw | 0.70 |
| BSM rate | — | 10 Hz |
| Prediction step | Δτ | 0.5 s |
| Affinity window | W | 300 s |
| Zipf exponent | α_z | 0.8 |
| Catalog size | — | 12,500 |
