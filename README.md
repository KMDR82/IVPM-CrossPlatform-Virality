# IVPM-EW: Online Early Warning for Digital Music Virality Under Chart Censoring

This repository contains the data, code, and results for the paper:

> *From Post-Hoc Explanation to Causal Early Warning: A Simulation-Calibrated
> Multi-Platform Framework for Detecting and Classifying Digital Music
> Virality Under Chart Censoring.*

IVPM-EW is an online early-warning system for viral resurgences of
back-catalog music. It monitors three platform signals (YouTube views,
Google Trends search interest, chart-derived Spotify streams), declares viral
**ignition** through a censoring-aware 2-of-3 cross-platform confirmation
rule, and classifies the viral **regime** (endogenous organic growth vs.
exogenous media shock) with an interpretable rise-ratio statistic.

## Key results

All numbers are produced by `notebooks/ivpm_ew_kaggle_v3.ipynb` and stored
in `results/`. Monte Carlo values are mean ± sd over 5 seeds (900 series per run).

| Finding | Result |
|---|---|
| Selected operating point (θ=1.75, 2-of-3, persistence=2) | TPR 98.8 ± 0.4 %, FA 2.1 ± 0.6 %, delay 1.9 ± 0.1 months |
| 3-of-3 rule under chart censoring | TPR collapses 99.5 % → 2.0 % (2-of-3 is a structural necessity) |
| Regime classification (rise ratio, h=12) | 84.2 ± 1.2 % vs. 61.2 ± 1.5 % for the original δ-jump rule |
| BOCPD comparison | Exogenous TPR 99.6 %; endogenous TPR 36.0 % (model-alignment effect) |
| Zero-shot real validation | All four documented ignition dates recovered; canonical regimes classified correctly; 9-month lead before endogenous streaming peak |

## Real-world archetypes

Four real-world cases spanning three catalyst types and a 42–91 % censoring range:

| Track | Catalyst | Ignition | Regime | SP censoring | Role |
|---|---|---|---|---|---|
| Simge — Aşkın Olayım | Icardi (Aug–Sep 2022) | 2022-09 | endogenous | 69 % | dev. |
| Kate Bush — Running Up That Hill | Stranger Things S4 (May 2022) | 2022-06 | exogenous | 76 % | dev. |
| Metallica — Master of Puppets | Stranger Things S4 (May 2022) | 2022-08 | exogenous | 91 % | held-out |
| Fleetwood Mac — Dreams | TikTok UGC (Sep–Oct 2020) | 2020-10 | exo. (signal) | 42 % | held-out |

*dev. = generator qualitatively informed; held-out = fully independent.*

## The censoring problem

A track is observed only while it charts; below the threshold the value is
zero. Censoring rates range from 42 % to 91 % across the four archetypes.
Zeros are treated as *unobserved*, not as zero consumption.

## Repository layout

```
├── notebooks/
│   └── ivpm_ew_kaggle_v3.ipynb   # canonical source of every number and figure
├── data/
│   ├── FINAL_ACADEMIC_DATASET_SIMGE.csv             # endogenous archetype (59 mo.)
│   ├── FINAL_ACADEMIC_DATASET_KATE_BUSH.csv         # exogenous archetype (184 mo.)
│   ├── FINAL_ACADEMIC_DATASET_METALLICA_MOP.csv     # held-out exogenous (66 mo.)
│   └── FINAL_ACADEMIC_DATASET_DREAMS_FLEETWOOD.csv  # held-out TikTok-catalytic (78 mo.)
├── results/             # CSV outputs of the full 5-seed v3 run
│   ├── results_seed_grid.csv          # per-seed operating characteristics
│   ├── results_censoring.csv          # 2-of-3 vs 3-of-3 censoring ablation
│   ├── results_seed_regime.csv        # per-seed regime classifier accuracies
│   ├── results_cut_sensitivity.csv    # rise-ratio cut sensitivity sweep (EXP4c)
│   ├── results_detector_replicated.csv# replicated detector comparison (EXP3b)
│   ├── results_sensitivity.csv        # generator sensitivity analysis
│   ├── results_missingness.csv        # broader missingness ablation
│   ├── results_real_cases.csv         # zero-shot validation (4 archetypes)
│   └── results_forecast_bench.csv     # expanding-window forecasting benchmark
├── outputs/figures/     # publication figures produced by the notebook
└── src/
    └── ivpm_ew_core.py  # core module (generator + detectors + classifiers)
```

## Reproducing the results

**On Kaggle (recommended).**
1. Create a Kaggle Dataset with the four CSV files in `data/` (keep filenames).
2. Open `notebooks/ivpm_ew_kaggle_v3.ipynb` and attach the dataset via *Add Input*.
3. Set Accelerator to *None* (CPU-parallel; no GPU needed) and *Run All*.
   Runtime ≈ 90–120 minutes.
4. All tables and figures are written to `/kaggle/working/` and bundled
   into `ivpm_ew_results_v3.zip`.

**Locally.**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/ivpm_ew_kaggle_v3.ipynb
```
Place the four CSV files in `data/` and set `OUT = "./results"` in Cell 1.

## Citation

If you use this code or data, please cite the accompanying paper (IEEE Access, under review).

## License

MIT
