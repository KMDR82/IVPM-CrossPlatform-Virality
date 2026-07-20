# IVPM-EW: Causal Early Warning for Digital Music Virality Under Chart Censoring

This repository contains the data, code, and results for the paper:

> *From Post-hoc Explanation to Causal Early Warning: A Simulation-Calibrated
> Multi-Platform Framework for Detecting and Classifying Digital Music
> Virality Under Chart Censoring.*

IVPM-EW is an online (causal) early-warning system for viral resurgences of
back-catalog music. It monitors three heterogeneous platform signals
(YouTube views, Google Trends search interest, chart-derived Spotify
streams), declares viral **ignition** through a censoring-aware 2-of-3
cross-platform confirmation rule, and classifies the viral **regime**
(endogenous organic growth vs. exogenous media shock) with an interpretable
rise-ratio statistic.

## Key results

All numbers below are produced by a single notebook
(`notebooks/ivpm_ew_kaggle_v2.ipynb`) and are stored in `results/`.
Monte Carlo values are mean ± sd over 5 seeds (900 series per run).

| Finding | Result |
|---|---|
| Selected operating point (θ=1.75, 2-of-3, persistence=2) | TPR 98.8 ± 0.4 %, false alarms 2.1 ± 0.6 %, delay 1.9 ± 0.1 months |
| Single-platform monitoring | 75.7 ± 1.4 % false alarms (unusable) |
| 3-of-3 rule under chart censoring | TPR collapses 99.5 % → 2.0 % (2-of-3 is a structural necessity) |
| Regime classification (rise ratio, h=12) | 84.2 ± 1.2 % vs. 61.2 ± 1.5 % for the original δ-jump rule (McNemar median p ≈ 3×10⁻²⁵) |
| Sim-to-real transfer | ML classifiers reach 99.7–99.9 % on synthetic data but misclassify a real archetype; the one-parameter rise ratio transfers to both |
| Zero-shot real validation | Both documented ignition dates recovered exactly (2022-09, 2022-06); both regimes correct; 9-month lead before the endogenous streaming peak |

## The censoring problem

The streaming columns are built from daily chart archives (kworb.net). A
track is observed only while it charts; below the chart threshold the value
is zero. In these datasets, 69 % (endogenous) and 76 % (exogenous) of
monthly streaming observations are censored zeros. Zeros must be treated as
*unobserved*, not as zero consumption. The framework and the synthetic
testbed are designed around this constraint.

## Repository layout

```
├── notebooks/
│   └── ivpm_ew_kaggle_v2.ipynb   # canonical source of every number and figure
├── data/
│   ├── FINAL_ACADEMIC_DATASET_SIMGE.csv       # endogenous archetype (59 months)
│   └── FINAL_ACADEMIC_DATASET_KATE_BUSH.csv   # exogenous archetype (184 months)
├── results/                      # CSV outputs of the full 5-seed run
│   ├── results_grid.csv                  # threshold × confirmation × persistence grid
│   ├── results_seed_grid.csv             # per-seed operating points
│   ├── results_seed_grid_stats.csv       # mean ± sd (± 95 % CI) per operating point
│   ├── results_censoring.csv             # censoring ablation
│   ├── results_detectors.csv             # persistent-exceedance vs CUSUM vs causal composite
│   ├── results_regime_horizon.csv        # regime accuracy vs confirmation horizon
│   ├── results_seed_regime.csv           # per-seed classifier accuracies + McNemar p-values
│   ├── results_real_cases.csv            # zero-shot validation on the two archetypes
│   ├── results_real_delta.csv            # δ-jump rule on the real archetypes
│   └── results_forecast_bench.csv        # expanding-window forecasting benchmark
├── figures/                      # publication figures produced by the notebook
└── src/
    └── ivpm_ew_core.py           # core module (generator + detectors + classifiers),
                                  # extracted verbatim from Cell 2 of the notebook
```

## Reproducing the results

**On Kaggle (recommended).**
1. Create a Kaggle Dataset containing the two CSV files in `data/`
   (keep the file names unchanged).
2. Create a new notebook from `notebooks/ivpm_ew_kaggle_v2.ipynb` and attach
   the dataset via *Add Input*.
3. Set Accelerator to *None* (the workload is CPU-parallel; no GPU is used)
   and select *Run All*. Runtime is roughly 35–60 minutes.
4. All tables and figures are written to `/kaggle/working/` and bundled
   into `ivpm_ew_results_v2.zip`.

**Locally.**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/ivpm_ew_kaggle_v2.ipynb
```
Place the two CSV files anywhere below the working directory; the notebook
locates them by file name.

Randomness is controlled by explicit seeds (7, 17, 27, 37, 47), so repeated
runs reproduce the published tables.

## Data description

Each CSV contains monthly rows with the raw platform columns and, for
transparency, the globally standardized z-scores used in the original IVPM
study. The paper documents why global z-scores must not be used for online
detection (look-ahead), and the notebook recomputes all normalizations
causally from the raw columns.

| Column | Description |
|---|---|
| `final_date` | month start date |
| `tr_streams` / `sp_streams` | monthly Spotify streams from daily chart archives (0 = not charting, i.e. censored) |
| `tr_rank` | chart rank where applicable (Simge dataset) |
| `yt_views` | monthly YouTube views |
| `google_search_interest` / `trends_interest` | Google Trends monthly interest (floor = 1) |
| `*_z` | legacy full-sample z-scores (do not use for online analysis) |

Note the measurement asymmetry: the Simge dataset uses Turkey chart streams,
while the Kate Bush dataset uses global streams.

## Legacy code

The original IVPM analysis scripts (post-hoc PCA/Granger/SVR–RF pipeline)
have been superseded by the notebook above. The descriptive analyses they
produced (Granger tables, cross-correlations, PCA weights) are retained in
the paper as in-sample descriptive statistics.

## Citation

If you use this code or data, please cite the paper (citation to be added
upon publication) and this repository.

## License

Code is released under the MIT License (see `LICENSE`). The datasets are
derived from publicly accessible sources (kworb.net chart archives, YouTube,
Google Trends) and are provided for research reproducibility.
