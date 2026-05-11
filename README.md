# IVPM: Integrated Viral Phenomenon Model

This repository contains the official implementation of the **Integrated Viral Phenomenon Model (IVPM)**, introduced in the paper:

> **"Modeling Endogenous and Exogenous Virality in the Digital Music Economy: The Integrated Viral Phenomenon Model (IVPM) via Multi-Platform Time Series Analysis"**

The framework distinguishes between **endogenous (organic growth)** and **exogenous (shock-driven)** cultural phenomena using multi-platform time series data (Spotify, YouTube, Google Trends).

## 🚀 Project Overview

The IVPM is a hybrid computational framework that maps the full lifecycle of digital music assets. It combines:
- PCA-based **CP-PSI** (Cross-Platform Parasocial Synchronization Index)
- Granger Causality for temporal precedence
- Interrupted Time Series (ITS) for structural break detection
- Hybrid modeling with **SVR** (continuous) and **Random Forest** (discrete shocks)

## 📂 Repository Structure

- **`01_PCA_Granger_Causality.py`**  
  Performs PCA-based CP-PSI construction and Granger Causality tests.

- **`02_IVPM_Ablation_Study.py`**  
  Rigorous ablation study evaluating the contribution of each platform (YouTube-only, Trends-only, Cross-Platform, Full Model).

- **`03_IVPM_CrossPlatform_Virality.py`**  
  Core modeling engine implementing the full IVPM architecture with automatic shock detection.

- **`data/`** (to be added)  
  Contains `FINAL_ACADEMIC_DATASET_SIMGE.csv` and `FINAL_ACADEMIC_DATASET_KATE_BUSH.csv`.

## 📊 Key Findings (from the paper)

- Validated a **~90-day (Lag-3)** incubation period for endogenous virality.
- **SVR** performs better on organic growth ($R^2 = 0.7717$), while **Random Forest** excels at exogenous shocks ($R^2 = 0.8660$).
- Demonstrated an **8.23-fold** revenue increase after the structural break in the endogenous case.
- Identified the "**Vector Collapse**" phenomenon during exogenous shocks.

## 🛠 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/KMDR82/IVPM-CrossPlatform-Virality.git
cd IVPM-CrossPlatform-Virality
