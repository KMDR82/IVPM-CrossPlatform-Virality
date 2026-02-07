# IVPM: Integrated Virality Prediction Model

This repository contains the official implementation of the **Integrated Virality Prediction Model (IVPM)**, as presented in the study of cross-platform digital virality triggered by external cultural catalysts (The "Icardi Effect" Case Study).

## Project Overview
IVPM is a hybrid machine learning architecture designed to forecast viral spikes by integrating multi-platform data (Spotify, YouTube, and Google Trends). The model utilizes:
- **Granger Causality** to identify lead-lag relationships.
- **PCA** for multi-dimensional index weighting (CP-PSI).
- **SVR & Random Forest** for non-linear prediction and feature importance analysis.
- **VADER Sentiment Analysis** for semantic validation.

## Key Features
- **Temporal Synchronization:** Identifies the 3-month incubation period between visual engagement and financial streams.
- **Structural Break Detection:** Quantifies systemic shocks in digital assets using Z-score volatility.
- **Comparative Validation:** Includes a control group analysis (e.g., "Öpücem") to distinguish organic growth from viral anomalies.

## Repository Content
- `01_IVPM_Viral_Prediction_Engine.py`: Core IVPM engine including PCA weighting, Granger tests, and SVR forecasting.
- `02_Semantic_Sentiment_Analysis.py`: Large-scale YouTube comment scraping and VADER sentiment velocity calculation.
- `03_Control_Group_Specificity_Test.py`: Benchmarking against non-viral assets for model specificity testing.

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/KMDR82/IVPM-Viral-Prediction.git]