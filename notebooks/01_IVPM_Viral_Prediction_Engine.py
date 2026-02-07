"""
IVPM: Integrated Virality Prediction Model - Core Engine
Part of the study: "Cross-Platform Digital Virality and External Cultural Catalysts"
Author: [Your Name]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, confusion_matrix
from statsmodels.tsa.stattools import grangercausalitytests

# Set academic plotting style
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def load_and_preprocess(filepath):
    """Load dataset and prepare standardized Z-scores."""
    df = pd.read_csv(filepath, parse_dates=['final_date'])
    df.set_index('final_date', inplace=True)
    return df

def perform_pca_analysis(df):
    """
    Construct CP-PSI Index using Principal Component Analysis.
    Reference: Table 2 and Figure 6 in the manuscript.
    """
    features = ['spotify_z', 'youtube_z', 'trends_z']
    pca = PCA(n_components=1)
    df['CP_PSI'] = pca.fit_transform(df[features])
    
    # Calculate weights based on PCA loadings
    weights = np.abs(pca.components_[0]) / np.sum(np.abs(pca.components_[0]))
    print(f"PCA Weights - Spotify: {weights[0]:.4f}, YouTube: {weights[1]:.4f}, Trends: {weights[2]:.4f}")
    print(f"Total Variance Explained by PC1: {pca.explained_variance_ratio_[0]:.4f}")
    return df

def granger_causality_analysis(df, max_lag=5):
    """
    Test lead-lag relationships between platforms.
    Reference: Section 4.1 and Figure 3.
    """
    print("\n--- Granger Causality Test: YouTube -> Spotify ---")
    grangercausalitytests(df[['spotify_z', 'youtube_z']], maxlag=max_lag)
    
    print("\n--- Granger Causality Test: Trends -> Spotify ---")
    grangercausalitytests(df[['spotify_z', 'trends_z']], maxlag=max_lag)

def train_ivpm_model(df):
    """
    Train SVR model using Granger-identified lagged features (Lag-3).
    Reference: Section 4.4 and Table 5 (Ablation Study).
    """
    # Feature Engineering
    df['youtube_lag3'] = df['youtube_z'].shift(3)
    df['trends_lag2'] = df['trends_z'].shift(2)
    model_data = df.dropna()
    
    X = model_data[['youtube_lag3', 'trends_lag2']]
    y = model_data['spotify_z']
    
    # Support Vector Regression (SVR) optimization
    svr = SVR(C=10.0, epsilon=0.1, kernel='rbf')
    svr.fit(X, y)
    
    y_pred = svr.predict(X)
    print(f"\nModel Performance (SVR): R2 = {r2_score(y, y_pred):.4f}")
    return svr

if __name__ == "__main__":
    # Standard implementation flow
    data = load_and_preprocess('FINAL_ACADEMIC_DATASET_SIMGE.csv')
    data = perform_pca_analysis(data)
    granger_causality_analysis(data)
    model = train_ivpm_model(data)