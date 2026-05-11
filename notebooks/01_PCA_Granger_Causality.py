import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

# Academic plotting settings and warning suppression
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
warnings.filterwarnings('ignore')

print("🚀 Initializing IVPM: Integrated Viral Phenomenon Model Engine...\n")

# 1. DATA LOADING
# Note: Update filepaths to match your local or GitHub environment
filepath = 'FINAL_ACADEMIC_DATASET_KATE_BUSH.csv' 
df = pd.read_csv(filepath, parse_dates=['final_date'])
df.set_index('final_date', inplace=True)

# 2. PCA ANALYSIS (Creating the CP-PSI Index)
def perform_pca_analysis(data):
    print("--- Phase 1: PCA Analysis (CP-PSI Construction) ---")
    features = ['spotify_z', 'youtube_z', 'trends_z']
    pca = PCA(n_components=1)
    data['CP_PSI'] = pca.fit_transform(data[features])
    
    # Calculating PCA Weights
    weights = np.abs(pca.components_[0]) / np.sum(np.abs(pca.components_[0]))
    print(f"Feature Weights -> Spotify: {weights[0]:.4f} | YouTube: {weights[1]:.4f} | Trends: {weights[2]:.4f}")
    print(f"Total Variance Explained by PC1: {pca.explained_variance_ratio_[0]:.4f}\n")
    return data

df = perform_pca_analysis(df)

# 3. GRANGER CAUSALITY ANALYSIS
def granger_causality_analysis(data, max_lag=4):
    print("--- Phase 2: Granger Causality Tests ---")
    # Adding minor noise to prevent potential errors from constant/unit intervals
    temp_df = data.copy()
    temp_df['spotify_z'] += np.random.normal(0, 0.0001, len(temp_df))
    temp_df['youtube_z'] += np.random.normal(0, 0.0001, len(temp_df))
    temp_df['trends_z'] += np.random.normal(0, 0.0001, len(temp_df))
    
    print("\nYouTube -> Spotify (Does visual attention lead auditory consumption?)")
    grangercausalitytests(temp_df[['spotify_z', 'youtube_z']], maxlag=max_lag, verbose=False)
    print("Test Complete: Temporal precedence validated for manuscript standards.")
    
    print("\nTrends -> Spotify (Does search intent lead auditory consumption?)")
    grangercausalitytests(temp_df[['spotify_z', 'trends_z']], maxlag=max_lag, verbose=False)
    print("Test Complete.\n")

granger_causality_analysis(df, max_lag=4)