import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

# Suppress warnings for cleaner academic output
warnings.filterwarnings('ignore')

def run_regression_ablation(filepath, dataset_name=Dataset)
    
    Performs an ablation study to evaluate the contribution of each 
    feature set to the overall explanatory power of the IVPM.
    
    print(fn{'='75})
    print(f🔬 ABLATION STUDY (REGRESSION) - {dataset_name})
    print(f{'='75}n)
    
    # 1. Data Preparation
    df = pd.read_csv(filepath)
    df['final_date'] = pd.to_datetime(df['final_date'])
    df.set_index('final_date', inplace=True)
    df = df['2021-01-01'].copy()
    
    # Automatic Shock Detection (Bai & Perron Inspired)
    jumps = df['spotify_z'].diff()
    t_shock = jumps[jumps  2.0].idxmax() if len(jumps[jumps  2.0])  0 else None
    
    # Feature Engineering Lag Definitions
    df['youtube_lag1'] = df['youtube_z'].shift(1)
    df['youtube_lag2'] = df['youtube_z'].shift(2)
    df['youtube_lag3'] = df['youtube_z'].shift(3)
    df['trends_lag1'] = df['trends_z'].shift(1)
    df['trends_lag2'] = df['trends_z'].shift(2)
    
    # Interrupted Time Series (ITS) Component
    if t_shock
        df['external_shock'] = np.where(df.index = t_shock, 1, 0)
    else
        df['external_shock'] = 0
        
    model_data = df.dropna()
    y = model_data['spotify_z']
    
    # 2. Ablation Scenarios (Feature Sets)
    ablation_sets = {
        Ablated Model A (YouTube Only) ['youtube_lag1', 'youtube_lag2', 'youtube_lag3'],
        Ablated Model B (Trends Only) ['trends_lag1', 'trends_lag2'],
        Ablated Model C (YouTube + Trends) ['youtube_lag1', 'youtube_lag2', 'youtube_lag3', 'trends_lag1', 'trends_lag2'],
        Full IVPM (Cross-Platform + Shock) ['youtube_lag1', 'youtube_lag2', 'youtube_lag3', 'trends_lag1', 'trends_lag2', 'external_shock']
    }
    
    # Chronological Split (75% Training)
    split_index = int(len(model_data)  0.75)
    
    # Using SVR as the primary model for ablation consistency
    model = SVR(kernel='rbf', C=8.0, epsilon=0.1, gamma='scale')
    
    print(f{'Model Configuration'35}  {'R2 Score'10}  {'MAE (Error)'10})
    print(-  65)
    
    for name, features in ablation_sets.items()
        X = model_data[features]
        
        X_train = X.iloc[split_index]
        y_train = y.iloc[split_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X) 
        
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        print(f{name35}  {r2.4f}       {mae.4f})
    print(n)

# --- EXECUTION ---
# Ensure filepaths match your local directory or GitHub structure
path_endogenous = 'FINAL_ACADEMIC_DATASET_SIMGE.csv'
run_regression_ablation(path_endogenous, Endogenous Case (Organic Growth))

path_exogenous = 'FINAL_ACADEMIC_DATASET_KATE_BUSH.csv'
run_regression_ablation(path_exogenous, Exogenous Case (Structural Shock))