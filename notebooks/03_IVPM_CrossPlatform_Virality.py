def run_generalized_ivpm(filepath, dataset_name="Dataset"):
    print(f"\n{'='*75}")
    print(f"🚀 IVPM CORE MODELING ALGORITHM - {dataset_name}")
    print(f"{'='*75}\n")
    
    df = pd.read_csv(filepath)
    df['final_date'] = pd.to_datetime(df['final_date'])
    df.set_index('final_date', inplace=True)
    
    # Focus on the relevant timeframe
    df = df['2021-01-01':].copy()
    
    # Automatic Shock Detection (Bai & Perron Inspired)
    jumps = df['spotify_z'].diff()
    t_shock = jumps[jumps > 2.0].idxmax() if len(jumps[jumps > 2.0]) > 0 else None
    
    # Lag Engineering (Granger-validated lags)
    df['youtube_lag1'] = df['youtube_z'].shift(1)
    df['youtube_lag2'] = df['youtube_z'].shift(2)
    df['youtube_lag3'] = df['youtube_z'].shift(3)
    df['trends_lag1'] = df['trends_z'].shift(1)
    df['trends_lag2'] = df['trends_z'].shift(2)
    
    # Interrupted Time Series (ITS) Dummy Variable
    if t_shock:
        df['external_shock'] = np.where(df.index >= t_shock, 1, 0)
    else:
        df['external_shock'] = 0
    
    model_data = df.dropna()
    
    X = model_data[['youtube_lag1', 'youtube_lag2', 'youtube_lag3', 
                    'trends_lag1', 'trends_lag2', 'external_shock']]
    y = model_data['spotify_z']
    
    # Chronological Split (75% Training)
    split_index = int(len(model_data) * 0.75)
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    
    # Support Vector Regression (Primary Non-linear Model)
    svr = SVR(kernel='rbf', C=8.0, epsilon=0.1, gamma='scale')
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X)
    
    # Random Forest Regressor (Control Baseline)
    rf = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X)
    
    # Performance Evaluation
    print(f"SVR Overall R2 Score : {r2_score(y, y_pred_svr):.4f}")
    print(f"RF  Overall R2 Score : {r2_score(y, y_pred_rf):.4f}")
    print(f"SVR MAE              : {mean_absolute_error(y, y_pred_svr):.4f}\n")

# EXECUTION
# Note: Ensure these filepaths are correct for your repository structure
path_endogenous = 'FINAL_ACADEMIC_DATASET_SIMGE.csv'
run_generalized_ivpm(path_endogenous, "Endogenous Organic Growth (Aşkın Olayım)")

path_exogenous = 'FINAL_ACADEMIC_DATASET_KATE_BUSH.csv'
run_generalized_ivpm(path_exogenous, "Exogenous Cultural Shock (Kate Bush)")