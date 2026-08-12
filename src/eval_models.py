import torch
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.models import LSTMForecaster, GRUForecaster, TransformerForecaster  # type: ignore
except ImportError:
    from models import LSTMForecaster, GRUForecaster, TransformerForecaster  # type: ignore
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "final_dataset", "city_day_cleaned.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'AQI']
TARGET_COL = 'AQI'
SEQ_LEN = 14

def create_sequences(df, feature_cols, target_col, seq_len=14):
    X, y = [], []
    for city, group in df.groupby('City'):
        group = group.sort_values('Date').reset_index(drop=True)
        feat_data = group[feature_cols].values
        target_data = group[target_col].values
        
        for i in range(len(group) - seq_len):
            X.append(feat_data[i:i+seq_len])
            y.append(target_data[i+seq_len])
            
    return np.array(X), np.array(y)

def evaluate():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    
    test_df = df[df['Date'] >= '2019-07-01'].copy()
    
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    test_df[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])
    
    X_test, y_test = create_sequences(test_df, FEATURE_COLS, TARGET_COL, SEQ_LEN)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    input_dim = len(FEATURE_COLS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = {
        'LSTM': LSTMForecaster(input_dim=input_dim, hidden_dim=64, num_layers=2),
        'GRU': GRUForecaster(input_dim=input_dim, hidden_dim=64, num_layers=2),
        'Transformer': TransformerForecaster(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
    }
    
    results = {}
    for name, model in models.items():
        weights_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pt")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                preds_scaled = model(X_tensor.to(device)).cpu().numpy().flatten()
                
            # Inverse scale predictions for AQI
            dummy = np.zeros((len(preds_scaled), len(FEATURE_COLS)))
            dummy[:, FEATURE_COLS.index('AQI')] = preds_scaled
            preds_aqi = scaler.inverse_transform(dummy)[:, FEATURE_COLS.index('AQI')]
            preds_aqi = np.clip(preds_aqi, 0, 1000)
            
            dummy_t = np.zeros((len(y_test), len(FEATURE_COLS)))
            dummy_t[:, FEATURE_COLS.index('AQI')] = y_test
            targets_aqi = scaler.inverse_transform(dummy_t)[:, FEATURE_COLS.index('AQI')]
            
            mae = mean_absolute_error(targets_aqi, preds_aqi)
            rmse = np.sqrt(mean_squared_error(targets_aqi, preds_aqi))
            r2 = r2_score(targets_aqi, preds_aqi)
            mape = np.mean(np.abs((targets_aqi - preds_aqi) / (targets_aqi + 1e-5))) * 100
            
            results[name] = {
                'MAE': round(float(mae), 2),
                'RMSE': round(float(rmse), 2),
                'R2': round(float(r2), 4),
                'MAPE': round(float(mape), 2)
            }
            print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}, MAPE={mape:.2f}%")
            
    with open(os.path.join(MODEL_DIR, "model_comparison.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Saved model_comparison.json successfully!")

if __name__ == '__main__':
    evaluate()
