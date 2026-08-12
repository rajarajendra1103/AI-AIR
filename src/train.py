import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
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
os.makedirs(MODEL_DIR, exist_ok=True)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

FEATURE_COLS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'AQI']
TARGET_COL = 'AQI'
SEQ_LEN = 14  # 14 days input history

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(df, feature_cols, target_col, seq_len=14):
    X, y = [], []
    for city, group in df.groupby('City'):
        group = group.sort_values('Date').reset_index(drop=True)
        feat_data = group[feature_cols].values
        target_data = group[target_col].values
        
        for i in range(len(group) - seq_len):
            X.append(feat_data[i:i+seq_len])
            y.append(target_data[i+seq_len]) # predict next day AQI
            
    return np.array(X), np.array(y)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())
            
    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    
    mae = mean_absolute_error(targets_arr, preds_arr)
    rmse = np.sqrt(mean_squared_error(targets_arr, preds_arr))
    r2 = r2_score(targets_arr, preds_arr)
    mape = np.mean(np.abs((targets_arr - preds_arr) / (targets_arr + 1e-5))) * 100
    
    return {
        'MAE': round(float(mae), 4),
        'RMSE': round(float(rmse), 4),
        'R2': round(float(r2), 4),
        'MAPE': round(float(mape), 4),
        'predictions': preds_arr.tolist(),
        'targets': targets_arr.tolist()
    }

def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Train / Val / Test split by Date
    train_df = df[df['Date'] < '2019-01-01'].copy()
    val_df = df[(df['Date'] >= '2019-01-01') & (df['Date'] < '2019-07-01')].copy()
    test_df = df[df['Date'] >= '2019-07-01'].copy()
    
    print(f"Dataset split counts: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Fit scaler on train features
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print("Saved feature scaler to scaler.pkl")
    
    # Transform datasets
    for d in [df, train_df, val_df, test_df]:
        d[FEATURE_COLS] = scaler.transform(d[FEATURE_COLS])
        
    X_train, y_train = create_sequences(train_df, FEATURE_COLS, TARGET_COL, SEQ_LEN)
    X_val, y_val = create_sequences(val_df, FEATURE_COLS, TARGET_COL, SEQ_LEN)
    X_test, y_test = create_sequences(test_df, FEATURE_COLS, TARGET_COL, SEQ_LEN)
    
    print(f"Sequence shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=128, shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_dim = len(FEATURE_COLS)
    
    models = {
        'LSTM': LSTMForecaster(input_dim=input_dim, hidden_dim=64, num_layers=2).to(device),
        'GRU': GRUForecaster(input_dim=input_dim, hidden_dim=64, num_layers=2).to(device),
        'Transformer': TransformerForecaster(input_dim=input_dim, d_model=64, nhead=4, num_layers=2).to(device)
    }
    
    epochs = 25
    comparison_results = {}
    
    for name, model in models.items():
        print(f"\n================ Training {name} Model ================")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_val_loss = float('inf')
        best_model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pt")
        
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validation loss
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    val_loss += criterion(model(X_b), y_b).item() * len(X_b)
            val_loss /= len(val_loader.dataset)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                
            if epoch % 5 == 0 or epoch == epochs:
                print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
        # Load best model for test evaluation
        model.load_state_dict(torch.load(best_model_path))
        metrics = evaluate_model(model, test_loader, device)
        
        print(f"--> {name} Test Performance: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, R²={metrics['R2']}")
        
        comparison_results[name] = {
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'R2': metrics['R2'],
            'MAPE': metrics['MAPE']
        }
        
    with open(os.path.join(MODEL_DIR, "model_comparison.json"), 'w') as f:
        json.dump(comparison_results, f, indent=4)
        
    print("\nAll models successfully trained and evaluated! Comparison saved to model_comparison.json.")

if __name__ == '__main__':
    main()
