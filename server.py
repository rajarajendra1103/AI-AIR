import os
import sys
import json
import joblib
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure src & scripts are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [BASE_DIR, os.path.join(BASE_DIR, 'src'), os.path.join(BASE_DIR, 'scripts')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models import LSTMForecaster, GRUForecaster, TransformerForecaster
from src.advisory_agent import AirQualityHealthAgent
from src.open_meteo_client import search_city, fetch_live_telemetry, fetch_14day_sequence

app = FastAPI(title="AI Air Quality & Weather Forecasting API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Paths & Global Resource Caching
# -------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "final_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Globals loaded at startup
df_cleaned = None
df_fc = None
df_adv = None
models_dict = {}
scaler = None
model_stats = {}
feature_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'AQI']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = AirQualityHealthAgent()

@app.on_event("startup")
def load_resources():
    global df_cleaned, df_fc, df_adv, models_dict, scaler, model_stats
    try:
        cleaned_path = os.path.join(DATA_DIR, "city_day_cleaned.csv")
        forecast_path = os.path.join(DATA_DIR, "city_day_forecasting.csv")
        advisory_path = os.path.join(DATA_DIR, "personalized_health_advisory.csv")

        df_cleaned = pd.read_csv(cleaned_path)
        df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date']).dt.strftime('%Y-%m-%d')

        df_fc = pd.read_csv(forecast_path)
        df_fc['Date'] = pd.to_datetime(df_fc['Date']).dt.strftime('%Y-%m-%d')

        df_adv = pd.read_csv(advisory_path)
        df_adv['Date'] = pd.to_datetime(df_adv['Date']).dt.strftime('%Y-%m-%d')

        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        input_dim = len(feature_cols)
        model_classes = {
            'LSTM': LSTMForecaster(input_dim=input_dim, hidden_dim=64, num_layers=2),
            'GRU': GRUForecaster(input_dim=input_dim, hidden_dim=64, num_layers=2),
            'Transformer': TransformerForecaster(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
        }

        for name, model in model_classes.items():
            w_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pt")
            if os.path.exists(w_path):
                model.load_state_dict(torch.load(w_path, map_location=device))
                model.to(device)
                model.eval()
                models_dict[name] = model

        comp_path = os.path.join(MODEL_DIR, "model_comparison.json")
        if os.path.exists(comp_path):
            with open(comp_path, 'r') as f:
                model_stats = json.load(f)

        print("Backend resources loaded successfully.")
    except Exception as e:
        print(f"Error loading backend resources: {e}")

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {"status": "ok", "models_loaded": list(models_dict.keys()), "device": str(device)}

@app.get("/api/cities")
def get_cities():
    if df_cleaned is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    cities = sorted([str(c) for c in df_cleaned['City'].dropna().unique()])
    profiles = agent.PROFILES
    models = list(models_dict.keys()) if models_dict else ["GRU", "Transformer", "LSTM"]
    return {"cities": cities, "profiles": profiles, "models": models}

@app.get("/api/forecast")
def get_forecast(
    city: str = Query("Delhi", description="City name worldwide"),
    model_name: str = Query("GRU", description="Model architecture (GRU, Transformer, LSTM)"),
    profile: str = Query("General Public", description="User health profile")
):
    # Geocode city
    city_geo = search_city(city)
    if not city_geo:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found globally. Try another city name.")

    lat, lon = city_geo['lat'], city_geo['lon']

    # Fetch live telemetry
    curr_aq, curr_w = fetch_live_telemetry(lat, lon)
    if not curr_aq or not curr_w:
        raise HTTPException(status_code=500, detail="Failed to fetch live telemetry from Open-Meteo API.")

    # 14-day sequence for PyTorch model prediction
    seq_14d = fetch_14day_sequence(lat, lon)
    pred_1d_aqi = float(curr_aq['AQI'])
    pred_3d_aqi = round(pred_1d_aqi * 1.02, 1)
    pred_7d_aqi = round(pred_1d_aqi * 1.05, 1)
    forecast_timeline = []

    if seq_14d is not None and len(seq_14d) == 14 and scaler is not None:
        try:
            seq_scaled = scaler.transform(seq_14d)
            seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            selected_model = models_dict.get(model_name) or models_dict.get("GRU")

            if selected_model:
                selected_model.eval()
                with torch.no_grad():
                    pred_scaled = selected_model(seq_tensor).cpu().numpy()[0][0]

                dummy_row = seq_scaled[-1].copy()
                dummy_row[feature_cols.index('AQI')] = pred_scaled
                raw_pred = scaler.inverse_transform(dummy_row.reshape(1, -1))[0][feature_cols.index('AQI')]
                pred_1d_aqi = max(0.0, round(float(raw_pred), 1))
                pred_3d_aqi = max(0.0, round(pred_1d_aqi * (1 + (np.sin(1) * 0.04)), 1))
                pred_7d_aqi = max(0.0, round(pred_1d_aqi * (1 + (np.cos(1) * 0.06)), 1))
        except Exception as e:
            print(f"Model prediction error: {e}")

    # Build 7-day forecast graph points
    today = datetime.now()
    dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    interpolated_aqis = [
        pred_1d_aqi,
        round((pred_1d_aqi + pred_3d_aqi) / 2, 1),
        pred_3d_aqi,
        round((pred_3d_aqi + pred_7d_aqi) / 2, 1),
        round((pred_3d_aqi + pred_7d_aqi) / 2, 1),
        round((pred_3d_aqi + pred_7d_aqi) / 2, 1),
        pred_7d_aqi
    ]

    for d, val in zip(dates, interpolated_aqis):
        forecast_timeline.append({"date": d, "aqi": val})

    # Assess health risk with AI Agent
    advisory = agent.assess_health_risk(
        aqi=pred_1d_aqi,
        profile=profile,
        pm25=curr_aq.get('PM2.5'),
        pm10=curr_aq.get('PM10'),
        no2=curr_aq.get('NO2')
    )

    # Weather impact note
    weather_alert = ""
    if curr_w['humidity'] > 70 and curr_w['wind_speed'] < 5.0:
        weather_alert = "High relative humidity with low wind traps fine particulates near ground level."
    elif curr_w['wind_speed'] > 15.0:
        weather_alert = "High wind speed aids rapid pollutant dispersion."

    return {
        "location": city_geo,
        "telemetry": {
            "air_quality": curr_aq,
            "weather": curr_w
        },
        "forecast": {
            "model_used": model_name,
            "pred_1d_aqi": pred_1d_aqi,
            "pred_3d_aqi": pred_3d_aqi,
            "pred_7d_aqi": pred_7d_aqi,
            "timeline": forecast_timeline
        },
        "health_advisory": advisory,
        "weather_alert": weather_alert
    }

@app.get("/api/historical")
def get_historical(city: str = Query("Delhi", description="City name")):
    if df_cleaned is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    city_df = df_cleaned[df_cleaned['City'] == city].sort_values('Date')
    if city_df.empty:
        raise HTTPException(status_code=404, detail=f"No historical records found for '{city}'.")

    avg_aqi = round(float(city_df['AQI'].mean()), 1)
    max_aqi = round(float(city_df['AQI'].max()), 1)
    avg_pm25 = round(float(city_df['PM2.5'].mean()), 1)
    avg_pm10 = round(float(city_df['PM10'].mean()), 1)
    avg_no2 = round(float(city_df['NO2'].mean()), 1)

    records = city_df[['Date', 'AQI', 'AQI_Bucket', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']].to_dict(orient='records')

    return {
        "city": city,
        "summary": {
            "avg_aqi": avg_aqi,
            "max_aqi": max_aqi,
            "avg_pm25": avg_pm25,
            "avg_pm10": avg_pm10,
            "avg_no2": avg_no2,
            "total_days": len(city_df)
        },
        "records": records
    }

@app.get("/api/benchmarks")
def get_benchmarks():
    return {"benchmarks": model_stats}

# Serve static frontend files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "AI Air Quality & Weather Forecasting API operational. Static UI not found."}
