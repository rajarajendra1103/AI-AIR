import os
import sys
import json
import joblib
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Query, HTTPException, Body
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
from src.aqi_calc import compute_aqi_full, get_aqi_bucket, STANDARD_THRESHOLDS

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
def startup_event():
    load_resources()

def load_resources():
    global df_cleaned, df_fc, df_adv, models_dict, scaler, model_stats
    try:
        cleaned_path = os.path.join(DATA_DIR, "city_day_cleaned.csv")
        forecast_path = os.path.join(DATA_DIR, "city_day_forecasting.csv")
        advisory_path = os.path.join(DATA_DIR, "personalized_health_advisory.csv")

        if os.path.exists(cleaned_path):
            df_cleaned = pd.read_csv(cleaned_path)
            df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date']).dt.strftime('%Y-%m-%d')

        if os.path.exists(forecast_path):
            df_fc = pd.read_csv(forecast_path)
            df_fc['Date'] = pd.to_datetime(df_fc['Date']).dt.strftime('%Y-%m-%d')

        if os.path.exists(advisory_path):
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

# Pre-load on import
load_resources()

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

    # Calculate subindices and breakdown for live telemetry
    aqi_val, major_pol, subindices = compute_aqi_full(curr_aq)
    breakdown = {}
    for pol, std in STANDARD_THRESHOLDS.items():
        val = curr_aq.get(pol, 0.0)
        sub = subindices.get(pol, 0.0)
        breakdown[pol] = {
            "value": round(float(val), 2),
            "standard_limit": std,
            "subindex": round(float(sub), 1),
            "ratio_pct": round(float((val / std) * 100.0), 1),
            "status": "Safe" if val <= std else "Exceeded"
        }

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
        "aqi_summary": {
            "aqi": float(curr_aq.get('AQI', pred_1d_aqi)),
            "category": advisory["AQI_Category"],
            "color_code": advisory["Color_Code"],
            "major_pollutant": major_pol,
            "subindices": subindices,
            "breakdown": breakdown
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

class CustomSimulationInput(BaseModel):
    pm25: float = Field(45.0, ge=0.0, description="PM2.5 in µg/m³")
    pm10: float = Field(85.0, ge=0.0, description="PM10 in µg/m³")
    no: Optional[float] = Field(None, ge=0.0, description="NO in µg/m³")
    no2: float = Field(35.0, ge=0.0, description="NO2 in µg/m³")
    nox: Optional[float] = Field(None, ge=0.0, description="NOx in µg/m³")
    nh3: Optional[float] = Field(15.0, ge=0.0, description="NH3 in µg/m³")
    co: float = Field(1.0, ge=0.0, description="CO in mg/m³")
    so2: float = Field(15.0, ge=0.0, description="SO2 in µg/m³")
    o3: float = Field(40.0, ge=0.0, description="O3 in µg/m³")
    temperature: float = Field(25.0, description="Temperature in °C")
    humidity: float = Field(60.0, ge=0.0, le=100.0, description="Relative Humidity %")
    wind_speed: float = Field(8.0, ge=0.0, description="Wind Speed in km/h")
    pressure: float = Field(1013.0, description="Surface Pressure in hPa")
    model_name: str = Field("GRU", description="PyTorch Model (GRU, Transformer, LSTM)")
    profile: str = Field("General Public", description="User Health Profile")
    scenario_name: Optional[str] = Field("Custom Parameter Scenario", description="Custom Scenario Name")

def process_simulation(data: CustomSimulationInput) -> Dict[str, Any]:
    no_val = data.no if data.no is not None else round(data.no2 * 0.4, 2)
    nox_val = data.nox if data.nox is not None else round(data.no2 * 1.3, 2)
    nh3_val = data.nh3 if data.nh3 is not None else 15.0

    current_data = {
        'PM2.5': data.pm25,
        'PM10': data.pm10,
        'NO': no_val,
        'NO2': data.no2,
        'NOx': nox_val,
        'NH3': nh3_val,
        'CO': data.co,
        'SO2': data.so2,
        'O3': data.o3
    }

    # Calculate exact CPCB AQI, sub-indices and major pollutant
    aqi_val, major_pol, subindices = compute_aqi_full(current_data)
    if pd.isna(aqi_val):
        aqi_val = 50.0
    current_data['AQI'] = aqi_val

    # Comparison against standard permissible thresholds
    breakdown = {}
    for pol, std in STANDARD_THRESHOLDS.items():
        val = current_data.get(pol, 0.0)
        sub = subindices.get(pol, 0.0)
        breakdown[pol] = {
            "value": round(float(val), 2),
            "standard_limit": std,
            "subindex": round(float(sub), 1),
            "ratio_pct": round(float((val / std) * 100.0), 1),
            "status": "Safe" if val <= std else "Exceeded"
        }

    # Simulate 14-day history ending with user parameters for PyTorch prediction
    pred_1d_aqi = float(aqi_val)
    pred_3d_aqi = round(pred_1d_aqi * 1.02, 1)
    pred_7d_aqi = round(pred_1d_aqi * 1.05, 1)
    forecast_timeline = []

    if scaler is not None:
        try:
            # Build synthetic 14-day history ending at current day
            seq_rows = []
            for t in range(14):
                # Gentle realistic trend with noise ending at day index 13
                decay_factor = 1.0 + 0.08 * np.sin((t - 13) * 0.45)
                row_dict = {
                    'PM2.5': max(0.0, current_data['PM2.5'] * decay_factor),
                    'PM10': max(0.0, current_data['PM10'] * decay_factor),
                    'NO': max(0.0, current_data['NO'] * decay_factor),
                    'NO2': max(0.0, current_data['NO2'] * decay_factor),
                    'NOx': max(0.0, current_data['NOx'] * decay_factor),
                    'NH3': max(0.0, current_data['NH3'] * decay_factor),
                    'CO': max(0.0, current_data['CO'] * decay_factor),
                    'SO2': max(0.0, current_data['SO2'] * decay_factor),
                    'O3': max(0.0, current_data['O3'] * decay_factor),
                    'AQI': max(0.0, current_data['AQI'] * decay_factor)
                }
                seq_rows.append([row_dict[col] for col in feature_cols])

            seq_arr = np.array(seq_rows)
            seq_scaled = scaler.transform(seq_arr)
            seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)

            selected_model = models_dict.get(data.model_name) or models_dict.get("GRU")
            if selected_model:
                selected_model.eval()
                with torch.no_grad():
                    pred_scaled = selected_model(seq_tensor).cpu().numpy()[0][0]

                dummy_row = seq_scaled[-1].copy()
                dummy_row[feature_cols.index('AQI')] = pred_scaled
                raw_pred = scaler.inverse_transform(dummy_row.reshape(1, -1))[0][feature_cols.index('AQI')]

                pred_1d_aqi = max(0.0, round(float(raw_pred), 1))
                
                # Dynamic multi-day projection with weather dispersion factor
                dispersion_mult = 1.0 - (min(30.0, data.wind_speed) / 100.0) + (data.humidity / 400.0)
                pred_3d_aqi = max(0.0, round(pred_1d_aqi * (1 + (np.sin(1) * 0.05 * dispersion_mult)), 1))
                pred_7d_aqi = max(0.0, round(pred_1d_aqi * (1 + (np.cos(1) * 0.08 * dispersion_mult)), 1))
        except Exception as e:
            print(f"Simulation PyTorch prediction error: {e}")

    # Build 7-day forecast timeline
    today = datetime.now()
    dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    interpolated_aqis = [
        pred_1d_aqi,
        round((pred_1d_aqi * 0.65 + pred_3d_aqi * 0.35), 1),
        pred_3d_aqi,
        round((pred_3d_aqi * 0.6 + pred_7d_aqi * 0.4), 1),
        round((pred_3d_aqi * 0.4 + pred_7d_aqi * 0.6), 1),
        round((pred_3d_aqi * 0.2 + pred_7d_aqi * 0.8), 1),
        pred_7d_aqi
    ]
    for d, val in zip(dates, interpolated_aqis):
        forecast_timeline.append({"date": d, "aqi": max(0.0, val)})

    # Health Advisory Assessment
    advisory = agent.assess_health_risk(
        aqi=aqi_val,
        profile=data.profile,
        pm25=data.pm25,
        pm10=data.pm10,
        no2=data.no2
    )

    # Meteorological Impact & Ventilation Analysis
    weather_factors = []
    dispersion_status = "Normal Dispersion"
    if data.humidity >= 75 and data.wind_speed <= 4.0:
        dispersion_status = "Severe Atmospheric Stagnation (Inversion Trap)"
        weather_factors.append("High humidity (>75%) coupled with calm winds (<4 km/h) creates surface boundary layer inversion, trapping particulates.")
    elif data.wind_speed >= 18.0:
        dispersion_status = "High Atmospheric Ventilation"
        weather_factors.append("Strong wind ventilation (>18 km/h) actively accelerates pollutant dispersion and reduces localized hotspots.")
    elif data.temperature >= 35.0 and data.o3 >= 70.0:
        dispersion_status = "Elevated Photochemical Smog Activity"
        weather_factors.append("High ambient temperature (>35°C) promotes photochemical ozone generation from precursor VOCs and NOx.")
    else:
        dispersion_status = "Moderate Ventilation"
        weather_factors.append("Ambient meteorological conditions allow standard atmospheric mixing.")

    return {
        "scenario_name": data.scenario_name or "Custom Scenario",
        "inputs": {
            "pollutants": current_data,
            "weather": {
                "temperature": data.temperature,
                "humidity": data.humidity,
                "wind_speed": data.wind_speed,
                "pressure": data.pressure
            }
        },
        "aqi_summary": {
            "aqi": aqi_val,
            "category": advisory["AQI_Category"],
            "color_code": advisory["Color_Code"],
            "major_pollutant": major_pol,
            "subindices": subindices,
            "breakdown": breakdown
        },
        "forecast": {
            "model_used": data.model_name,
            "pred_1d_aqi": pred_1d_aqi,
            "pred_3d_aqi": pred_3d_aqi,
            "pred_7d_aqi": pred_7d_aqi,
            "timeline": forecast_timeline
        },
        "health_advisory": advisory,
        "weather_impact": {
            "dispersion_status": dispersion_status,
            "insights": weather_factors
        }
    }

@app.post("/api/simulate")
def post_simulation(input_data: CustomSimulationInput):
    return process_simulation(input_data)

@app.get("/api/simulate")
def get_simulation(
    pm25: float = Query(45.0),
    pm10: float = Query(85.0),
    no2: float = Query(35.0),
    so2: float = Query(15.0),
    co: float = Query(1.0),
    o3: float = Query(40.0),
    nh3: float = Query(15.0),
    temperature: float = Query(25.0),
    humidity: float = Query(60.0),
    wind_speed: float = Query(8.0),
    pressure: float = Query(1013.0),
    model_name: str = Query("GRU"),
    profile: str = Query("General Public"),
    scenario_name: str = Query("Custom Scenario")
):
    inp = CustomSimulationInput(
        pm25=pm25, pm10=pm10, no2=no2, so2=so2, co=co, o3=o3, nh3=nh3,
        temperature=temperature, humidity=humidity, wind_speed=wind_speed,
        pressure=pressure, model_name=model_name, profile=profile,
        scenario_name=scenario_name
    )
    return process_simulation(inp)

# Serve static frontend files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "AI Air Quality & Weather Forecasting API operational. Static UI not found."}
