import requests
import pandas as pd
import numpy as np
import os
import sys
import time

# Add project root, scripts & src to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.advisory_agent import AirQualityHealthAgent
    from src.aqi_calc import compute_aqi_details
except ImportError:
    from advisory_agent import AirQualityHealthAgent  # type: ignore
    from aqi_calc import compute_aqi_details  # type: ignore

FEATURE_COLS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'AQI']

def search_city(city_name):
    """
    Geocodes a city name using Open-Meteo Geocoding API.
    """
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
        res = requests.get(url, timeout=10).json()
        if res.get('results') and len(res['results']) > 0:
            result = res['results'][0]
            return {
                'name': result.get('name'),
                'country': result.get('country', ''),
                'admin1': result.get('admin1', ''),
                'lat': result.get('latitude'),
                'lon': result.get('longitude')
            }
    except Exception as e:
        print(f"Geocoding error: {e}")
    return None

def fetch_live_telemetry(lat, lon, max_retries=3):
    """
    Fetches live weather and air quality telemetry from Open-Meteo API.
    Retries up to max_retries times with exponential backoff on failure.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(0.5 * (2 ** (attempt - 1)))  # 0.5s, 1s backoff

            # Weather
            w_url = (
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                "&current=temperature_2m,relative_humidity_2m,surface_pressure,"
                "wind_speed_10m,wind_direction_10m,weather_code"
            )
            w_res = requests.get(w_url, timeout=15).json().get('current', {})

            # Air Quality (ammonia removed — not reliably available everywhere)
            aq_url = (
                f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
                "&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,ammonia"
            )
            aq_json = requests.get(aq_url, timeout=15).json()
            aq_res = aq_json.get('current', {})

            # Validate we got at least some data
            if not w_res or not aq_res:
                raise ValueError("Empty response from Open-Meteo API")

            # Unit conversion & safe null handling
            def _safe(v, default=0.0):
                return float(v) if v is not None else default

            co_mg = _safe(aq_res.get('carbon_monoxide'), 0.0) / 1000.0

            current_data = {
                'PM2.5': _safe(aq_res.get('pm2_5')),
                'PM10':  _safe(aq_res.get('pm10')),
                'NO':    _safe(aq_res.get('nitrogen_dioxide')) * 0.4,
                'NO2':   _safe(aq_res.get('nitrogen_dioxide')),
                'NOx':   _safe(aq_res.get('nitrogen_dioxide')) * 1.3,
                'NH3':   _safe(aq_res.get('ammonia')),
                'CO':    co_mg,
                'SO2':   _safe(aq_res.get('sulphur_dioxide')),
                'O3':    _safe(aq_res.get('ozone')),
            }

            # Calculate CPCB AQI
            aqi_val, major_pol = compute_aqi_details(current_data)
            current_data['AQI'] = float(aqi_val) if (aqi_val is not None and not pd.isna(aqi_val)) else 50.0
            current_data['Major_Pollutant'] = major_pol or 'PM2.5'

            weather_data = {
                'temperature':   _safe(w_res.get('temperature_2m'), 25.0),
                'humidity':      _safe(w_res.get('relative_humidity_2m'), 60.0),
                'pressure':      _safe(w_res.get('surface_pressure'), 1013.0),
                'wind_speed':    _safe(w_res.get('wind_speed_10m'), 5.0),
                'wind_direction':_safe(w_res.get('wind_direction_10m'), 0.0),
                'weather_code':  int(w_res.get('weather_code') or 0),
            }

            return current_data, weather_data

        except Exception as e:
            last_exc = e
            print(f"Telemetry fetch error (attempt {attempt + 1}/{max_retries}): {e}")

    print(f"All {max_retries} telemetry fetch attempts failed. Last error: {last_exc}")
    return None, None

def fetch_14day_sequence(lat, lon):
    """
    Retrieves the last 14 days of hourly air quality data from Open-Meteo,
    resamples to daily averages, computes CPCB AQI, and returns a pandas DataFrame
    matching FEATURE_COLS.
    """
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&past_days=14&forecast_days=1&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,ammonia"
        res = requests.get(url, timeout=10).json()
        
        hourly = res.get('hourly', {})
        if not hourly or 'time' not in hourly:
            return None
            
        time_arr = pd.to_datetime(hourly['time'])
        n_rows = len(time_arr)
        
        pm25 = hourly.get('pm2_5', [0]*n_rows)
        pm10 = hourly.get('pm10', [0]*n_rows)
        no2 = hourly.get('nitrogen_dioxide', [0]*n_rows)
        so2 = hourly.get('sulphur_dioxide', [0]*n_rows)
        o3 = hourly.get('ozone', [0]*n_rows)
        nh3 = hourly.get('ammonia', [0]*n_rows) if 'ammonia' in hourly else [0]*n_rows
        co_raw = hourly.get('carbon_monoxide', [0]*n_rows)
        
        co = [val / 1000.0 if val is not None else 0 for val in co_raw]
        
        df_h = pd.DataFrame({
            'time': time_arr,
            'PM2.5': [v if v is not None else 0 for v in pm25],
            'PM10': [v if v is not None else 0 for v in pm10],
            'NO2': [v if v is not None else 0 for v in no2],
            'SO2': [v if v is not None else 0 for v in so2],
            'O3': [v if v is not None else 0 for v in o3],
            'NH3': [v if v is not None else 0 for v in nh3],
            'CO': co
        })
        
        df_h['NO'] = df_h['NO2'] * 0.4
        df_h['NOx'] = df_h['NO2'] * 1.3
        
        df_h['Date'] = df_h['time'].dt.date
        df_d = df_h.groupby('Date').mean(numeric_only=True).reset_index()
        
        # Take the last 14 days
        df_d = df_d.tail(14).copy()
        
        # Calculate CPCB AQI for each day
        aqi_list = []
        for idx, row in df_d.iterrows():
            aqi_val, _ = compute_aqi_details(row.to_dict())
            aqi_list.append(aqi_val if not pd.isna(aqi_val) else 50)
            
        df_d['AQI'] = aqi_list
        df_d = df_d.bfill().ffill().fillna(0)
        
        # Ensure exact column ordering
        return df_d[FEATURE_COLS].values
    except Exception as e:
        print(f"14-day sequence fetch error: {e}")
        return None
