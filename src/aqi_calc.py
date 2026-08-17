import pandas as pd
import numpy as np

# -------------------------------------------------------------
# CPCB Sub-Index & AQI Helper Functions
# -------------------------------------------------------------
STANDARD_THRESHOLDS = {
    'PM2.5': 60.0,   # µg/m³ (24-hr standard)
    'PM10': 100.0,   # µg/m³ (24-hr standard)
    'NO2': 80.0,     # µg/m³ (24-hr standard)
    'SO2': 80.0,     # µg/m³ (24-hr standard)
    'CO': 2.0,       # mg/m³ (8-hr standard)
    'O3': 100.0,     # µg/m³ (8-hr standard)
    'NH3': 400.0     # µg/m³ (24-hr standard)
}

def get_pm25_subindex(x):
    if x is None or pd.isna(x) or x < 0: return np.nan
    if x <= 30: return x * 50 / 30
    elif x <= 60: return 50 + (x - 30) * 50 / 30
    elif x <= 90: return 100 + (x - 60) * 100 / 30
    elif x <= 120: return 200 + (x - 90) * 100 / 30
    elif x <= 250: return 300 + (x - 120) * 100 / 130
    else: return 400 + (x - 250) * 100 / 130

def get_pm10_subindex(x):
    if x is None or pd.isna(x) or x < 0: return np.nan
    if x <= 50: return x
    elif x <= 100: return 50 + (x - 50)
    elif x <= 250: return 100 + (x - 100) * 100 / 150
    elif x <= 350: return 200 + (x - 250) * 100 / 100
    elif x <= 430: return 300 + (x - 350) * 100 / 80
    else: return 400 + (x - 430) * 100 / 80

def get_no2_subindex(x):
    if x is None or pd.isna(x) or x < 0: return np.nan
    if x <= 40: return x * 50 / 40
    elif x <= 80: return 50 + (x - 40) * 50 / 40
    elif x <= 180: return 100 + (x - 80) * 100 / 100
    elif x <= 280: return 200 + (x - 180) * 100 / 100
    elif x <= 400: return 300 + (x - 280) * 100 / 120
    else: return 400 + (x - 400) * 100 / 120

def get_so2_subindex(x):
    if x is None or pd.isna(x) or x < 0: return np.nan
    if x <= 40: return x * 50 / 40
    elif x <= 80: return 50 + (x - 40) * 50 / 40
    elif x <= 380: return 100 + (x - 80) * 100 / 300
    elif x <= 800: return 200 + (x - 380) * 100 / 420
    elif x <= 1600: return 300 + (x - 800) * 100 / 800
    else: return 400 + (x - 1600) * 100 / 800

def get_co_subindex(x):
    if x is None or pd.isna(x) or x < 0: return np.nan
    if x <= 1: return x * 50
    elif x <= 2: return 50 + (x - 1) * 50
    elif x <= 10: return 100 + (x - 2) * 100 / 8
    elif x <= 17: return 200 + (x - 10) * 100 / 7
    elif x <= 34: return 300 + (x - 17) * 100 / 17
    else: return 400 + (x - 34) * 100 / 17

def get_o3_subindex(x):
    if x is None or pd.isna(x) or x < 0: return np.nan
    if x <= 50: return x * 50 / 50
    elif x <= 100: return 50 + (x - 50) * 50 / 50
    elif x <= 168: return 100 + (x - 100) * 100 / 68
    elif x <= 208: return 200 + (x - 168) * 100 / 40
    elif x <= 748: return 300 + (x - 208) * 100 / 540
    else: return 400 + (x - 748) * 100 / 540

def get_nh3_subindex(x):
    if x is None or pd.isna(x) or x < 0: return np.nan
    if x <= 200: return x * 50 / 200
    elif x <= 400: return 50 + (x - 200) * 50 / 200
    elif x <= 800: return 100 + (x - 400) * 100 / 400
    elif x <= 1200: return 200 + (x - 800) * 100 / 400
    elif x <= 1800: return 300 + (x - 1200) * 100 / 600
    else: return 400 + (x - 1800) * 100 / 600

def get_aqi_bucket(aqi):
    if aqi is None or pd.isna(aqi): return "Unknown"
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def compute_aqi_full(row):
    subs = {
        'PM2.5': get_pm25_subindex(row.get('PM2.5', np.nan)),
        'PM10': get_pm10_subindex(row.get('PM10', np.nan)),
        'NO2': get_no2_subindex(row.get('NO2', np.nan)),
        'SO2': get_so2_subindex(row.get('SO2', np.nan)),
        'CO': get_co_subindex(row.get('CO', np.nan)),
        'O3': get_o3_subindex(row.get('O3', np.nan)),
        'NH3': get_nh3_subindex(row.get('NH3', np.nan))
    }
    valid_subs = {k: float(v) for k, v in subs.items() if not pd.isna(v)}
    
    if len(valid_subs) >= 1:
        major_pol = max(valid_subs, key=valid_subs.get)
        aqi_val = float(np.round(valid_subs[major_pol]))
        return aqi_val, major_pol, valid_subs
    else:
        return np.nan, "None", {}

def compute_aqi_details(row):
    aqi_val, major_pol, _ = compute_aqi_full(row)
    return aqi_val, major_pol
