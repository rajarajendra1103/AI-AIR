import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR = r"c:\Users\Thilak chodagiri\Desktop\AIR\dataset"
OUTPUT_DIR = r"c:\Users\Thilak chodagiri\Desktop\AIR\final_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting Air Quality Data Cleaning & Feature Pipeline...")

# -------------------------------------------------------------
# CPCB Sub-Index & AQI Helper Functions
# -------------------------------------------------------------
def get_pm25_subindex(x):
    if pd.isna(x) or x < 0: return np.nan
    if x <= 30: return x * 50 / 30
    elif x <= 60: return 50 + (x - 30) * 50 / 30
    elif x <= 90: return 100 + (x - 60) * 100 / 30
    elif x <= 120: return 200 + (x - 90) * 100 / 30
    elif x <= 250: return 300 + (x - 120) * 100 / 130
    else: return 400 + (x - 250) * 100 / 130

def get_pm10_subindex(x):
    if pd.isna(x) or x < 0: return np.nan
    if x <= 50: return x
    elif x <= 100: return 50 + (x - 50)
    elif x <= 250: return 100 + (x - 100) * 100 / 150
    elif x <= 350: return 200 + (x - 250) * 100 / 100
    elif x <= 430: return 300 + (x - 350) * 100 / 80
    else: return 400 + (x - 430) * 100 / 80

def get_no2_subindex(x):
    if pd.isna(x) or x < 0: return np.nan
    if x <= 40: return x * 50 / 40
    elif x <= 80: return 50 + (x - 40) * 50 / 40
    elif x <= 180: return 100 + (x - 80) * 100 / 100
    elif x <= 280: return 200 + (x - 180) * 100 / 100
    elif x <= 400: return 300 + (x - 280) * 100 / 120
    else: return 400 + (x - 400) * 100 / 120

def get_so2_subindex(x):
    if pd.isna(x) or x < 0: return np.nan
    if x <= 40: return x * 50 / 40
    elif x <= 80: return 50 + (x - 40) * 50 / 40
    elif x <= 380: return 100 + (x - 80) * 100 / 300
    elif x <= 800: return 200 + (x - 380) * 100 / 420
    elif x <= 1600: return 300 + (x - 800) * 100 / 800
    else: return 400 + (x - 1600) * 100 / 800

def get_co_subindex(x):
    if pd.isna(x) or x < 0: return np.nan
    if x <= 1: return x * 50
    elif x <= 2: return 50 + (x - 1) * 50
    elif x <= 10: return 100 + (x - 2) * 100 / 8
    elif x <= 17: return 200 + (x - 10) * 100 / 7
    elif x <= 34: return 300 + (x - 17) * 100 / 17
    else: return 400 + (x - 34) * 100 / 17

def get_o3_subindex(x):
    if pd.isna(x) or x < 0: return np.nan
    if x <= 50: return x * 50 / 50
    elif x <= 100: return 50 + (x - 50) * 50 / 50
    elif x <= 168: return 100 + (x - 100) * 100 / 68
    elif x <= 208: return 200 + (x - 168) * 100 / 40
    elif x <= 748: return 300 + (x - 208) * 100 / 540
    else: return 400 + (x - 748) * 100 / 540

def get_nh3_subindex(x):
    if pd.isna(x) or x < 0: return np.nan
    if x <= 200: return x * 50 / 200
    elif x <= 400: return 50 + (x - 200) * 50 / 200
    elif x <= 800: return 100 + (x - 400) * 100 / 400
    elif x <= 1200: return 200 + (x - 800) * 100 / 400
    elif x <= 1800: return 300 + (x - 1200) * 100 / 600
    else: return 400 + (x - 1800) * 100 / 600

def get_aqi_bucket(aqi):
    if pd.isna(aqi): return "Unknown"
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def compute_aqi_details(row):
    subs = {
        'PM2.5': get_pm25_subindex(row.get('PM2.5', np.nan)),
        'PM10': get_pm10_subindex(row.get('PM10', np.nan)),
        'NO2': get_no2_subindex(row.get('NO2', np.nan)),
        'SO2': get_so2_subindex(row.get('SO2', np.nan)),
        'CO': get_co_subindex(row.get('CO', np.nan)),
        'O3': get_o3_subindex(row.get('O3', np.nan)),
        'NH3': get_nh3_subindex(row.get('NH3', np.nan))
    }
    valid_subs = {k: v for k, v in subs.items() if not pd.isna(v)}
    has_pm = ('PM2.5' in valid_subs) or ('PM10' in valid_subs)
    
    if len(valid_subs) >= 3 and has_pm:
        major_pol = max(valid_subs, key=valid_subs.get)
        aqi_val = np.round(valid_subs[major_pol])
        return aqi_val, major_pol
    elif len(valid_subs) >= 1:
        major_pol = max(valid_subs, key=valid_subs.get)
        aqi_val = np.round(valid_subs[major_pol])
        return aqi_val, major_pol
    else:
        return np.nan, "None"

def get_season(month):
    if month in [12, 1, 2]: return "Winter"
    elif month in [3, 4, 5]: return "Summer"
    elif month in [6, 7, 8, 9]: return "Monsoon"
    else: return "Post-Monsoon"

# -------------------------------------------------------------
# Health Advisory Mapping
# -------------------------------------------------------------
def generate_health_advisory(aqi, major_pol):
    bucket = get_aqi_bucket(aqi)
    
    if bucket == "Good":
        return {
            "Health_Risk_Level": "Low",
            "Advisory_General": "Air quality is clean and ideal for outdoor activities.",
            "Advisory_Asthma_Respiratory": "No precautions needed. Enjoy fresh air.",
            "Advisory_Cardiovascular": "Safe for normal physical exertion outdoor.",
            "Advisory_Children_Elderly": "Safe for outdoor play and walking.",
            "Mask_Recommendation": "Not required",
            "Air_Purifier_Need": "Not required",
            "Outdoor_Activity_Recommendation": "Enjoy normal outdoor activities"
        }
    elif bucket == "Satisfactory":
        return {
            "Health_Risk_Level": "Minor Risk",
            "Advisory_General": "Air quality is acceptable. Minor discomfort to sensitive individuals.",
            "Advisory_Asthma_Respiratory": "Keep rescue inhaler handy if unusually sensitive to dust.",
            "Advisory_Cardiovascular": "Normal outdoor activity is generally safe.",
            "Advisory_Children_Elderly": "Safe for outdoor activities, monitor for cough or irritation.",
            "Mask_Recommendation": "Optional for sensitive individuals",
            "Air_Purifier_Need": "Not required",
            "Outdoor_Activity_Recommendation": "Normal outdoor activities"
        }
    elif bucket == "Moderate":
        return {
            "Health_Risk_Level": "Moderate Risk",
            "Advisory_General": "May cause breathing discomfort to people with lungs, asthma, and heart diseases.",
            "Advisory_Asthma_Respiratory": "Reduce prolonged heavy outdoor exertion. Keep medication ready.",
            "Advisory_Cardiovascular": "Avoid heavy workouts outdoors near traffic zones.",
            "Advisory_Children_Elderly": "Limit outdoor strenuous play during peak afternoon hours.",
            "Mask_Recommendation": "Cloth / Surgical mask recommended outdoors near busy roads",
            "Air_Purifier_Need": "Recommended for sensitive individuals indoors",
            "Outdoor_Activity_Recommendation": "Reduce intense outdoor workouts"
        }
    elif bucket == "Poor":
        return {
            "Health_Risk_Level": "High Risk",
            "Advisory_General": "Breathing discomfort to most people on prolonged exposure.",
            "Advisory_Asthma_Respiratory": "Avoid outdoor exertion. Wear N95 mask outside.",
            "Advisory_Cardiovascular": "Avoid outdoor exercising. Stay indoors during morning smog.",
            "Advisory_Children_Elderly": "Children and elderly should stay indoors during high smog.",
            "Mask_Recommendation": "N95 / FFP2 mask strongly recommended outdoors",
            "Air_Purifier_Need": "Highly recommended indoors",
            "Outdoor_Activity_Recommendation": "Avoid morning & evening outdoor jogging"
        }
    elif bucket == "Very Poor":
        return {
            "Health_Risk_Level": "Very High Risk",
            "Advisory_General": "Respiratory illness on prolonged exposure. Significant health impact.",
            "Advisory_Asthma_Respiratory": "STRICT WARNING: Remain indoors with air purifier. High risk of asthma attack.",
            "Advisory_Cardiovascular": "Avoid all physical outdoor activity. High blood pressure/cardiac risk.",
            "Advisory_Children_Elderly": "Keep children and elderly indoors. Keep windows and doors shut.",
            "Mask_Recommendation": "N95 / N99 mask mandatory outdoors",
            "Air_Purifier_Need": "Essential for all indoor spaces",
            "Outdoor_Activity_Recommendation": "Cancel outdoor sports and strenuous work"
        }
    else: # Severe
        return {
            "Health_Risk_Level": "Emergency Risk",
            "Advisory_General": "EMERGENCY: Affects healthy people and seriously impacts those with existing diseases.",
            "Advisory_Asthma_Respiratory": "CRITICAL: Stay indoors. Use HEPA air purifiers. Emergency medical contact ready.",
            "Advisory_Cardiovascular": "CRITICAL: Strict indoor isolation. Avoid any exertion.",
            "Advisory_Children_Elderly": "CRITICAL: Do not allow children or elderly outside under any circumstances.",
            "Mask_Recommendation": "N99 / Respirator mandatory for short outdoor exposure",
            "Air_Purifier_Need": "Critical - HEPA Air Purifier at maximum speed",
            "Outdoor_Activity_Recommendation": "Complete shutdown of outdoor activities"
        }

# -------------------------------------------------------------
# 1. Processing city_day.csv
# -------------------------------------------------------------
print("Processing city_day.csv...")
city_day = pd.read_csv(os.path.join(INPUT_DIR, "city_day.csv"))
city_day['Date'] = pd.to_datetime(city_day['Date'])
city_day = city_day.sort_values(['City', 'Date']).reset_index(drop=True)

pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

# Clip negative values if any
for col in pollutants:
    if col in city_day.columns:
        city_day[col] = city_day[col].clip(lower=0)

# Step A: Time series linear interpolation (gaps <= 7 days) within city
for col in pollutants + ['AQI']:
    city_day[col] = city_day.groupby('City')[col].transform(lambda x: x.interpolate(method='linear', limit=7).ffill().bfill())

# Step B: Monthly city median fallback for any remaining NaNs
city_day['Month'] = city_day['Date'].dt.month
for col in pollutants + ['AQI']:
    city_day[col] = city_day.groupby(['City', 'Month'])[col].transform(lambda x: x.fillna(x.median()))
    # Global median fallback if a city has no data for that month
    city_day[col] = city_day[col].fillna(city_day[col].median())

# Step C: Re-calculate CPCB AQI and Major Pollutant for guaranteed consistency
calc_results = city_day.apply(compute_aqi_details, axis=1)
calc_aqi = [r[0] for r in calc_results]
calc_major = [r[1] for r in calc_results]

# Fill missing AQI with calculated AQI where necessary
city_day['AQI'] = city_day['AQI'].fillna(pd.Series(calc_aqi))
city_day['AQI_Bucket'] = city_day['AQI'].apply(get_aqi_bucket)
city_day['Major_Pollutant'] = calc_major

# Save city_day_cleaned.csv
cleaned_cols = ['City', 'Date'] + pollutants + ['AQI', 'AQI_Bucket', 'Major_Pollutant']
city_day_cleaned = city_day[cleaned_cols].copy()
city_day_cleaned.to_csv(os.path.join(OUTPUT_DIR, "city_day_cleaned.csv"), index=False)
print("Saved city_day_cleaned.csv (Shape:", city_day_cleaned.shape, ")")

# -------------------------------------------------------------
# 2. Engineering Features for Forecasting (city_day_forecasting.csv)
# -------------------------------------------------------------
print("Generating city_day_forecasting.csv features...")
df_fc = city_day_cleaned.copy()
df_fc['Date'] = pd.to_datetime(df_fc['Date'])

# Calendar features
df_fc['Year'] = df_fc['Date'].dt.year
df_fc['Month'] = df_fc['Date'].dt.month
df_fc['Day'] = df_fc['Date'].dt.day
df_fc['DayOfWeek'] = df_fc['Date'].dt.dayofweek
df_fc['DayOfYear'] = df_fc['Date'].dt.dayofyear
df_fc['IsWeekend'] = df_fc['DayOfWeek'].isin([5, 6]).astype(int)
df_fc['Season'] = df_fc['Month'].apply(get_season)

# Ratios
df_fc['PM2.5_PM10_ratio'] = (df_fc['PM2.5'] / (df_fc['PM10'] + 1e-5)).clip(0, 1)
df_fc['NO2_NOx_ratio'] = (df_fc['NO2'] / (df_fc['NOx'] + 1e-5)).clip(0, 1)

# Grouped Time-Series Features per City
def add_time_series_features(group):
    group = group.sort_values('Date').reset_index(drop=True)
    
    # Lags
    for lag in [1, 2, 7]:
        group[f'AQI_lag_{lag}d'] = group['AQI'].shift(lag)
        group[f'PM2.5_lag_{lag}d'] = group['PM2.5'].shift(lag)
        group[f'NO2_lag_{lag}d'] = group['NO2'].shift(lag)
    
    # Rolling Statistics
    for w in [3, 7, 14, 30]:
        group[f'AQI_roll_mean_{w}d'] = group['AQI'].shift(1).rolling(w, min_periods=1).mean()
        group[f'AQI_roll_std_{w}d'] = group['AQI'].shift(1).rolling(w, min_periods=1).std().fillna(0)
        group[f'PM2.5_roll_mean_{w}d'] = group['PM2.5'].shift(1).rolling(w, min_periods=1).mean()
    
    # Target Horizons
    group['AQI_target_1d'] = group['AQI'].shift(-1)
    group['AQI_target_3d'] = group['AQI'].shift(-3)
    group['AQI_target_7d'] = group['AQI'].shift(-7)
    
    return group

df_fc = df_fc.groupby('City', group_keys=False).apply(add_time_series_features)

# Drop initial lag NaNs per city
df_fc = df_fc.dropna(subset=['AQI_lag_7d', 'AQI_target_1d']).reset_index(drop=True)

df_fc.to_csv(os.path.join(OUTPUT_DIR, "city_day_forecasting.csv"), index=False)
print("Saved city_day_forecasting.csv (Shape:", df_fc.shape, ")")

# -------------------------------------------------------------
# 3. Generating Personalized Health Advisory Dataset
# -------------------------------------------------------------
print("Generating personalized_health_advisory.csv...")
advisory_list = []
for idx, row in city_day_cleaned.iterrows():
    adv = generate_health_advisory(row['AQI'], row['Major_Pollutant'])
    entry = {
        'City': row['City'],
        'Date': row['Date'].strftime('%Y-%m-%d'),
        'AQI': row['AQI'],
        'AQI_Bucket': row['AQI_Bucket'],
        'Major_Pollutant': row['Major_Pollutant'],
        'PM2.5': row['PM2.5'],
        'PM10': row['PM10'],
        'NO2': row['NO2']
    }
    entry.update(adv)
    advisory_list.append(entry)

df_advisory = pd.DataFrame(advisory_list)
df_advisory.to_csv(os.path.join(OUTPUT_DIR, "personalized_health_advisory.csv"), index=False)
print("Saved personalized_health_advisory.csv (Shape:", df_advisory.shape, ")")

# Save reference lookup matrix
lookup_rows = []
for bucket in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]:
    adv = generate_health_advisory(
        {"Good": 25, "Satisfactory": 75, "Moderate": 150, "Poor": 250, "Very Poor": 350, "Severe": 450}[bucket],
        "PM2.5"
    )
    entry = {"AQI_Bucket": bucket}
    entry.update(adv)
    lookup_rows.append(entry)

df_lookup = pd.DataFrame(lookup_rows)
df_lookup.to_csv(os.path.join(OUTPUT_DIR, "health_advisory_lookup.csv"), index=False)
print("Saved health_advisory_lookup.csv")

# -------------------------------------------------------------
# 4. Processing station_day.csv & city_hour.csv
# -------------------------------------------------------------
print("Processing station_day.csv...")
s_day = pd.read_csv(os.path.join(INPUT_DIR, "station_day.csv"))
s_day['Date'] = pd.to_datetime(s_day['Date'])
s_day = s_day.sort_values(['StationId', 'Date']).reset_index(drop=True)

for col in pollutants:
    if col in s_day.columns:
        s_day[col] = s_day[col].clip(lower=0)

for col in pollutants + ['AQI']:
    s_day[col] = s_day.groupby('StationId')[col].transform(lambda x: x.interpolate(method='linear', limit=7).ffill().bfill())

s_day['Month'] = s_day['Date'].dt.month
for col in pollutants + ['AQI']:
    s_day[col] = s_day.groupby(['StationId', 'Month'])[col].transform(lambda x: x.fillna(x.median()))
    s_day[col] = s_day[col].fillna(s_day[col].median())

s_day['AQI_Bucket'] = s_day['AQI'].apply(get_aqi_bucket)
s_day.to_csv(os.path.join(OUTPUT_DIR, "station_day_cleaned.csv"), index=False)
print("Saved station_day_cleaned.csv (Shape:", s_day.shape, ")")

print("Processing city_hour.csv (sampled hourly aggregation)...")
c_hour = pd.read_csv(os.path.join(INPUT_DIR, "city_hour.csv"))
c_hour['Datetime'] = pd.to_datetime(c_hour['Datetime'])
c_hour = c_hour.sort_values(['City', 'Datetime']).reset_index(drop=True)

for col in ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']:
    if col in c_hour.columns:
        c_hour[col] = c_hour[col].clip(lower=0)
        c_hour[col] = c_hour.groupby('City')[col].transform(lambda x: x.interpolate(method='linear', limit=12).ffill().bfill())
        c_hour[col] = c_hour[col].fillna(c_hour[col].median())

c_hour['AQI_Bucket'] = c_hour['AQI'].apply(get_aqi_bucket)
c_hour.to_csv(os.path.join(OUTPUT_DIR, "city_hour_cleaned.csv"), index=False)
print("Saved city_hour_cleaned.csv (Shape:", c_hour.shape, ")")

print("\nDataset cleaning and feature engineering successfully completed!")
