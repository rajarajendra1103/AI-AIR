# Dataset & System Documentation: Air Quality Forecasting and Personalized Health Advisory

This document provides a comprehensive technical overview of the cleaned dataset, feature engineering pipeline, deep learning models (**LSTM**, **GRU**, **Transformer**), and Personalized Health Advisory AI Agent.

---

## 📁 Directory Structure

```
AIR/
├── dataset/                         # Original raw CPCB dataset
│   ├── city_day.csv
│   ├── city_hour.csv
│   ├── station_day.csv
│   ├── station_hour.csv
│   └── stations.csv
├── final_dataset/                   # Cleaned & feature-engineered final dataset
│   ├── city_day_cleaned.csv         # Continuous daily air quality for 26 cities
│   ├── city_day_forecasting.csv     # Features & multi-step targets for ML/DL models
│   ├── personalized_health_advisory.csv # Pre-mapped health advisories per city/day
│   ├── health_advisory_lookup.csv   # Fast reference lookup matrix for apps
│   ├── station_day_cleaned.csv      # Cleaned station-level daily dataset
│   └── city_hour_cleaned.csv        # Cleaned city-level hourly dataset
├── models/                          # Trained PyTorch models & artifacts
│   ├── lstm_model.pt                # Trained LSTM model weights
│   ├── gru_model.pt                 # Trained GRU model weights
│   ├── transformer_model.pt         # Trained Time-Series Transformer model weights
│   ├── scaler.pkl                   # StandardScaler object
│   └── model_comparison.json        # MAE, RMSE, R2, MAPE benchmark report
├── scripts/
│   └── clean_and_process.py         # End-to-end dataset cleaning pipeline
├── src/
│   ├── models.py                    # PyTorch model architectures (LSTM, GRU, Transformer)
│   ├── train.py                     # Training & evaluation pipeline
│   └── advisory_agent.py            # AI Agent Personalized Health Advisory engine
└── app.py                           # Interactive Streamlit Web Dashboard
```

---

## 📊 Final Cleaned Datasets Data Dictionary

### `city_day_cleaned.csv`
Contains continuous daily ambient air pollutant concentrations across 26 Indian cities (2015–2020) with fully imputed values and recalculated CPCB AQI.

| Column Name | Data Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `City` | Categorical | - | Name of the Indian city (26 unique cities) |
| `Date` | Datetime | YYYY-MM-DD | Date of observation |
| `PM2.5` | Float | µg/m³ | Fine Particulate Matter (<2.5 µm diameter) |
| `PM10` | Float | µg/m³ | Coarse Particulate Matter (<10 µm diameter) |
| `NO` | Float | µg/m³ | Nitric Oxide concentration |
| `NO2` | Float | µg/m³ | Nitrogen Dioxide concentration |
| `NOx` | Float | ppb / µg/m³ | Nitrogen Oxides total concentration |
| `NH3` | Float | µg/m³ | Ammonia concentration |
| `CO` | Float | mg/m³ | Carbon Monoxide concentration |
| `SO2` | Float | µg/m³ | Sulfur Dioxide concentration |
| `O3` | Float | µg/m³ | Ground-level Ozone concentration |
| `Benzene` | Float | µg/m³ | Benzene chemical concentration |
| `Toluene` | Float | µg/m³ | Toluene chemical concentration |
| `Xylene` | Float | µg/m³ | Xylene chemical concentration |
| `AQI` | Float | Index (0-500+) | Calculated Indian National Air Quality Index |
| `AQI_Bucket` | Categorical | Category | CPCB Category (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) |
| `Major_Pollutant`| Categorical | Pollutant | Dominant pollutant contributing most to the daily AQI sub-index |

---

### `city_day_forecasting.csv`
Enriched time-series forecasting dataset containing engineered features and target horizons.

| Feature Group | Features Included | Description |
| :--- | :--- | :--- |
| **Calendar & Temporal** | `Year`, `Month`, `Day`, `DayOfWeek`, `DayOfYear`, `IsWeekend`, `Season` | Encodes seasonal and weekly patterns (Winter, Summer, Monsoon, Post-Monsoon). |
| **Ratios** | `PM2.5_PM10_ratio`, `NO2_NOx_ratio` | Indicates combustion vs dust ratio and photochemical activity. |
| **Lags (1d, 2d, 7d)** | `AQI_lag_1d`, `AQI_lag_2d`, `AQI_lag_7d`, `PM2.5_lag_1d`, `NO2_lag_1d` | Captures autocorrelation and momentum. |
| **Rolling Windows** | `AQI_roll_mean_3d`, `AQI_roll_mean_7d`, `AQI_roll_mean_14d`, `AQI_roll_std_7d`, `PM2.5_roll_mean_7d` | Moving averages and volatility over 3, 7, 14, and 30 days. |
| **Target Horizons** | `AQI_target_1d`, `AQI_target_3d`, `AQI_target_7d` | Multi-step forward targets for direct forecasting. |

---

## 🩺 Personalized Health Advisory System

The AI Agent maps environmental readings to 6 distinct health profiles:

1. **General Public**: Standard ambient precautions.
2. **Asthma / Respiratory Conditions**: Strict inhaler readiness, N95/N99 mask guidance, HEPA purification.
3. **Cardiovascular / Heart Disease**: Workout intensity caps, indoor isolation during peak smog.
4. **Elderly (65+ years)**: High risk protection against respiratory distress.
5. **Children & Infants**: School play & outdoor activity restriction during high AQI.
6. **Outdoor Athletes / Workers**: Shift high-exertion training away from peak traffic/smog hours.

---

## 🤖 Deep Learning Models (PyTorch)

Three deep learning sequence architectures are trained on 14-day sliding windows (`SEQ_LEN = 14`):

1. **LSTM (`LSTMForecaster`)**: 2-layer stacked LSTM with dropout (0.2) and dense regression head.
2. **GRU (`GRUForecaster`)**: 2-layer Gated Recurrent Unit network for efficient temporal representations.
3. **Transformer (`TransformerForecaster`)**: Positional Encoding + 2-layer Multi-Head Self-Attention (`nhead=4`) Encoder.

---

## 🚀 Quickstart Usage Guide

### 1. Re-running the Cleaning Pipeline
```bash
python scripts/clean_and_process.py
```

### 2. Training Deep Learning Models
```bash
python src/train.py
```

### 3. Launching the Interactive Web Dashboard
```bash
streamlit run app.py
```
