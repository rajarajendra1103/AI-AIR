# 🌍 AI Air Quality & Weather Forecasting | Personalized Health Advisory Platform

An end-to-end Deep Learning & AI-powered Air Quality Index (AQI) forecasting and simulation platform featuring live global telemetry integration (Open-Meteo API), PyTorch neural architectures (LSTM, GRU, Transformer), CPCB sub-index calculation engines, and weather-aware clinical health advisories.

---

## 🚀 Quickstart: How to Run

### 1. Prerequisites & Environment Setup
Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
# Optional: Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Key libraries: `torch`, `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `requests`, `joblib`, `plotly`, `streamlit`)*

---

### 2. Launch the Web Application

#### Option A: High-Performance Modern Web Dashboard (Recommended) ⭐
Runs the FastAPI asynchronous backend paired with the rich animated web interface:

```bash
python run.py
```
👉 Open your browser at: **`http://127.0.0.1:8000`**

#### Option B: Alternative Streamlit Dashboard
Runs the interactive Streamlit analytics view:

```bash
streamlit run app.py
```
👉 Open your browser at: **`http://localhost:8501`**

---

### 3. Model Training & Pipeline Scripts

To re-train or re-evaluate the deep learning models from scratch:

```bash
# 1. Clean raw dataset & generate lag features
python scripts/clean_and_process.py

# 2. Train PyTorch models (LSTM, GRU, Transformer)
python src/train.py

# 3. Evaluate models & generate benchmark metrics
python src/eval_models.py
```

---

## 📱 Application Dashboard Tour (5 Interactive Modules)

### 1. 🌍 Live City Search & AI Air Quality Forecaster
* **Global Live Telemetry**: Geocoding + live weather & 7 criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3, NH3) via Open-Meteo.
* **3D Earth Startup Loader**: High-impact rotating Earth animation with 4-stage pipeline checklist.
* **Dynamic Pipeline Fetch Button**: Shows progress through location resolution, telemetry fetch, and PyTorch model evaluation.
* **Sub-Index & Compliance Table**: Live CPCB sub-index breakdown with animated percentage bars against 24-hr safe thresholds.
* **PyTorch 7-Day Forecast**: Multi-step deep learning forecast timeline with progressive line drawing.
* **Health Advisory Sequence**: Staggered clinical directives, mask recommendations, and air purifier settings.

### 2. 🧪 Custom Parameters & Simulation Studio
* **Quick Scenario Presets**: Instant loading for *Clean Alpine Day*, *Moderate Urban*, *Severe Smog*, *Industrial Hotspot*, and *High-Wind Dust*.
* **Real-Time Air Quality Impact Meter**: Instant feedback on environmental risk as sliders are adjusted.
* **Simulation Pipeline Visualizer**: Step-by-step visual indicator tracking data normalization, CPCB calculation, PyTorch GRU inference, and health advisory generation.
* **Simulation Processing Modal**: Neural tensor visualization overlay during computation.

### 3. 📈 Historical Air Quality Explorer
* **26 Indian Cities (2015–2020)**: Imputed continuous daily air quality records.
* **Interactive Chart.js Timeline**: Daily AQI variation with average/max metric counters.

### 4. 🧠 PyTorch Deep Learning Model Benchmarks
* **Model Comparison**: Evaluation table comparing **LSTM**, **GRU**, and **Transformer** architectures.
* **Total Prediction Accuracy Column**: Displays overall prediction accuracy (`100 - MAPE`) and R² variance fit.
* **Optimal Model Highlight**: Highlights **GRU Forecaster** as production default with lowest MAE and highest accuracy.

### 5. ⚙️ System Architecture & Data Dictionary
* **End-to-End Pipeline Flow Diagram**: Animated data flow tracking requests from City Input down to AI Health Advisory.
* **Neural Architecture Visualizer**: Layer-by-layer tensor dimensions for LSTM, GRU, and Transformer encoders.
* **Interactive Expandable Data Dictionary**: Accordions detailing scientific definitions, measurement units, and CPCB safe standard thresholds.

---

## 📊 PyTorch Model Benchmark Results

| Model Architecture | Total Accuracy (%) <br> `(100 - MAPE)` | R² Score <br> `(Variance Fit)` | MAE <br> `(Lower is Better)` | RMSE <br> `(Lower is Better)` | MAPE (%) | Benchmark Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🤖 **LSTM Forecaster** | **76.90%** | **0.8537** (85.37%) | 22.460 | 40.250 | 23.10% | ✓ BENCHMARKED |
| ⚡ **GRU Forecaster** | **83.52%** | **0.8616** (86.16%) | **19.410** | **39.160** | **16.48%** | **⭐ OPTIMAL (83.52%)** |
| 🔮 **Transformer Forecaster** | **79.71%** | **0.8589** (85.89%) | 21.210 | 39.530 | 20.29% | ✓ BENCHMARKED |

---

## 📁 Repository Structure

```
AIR/
├── dataset/                         # Original raw CPCB dataset (city_day, city_hour, etc.)
├── final_dataset/                   # Cleaned & feature-engineered dataset files
│   ├── city_day_cleaned.csv         # Imputed daily air quality for 26 cities
│   └── city_day_forecasting.csv     # Features & multi-step targets for PyTorch models
├── models/                          # Trained model checkpoints & artifacts
│   ├── lstm_model.pt                # Trained LSTM model weights
│   ├── gru_model.pt                 # Trained GRU model weights
│   ├── transformer_model.pt         # Trained Transformer model weights
│   ├── scaler.pkl                   # StandardScaler feature normalizer
│   └── model_comparison.json        # Benchmark metrics report
├── static/                          # Frontend assets for Modern Web App
│   ├── index.html                   # 5-Tab Modern Dashboard HTML
│   ├── styles.css                   # Dual-Theme (Light/Dark) Design System & Animations
│   └── app.js                       # Frontend Controller, Chart.js & Count-Up Engine
├── src/                             # Core Deep Learning & Engine Modules
│   ├── models.py                    # PyTorch Neural Architectures (LSTM, GRU, Transformer)
│   ├── train.py                     # Model training pipeline
│   ├── eval_models.py               # Benchmark evaluation engine
│   ├── aqi_calc.py                  # CPCB sub-index & category calculation engine
│   ├── advisory_agent.py            # Weather-Aware AI Health Advisory Agent
│   └── open_meteo_client.py         # Open-Meteo Geocoding & Telemetry Client
├── server.py                        # FastAPI REST API Backend
├── run.py                           # Application Launcher Script (Port 8000)
├── app.py                           # Streamlit Interactive Web Application
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation
```

---

## 🩺 Supported Health Profiles

The AI Health Advisory Agent tailors actionable directives across 6 distinct profiles:
1. **General Public**: Ambient guidance, outdoor exercise timing, and window ventilation advice.
2. **Asthma / Respiratory Conditions**: Inhaler readiness, N95/N99 mask directives, and HEPA purifier settings.
3. **Cardiovascular / Heart Disease**: Exercise intensity limits and peak smog avoidance.
4. **Elderly (65+ years)**: Targeted protection against respiratory distress and temperature extremes.
5. **Children & Infants**: School recess safety, indoor play alerts, and hydration recommendations.
6. **Outdoor Athletes**: High-ventilation workout safety and optimal training time windows.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).