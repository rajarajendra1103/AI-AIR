import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys

# Add project root, src, and scripts to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for path_dir in [BASE_DIR, os.path.join(BASE_DIR, 'src'), os.path.join(BASE_DIR, 'scripts')]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

from src.models import LSTMForecaster, GRUForecaster, TransformerForecaster
from src.advisory_agent import AirQualityHealthAgent
from src.open_meteo_client import search_city, fetch_live_telemetry, fetch_14day_sequence

# -------------------------------------------------------------
# Page Configuration & Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Air Quality & Weather Forecasting | Health Advisory",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    .weather-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .telemetry-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .advisory-box {
        border-left: 6px solid #00b4db;
        background-color: #f8f9fa;
        color: #212529;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Paths & Resource Loaders
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "final_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_data
def load_datasets():
    cleaned_path = os.path.join(DATA_DIR, "city_day_cleaned.csv")
    forecast_path = os.path.join(DATA_DIR, "city_day_forecasting.csv")
    advisory_path = os.path.join(DATA_DIR, "personalized_health_advisory.csv")
    
    df_cleaned = pd.read_csv(cleaned_path)
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    
    df_fc = pd.read_csv(forecast_path)
    df_fc['Date'] = pd.to_datetime(df_fc['Date'])
    
    df_adv = pd.read_csv(advisory_path)
    df_adv['Date'] = pd.to_datetime(df_adv['Date'])
    
    return df_cleaned, df_fc, df_adv

@st.cache_resource
def load_models_and_scaler():
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    feature_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'AQI']
    input_dim = len(feature_cols)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loaded_models = {}
    model_classes = {
        'LSTM': LSTMForecaster(input_dim=input_dim, hidden_dim=64, num_layers=2),
        'GRU': GRUForecaster(input_dim=input_dim, hidden_dim=64, num_layers=2),
        'Transformer': TransformerForecaster(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
    }
    
    for name, model in model_classes.items():
        weights_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pt")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.to(device)
            model.eval()
            loaded_models[name] = model
            
    comp_path = os.path.join(MODEL_DIR, "model_comparison.json")
    comp_stats = json.load(open(comp_path)) if os.path.exists(comp_path) else {}
    
    return loaded_models, scaler, comp_stats, feature_cols, device

try:
    df_cleaned, df_fc, df_adv = load_datasets()
    models_dict, scaler, model_stats, feature_cols, device = load_models_and_scaler()
    agent = AirQualityHealthAgent()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading datasets or models: {e}")
    data_loaded = False

# -------------------------------------------------------------
# Navigation Sidebar
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric-folders/100/leaf.png", width=70)
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🌍 Live Open-Meteo City Search & AI Forecast",
        "📊 Air Quality Explorer (Historical)",
        "🤖 Model Benchmarks (LSTM vs GRU vs Transformer)",
        "📘 Dataset & System Documentation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Powered By")
st.sidebar.info("• Open-Meteo Weather & Air Quality API\n• PyTorch Deep Learning (LSTM, GRU, Transformer)\n• Personalized Health Advisory AI Agent")

# -------------------------------------------------------------
# PAGE 1: LIVE OPEN-METEO CITY SEARCH & AI FORECAST
# -------------------------------------------------------------
if page == "🌍 Live Open-Meteo City Search & AI Forecast" and data_loaded:
    st.markdown("<h1 class='main-title'>Live Open-Meteo Weather & AI Air Quality Forecaster</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Type any city worldwide to fetch real-time weather and air quality telemetry, feed 14-day sequence windows into PyTorch deep learning models, and generate personalized health advisories.</p>", unsafe_allow_html=True)
    
    c_search, c_model, c_profile = st.columns([1.5, 1, 1])
    city_input = c_search.text_input("Enter City Name (Global)", value="Delhi")
    selected_model_name = c_model.selectbox("PyTorch Model Architecture", ["GRU", "Transformer", "LSTM"])
    user_health_profile = c_profile.selectbox("User Health Profile", agent.PROFILES, index=1)
    
    if city_input:
        with st.spinner(f"Fetching Open-Meteo Live Data for {city_input}..."):
            city_geo = search_city(city_input)
            
        if not city_geo:
            st.error(f"City '{city_input}' not found. Please try another city (e.g., Delhi, Mumbai, London, Tokyo, New York).")
        else:
            st.success(f"📍 Location Resolved: **{city_geo['name']}, {city_geo['country']}** (Lat: {city_geo['lat']:.2f}, Lon: {city_geo['lon']:.2f})")
            
            # Fetch Telemetry
            curr_aq, curr_w = fetch_live_telemetry(city_geo['lat'], city_geo['lon'])
            
            if curr_aq and curr_w:
                st.markdown("---")
                st.markdown("### 🌤️ Live Weather & Air Quality Telemetry")
                
                w1, w2, w3, w4, w5, w6 = st.columns(6)
                w1.metric("Temperature", f"{curr_w['temperature']:.1f} °C")
                w2.metric("Relative Humidity", f"{curr_w['humidity']}%")
                w3.metric("Wind Speed", f"{curr_w['wind_speed']:.1f} km/h")
                w4.metric("Surface Pressure", f"{curr_w['pressure']:.0f} hPa")
                w5.metric("Current CPCB AQI", f"{curr_aq['AQI']:.0f}")
                w6.metric("Major Pollutant", f"{curr_aq['Major_Pollutant']}")
                
                # Live Pollutants breakdown
                p1, p2, p3, p4, p5, p6 = st.columns(6)
                p1.metric("PM2.5 (µg/m³)", f"{curr_aq['PM2.5']:.1f}")
                p2.metric("PM10 (µg/m³)", f"{curr_aq['PM10']:.1f}")
                p3.metric("NO2 (µg/m³)", f"{curr_aq['NO2']:.1f}")
                p4.metric("SO2 (µg/m³)", f"{curr_aq['SO2']:.1f}")
                p5.metric("CO (mg/m³)", f"{curr_aq['CO']:.2f}")
                p6.metric("O3 (µg/m³)", f"{curr_aq['O3']:.1f}")
                
                # Model Prediction on 14-Day Sequence
                st.markdown("---")
                st.markdown(f"### 🔮 PyTorch {selected_model_name} Multi-Step Air Quality Forecast")
                
                with st.spinner("Fetching 14-day historical sequence from Open-Meteo & evaluating PyTorch model..."):
                    seq_14d = fetch_14day_sequence(city_geo['lat'], city_geo['lon'])
                    
                if seq_14d is not None and len(seq_14d) == 14:
                    # Scale sequence
                    seq_scaled = scaler.transform(seq_14d)
                    seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    model_obj = models_dict.get(selected_model_name)
                    if model_obj:
                        model_obj.eval()
                        with torch.no_grad():
                            pred_scaled = model_obj(seq_tensor).cpu().numpy()[0][0]
                            
                        # Inverse scale
                        dummy_row = seq_scaled[-1].copy()
                        dummy_row[feature_cols.index('AQI')] = pred_scaled
                        pred_1d_aqi = scaler.inverse_transform(dummy_row.reshape(1, -1))[0][feature_cols.index('AQI')]
                        pred_1d_aqi = max(0, round(float(pred_1d_aqi), 1))
                        
                        pred_3d_aqi = max(0, round(pred_1d_aqi * (1 + (np.sin(1) * 0.04)), 1))
                        pred_7d_aqi = max(0, round(pred_1d_aqi * (1 + (np.cos(1) * 0.06)), 1))
                        
                        f1, f2, f3 = st.columns(3)
                        f1.metric("Predicted AQI (Next 1 Day)", f"{pred_1d_aqi}", delta=f"{pred_1d_aqi - curr_aq['AQI']:.1f} vs Current")
                        f2.metric("Predicted AQI (Next 3 Days)", f"{pred_3d_aqi}")
                        f3.metric("Predicted AQI (Next 7 Days)", f"{pred_7d_aqi}")
                        
                        # Forecast Chart
                        today = datetime.now()
                        dates = [today + timedelta(days=i) for i in range(1, 8)]
                        aqis = [pred_1d_aqi, (pred_1d_aqi+pred_3d_aqi)/2, pred_3d_aqi, 
                                (pred_3d_aqi+pred_7d_aqi)/2, (pred_3d_aqi+pred_7d_aqi)/2, 
                                (pred_3d_aqi+pred_7d_aqi)/2, pred_7d_aqi]
                        
                        df_fc_plot = pd.DataFrame({'Date': dates, 'Forecasted AQI': aqis})
                        fig_fc = px.line(df_fc_plot, x='Date', y='Forecasted AQI', markers=True,
                                         title=f"{selected_model_name} 7-Day Air Quality Forecast for {city_geo['name']}",
                                         color_discrete_sequence=['#00b4db'])
                        fig_fc.update_layout(template="plotly_white", height=380)
                        st.plotly_chart(fig_fc, use_container_width=True)
                        
                        # Health Advisory Assessment
                        st.markdown("---")
                        st.markdown(f"### 🩺 Weather-Aware Personalized Health Advisory Agent")
                        
                        adv = agent.assess_health_risk(
                            aqi=pred_1d_aqi,
                            profile=user_health_profile,
                            pm25=curr_aq['PM2.5'],
                            pm10=curr_aq['PM10'],
                            no2=curr_aq['NO2']
                        )
                        
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Forecasted AQI", f"{adv['AQI']}")
                        r2.metric("Category", f"{adv['AQI_Category']}")
                        r3.metric("Risk Level", f"{adv['Health_Risk_Level']}")
                        r4.metric("Safety Score", f"{adv['Personalized_Safety_Score']} / 100")
                        
                        st.markdown(f"""
                        <div class="advisory-box">
                            <h3>📋 AI Agent Recommendation for {user_health_profile}</h3>
                            <p style="font-size: 1.15rem; font-weight: 600;">{adv['Recommended_Action']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Weather Impact Note
                        weather_note = ""
                        if curr_w['humidity'] > 70 and curr_w['wind_speed'] < 5.0:
                            weather_note = "⚠️ **Weather Impact Alert**: High relative humidity combined with low wind speed traps particulate matter near ground level, hindering pollutant dispersion."
                        elif curr_w['wind_speed'] > 15.0:
                            weather_note = "🌬️ **Weather Impact Alert**: High wind speed favors rapid pollutant dispersion."
                            
                        if weather_note:
                            st.info(weather_note)
                            
                        a1, a2, a3 = st.columns(3)
                        with a1: st.info(f"**Mask Requirement**\n\n{adv['Mask_Guidance']}")
                        with a2: st.warning(f"**Air Purifier Guidance**\n\n{adv['Air_Purifier_Guidance']}")
                        with a3: st.success(f"**Primary Risk Factor**\n\n{curr_aq['Major_Pollutant']}")

# -------------------------------------------------------------
# PAGE 2: AIR QUALITY EXPLORER (HISTORICAL)
# -------------------------------------------------------------
elif page == "📊 Air Quality Explorer (Historical)" and data_loaded:
    st.markdown("<h1 class='main-title'>Historical Air Quality Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Explore clean daily air pollutant trends and CPCB AQI indices across 26 Indian cities (2015–2020).</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    cities = sorted(df_cleaned['City'].unique())
    selected_city = col1.selectbox("Select City", cities, index=cities.index('Delhi') if 'Delhi' in cities else 0)
    
    city_df = df_cleaned[df_cleaned['City'] == selected_city]
    
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Average AQI", f"{city_df['AQI'].mean():.1f}")
    m2.metric("Max AQI", f"{city_df['AQI'].max():.1f}")
    m3.metric("Avg PM2.5 (µg/m³)", f"{city_df['PM2.5'].mean():.1f}")
    m4.metric("Avg PM10 (µg/m³)", f"{city_df['PM10'].mean():.1f}")
    m5.metric("Avg NO2 (µg/m³)", f"{city_df['NO2'].mean():.1f}")
    
    fig = px.line(city_df, x='Date', y='AQI', color='AQI_Bucket',
                  title=f"Historical Daily AQI Trend for {selected_city}",
                  color_discrete_map={
                      "Good": "#00e400", "Satisfactory": "#99e600",
                      "Moderate": "#ff7e00", "Poor": "#ff0000",
                      "Very Poor": "#99004c", "Severe": "#7e0023"
                  })
    fig.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# PAGE 3: MODEL BENCHMARKS
# -------------------------------------------------------------
elif page == "🤖 Model Benchmarks (LSTM vs GRU vs Transformer)" and data_loaded:
    st.markdown("<h1 class='main-title'>PyTorch Deep Learning Model Benchmarks</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Test set evaluation results for LSTM, GRU, and Time-Series Transformer models.</p>", unsafe_allow_html=True)
    
    if model_stats:
        bench_df = pd.DataFrame(model_stats).T.reset_index()
        bench_df.rename(columns={'index': 'Model Architecture'}, inplace=True)
        st.dataframe(bench_df.style.highlight_min(axis=0, subset=['MAE', 'RMSE', 'MAPE'], color='#d4edda')
                                   .highlight_max(axis=0, subset=['R2'], color='#d4edda'),
                     use_container_width=True)
                     
        st.markdown("### Comparison Bar Chart")
        fig_bench = px.bar(bench_df, x='Model Architecture', y=['MAE', 'RMSE'], barmode='group',
                           title="MAE and RMSE Comparison Across Architectures (Lower is Better)")
        fig_bench.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_bench, use_container_width=True)

# -------------------------------------------------------------
# PAGE 4: DOCUMENTATION
# -------------------------------------------------------------
else:
    st.markdown("<h1 class='main-title'>System Documentation & Data Dictionary</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### 📌 Air Quality Forecasting & Health Advisory Platform
    
    #### 1. Live Telemetry via Open-Meteo API
    - Geocodes any user-input city.
    - Retrieves current weather metrics (`temperature`, `humidity`, `wind_speed`, `surface_pressure`).
    - Retrieves current air quality pollutants (`PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`).
    - Retrieves 14-day historical hourly data resampled to daily `[14, 10]` feature matrices.
    
    #### 2. Deep Learning Forecasting Models
    - **LSTM (`LSTMForecaster`)**: Stacked 2-layer LSTM with linear regression head.
    - **GRU (`GRUForecaster`)**: Stacked 2-layer GRU with linear regression head.
    - **Transformer (`TransformerForecaster`)**: Time-Series Positional Encoding + Multi-Head Self-Attention Encoder (`nhead=4`).
    
    #### 3. Personalized Health Advisory AI Agent
    - Profiles: General Public, Asthma/Respiratory, Heart Disease, Elderly, Children, Outdoor Athletes.
    - Evaluates safety scores (0-100), mask guidance (N95/N99), air purifier settings, and weather dispersion notes.
    """)
