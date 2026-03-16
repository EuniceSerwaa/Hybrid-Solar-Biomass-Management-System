import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from scipy.optimize import linprog
import plotly.graph_objects as go

st.set_page_config(
    page_title="Hybrid Solar-Biomass EMS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM STYLE
# ---------------------------------------------------
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    h1 {
        color: #12355b;
        font-weight: 800;
        letter-spacing: 0.3px;
    }

    h2, h3 {
        color: #1d4e89;
        font-weight: 700;
    }

    .subtle-text {
        color: #4f5d75;
        font-size: 1.05rem;
        margin-top: -8px;
        margin-bottom: 18px;
    }

    .info-box {
        background: white;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 6px 18px rgba(18, 53, 91, 0.08);
        border-left: 6px solid #2a6fdb;
        margin-bottom: 18px;
    }

    .footer-box {
        background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 100%);
        border-radius: 18px;
        padding: 18px 22px;
        box-shadow: 0 6px 18px rgba(18, 53, 91, 0.08);
        border-left: 6px solid #2a6fdb;
        margin-top: 24px;
    }

    [data-testid="stMetric"] {
        background: white;
        border-radius: 16px;
        padding: 12px 14px;
        box-shadow: 0 6px 16px rgba(18, 53, 91, 0.08);
        border: 1px solid #e9f0fb;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7faff 0%, #edf4ff 100%);
    }

    .chart-box {
        background: white;
        padding: 10px 14px 18px 14px;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(18, 53, 91, 0.08);
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
solar_model = joblib.load("saved_models/solar_model.joblib")
demand_model = joblib.load("saved_models/demand_model.joblib")

with open("saved_models/model_metadata.json", "r") as f:
    metadata = json.load(f)

solar_features = metadata["solar_features"]
demand_features = metadata["demand_features"]

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.markdown("## Input Parameters")

solar_irradiance_W_m2 = st.sidebar.slider("Solar Irradiance (W/m²)", 0.0, 1200.0, 700.0)
temperature_C = st.sidebar.slider("Temperature (°C)", 10.0, 45.0, 27.0)
humidity_percent = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0)
wind_speed_m_s = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 3.0)
cloud_cover_percent = st.sidebar.slider("Cloud Cover (%)", 0.0, 100.0, 25.0)

hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
month = st.sidebar.slider("Month", 1, 12, 6)

biomass_available_kg = st.sidebar.slider("Biomass Available (kg)", 0.0, 500.0, 120.0)
calorific_value_MJ_kg = st.sidebar.slider("Calorific Value (MJ/kg)", 5.0, 30.0, 15.0)
conversion_efficiency_percent = st.sidebar.slider("Conversion Efficiency (%)", 1.0, 100.0, 30.0)

# ---------------------------------------------------
# HIDDEN / AUTO FEATURES
# ---------------------------------------------------
day_of_week = 2
is_weekend = 0

load_lag_1 = 3.0
load_lag_24 = 3.0
load_rolling_mean_6 = 3.0
load_rolling_mean_24 = 3.0

hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

irradiance_cloud_interaction = solar_irradiance_W_m2 * (1 - cloud_cover_percent / 100)
temp_humidity_interaction = temperature_C * humidity_percent

biomass_energy_MJ = biomass_available_kg * calorific_value_MJ_kg * (conversion_efficiency_percent / 100)
biomass_energy_kWh = biomass_energy_MJ / 3.6
biomass_max_possible_kW = biomass_energy_kWh

# ---------------------------------------------------
# MODEL INPUTS
# ---------------------------------------------------
solar_input = pd.DataFrame([{
    "solar_irradiance_W_m2": solar_irradiance_W_m2,
    "temperature_C": temperature_C,
    "humidity_percent": humidity_percent,
    "wind_speed_m_s": wind_speed_m_s,
    "cloud_cover_percent": cloud_cover_percent,
    "hour": hour,
    "month": month,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "month_sin": month_sin,
    "month_cos": month_cos,
    "irradiance_cloud_interaction": irradiance_cloud_interaction,
    "temp_humidity_interaction": temp_humidity_interaction
}])[solar_features]

demand_input = pd.DataFrame([{
    "temperature_C": temperature_C,
    "humidity_percent": humidity_percent,
    "wind_speed_m_s": wind_speed_m_s,
    "cloud_cover_percent": cloud_cover_percent,
    "hour": hour,
    "month": month,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "month_sin": month_sin,
    "month_cos": month_cos,
    "load_lag_1": load_lag_1,
    "load_lag_24": load_lag_24,
    "load_rolling_mean_6": load_rolling_mean_6,
    "load_rolling_mean_24": load_rolling_mean_24
}])[demand_features]

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------
predicted_solar_kW = float(solar_model.predict(solar_input)[0])
predicted_demand_kW = float(demand_model.predict(demand_input)[0])

# ---------------------------------------------------
# OPTIMIZATION
# ---------------------------------------------------
c = [1.0, 1000.0, 0.1]
A_eq = [[1, 1, -1]]
b_eq = [predicted_demand_kW - predicted_solar_kW]
bounds = [
    (0, biomass_max_possible_kW),  # biomass dispatch
    (0, None),                     # unserved energy
    (0, None)                      # curtailment
]

result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

if result.success:
    biomass_dispatch_kW = float(result.x[0])
    unserved_energy_kW = float(result.x[1])
    curtailment_kW = float(result.x[2])
else:
    biomass_dispatch_kW = 0.0
    unserved_energy_kW = 0.0
    curtailment_kW = 0.0

solar_contribution_pct = (predicted_solar_kW / predicted_demand_kW * 100) if predicted_demand_kW > 0 else 0.0

# ---------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------
st.title("Hybrid Solar–Biomass Energy Management System")
st.markdown('<div class="subtle-text">Smart Mini-Grid Control Interface</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
This interface forecasts <b>solar power</b>, estimates <b>electricity demand</b>,
and determines the <b>biomass backup required</b> to maintain reliable energy supply.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Solar Power (kW)", f"{predicted_solar_kW:.2f}")
col2.metric("Predicted Demand (kW)", f"{predicted_demand_kW:.2f}")
col3.metric("Biomass Dispatch (kW)", f"{biomass_dispatch_kW:.2f}")

col4, col5, col6 = st.columns(3)
col4.metric("Biomass Max Possible (kW)", f"{biomass_max_possible_kW:.2f}")
col5.metric("Unserved Energy (kW)", f"{unserved_energy_kW:.2f}")
col6.metric("Curtailment (kW)", f"{curtailment_kW:.2f}")

# ---------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------
st.subheader("System Interpretation")

if predicted_solar_kW >= predicted_demand_kW:
    st.success("Solar generation is enough to meet the demand. Biomass is not required.")
elif biomass_dispatch_kW > 0 and unserved_energy_kW == 0:
    st.info("Solar is not enough, so biomass is dispatched to cover the shortfall.")
else:
    st.warning("Even after biomass dispatch, some demand remains unmet.")

# ---------------------------------------------------
# CHARTS
# ---------------------------------------------------
st.subheader("Energy Balance Visualization")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=["Solar Power", "Demand", "Biomass Dispatch"],
        y=[predicted_solar_kW, predicted_demand_kW, biomass_dispatch_kW],
        text=[round(predicted_solar_kW, 2), round(predicted_demand_kW, 2), round(biomass_dispatch_kW, 2)],
        textposition="outside",
        marker=dict(
            color=["#2a6fdb", "#f4a261", "#2a9d8f"],
            line=dict(color="#ffffff", width=1.5)
        )
    ))
    fig_bar.update_layout(
        title="Predicted Supply and Demand Comparison",
        template="plotly_white",
        xaxis_title="Energy Component",
        yaxis_title="Power (kW)",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=solar_contribution_pct,
        title={"text": "Solar Contribution to Demand (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2a9d8f"},
            "steps": [
                {"range": [0, 40], "color": "#fde2e4"},
                {"range": [40, 70], "color": "#fff1c1"},
                {"range": [70, 100], "color": "#d8f3dc"}
            ]
        }
    ))
    fig_gauge.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

# ---------------------------------------------------
# MINI DONUT CHART
# ---------------------------------------------------
st.subheader("Supply Composition")

fig_donut = go.Figure(data=[go.Pie(
    labels=["Solar Power", "Biomass Dispatch"],
    values=[max(predicted_solar_kW, 0), max(biomass_dispatch_kW, 0)],
    hole=0.6,
    marker=dict(colors=["#2a6fdb", "#2a9d8f"])
)])
fig_donut.update_layout(
    template="plotly_white",
    height=420,
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig_donut, use_container_width=True)

# ---------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------
st.subheader("Summary Table")

results_df = pd.DataFrame({
    "Metric": [
        "Predicted Solar Power (kW)",
        "Predicted Demand (kW)",
        "Biomass Dispatch (kW)",
        "Biomass Max Possible (kW)",
        "Unserved Energy (kW)",
        "Curtailment (kW)"
    ],
    "Value": [
        round(predicted_solar_kW, 4),
        round(predicted_demand_kW, 4),
        round(biomass_dispatch_kW, 4),
        round(biomass_max_possible_kW, 4),
        round(unserved_energy_kW, 4),
        round(curtailment_kW, 4)
    ]
})

st.dataframe(results_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("""
<div class="footer-box">
<b>This dashboard predicts:</b>
<ol>
<li>Solar power generation</li>
<li>Community electricity demand</li>
<li>Biomass dispatch required to meet the energy gap</li>
</ol>
</div>
""", unsafe_allow_html=True)