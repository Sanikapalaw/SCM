import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE SETTINGS ---
st.set_page_config(page_title="Supply Chain Assistant", layout="wide")

# --- CUSTOM DESIGN (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA & MODEL LOADING ---
@st.cache_resource
def load_and_train():
    # Load your dataset
    df = pd.read_csv('data.csv')
    
    # Define simple features for the AI
    features = ['Demand', 'Forecast', 'NS', 'LTD', 'SS', 'OUT']
    X = df[features]
    y = df['Order']
    
    # Train the AI Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return df, model, features

try:
    df, model, features = load_and_train()
except FileNotFoundError:
    st.error("❌ 'data.csv' not found. Please make sure it's in the same folder as this script!")
    st.stop()

# --- HEADER ---
st.title("📦 Smart Supply Chain Assistant")
st.markdown("""
This app uses **Artificial Intelligence** to help you manage the 'Bullwhip Effect'—the phenomenon 
where small changes in customer demand create massive waves of overstock or shortages.
""")

# --- KPI SECTION (The 'Ripple' Meter) ---
st.divider()
var_demand = df['Demand'].var()
var_order = df['Order'].var()
bw_ratio = var_order / var_demand

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Supply Chain Health", "✅ Stable" if bw_ratio < 2.5 else "⚠️ High Ripple")
with col2:
    st.metric("Ripple Factor", f"{bw_ratio:.2f}x", help="Shows how much orders vary compared to demand.")
with col3:
    st.metric("Data Points Analyzed", len(df))

# --- INTERACTIVE PREDICTOR ---
st.subheader("🔮 AI Order Predictor")
st.info("What is the current situation? Select a scenario or use the sliders below.")

# Scenario Buttons for quick use
c1, c2, c3 = st.columns(3)
selected_demand = df['Demand'].mean()
selected_forecast = df['Forecast'].mean()

if c1.button("📉 Slow Season (Low Demand)"):
    selected_demand, selected_forecast = 60.0, 65.0
if c2.button("🏠 Business as Usual"):
    selected_demand, selected_forecast = 100.0, 105.0
if c3.button("🚀 Holiday Peak (High Demand)"):
    selected_demand, selected_forecast = 160.0, 180.0

# Input Sliders
with st.container():
    col_left, col_right = st.columns(2)
    with col_left:
        in_demand = st.slider("Customer Demand (What people want)", 30.0, 200.0, float(selected_demand))
        in_forecast = st.slider("Your Sales Forecast", 30.0, 200.0, float(selected_forecast))
    with col_right:
        in_ss = st.slider("Safety Stock (Emergency Backup)", 0.0, 50.0, 32.0)
        # Use averages for background variables to keep it simple for the user
        in_ns = 32.0
        in_ltd = in_forecast
        in_out = in_forecast + in_ss

# AI Prediction logic
input_data = pd.DataFrame([[in_demand, in_forecast, in_ns, in_ltd, in_ss, in_out]], columns=features)
prediction = model.predict(input_data)[0]

st.success(f"### 🤖 Recommended Order Quantity: **{prediction:.2f} units**")

# --- VISUALIZATION ---
st.divider()
st.subheader("📊 Visualizing the Bullwhip Effect")
st.write("Compare the stable Customer Demand (Blue) against the volatile Warehouse Orders (Orange).")

# Show only first 100 rows for clarity
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=df.head(100)[['Demand', 'Order']], palette=['#1f77b4', '#ff7f0e'], ax=ax)
ax.set_ylabel("Quantity")
ax.set_xlabel("Time (Days)")
st.pyplot(fig)

st.markdown("""
---
**How to interpret this?**
If the orange line (Orders) has much higher peaks than the blue line (Demand), you are experiencing the **Bullwhip Effect**. 
Our AI predictor helps flatten these peaks to save money on storage!
""")
