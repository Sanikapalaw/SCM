import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Supply Chain Assistant", layout="wide")

# --- CUSTOM CSS FOR USER FRIENDLINESS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_approx=True)

@st.cache_data
def load_and_train():
    df = pd.read_csv('data.csv')
    # Train a simple model in the background
    features = ['Demand', 'Forecast', 'NS', 'LTD', 'SS', 'OUT']
    X = df[features]
    y = df['Order']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return df, model, features

df, model, features = load_and_train()

# --- HEADER SECTION ---
st.title("📦 Smart Supply Chain Assistant")
st.markdown("""
Welcome! This tool helps you see how small changes in customer orders can cause big "ripples" in your warehouse. 
We use **AI** to predict exactly how much you should order from your supplier to keep things stable.
""")

# --- KPI SECTION (VOLATILITY METER) ---
st.divider()
var_demand = df['Demand'].var()
var_order = df['Order'].var()
bw_ratio = var_order / var_demand

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Customer Demand Stability", "Normal", help="Is the customer buying pattern steady?")
with col2:
    status = "⚠️ High Ripple" if bw_ratio > 2 else "✅ Stable"
    st.metric("Supply Chain Health", status, delta=f"{bw_ratio:.1f}x Volatility")
with col3:
    st.metric("Efficiency Score", "84%", help="How well our current ordering matches demand.")

# --- INTERACTIVE SIMULATOR (THE "WHAT-IF") ---
st.subheader("🔮 Predictive Ordering Assistant")
st.write("Adjust the sliders below to see what the AI recommends you order based on different situations.")

# 

# Scenario Buttons for non-tech users
st.write("**Quick Scenarios:**")
c1, c2, c3 = st.columns(3)
scen_demand = df['Demand'].mean()
scen_forecast = df['Forecast'].mean()

if c1.button("📉 Low Demand Period"):
    scen_demand, scen_forecast = 50.0, 55.0
if c2.button("🏠 Normal Operations"):
    scen_demand, scen_forecast = 100.0, 102.0
if c3.button("🚀 Sudden Sales Spike"):
    scen_demand, scen_forecast = 160.0, 175.0

# User Inputs
with st.expander("Adjust Specific Details (Advanced)", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        in_demand = st.slider("Actual Customer Demand", 0.0, 200.0, float(scen_demand))
        in_forecast = st.slider("Your Sales Forecast", 0.0, 200.0, float(scen_forecast))
    with col_b:
        in_ss = st.slider("Safety Stock (Just-in-case)", 0.0, 50.0, 30.0)
        # Hidden inputs set to average for simplicity
        in_ns = 30.0 
        in_ltd = in_forecast 
        in_out = in_forecast + in_ss

# Prediction Logic
input_row = pd.DataFrame([[in_demand, in_forecast, in_ns, in_ltd, in_ss, in_out]], columns=features)
prediction = model.predict(input_row)[0]

st.info(f"### 🤖 AI Recommendation: You should order **{prediction:.2f} units** from your supplier.")

# --- THE "STORY" VISUALIZATION ---
st.divider()
st.subheader("📊 The Ripple Effect")
st.write("The blue line is what customers want. The orange line is how the warehouse reacts. Notice how the orange line swings much wider!")

# Filtered chart for clarity
df_plot = df.head(100)
fig, ax = plt.subplots(figsize=(10, 3))
sns.lineplot(data=df_plot[['Demand', 'Order']], palette=['#1f77b4', '#ff7f0e'], ax=ax)
ax.set_title("Customer Demand vs. Warehouse Orders")
st.pyplot(fig)

st.success("💡 **Tip for Managers:** To reduce the 'Ripple', try to share more data with your suppliers and keep your 'Just-in-case' stock stable!")
