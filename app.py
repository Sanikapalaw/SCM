import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from openai import OpenAI

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Bullwhip Advisor", layout="wide")

# ---------- LOAD MODEL (SAFE MODE) ----------
model = load_model("bullwhip_lstm_model.h5", compile=False)
model.compile(optimizer="adam", loss="mse")

scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- UI ----------
st.title("📦 AI Bullwhip Risk Advisor")
st.markdown("### LSTM Forecast + GenAI Decision Assistant")

uploaded = st.file_uploader("Upload supply chain CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    features = ['Demand','Receive','Forecast','NS','LTD','SS','OUT']
    X = df[features]

    X_scaled = scaler_X.transform(X)

    # -------- create sequences --------
    def make_seq(X, window=20):
        return np.array([X[i:i+window] for i in range(len(X)-window)])

    X_seq = make_seq(X_scaled)

    # -------- predict --------
    preds = model.predict(X_seq)
    preds = scaler_y.inverse_transform(preds)

    df_res = df.iloc[20:].copy()
    df_res["Predicted_Order"] = preds[:,0]
    df_res["Predicted_Bullwhip"] = preds[:,1]

    st.success("Prediction completed!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Predicted Orders")
        st.line_chart(df_res["Predicted_Order"])

    with col2:
        st.subheader("📉 Predicted Bullwhip")
        st.line_chart(df_res["Predicted_Bullwhip"])

    st.subheader("📊 Result Preview")
    st.dataframe(df_res.head())

    # ---------- GENAI ----------
    avg_bw = df_res["Predicted_Bullwhip"].mean()
    avg_order = df_res["Predicted_Order"].mean()
    avg_cost = df_res["Cost"].mean()

    if st.button("🧠 Generate AI Insight"):
        with st.spinner("Analyzing..."):
            prompt = f"""
            You are a supply chain expert.
            Predicted Bullwhip: {avg_bw:.2f}
            Avg Order: {avg_order:.2f}
            Avg Cost: {avg_cost:.2f}

            Explain the causes and suggest 3 actions.
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            st.markdown("### 🤖 AI Insight")
            st.write(response.choices[0].message.content)

    # ---------- CHAT ----------
    st.markdown("---")
    st.subheader("💬 Ask the AI Advisor")

    user_q = st.text_input("Ask about your supply chain:")

    if st.button("Ask AI"):
        with st.spinner("Thinking..."):
            chat_prompt = f"""
            You are a supply chain AI advisor.
            Bullwhip: {avg_bw:.2f}
            Avg Order: {avg_order:.2f}
            Avg Cost: {avg_cost:.2f}

            User question: {user_q}
            """

            reply = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": chat_prompt}]
            )

            st.write("### 🤖 AI Response")
            st.write(reply.choices[0].message.content)
