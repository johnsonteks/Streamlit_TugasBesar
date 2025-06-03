import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

st.title("📈 Prediksi Harga Rumah (Regresi Linear)")

@st.cache_data
def load_data():
    path = "save_model/data.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_resource
def load_model():
    path = "save_model/model.pkl"
    return joblib.load(path) if os.path.exists(path) else None

df = load_data()
model = load_model()

if model is None:
    st.error("❌ Model tidak ditemukan! Harap letakkan `model.pkl` di folder save_model.")
elif df.empty:
    st.error("❌ Data tidak ditemukan! Harap simpan `data.csv` di folder save_model.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        kt = st.number_input("Kamar Tidur", min_value=1, max_value=10, value=3)
        km = st.number_input("Kamar Mandi", min_value=1, max_value=10, value=2)
    with col2:
        garasi = st.number_input("Garasi", min_value=0, max_value=5, value=1)
        lb = st.number_input("Luas Bangunan (m²)", min_value=20, max_value=1000, value=100)
    with col3:
        lt = st.number_input("Luas Tanah (m²)", min_value=20, max_value=1000, value=100)

    if st.button("Prediksi Harga"):
        pred = model.predict([[kt, km, garasi, lb, lt]])[0]
        st.success(f"💰 Estimasi Harga: Rp {pred:,.0f}")

    fig = px.scatter(
        df,
        x="Luas_Bangunan",
        y="Harga",
        color="Kamar_Tidur",
        hover_data=["Kamar_Mandi", "Garasi", "Luas_Tanah"],
        title="Scatter Plot Harga Berdasarkan Luas Bangunan dan Kamar Tidur"
    )

    st.plotly_chart(fig, use_container_width=True)

    acc_path = "save_model/accuracy.txt"
    if os.path.exists(acc_path):
        with open(acc_path) as f:
            acc = f.read().strip()
        st.markdown(f"🎯 **Akurasi Model (R²): {acc}%**")
