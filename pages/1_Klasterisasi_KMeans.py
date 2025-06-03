import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import os

st.title("🔍 Klasterisasi Harga Rumah (KMeans)")

@st.cache_data
def load_data():
    path = "save_model/data.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

df = load_data()

if df.empty:
    st.error("❌ Data tidak ditemukan! Harap simpan `data.csv` di folder `save_model`.")
else:
    features = df[['Kamar_Tidur', 'Kamar_Mandi', 'Garasi', 'Luas_Bangunan', 'Luas_Tanah']]

    n_clusters = st.slider("Jumlah Klaster", 2, 10, 4)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df['Kluster'] = kmeans.fit_predict(scaled_features)

    fig = px.scatter(
        df,
        x="Luas_Bangunan",
        y="Harga",
        color=df['Kluster'].astype(str),  # pastikan bertipe string untuk warna
        hover_data=["Kamar_Tidur", "Kamar_Mandi", "Garasi", "Luas_Tanah"],
        title="Scatter Plot Klasterisasi: Luas Bangunan vs Harga"
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Rangkuman Klaster"):
        st.dataframe(
            df.groupby("Kluster").agg({
                'Harga': 'mean',
                'Luas_Bangunan': 'mean',
                'Luas_Tanah': 'mean',
                'Kamar_Tidur': 'mean',
                'Kamar_Mandi': 'mean',
                'Garasi': 'mean'
            }).reset_index().round(2)
        )

    with st.expander("🔎 Lihat Data per Klaster"):
        klaster_terpilih = st.selectbox("Pilih Klaster", sorted(df['Kluster'].unique()))
        st.dataframe(df[df['Kluster'] == klaster_terpilih].reset_index(drop=True))
