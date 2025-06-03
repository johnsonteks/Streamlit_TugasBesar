import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Dashboard Analisis Rumah", page_icon="🏠", layout="wide")

st.title("🏠 Aplikasi Analisis Harga Rumah")

st.markdown("""
Selamat datang di aplikasi analisis harga rumah!  
Gunakan menu di sebelah kiri untuk:
- 🔹 **Klasterisasi (KMeans)** rumah berdasarkan fitur
- 🔹 **Prediksi Harga (Regresi Linear)** berdasarkan input
""")

# Load data
@st.cache_data
def load_data():
    path = "save_model/data.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("⚠️ Data tidak ditemukan. Harap pastikan file `data.csv` tersedia di folder `save_model`.")
else:
    st.subheader("📋 Data Rumah Lengkap")

    # Input search
    keyword = st.text_input("🔎 Cari berdasarkan Nama Rumah").lower()

    # Kolom yang akan ditampilkan
    show_columns = ['NO', 'NAMA RUMAH', 'Harga', 'Luas_Bangunan', 'Luas_Tanah',
                    'Kamar_Tidur', 'Kamar_Mandi', 'Garasi']

    if keyword:
        filtered = df[df['NAMA RUMAH'].str.lower().str.contains(keyword)]
    else:
        filtered = df

    st.dataframe(filtered[show_columns].reset_index(drop=True), use_container_width=True)
