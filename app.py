# ==========================================================
# IMPORT LIBRARY
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================

st.set_page_config(
    page_title="Prediksi Stunting Jawa Barat",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# LOKASI FILE
# ==========================================================

BASE_DIR = Path(__file__).parent

# ==========================================================
# MEMBACA DATA
# ==========================================================

dataset = pd.read_csv(BASE_DIR / "dataset_final_stunting.csv")
evaluasi = pd.read_csv(BASE_DIR / "hasil_evaluasi.csv")
hasil = pd.read_csv(BASE_DIR / "hasil_prediksi_rf.csv")
importance = pd.read_csv(BASE_DIR / "feature_importance.csv")

# Jika file tidak ada, aplikasi tetap berjalan
try:
    best_parameter = pd.read_csv(BASE_DIR / "best_parameter_rf.csv")
except:
    best_parameter = None

# ==========================================================
# MEMBANGUN MODEL RANDOM FOREST
# ==========================================================

fitur = [
    "persentase_penduduk_miskin",
    "garis_kemiskinan",
    "persentase_sanitasi_layak",
    "jumlah_tenaga_gizi"
]

target = "jumlah_balita_stunting"

X = dataset[fitur]
y = dataset[target]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📋 Menu")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "🏠 Beranda",
        "📊 Dataset",
        "🤖 Prediksi",
        "📉 Evaluasi",
        "⭐ Feature Importance",
        "ℹ️ Tentang"
    ]
)
# ==========================================================
# BERANDA
# ==========================================================

if menu == "🏠 Beranda":

    st.title("📊 Prediksi Jumlah Balita Stunting di Provinsi Jawa Barat")

    st.markdown(
        """
        Aplikasi ini digunakan untuk memprediksi jumlah balita stunting
        di Provinsi Jawa Barat menggunakan algoritma **Random Forest Regressor**.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Jumlah Data",
            len(dataset)
        )

    with col2:
        st.metric(
            "Jumlah Variabel",
            len(dataset.columns)
        )

    with col3:
        st.metric(
            "Algoritma",
            "Random Forest"
        )

    st.divider()

    st.subheader("Preview Dataset")

    st.dataframe(
        dataset.head(10),
        use_container_width=True
    )

    st.divider()

    st.subheader("Statistik Deskriptif")

    st.dataframe(
        dataset.describe(),
        use_container_width=True
    )

    st.divider()

    st.subheader("Distribusi Jumlah Balita Stunting")

    fig = px.histogram(
        dataset,
        x="jumlah_balita_stunting",
        nbins=20,
        title="Distribusi Jumlah Balita Stunting"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
  # ==========================================================
# DATASET
# ==========================================================

elif menu == "📊 Dataset":

    st.title("📊 Dataset Penelitian")

    st.write(
        "Dataset yang digunakan dalam penelitian prediksi jumlah balita stunting di Provinsi Jawa Barat."
    )

    st.subheader("Dataset")

    st.dataframe(
        dataset,
        use_container_width=True
    )

    st.divider()

    st.subheader("Informasi Dataset")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Jumlah Baris",
            dataset.shape[0]
        )

        st.metric(
            "Jumlah Kolom",
            dataset.shape[1]
        )

    with col2:

        st.metric(
            "Jumlah Missing Value",
            dataset.isnull().sum().sum()
        )

        st.metric(
            "Jumlah Fitur",
            len(fitur)
        )

    st.divider()

    st.subheader("Statistik Deskriptif")

    st.dataframe(
        dataset.describe(),
        use_container_width=True
    )

    st.divider()

    st.subheader("Distribusi Jumlah Balita Stunting")

    fig = px.histogram(
        dataset,
        x="jumlah_balita_stunting",
        nbins=20,
        title="Distribusi Jumlah Balita Stunting"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Hubungan Persentase Penduduk Miskin dengan Jumlah Balita Stunting")

    fig2 = px.scatter(
        dataset,
        x="persentase_penduduk_miskin",
        y="jumlah_balita_stunting",
        color="jumlah_balita_stunting",
        title="Scatter Plot"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    st.subheader("Hubungan Sanitasi Layak dengan Jumlah Balita Stunting")

    fig3 = px.scatter(
        dataset,
        x="persentase_sanitasi_layak",
        y="jumlah_balita_stunting",
        color="jumlah_balita_stunting"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )
  # ==========================================================
# PREDIKSI
# ==========================================================

elif menu == "🤖 Prediksi":

    st.title("🤖 Prediksi Jumlah Balita Stunting")

    st.write(
        "Masukkan nilai setiap variabel untuk melakukan prediksi menggunakan algoritma Random Forest."
    )

    col1, col2 = st.columns(2)

    with col1:

        miskin = st.number_input(
            "Persentase Penduduk Miskin",
            min_value=0.0,
            max_value=100.0,
            value=7.5,
            step=0.1
        )

        garis = st.number_input(
            "Garis Kemiskinan",
            min_value=0.0,
            value=450000.0,
            step=1000.0
        )

    with col2:

        sanitasi = st.number_input(
            "Persentase Sanitasi Layak",
            min_value=0.0,
            max_value=100.0,
            value=90.0,
            step=0.1
        )

        gizi = st.number_input(
            "Jumlah Tenaga Gizi",
            min_value=0,
            value=40,
            step=1
        )

    st.divider()

    if st.button("🔍 Prediksi", use_container_width=True):

        data = pd.DataFrame({

            "persentase_penduduk_miskin": [miskin],
            "garis_kemiskinan": [garis],
            "persentase_sanitasi_layak": [sanitasi],
            "jumlah_tenaga_gizi": [gizi]

        })

        prediksi = model.predict(data)[0]

        st.success(
            f"Prediksi Jumlah Balita Stunting: **{prediksi:,.0f} Balita**"
        )

        st.subheader("Data Input")

        st.dataframe(
            data,
            use_container_width=True
        )

        st.subheader("Hasil Prediksi")

        hasil_prediksi = pd.DataFrame({

            "Prediksi Jumlah Balita Stunting": [round(prediksi)]

        })

        st.dataframe(
            hasil_prediksi,
            use_container_width=True
        )

        csv = hasil_prediksi.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Hasil Prediksi",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv",
            use_container_width=True
        )
      # ==========================================================
# EVALUASI
# ==========================================================

elif menu == "📉 Evaluasi":

    st.title("📉 Evaluasi Model Random Forest")

    st.subheader("Hasil Evaluasi")

    st.dataframe(
        evaluasi,
        use_container_width=True
    )

    st.divider()

    if {"Aktual", "Prediksi"}.issubset(hasil.columns):

        st.subheader("Grafik Aktual vs Prediksi")

        fig = px.scatter(
            hasil,
            x="Aktual",
            y="Prediksi",
            title="Aktual vs Prediksi",
            trendline="ols"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    if "Residual" in hasil.columns:

        st.subheader("Residual Plot")

        fig2 = px.scatter(
            hasil,
            x="Prediksi",
            y="Residual",
            title="Residual Plot"
        )

        fig2.add_hline(
            y=0,
            line_dash="dash"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.divider()

    csv = evaluasi.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Hasil Evaluasi",
        data=csv,
        file_name="hasil_evaluasi.csv",
        mime="text/csv",
        use_container_width=True
    )

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

elif menu == "⭐ Feature Importance":

    st.title("⭐ Feature Importance")

    st.dataframe(
        importance,
        use_container_width=True
    )

    kolom_fitur = importance.columns[0]
    kolom_nilai = importance.columns[1]

    fig = px.bar(
        importance,
        x=kolom_nilai,
        y=kolom_fitur,
        orientation="h",
        title="Feature Importance Random Forest"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# TENTANG
# ==========================================================

elif menu == "ℹ️ Tentang":

    st.title("ℹ️ Tentang Aplikasi")

    st.markdown("""
### Prediksi Jumlah Balita Stunting Provinsi Jawa Barat

Aplikasi ini dibuat sebagai implementasi penelitian skripsi menggunakan algoritma **Random Forest Regressor**.

### Dataset
- Data indikator stunting Provinsi Jawa Barat
- Variabel:
  - Persentase Penduduk Miskin
  - Garis Kemiskinan
  - Persentase Sanitasi Layak
  - Jumlah Tenaga Gizi
  - Jumlah Balita Stunting

### Algoritma
- Random Forest Regressor
- Scikit-Learn

### Tools
- Python 3.11
- Streamlit
- Pandas
- Plotly
- Scikit-Learn

---
**Universitas Gunadarma**
""")
