# ==========================================================
# IMPORT LIBRARY
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================

st.set_page_config(
    page_title="Prediksi Stunting Jawa Barat",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# LOAD DATASET
# ==========================================================

BASE_DIR = Path(__file__).parent

dataset = pd.read_csv(
    BASE_DIR / "dataset_final_stunting.csv"
)

# ==========================================================
# FITUR DAN TARGET
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

# ==========================================================
# SPLIT DATA
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# RANDOM FOREST
# ==========================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ==========================================================
# HASIL PREDIKSI
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# EVALUASI
# ==========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({
    "Fitur": fitur,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📋 Menu")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "🏠 Beranda",
        "📊 Dataset",
        "🤖 Prediksi",
        "📈 Evaluasi",
        "⭐ Feature Importance",
        "ℹ️ Tentang"
    ]
)
# ==========================================================
# BERANDA
# ==========================================================

if menu == "🏠 Beranda":

    st.title("📊 Prediksi Jumlah Balita Stunting Provinsi Jawa Barat")

    st.write("""
    Aplikasi ini dibuat untuk memprediksi jumlah balita stunting
    di Provinsi Jawa Barat menggunakan algoritma
    **Random Forest Regressor**.
    """)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Jumlah Data",
            len(dataset)
        )

    with col2:
        st.metric(
            "Jumlah Fitur",
            len(fitur)
        )

    with col3:
        st.metric(
            "Algoritma",
            "Random Forest"
        )

    with col4:
        st.metric(
            "R² Score",
            f"{r2:.3f}"
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

    st.divider()

    st.subheader("Lima Data Pertama")

    st.table(
        dataset.head()
    )

    st.info(
        "Silakan pilih menu di sebelah kiri untuk melihat Dataset, Prediksi, Evaluasi, dan Feature Importance."
    )
    # ==========================================================
# DATASET
# ==========================================================

elif menu == "📊 Dataset":

    st.title("📊 Dataset Penelitian")

    st.write(
        "Dataset yang digunakan untuk membangun model Random Forest Regressor."
    )

    st.divider()

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
            "Jumlah Fitur",
            len(fitur)
        )

        st.metric(
            "Target",
            target
        )

    st.divider()

    st.subheader("Statistik Deskriptif")

    st.dataframe(
        dataset.describe(),
        use_container_width=True
    )

    st.divider()

    st.subheader("Missing Value")

    missing = pd.DataFrame({

        "Kolom": dataset.columns,
        "Jumlah Missing": dataset.isnull().sum().values

    })

    st.dataframe(
        missing,
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

    st.subheader("Hubungan Kemiskinan dengan Stunting")

    fig = px.scatter(
        dataset,
        x="persentase_penduduk_miskin",
        y="jumlah_balita_stunting",
        title="Persentase Penduduk Miskin vs Jumlah Balita Stunting"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Hubungan Sanitasi Layak dengan Stunting")

    fig = px.scatter(
        dataset,
        x="persentase_sanitasi_layak",
        y="jumlah_balita_stunting",
        title="Persentase Sanitasi Layak vs Jumlah Balita Stunting"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    csv = dataset.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Dataset",
        data=csv,
        file_name="dataset_final_stunting.csv",
        mime="text/csv",
        use_container_width=True
    )
    # ==========================================================
# PREDIKSI
# ==========================================================

elif menu == "🤖 Prediksi":

    st.title("🤖 Prediksi Jumlah Balita Stunting")

    st.write(
        "Masukkan nilai setiap variabel untuk memperoleh prediksi jumlah balita stunting."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        persentase_penduduk_miskin = st.number_input(
            "Persentase Penduduk Miskin",
            min_value=0.0,
            value=float(dataset["persentase_penduduk_miskin"].mean())
        )

        garis_kemiskinan = st.number_input(
            "Garis Kemiskinan",
            min_value=0.0,
            value=float(dataset["garis_kemiskinan"].mean())
        )

    with col2:

        persentase_sanitasi_layak = st.number_input(
            "Persentase Sanitasi Layak",
            min_value=0.0,
            value=float(dataset["persentase_sanitasi_layak"].mean())
        )

        jumlah_tenaga_gizi = st.number_input(
            "Jumlah Tenaga Gizi",
            min_value=0,
            value=int(dataset["jumlah_tenaga_gizi"].mean())
        )

    st.divider()

    if st.button("🔍 Prediksi", use_container_width=True):

        data_input = pd.DataFrame({
            "persentase_penduduk_miskin": [persentase_penduduk_miskin],
            "garis_kemiskinan": [garis_kemiskinan],
            "persentase_sanitasi_layak": [persentase_sanitasi_layak],
            "jumlah_tenaga_gizi": [jumlah_tenaga_gizi]
        })

        hasil_prediksi = model.predict(data_input)[0]

        st.success(
            f"Prediksi jumlah balita stunting adalah **{hasil_prediksi:.2f} balita**"
        )

        st.subheader("Data Input")

        st.dataframe(
            data_input,
            use_container_width=True
        )

        st.subheader("Hasil Prediksi")

        hasil_df = pd.DataFrame({
            "Prediksi Jumlah Balita Stunting": [round(hasil_prediksi, 2)]
        })

        st.dataframe(
            hasil_df,
            use_container_width=True
        )

        csv = hasil_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Hasil Prediksi",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv",
            use_container_width=True
        )
        # ==========================================================
# EVALUASI MODEL
# ==========================================================

elif menu == "📈 Evaluasi":

    st.title("📈 Evaluasi Model Random Forest")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "MAE",
            f"{mae:.2f}"
        )

        st.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

    with col2:

        st.metric(
            "MSE",
            f"{mse:.2f}"
        )

        st.metric(
            "R² Score",
            f"{r2:.4f}"
        )

    st.divider()

    st.subheader("Aktual vs Prediksi")

    hasil = pd.DataFrame({

        "Aktual": y_test.values,
        "Prediksi": y_pred

    })

    fig = px.scatter(
        hasil,
        x="Aktual",
        y="Prediksi",
        title="Grafik Aktual vs Prediksi"
    )

    fig.add_shape(
        type="line",
        x0=hasil["Aktual"].min(),
        y0=hasil["Aktual"].min(),
        x1=hasil["Aktual"].max(),
        y1=hasil["Aktual"].max(),
        line=dict(
            color="red",
            dash="dash"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Residual Plot")

    residual = y_test.values - y_pred

    residual_df = pd.DataFrame({

        "Prediksi": y_pred,
        "Residual": residual

    })

    fig2 = px.scatter(
        residual_df,
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

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

elif menu == "⭐ Feature Importance":

    st.title("⭐ Feature Importance")

    st.dataframe(
        importance,
        use_container_width=True
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Fitur",
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
## Prediksi Jumlah Balita Stunting Provinsi Jawa Barat

Aplikasi ini dibuat sebagai implementasi penelitian skripsi
menggunakan algoritma **Random Forest Regressor**.

### Variabel

- Persentase Penduduk Miskin
- Garis Kemiskinan
- Persentase Sanitasi Layak
- Jumlah Tenaga Gizi

### Target

Jumlah Balita Stunting

### Algoritma

- Random Forest Regressor

### Tools

- Python
- Streamlit
- Scikit-Learn
- Plotly
- Pandas

---
**Universitas Gunadarma**
""")
