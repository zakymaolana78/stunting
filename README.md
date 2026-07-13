---
title: Prediksi Stunting Jawa Barat
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.46.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Prediksi Jumlah Balita Stunting Provinsi Jawa Barat

Aplikasi ini dibuat untuk memprediksi jumlah balita stunting di Provinsi Jawa Barat menggunakan algoritma **Random Forest Regressor**.

## Dataset

Dataset yang digunakan merupakan data indikator stunting Provinsi Jawa Barat dengan variabel:

- Persentase Penduduk Miskin
- Garis Kemiskinan
- Persentase Sanitasi Layak
- Jumlah Tenaga Gizi
- Jumlah Balita Stunting

## Algoritma

- Random Forest Regressor

## Library

- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-Learn
- Matplotlib

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Struktur Project

```
├── app.py
├── requirements.txt
├── dataset_final_stunting.csv
├── README.md
```

## Developer

**Zaky Maolana Al Rasid**  
Universitas Gunadarma
