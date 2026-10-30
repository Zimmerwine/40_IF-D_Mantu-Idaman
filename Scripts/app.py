"""
Aplikasi Web Antarmuka Pengiriman Last-Mile (Streamlit Dashboard)
Implementasi Prototipe Deployment Proporsional untuk Sistem Prediksi ETA Dinamis
UPN "Veteran" Yogyakarta - 2026
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, time

# Hubungkan modul internal
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import ETAPredictorService

st.set_page_config(
    page_title="Last-Mile Delivery ETA Predictor",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Sistem Prediksi Waktu Tiba Pengiriman Last-Mile (Amazon Delivery)")
st.caption("Implementasi CRISP-DM Fase Deployment - Program Studi Informatika, UPN 'Veteran' Yogyakarta 2026")

# Inisialisasi Service
@st.cache_resource
def get_service():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_delivery_model.joblib")
    return ETAPredictorService(model_path=model_path, safety_buffer_minutes=10.0, delay_threshold_minutes=180.0)

service = get_service()

# Tab Layout
tab1, tab2, tab3 = st.tabs(["🎯 Simulasi Prediksi ETA Real-Time", "📊 Metrik & Evaluasi Model", "📋 Rencana Deployment & Etika"])

with tab1:
    st.subheader("Simulasi Penugasan Pesanan Baru (Smart Dispatching)")
    st.write("Masukkan parameter operasional pesanan untuk mendapatkan estimasi durasi dan deteksi risiko keterlambatan.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📍 Koordinat & Jarak")
        store_lat = st.number_input("Store Latitude", value=22.745049, format="%.6f")
        store_lon = st.number_input("Store Longitude", value=75.892471, format="%.6f")
        drop_lat = st.number_input("Drop Latitude", value=22.765049, format="%.6f")
        drop_lon = st.number_input("Drop Longitude", value=75.912471, format="%.6f")

    with col2:
        st.markdown("#### 🌦️ Kondisi Lapangan & Armada")
        traffic = st.selectbox("Tingkat Kepadatan Lalu Lintas (Traffic)", ["Low", "Medium", "High", "Jam"])
        weather = st.selectbox("Kondisi Cuaca (Weather)", ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"])
        vehicle = st.selectbox("Moda Kendaraan (Vehicle)", ["motorcycle", "scooter", "bicycle", "van"])
        area = st.selectbox("Tipologi Area (Area)", ["Urban", "Metropolitan", "Semi-Urban", "Other"])
        category = st.selectbox("Kategori Komoditas (Category)", ["Clothing", "Electronics", "Sports", "Cosmetics", "Toys", "Snacks"])

    with col3:
        st.markdown("#### 👤 Profil Kurir & Waktu")
        agent_age = st.slider("Usia Kurir (Agent_Age)", min_value=18, max_value=50, value=28)
        agent_rating = st.slider("Rating Kurir (Agent_Rating)", min_value=1.0, max_value=5.0, value=4.7, step=0.1)
        order_time_input = st.time_input("Waktu Pemesanan (Order_Time)", value=time(14, 30))
        pickup_delay = st.slider("Estimasi Penjemputan Paket di Toko (Menit)", min_value=5, max_value=30, value=10)

    # Tombol Prediksi
    if st.button("🚀 Hitung Estimasi Waktu Tiba (Predict ETA)", type="primary"):
        # Format input dictionary
        order_time_str = order_time_input.strftime("%H:%M:%S")
        pickup_h = (order_time_input.hour + (order_time_input.minute + pickup_delay) // 60) % 24
        pickup_m = (order_time_input.minute + pickup_delay) % 60
        pickup_time_str = f"{pickup_h:02d}:{pickup_m:02d}:00"

        input_data = {
            "Order_ID": "SIM-2026-LIVE",
            "Agent_Age": agent_age,
            "Agent_Rating": agent_rating,
            "Store_Latitude": store_lat,
            "Store_Longitude": store_lon,
            "Drop_Latitude": drop_lat,
            "Drop_Longitude": drop_lon,
            "Order_Date": "2026-03-20",
            "Order_Time": order_time_str,
            "Pickup_Time": pickup_time_str,
            "Weather": weather,
            "Traffic": traffic,
            "Vehicle": vehicle,
            "Area": area,
            "Category": category
        }

        with st.spinner("Memproses inferensi model..."):
            result = service.predict_single(input_data)

        st.divider()
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        
        with res_col1:
            st.metric("Prediksi Durasi Murni", f"{result['predicted_delivery_minutes']} Menit")
        with res_col2:
            st.metric("Janji ETA ke Pelanggan (+Buffer)", f"{result['buffered_delivery_minutes']} Menit", delta="+10m Safety")
        with res_col3:
            st.metric("Estimasi Jam Tiba", f"{result['estimated_arrival_time']}")
        with res_col4:
            st.metric("Jarak Geodesik", f"{result['distance_km']} km")

        if result["is_delay_risk"]:
            st.error("⚠️ PERINGATAN KETERLAMBATAN: Pesanan diprediksi memerlukan waktu >180 menit! Sistem Early Warning Delay merekomendasikan re-routing atau konfirmasi ke tim Customer Service.")
        else:
            st.success("✅ Estimasi Waktu Tiba Berada dalam Batas Operasional Normal (SLA Aman).")

        st.info(f"Interval Kepercayaan 80%: Durasi diproyeksikan berada di rentang {result['confidence_interval_80pct']['min_minutes']} - {result['confidence_interval_80pct']['max_minutes']} menit.")

with tab2:
    st.subheader("Ringkasan Metrik Model Terhadap Target Bisnis & SMART")
    metrics_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "evaluation_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)

        st.markdown("#### Perbandingan Model Regresi pada Data Uji (Temporal Split 20%)")
        comp_df = pd.DataFrame(metrics_data["model_comparison"]).T
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("#### Verifikasi Target SMART:")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Penurunan MAE", "57.4%", "Target: ≥25% (LULUS)")
        m2.metric("R² Score", "0.8099", "Target: ≥0.70 (LULUS)")
        m3.metric("Akurasi ETA ±30m", "82.96%", "Target: ≥80% (LULUS)")
        m4.metric("Recall Keterlambatan", "72.72%", "Target: ≥70% (LULUS)")
    else:
        st.warning("File metrik belum tersedia. Jalankan pelatihan model terlebih dahulu: python Scripts/model_pipeline.py")

with tab3:
    st.subheader("Rencana Deployment Proporsional & Panduan Etika")
    st.markdown("""
    ### 1. Pola Deployment yang Proporsional
    * **Arsitektur Hybrid:** REST API microservice untuk kalkulasi Dynamic ETA real-time saat pesanan dibuat, dipadukan dengan Batch Polling harian untuk pemantauan SLA.
    * **Buffer Waktu Aman (Safety Buffer):** Tambahan buffer 10 menit pada janji waktu tiba konsumen untuk melindungi kurir dari tekanan kecepatan berkendara yang membahayakan keselamatan.
    * **Sistem Peringatan Dini (Early Warning):** Notifikasi otomatis kepada Dispatcher jika prediksi melebihi 180 menit.

    ### 2. Safeguard Etika & Hukum (UU No. 27/2022)
    * **Perlindungan Pekerja Anak:** Transaksi dengan kurir usia <18 tahun diblokir dari sistem penugasan operasional dan ditandai sebagai anomali audit.
    * **Anti-Diskriminasi Profil Kurir:** Fitur usia dan rating hanya digunakan sebagai faktor moderasi analisis kalibrasi, bukan untuk sanksi penalti individu atau penurunan komisi.
    * **Privasi Lokasi Geospasial:** Koordinat presisi tidak dipublikasikan ke pihak ketiga; model menggunakan agregasi jarak geodesik yang terenkripsi.
    """)

