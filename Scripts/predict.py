"""
Modul Inferensi & Deployment Layanan Prediksi ETA Pengiriman Last-Mile (Amazon Delivery)
Menyediakan antarmuka inferensi (Programmatic & CLI) untuk Use Case Utama (Dynamic ETA)
dan Use Case Pendukung (Early Warning Delay System & Smart Dispatching).
"""

import os
import sys
import argparse
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Union, List

# Tambahkan direktori skrip ke path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import clean_raw_data, engineer_features


class ETAPredictorService:
    """
    Layanan inferensi siap produksi untuk memprediksi estimasi waktu tiba (ETA).
    """

    def __init__(
        self,
        model_path: str = "Scripts/models/best_delivery_model.joblib",
        safety_buffer_minutes: float = 10.0,
        delay_threshold_minutes: float = 180.0
    ):
        self.model_path = model_path
        self.safety_buffer_minutes = safety_buffer_minutes
        self.delay_threshold_minutes = delay_threshold_minutes

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Please train the model first.")

        self.pipeline = joblib.load(model_path)
        self.cat_features = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
        self.num_features = [
            "Distance_km", "Pickup_Delay_min", "Order_Hour",
            "Agent_Age_Imputed", "Agent_Rating_Clipped"
        ]

    def predict_single(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Melakukan inferensi untuk 1 transaksi pesanan baru.
        Menerima dictionary data operasional pesanan.
        """
        df_input = pd.DataFrame([order_data])
        results = self.predict_batch(df_input)
        return results[0]

    def predict_batch(self, df_input: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Melakukan inferensi untuk kumpulan pesanan (batch) secara efisien.
        """
        # Bersihkan dan rekayasa fitur
        df_clean = clean_raw_data(df_input)
        df_features = engineer_features(df_clean)

        X = df_features[self.cat_features + self.num_features]
        raw_predictions = self.pipeline.predict(X)

        output_records = []
        for i, pred_duration in enumerate(raw_predictions):
            pred_minutes = round(float(pred_duration), 1)
            buffered_minutes = round(pred_minutes + self.safety_buffer_minutes, 1)

            # Estimasi waktu tiba dalam jam:menit
            order_time_str = str(df_features.iloc[i].get("Order_Time", "12:00:00")).strip()
            est_arrival_time = None
            try:
                parts = order_time_str.split(":")
                if len(parts) >= 2:
                    h, m = int(parts[0]), int(parts[1])
                    base_dt = datetime(2026, 1, 1, h, m)
                    arrival_dt = base_dt + timedelta(minutes=pred_minutes)
                    est_arrival_time = arrival_dt.strftime("%H:%M:%S")
            except Exception:
                est_arrival_time = "N/A"

            # Klasifikasi Early Warning Keterlambatan
            is_delay_risk = pred_minutes > self.delay_threshold_minutes

            # Confidence interval (~80% error margin +-18 menit berdasarkan MAE model)
            lower_bound = max(10.0, round(pred_minutes - 18.0, 1))
            upper_bound = round(pred_minutes + 18.0, 1)

            output_records.append({
                "Order_ID": str(df_features.iloc[i].get("Order_ID", f"ORD_{i+1}")),
                "predicted_delivery_minutes": pred_minutes,
                "buffered_delivery_minutes": buffered_minutes,
                "confidence_interval_80pct": {
                    "min_minutes": lower_bound,
                    "max_minutes": upper_bound
                },
                "estimated_arrival_time": est_arrival_time,
                "distance_km": round(float(df_features.iloc[i]["Distance_km"]), 2),
                "delay_risk_status": "HIGH_DELAY_RISK (>180m)" if is_delay_risk else "NORMAL",
                "is_delay_risk": bool(is_delay_risk),
                "operational_notes": {
                    "traffic": str(df_features.iloc[i]["Traffic"]),
                    "weather": str(df_features.iloc[i]["Weather"]),
                    "vehicle": str(df_features.iloc[i]["Vehicle"]),
                    "area": str(df_features.iloc[i]["Area"])
                }
            })

        return output_records


def run_cli_demo():
    print("=" * 70)
    print("DEMO INFERENSI LAYANAN PREDIKSI ETA PENGIRIMAN LAST-MILE")
    print("=" * 70)

    service = ETAPredictorService()

    # Simulasi 2 skenario pengiriman:
    # 1. Skenario lancar (Cuaca cerah, lalu lintas Low, motor, jarak sedang)
    sample_order_normal = {
        "Order_ID": "TEST_ORDER_001",
        "Agent_Age": 28,
        "Agent_Rating": 4.8,
        "Store_Latitude": 22.745049,
        "Store_Longitude": 75.892471,
        "Drop_Latitude": 22.765049,
        "Drop_Longitude": 75.912471,
        "Order_Date": "2026-03-20",
        "Order_Time": "14:15:00",
        "Pickup_Time": "14:25:00",
        "Weather": "Sunny",
        "Traffic": "Low",
        "Vehicle": "motorcycle",
        "Area": "Urban",
        "Category": "Clothing"
    }

    # 2. Skenario macet & badai ekstrem (Berisiko keterlambatan >180 menit)
    sample_order_jam = {
        "Order_ID": "TEST_ORDER_002",
        "Agent_Age": 32,
        "Agent_Rating": 4.5,
        "Store_Latitude": 22.715000,
        "Store_Longitude": 75.850000,
        "Drop_Latitude": 22.825000,
        "Drop_Longitude": 75.950000,
        "Order_Date": "2026-03-20",
        "Order_Time": "18:30:00",
        "Pickup_Time": "18:45:00",
        "Weather": "Stormy",
        "Traffic": "Jam",
        "Vehicle": "scooter",
        "Area": "Metropolitian ",  # sengaja ada typo & trailing spasi untuk uji ketahanan
        "Category": "Electronics"
    }

    for idx, sample in enumerate([sample_order_normal, sample_order_jam], 1):
        print(f"\n--- Pengujian Pesanan Contoh #{idx} [{sample['Order_ID']}] ---")
        res = service.predict_single(sample)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferensi Model ETA Amazon Delivery")
    parser.add_argument("--demo", action="store_true", default=True, help="Jalankan simulasi demo")
    args = parser.parse_args()

    run_cli_demo()

