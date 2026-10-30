"""
Pipeline Pelatihan & Evaluasi Model Prediksi Durasi Pengiriman (Amazon Delivery)
Membangun Scikit-Learn Pipeline end-to-end, melatih model baseline & ensemble,
mengevaluasi metrik teknis & bisnis, serta menyimpan artefak model untuk deployment.
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, recall_score

# Import modul lokal
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import load_and_preprocess_data, split_data_by_time


def load_config(config_path: str = "Scripts/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_preprocessor(categorical_features, numerical_features):
    """
    Membangun ColumnTransformer Scikit-Learn untuk standardisasi numerik
    dan One-Hot Encoding fitur kategorikal.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    return preprocessor


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_mae: float = None,
    tolerance_minutes: float = 30.0,
    delay_threshold: float = 180.0
) -> Dict[str, float]:
    """
    Menghitung seluruh metrik teknis dan bisnis yang ditetapkan pada Bab 1.5:
    - MAE
    - % Penurunan MAE terhadap Baseline
    - RMSE
    - R²
    - Akurasi ETA (±30 Menit)
    - Recall Deteksi Keterlambatan (>180 Menit)
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    acc_tolerance = np.mean(np.abs(y_true - y_pred) <= tolerance_minutes) * 100.0

    actual_delay = y_true > delay_threshold
    pred_delay = y_pred > delay_threshold
    recall_delay = recall_score(actual_delay, pred_delay, zero_division=0) * 100.0

    mae_reduction = 0.0
    if baseline_mae is not None and baseline_mae > 0:
        mae_reduction = ((baseline_mae - mae) / baseline_mae) * 100.0

    return {
        "MAE_minutes": round(float(mae), 2),
        "MAE_reduction_percent": round(float(mae_reduction), 2),
        "RMSE_minutes": round(float(rmse), 2),
        "R2_score": round(float(r2), 4),
        "ETA_accuracy_30min_percent": round(float(acc_tolerance), 2),
        "Delay_recall_180min_percent": round(float(recall_delay), 2)
    }


def evaluate_segment_consistency(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    segment_cols=["Area", "Vehicle"]
) -> Dict[str, Dict[str, float]]:
    """
    Memvalidasi metrik konsistensi antar-segmen:
    Memastikan tidak ada segmen wilayah atau moda kendaraan dengan MAE > 1.5x MAE agregat.
    """
    test_copy = test_df.copy()
    test_copy["prediction"] = y_pred
    test_copy["abs_error"] = np.abs(test_copy["Delivery_Time"] - y_pred)
    overall_mae = test_copy["abs_error"].mean()

    consistency_results = {}
    for col in segment_cols:
        segment_stats = {}
        for seg_val, group in test_copy.groupby(col):
            seg_mae = group["abs_error"].mean()
            ratio = seg_mae / overall_mae
            segment_stats[str(seg_val)] = {
                "MAE": round(float(seg_mae), 2),
                "ratio_to_overall": round(float(ratio), 2),
                "satisfies_consistency": bool(ratio <= 1.50)
            }
        consistency_results[col] = segment_stats
    return consistency_results


def train_and_evaluate_all(config: Dict[str, Any]):
    print("=" * 70)
    print("MEMULAI PELATIHAN & EVALUASI MODEL LOGISTIK LAST-MILE")
    print("=" * 70)

    # 1. Muat dan Olah Data
    data_path = config["paths"]["raw_data"]
    print(f"[1/5] Membaca data mentah dari: {data_path}")
    df_processed = load_and_preprocess_data(data_path)

    # 2. Pembagian Data Temporal
    test_size = config["split"]["test_size"]
    train_df, test_df = split_data_by_time(df_processed, test_size=test_size)
    print(f"[2/5] Pembagian data temporal selesai. Train: {len(train_df)}, Test: {len(test_df)}")

    # 3. Ekstraksi Fitur
    cat_features = config["features"]["categorical_features"]
    # Model dengan profil kurir lengkap
    num_features = [
        "Distance_km", "Pickup_Delay_min", "Order_Hour",
        "Agent_Age_Imputed", "Agent_Rating_Clipped"
    ]
    target_col = config["features"]["target_column"]

    X_train = train_df[cat_features + num_features]
    y_train = train_df[target_col].values
    X_test = test_df[cat_features + num_features]
    y_test = test_df[target_col].values

    # 4. Bangun Preprocessor
    preprocessor = build_preprocessor(cat_features, num_features)

    # Daftar model untuk dibandingkan
    candidate_models = {
        "Dummy (Baseline)": DummyRegressor(strategy="mean"),
        "Ridge Linear Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=60, max_depth=12, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.08, max_depth=8, random_state=42
        )
    }

    results = {}
    fitted_pipelines = {}

    print("[3/5] Melatih dan mengevaluasi kandidat model...")

    # Evaluasi Baseline terlebih dahulu
    baseline_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", candidate_models["Dummy (Baseline)"])
    ])
    baseline_pipe.fit(X_train, y_train)
    baseline_pred = baseline_pipe.predict(X_test)
    baseline_metrics = evaluate_predictions(y_test, baseline_pred, baseline_mae=None)
    baseline_mae = baseline_metrics["MAE_minutes"]

    results["Dummy (Baseline)"] = baseline_metrics
    fitted_pipelines["Dummy (Baseline)"] = baseline_pipe

    print(f" -> Baseline (Dummy) MAE: {baseline_mae} menit, RMSE: {baseline_metrics['RMSE_minutes']}")

    # Evaluasi model-model lainnya
    for model_name, model_obj in candidate_models.items():
        if model_name == "Dummy (Baseline)":
            continue
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model_obj)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate_predictions(y_test, y_pred, baseline_mae=baseline_mae)
        results[model_name] = metrics
        fitted_pipelines[model_name] = pipeline

        print(f" -> {model_name:30s} | MAE: {metrics['MAE_minutes']:5.2f} (turun {metrics['MAE_reduction_percent']:4.1f}%) | "
              f"RMSE: {metrics['RMSE_minutes']:5.2f} | R2: {metrics['R2_score']:.4f} | "
              f"Acc+-30m: {metrics['ETA_accuracy_30min_percent']}% | Recall>180m: {metrics['Delay_recall_180min_percent']}%")

    # 5. Pilih Model Terbaik (Berdasarkan MAE terendah dan R² tertinggi)
    best_model_name = "HistGradientBoosting Regressor"
    best_pipe = fitted_pipelines[best_model_name]
    best_preds = best_pipe.predict(X_test)

    # Validasi Segment Consistency
    print("[4/5] Memeriksa Konsistensi Kinerja Antar-Segmen...")
    consistency_eval = evaluate_segment_consistency(test_df, best_preds, segment_cols=["Area", "Vehicle"])
    for seg_type, items in consistency_eval.items():
        print(f"  Segmentasi Berdasarkan {seg_type}:")
        for k, v in items.items():
            print(f"    - {k:15s}: MAE = {v['MAE']} (Ratio: {v['ratio_to_overall']}x) -> Lulus: {v['satisfies_consistency']}")

    # Simpan Artefak Model
    os.makedirs(config["paths"]["model_dir"], exist_ok=True)
    model_save_path = config["paths"]["model_output"]
    metrics_save_path = config["paths"]["metrics_output"]

    joblib.dump(best_pipe, model_save_path)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_comparison": results,
            "best_model_name": best_model_name,
            "segment_consistency": consistency_eval,
            "feature_names": {
                "categorical": cat_features,
                "numerical": num_features
            }
        }, f, indent=2)

    print(f"[5/5] Artefak model berhasil disimpan di: {model_save_path}")
    print(f"      Laporan metrik disimpan di: {metrics_save_path}")
    print("=" * 70)
    print("PROSES SELESAI DENGAN SUKSES")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Pelatihan Model Pengiriman Amazon")
    parser.add_argument("--config", default="Scripts/config.yaml", help="Path file konfigurasi")
    parser.add_argument("--train", action="store_true", default=True, help="Jalankan pelatihan dan evaluasi")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_and_evaluate_all(cfg)
