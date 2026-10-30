"""
Module Pemrosesan Data & Rekayasa Fitur untuk Logistik Last-Mile (Amazon Delivery)
Mendukung pembersihan data, penanganan anomali geospasial, koreksi cross-midnight,
dan penyusunan fitur operasional sesuai temuan Business & Data Understanding.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    radius_km: float = 6371.0
) -> np.ndarray:
    """
    Menghitung jarak lingkaran besar (geodesik) antara koordinat toko dan drop
    menggunakan rumus Haversine.
    """
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius_km * c


def parse_time_string_to_minutes(time_val: Any) -> float:
    """
    Mengonversi string waktu format 'HH:MM:SS' atau 'HH:MM' ke total menit sejak tengah malam.
    """
    if pd.isna(time_val):
        return np.nan
    s = str(time_val).strip()
    if not s or s.lower() in ('nan', 'null', 'none'):
        return np.nan
    try:
        parts = s.split(':')
        if len(parts) >= 2:
            h = float(parts[0])
            m = float(parts[1])
            sec = float(parts[2]) if len(parts) > 2 else 0.0
            return h * 60.0 + m + (sec / 60.0)
    except Exception:
        pass
    return np.nan


def clean_raw_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pembersihan string kategorikal, perbaikan spasi berlebih (trailing whitespaces),
    penanganan nilai string 'NaN', serta koreksi saltik nama kelas (e.g., 'Metropolitian').
    """
    df = df_raw.copy()

    # 1. Bersihkan spasi dan konversi string 'NaN' menjadi np.nan
    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'NaN': np.nan, 'nan': np.nan, 'null': np.nan, 'None': np.nan})

    # 2. Koreksi typographical error pada Area
    df['Area'] = df['Area'].replace({'Metropolitian': 'Metropolitan'})

    return df


def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Rekayasa fitur geospasial, temporal, dan penanganan anomali nilai:
    - Koreksi tanda koordinat negatif (pembalikan tanda akibat galat pengkodean GPS)
    - Jarak Haversine dan penanganan kegagalan sinyal GPS (0.01 / 0)
    - Koreksi selisih waktu cross-midnight (+1440 menit) antara Pickup_Time dan Order_Time
    - Normalisasi/clipping rating kurir ke rentang [1.0, 5.0]
    - Penandaan audit kurir di bawah umur (< 18 tahun)
    - Ekstraksi jam pemesanan dan fitur kalender
    """
    df = df_clean.copy()

    # 1. Koreksi Geospasial
    # Nilai negatif Store_Latitude/Store_Longitude adalah cerminan tanda koordinat daratan India
    store_lat_abs = df['Store_Latitude'].abs()
    store_lon_abs = df['Store_Longitude'].abs()
    drop_lat_abs = df['Drop_Latitude'].abs()
    drop_lon_abs = df['Drop_Longitude'].abs()

    # Hitung jarak garis lurus Haversine
    dist = haversine_distance(store_lat_abs, store_lon_abs, drop_lat_abs, drop_lon_abs)

    # Identifikasi kegagalan koordinat GPS (titik 0 atau mendekati 0.01)
    gps_failure = (store_lat_abs < 1.0) | (drop_lat_abs < 1.0) | (store_lon_abs < 1.0) | (drop_lon_abs < 1.0)
    dist[gps_failure] = np.nan
    df['Distance_km'] = dist
    df['Is_GPS_Failure'] = gps_failure.astype(int)

    # Imputasi jarak berdasarkan median per tipe Area
    df['Distance_km'] = df.groupby('Area')['Distance_km'].transform(
        lambda s: s.fillna(s.median() if not s.median() != s.median() else 9.5)
    )

    # 2. Koreksi Temporal Cross-Midnight
    order_min = df['Order_Time'].apply(parse_time_string_to_minutes)
    pickup_min = df['Pickup_Time'].apply(parse_time_string_to_minutes)

    diff_min = pickup_min - order_min
    # Jika bernilai negatif (pesanan dibuat sebelum tengah malam dan dijemput setelah tengah malam)
    is_cross_midnight = diff_min < 0
    diff_min[is_cross_midnight] += 1440.0

    df['Pickup_Delay_min'] = diff_min.fillna(10.0)  # default median delay 10 menit
    df['Is_Cross_Midnight'] = is_cross_midnight.astype(int)
    df['Order_Hour'] = (order_min // 60.0).fillna(12.0)
    df['Is_Rush_Hour'] = df['Order_Hour'].isin([11, 12, 13, 18, 19, 20, 21]).astype(int)

    # Ekstraksi informasi Order_Date
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df['Day_of_Week'] = df['Order_Date'].dt.dayofweek.fillna(0).astype(int)
    df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)

    # 3. Penanganan Anomali Agen Kurir
    # Peringkat kurir di atas skala Likert 5.0 (ada nilai 6.0) dipotong ke 5.0
    df['Agent_Rating_Clipped'] = df['Agent_Rating'].clip(lower=1.0, upper=5.0)
    df['Agent_Rating_Clipped'] = df['Agent_Rating_Clipped'].fillna(df['Agent_Rating_Clipped'].median())

    # Usia kurir: flag kepatuhan regulasi (< 18 tahun)
    df['Is_Minor_Courier'] = (df['Agent_Age'] < 18).astype(int)
    df['Agent_Age_Imputed'] = df['Agent_Age'].fillna(df['Agent_Age'].median())

    # 4. Imputasi Modus Cuaca dan Kepadatan Lalu Lintas
    mode_weather = df['Weather'].mode()[0] if not df['Weather'].dropna().empty else 'Sunny'
    mode_traffic = df['Traffic'].mode()[0] if not df['Traffic'].dropna().empty else 'Low'
    df['Weather'] = df['Weather'].fillna(mode_weather)
    df['Traffic'] = df['Traffic'].fillna(mode_traffic)

    return df


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Fungsi hulu-ke-hilir untuk membaca file CSV mentah dan menghasilkan
    DataFrame yang telah bersih dan diperkaya fitur.
    """
    raw_df = pd.read_csv(csv_path)
    clean_df = clean_raw_data(raw_df)
    processed_df = engineer_features(clean_df)
    return processed_df


def split_data_by_time(
    df: pd.DataFrame,
    test_size: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Membagi dataset secara temporal (berdasarkan urutan Order_Date) untuk mencegah
    kebocoran data masa depan (future data leakage).
    """
    df_sorted = df.sort_values(by=['Order_Date', 'Order_ID']).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1.0 - test_size))
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    return train_df, test_df

