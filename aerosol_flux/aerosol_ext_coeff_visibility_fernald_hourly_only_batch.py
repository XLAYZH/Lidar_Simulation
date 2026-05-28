# -*- coding: utf-8 -*-
"""
aerosol_ext_coeff_visibility_fernald_hourly_only_batch.py

Batch Fernald inversion driven by hourly visibility from an Excel table.

This script is intentionally simplified from the target-profile version:
1. Read all *_preprocessed.npz files under date subfolders.
2. Use raw timestamp saved in each npz as the only time basis.
3. Read hourly visibility from an xlsx table.
4. Interpolate missing hourly visibility values in time.
5. Interpolate hourly visibility to each lidar profile time.
6. Perform Fernald inversion with time-varying reference values.
7. Output only:
   - single-radial time-height extinction curtain
   - file-internal VAD-16 time-height extinction curtain
   - result npz for each date folder

No target-time profile matching is performed here.
This version is intended only to verify the hourly inversion chain first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =========================================================
# Matplotlib style: English + Times New Roman
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


# =========================================================
# User settings
# =========================================================
PREPROCESSED_ROOT = Path(r"E:\测风组实验数据\气溶胶反演\预处理")
VISIBILITY_XLSX_PATH = Path(r"F:\3220240787\DataShareClub\weather_data\beijing_weather.xlsx")
OUTPUT_ROOT = Path(r"E:\测风组实验数据\气溶胶反演\fernald_output_hourly_only")

# Optional whitelist of date folders. Example: ["2026_04_20", "2026_04_21"]
DATE_FOLDER_WHITELIST: Optional[list[str]] = ["2024_10_26", "2026_04_19", "2026_04_20"]

# Gap thresholds for blank display in curtains
SINGLE_GAP_THRESHOLD_SECONDS = 30.0
VAD16_GAP_THRESHOLD_SECONDS = 180.0

# Curtain color scale
VMIN_ALPHA = 0.0
VMAX_ALPHA = None


# =========================================================
# System / inversion parameters
# =========================================================
LAMBDA_NM = 1550.0
SA_AER = 29.978
SM_MOL = 8.0 * np.pi / 3.0
K_BETA = 0.2165

ELEV_DEG = 72.0
LIDAR_ALT_M = 0.0
N_AZIMUTH_PER_VAD = 16


# =========================================================
# Date / time utilities
# =========================================================
def parse_date_like_to_date(value):
    """
    Supported formats:
    - YYYY/M/D
    - YYYY-MM-DD
    - YYYY_MM_DD
    - YYYYMMDD
    - date strings with time
    """
    if pd.isna(value):
        raise ValueError("Date value is empty")

    s = str(value).strip()
    if s == "":
        raise ValueError("Date value is empty string")

    s_norm = s.replace("_", "-").replace(".", "-").replace("/", "-")

    if s_norm.isdigit() and len(s_norm) == 8:
        dt = pd.to_datetime(s_norm, format="%Y%m%d", errors="coerce")
    else:
        dt = pd.to_datetime(s_norm, format="%Y-%m-%d", errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(s_norm, errors="coerce")

    if pd.isna(dt):
        raise ValueError(f"Cannot parse date: {value}")

    return dt.date()



def get_date_from_folder_name(folder: str | Path):
    folder = Path(folder)
    return parse_date_like_to_date(folder.name)



def utc_seconds_to_local_datetime_index(utc_seconds: np.ndarray) -> pd.DatetimeIndex:
    utc_seconds = np.asarray(utc_seconds, dtype=np.float64)
    dt_utc = pd.to_datetime(utc_seconds, unit="s", utc=True)
    dt_local = dt_utc.tz_convert("Asia/Shanghai")
    return dt_local



def datetime_index_to_plot_num(dt_index: pd.DatetimeIndex) -> np.ndarray:
    dt_naive = dt_index.tz_localize(None)
    return mdates.date2num(dt_naive.to_pydatetime())


# =========================================================
# Geometry and molecular backscatter
# =========================================================
def slant_range_to_height_km(
    range_m: np.ndarray,
    elev_deg: float = ELEV_DEG,
    lidar_alt_m: float = LIDAR_ALT_M,
) -> np.ndarray:
    range_m = np.asarray(range_m, dtype=np.float64)
    height_m = lidar_alt_m + range_m * np.sin(np.deg2rad(elev_deg))
    return height_m / 1000.0



def molecular_backscatter_km(
    height_km: np.ndarray,
    lambda_nm: float = LAMBDA_NM,
) -> np.ndarray:
    """
    beta_m(z, lambda) = 1.54e-3 * (532 / lambda)^4 * exp(-z / 7)

    Units:
    - z: km
    - lambda: nm
    - beta_m: km^-1 sr^-1
    """
    z_km = np.asarray(height_km, dtype=np.float64)
    beta_m = 1.54e-3 * (532.0 / lambda_nm) ** 4 * np.exp(-z_km / 7.0)
    return beta_m


# =========================================================
# Visibility xlsx reader
# =========================================================
def _parse_hourly_local_time_with_year(value, year: int) -> pd.Timestamp:
    """
    Parse time strings like '10-26 20:00' using the given year.
    Return timezone-aware local time (Asia/Shanghai).
    """
    if pd.isna(value):
        return pd.NaT

    s = str(value).strip()
    if s == "":
        return pd.NaT

    s_norm = s.replace("/", "-").replace("_", "-").replace(".", "-")
    dt = pd.to_datetime(f"{year}-{s_norm}", format="%Y-%m-%d %H:%M", errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(f"{year}-{s_norm}", errors="coerce")
    if pd.isna(dt):
        return pd.NaT
    return dt.tz_localize("Asia/Shanghai")



def load_hourly_weather_table_for_year(
    xlsx_path: str | Path,
    year: int,
) -> pd.DataFrame:
    """
    Read the xlsx weather table.

    Table layout (0-based column indices):
    - col 0  : local time string MM-DD HH:00
    - col 9  : visibility in km
    - col 13 : AQI
    - col 14 : PM2.5 index
    - col 15 : PM10 index
    """
    xlsx_path = Path(xlsx_path)
    raw = pd.read_excel(xlsx_path, header=None)

    if raw.shape[0] < 3:
        raise ValueError("The xlsx table does not have enough rows")

    data = raw.iloc[2:].copy()

    required_max_col = 15
    if data.shape[1] <= required_max_col:
        raise ValueError(
            f"The xlsx table must contain at least {required_max_col + 1} columns"
        )

    df = pd.DataFrame({
        "time_local": data.iloc[:, 0].apply(lambda x: _parse_hourly_local_time_with_year(x, year)),
        "visibility_km": pd.to_numeric(data.iloc[:, 9], errors="coerce"),
        "AQI": pd.to_numeric(data.iloc[:, 13], errors="coerce"),
        "PM25_index": pd.to_numeric(data.iloc[:, 14], errors="coerce"),
        "PM10_index": pd.to_numeric(data.iloc[:, 15], errors="coerce"),
    })

    df = df.dropna(subset=["time_local"]).sort_values("time_local").drop_duplicates(subset=["time_local"])
    df = df.set_index("time_local")
    return df



def build_hourly_weather_for_date(
    xlsx_path: str | Path,
    target_date,
) -> pd.DataFrame:
    """
    For one date, build a complete hourly local-time table (00:00~23:00).
    Missing visibility values are interpolated in time.
    """
    target_date = pd.Timestamp(target_date).date()
    year = target_date.year

    df_year = load_hourly_weather_table_for_year(xlsx_path, year)

    day_start = pd.Timestamp(target_date).tz_localize("Asia/Shanghai")
    hourly_index = pd.date_range(
        start=day_start,
        periods=24,
        freq="1h",
        tz="Asia/Shanghai",
    )

    df_day = df_year[df_year.index.date == target_date].copy()
    df_day = df_day.reindex(hourly_index)

    vis = df_day["visibility_km"].astype(float)
    vis = vis.interpolate(method="time", limit_direction="both")
    df_day["visibility_km"] = vis

    for col in ["AQI", "PM25_index", "PM10_index"]:
        s = df_day[col].astype(float)
        s = s.interpolate(method="time", limit_direction="both")
        df_day[col] = s

    if df_day["visibility_km"].isna().all():
        raise ValueError(f"No valid hourly visibility found for date {target_date}")

    return df_day



def interpolate_hourly_visibility_to_times(
    hourly_visibility_km: pd.Series,
    target_times_local: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Interpolate hourly visibility to arbitrary profile times (UTC+8).
    """
    hourly_visibility_km = hourly_visibility_km.copy().sort_index()
    target_times_local = pd.DatetimeIndex(target_times_local)

    union_index = hourly_visibility_km.index.union(target_times_local).sort_values()
    union_series = hourly_visibility_km.reindex(union_index)
    union_series = union_series.interpolate(method="time", limit_direction="both")

    out = union_series.reindex(target_times_local).to_numpy(dtype=np.float64)
    return out



def q_from_visibility_km(u_km: np.ndarray | float) -> np.ndarray:
    u = np.asarray(u_km, dtype=np.float64)
    q = np.empty_like(u, dtype=np.float64)
    q[u > 50.0] = 1.6
    mask_mid = (u > 6.0) & (u <= 50.0)
    q[mask_mid] = 1.3
    mask_low = u <= 6.0
    q[mask_low] = 0.585 * np.power(u[mask_low], 1.0 / 3.0)
    return q



def visibility_to_reference_beta_alpha_from_km(
    visibility_km: np.ndarray | float,
    lambda_nm: float = LAMBDA_NM,
    sa_aer: float = SA_AER,
    k_beta: float = K_BETA,
):
    """
    Convert visibility (km) to:
    - beta_near_ground : km^-1 sr^-1
    - beta_a_ref       : km^-1 sr^-1
    - alpha_a_ref      : km^-1
    - q                : unitless
    """
    u_km = np.asarray(visibility_km, dtype=np.float64)
    if np.any(u_km <= 0):
        raise ValueError("Visibility must be positive")

    q = q_from_visibility_km(u_km)
    beta_near_ground = 3.91 / (sa_aer * u_km) * np.power((lambda_nm / 550.0), -q)
    beta_a_ref = k_beta * beta_near_ground
    alpha_a_ref = sa_aer * beta_a_ref
    return beta_near_ground, beta_a_ref, alpha_a_ref, q


# =========================================================
# Read one preprocessed file
# =========================================================
def load_one_preprocessed_npz(npz_path: str | Path):
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        timestamp = np.asarray(data["timestamp"], dtype=np.float64).reshape(-1)
        p_rcs = np.asarray(data["p_rcs"], dtype=np.float64)
        s_rcs = np.asarray(data["s_rcs"], dtype=np.float64)
        range_m = np.asarray(data["range_m"], dtype=np.float64).reshape(-1)
        azimuth = np.asarray(data["azimuth"], dtype=np.float64).reshape(-1)

    if p_rcs.shape != s_rcs.shape:
        raise ValueError(f"{npz_path.name}: p_rcs and s_rcs shape mismatch")
    if p_rcs.shape[0] != timestamp.size:
        raise ValueError(f"{npz_path.name}: timestamp length mismatch")
    if p_rcs.shape[1] != range_m.size:
        raise ValueError(f"{npz_path.name}: range axis mismatch")

    x_total = np.maximum(p_rcs + s_rcs, 1e-12)

    return {
        "timestamp": timestamp,
        "time_local": utc_seconds_to_local_datetime_index(timestamp),
        "azimuth": azimuth,
        "range_m": range_m,
        "x_total": x_total,
        "npz_path": str(npz_path),
    }


# =========================================================
# Collect whole-day datasets
# =========================================================
def collect_one_day_single_radials(preprocessed_date_dir: str | Path):
    preprocessed_date_dir = Path(preprocessed_date_dir)
    npz_files = sorted(preprocessed_date_dir.glob("*_preprocessed.npz"))

    if len(npz_files) == 0:
        raise FileNotFoundError(f"No *_preprocessed.npz found in {preprocessed_date_dir}")

    timestamps_all = []
    azimuth_all = []
    x_total_all = []
    source_file_all = []
    range_ref = None

    for npz_path in npz_files:
        d = load_one_preprocessed_npz(npz_path)

        if range_ref is None:
            range_ref = d["range_m"]
        else:
            if not np.allclose(range_ref, d["range_m"], rtol=0, atol=1e-6):
                raise ValueError(f"Range axis mismatch in {npz_path}")

        timestamps_all.append(d["timestamp"])
        azimuth_all.append(d["azimuth"])
        x_total_all.append(d["x_total"])
        source_file_all.extend([str(npz_path)] * d["x_total"].shape[0])

    timestamps_all = np.concatenate(timestamps_all, axis=0)
    azimuth_all = np.concatenate(azimuth_all, axis=0)
    x_total_all = np.concatenate(x_total_all, axis=0)
    source_file_all = np.asarray(source_file_all)

    sort_idx = np.argsort(timestamps_all)
    timestamps_all = timestamps_all[sort_idx]
    azimuth_all = azimuth_all[sort_idx]
    x_total_all = x_total_all[sort_idx]
    source_file_all = source_file_all[sort_idx]

    time_local_all = utc_seconds_to_local_datetime_index(timestamps_all)

    return {
        "range_m": range_ref,
        "timestamp_utc_s": timestamps_all,
        "time_local": time_local_all,
        "azimuth_deg": azimuth_all,
        "x_total": x_total_all,
        "source_file": source_file_all,
    }



def average_one_file_by_vad_cycle(
    x_total: np.ndarray,
    timestamp_utc_s: np.ndarray,
    azimuth_deg: np.ndarray,
    n_azimuth_per_vad: int = N_AZIMUTH_PER_VAD,
):
    """
    VAD-16 averaging within one file only. No cross-file grouping.
    """
    x_total = np.asarray(x_total, dtype=np.float64)
    timestamp_utc_s = np.asarray(timestamp_utc_s, dtype=np.float64)
    azimuth_deg = np.asarray(azimuth_deg, dtype=np.float64)

    sort_idx = np.argsort(timestamp_utc_s)
    x_total = x_total[sort_idx]
    timestamp_utc_s = timestamp_utc_s[sort_idx]
    azimuth_deg = azimuth_deg[sort_idx]

    n_radial, n_gate = x_total.shape
    n_cycle = n_radial // n_azimuth_per_vad
    n_used = n_cycle * n_azimuth_per_vad
    n_discard = n_radial - n_used

    if n_cycle == 0:
        return {
            "x_vad": np.empty((0, n_gate), dtype=np.float64),
            "timestamp_utc_s_vad": np.empty((0,), dtype=np.float64),
            "azimuth_mean_vad": np.empty((0,), dtype=np.float64),
            "n_discard": n_discard,
        }

    x_trim = x_total[:n_used]
    t_trim = timestamp_utc_s[:n_used]
    az_trim = azimuth_deg[:n_used]

    x_vad = x_trim.reshape(n_cycle, n_azimuth_per_vad, n_gate).mean(axis=1)
    t_vad = t_trim.reshape(n_cycle, n_azimuth_per_vad).mean(axis=1)
    az_vad_mean = az_trim.reshape(n_cycle, n_azimuth_per_vad).mean(axis=1)

    return {
        "x_vad": x_vad,
        "timestamp_utc_s_vad": t_vad,
        "azimuth_mean_vad": az_vad_mean,
        "n_discard": n_discard,
    }



def collect_one_day_file_internal_vad(preprocessed_date_dir: str | Path):
    preprocessed_date_dir = Path(preprocessed_date_dir)
    npz_files = sorted(preprocessed_date_dir.glob("*_preprocessed.npz"))

    if len(npz_files) == 0:
        raise FileNotFoundError(f"No *_preprocessed.npz found in {preprocessed_date_dir}")

    timestamps_vad_all = []
    azimuth_vad_all = []
    x_vad_all = []
    source_file_all = []
    range_ref = None
    total_discard = 0

    for npz_path in npz_files:
        d = load_one_preprocessed_npz(npz_path)

        if range_ref is None:
            range_ref = d["range_m"]
        else:
            if not np.allclose(range_ref, d["range_m"], rtol=0, atol=1e-6):
                raise ValueError(f"Range axis mismatch in {npz_path}")

        vad_d = average_one_file_by_vad_cycle(
            x_total=d["x_total"],
            timestamp_utc_s=d["timestamp"],
            azimuth_deg=d["azimuth"],
            n_azimuth_per_vad=N_AZIMUTH_PER_VAD,
        )

        if vad_d["x_vad"].shape[0] > 0:
            timestamps_vad_all.append(vad_d["timestamp_utc_s_vad"])
            azimuth_vad_all.append(vad_d["azimuth_mean_vad"])
            x_vad_all.append(vad_d["x_vad"])
            source_file_all.extend([str(npz_path)] * vad_d["x_vad"].shape[0])

        total_discard += vad_d["n_discard"]

    if len(x_vad_all) == 0:
        raise ValueError("No valid VAD-16 groups found for the day")

    timestamps_vad_all = np.concatenate(timestamps_vad_all, axis=0)
    azimuth_vad_all = np.concatenate(azimuth_vad_all, axis=0)
    x_vad_all = np.concatenate(x_vad_all, axis=0)
    source_file_all = np.asarray(source_file_all)

    sort_idx = np.argsort(timestamps_vad_all)
    timestamps_vad_all = timestamps_vad_all[sort_idx]
    azimuth_vad_all = azimuth_vad_all[sort_idx]
    x_vad_all = x_vad_all[sort_idx]
    source_file_all = source_file_all[sort_idx]

    time_local_vad = utc_seconds_to_local_datetime_index(timestamps_vad_all)

    return {
        "range_m": range_ref,
        "timestamp_utc_s_vad": timestamps_vad_all,
        "time_local_vad": time_local_vad,
        "azimuth_deg_vad": azimuth_vad_all,
        "x_vad": x_vad_all,
        "source_file_vad": source_file_all,
        "n_discard_after_vad_mean": total_discard,
    }


# =========================================================
# Fernald inversion
# =========================================================
def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    out = np.zeros_like(y, dtype=np.float64)
    if y.size <= 1:
        return out

    dx = np.diff(x)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)
    return out



def fernald_forward_single(
    x_profile: np.ndarray,
    range_km: np.ndarray,
    beta_m_profile: np.ndarray,
    beta_a_ref: float,
    sa_aer: float = SA_AER,
    sm_mol: float = SM_MOL,
    ref_index: int = 0,
    clip_negative: bool = True,
):
    """
    Units:
    - range_km: km
    - beta_m_profile: km^-1 sr^-1
    - beta_a_ref: km^-1 sr^-1
    - beta_a output: km^-1 sr^-1
    - alpha_a output: km^-1
    """
    x = np.asarray(x_profile, dtype=np.float64)
    r = np.asarray(range_km, dtype=np.float64)
    beta_m = np.asarray(beta_m_profile, dtype=np.float64)

    x_seg = np.maximum(x[ref_index:], 1e-12)
    r_seg = r[ref_index:]
    beta_m_seg = beta_m[ref_index:]

    i_m = cumulative_trapezoid(beta_m_seg, r_seg)
    g_r = x_seg * np.exp(-2.0 * (sa_aer - sm_mol) * i_m)
    j_r = cumulative_trapezoid(g_r, r_seg)

    denom = x_seg[0] / (beta_a_ref + beta_m_seg[0]) - 2.0 * sa_aer * j_r

    beta_total_seg = np.full_like(g_r, np.nan, dtype=np.float64)
    valid = denom > 0.0
    beta_total_seg[valid] = g_r[valid] / denom[valid]

    beta_a_seg = beta_total_seg - beta_m_seg
    if clip_negative:
        beta_a_seg = np.maximum(beta_a_seg, 0.0)

    alpha_a_seg = sa_aer * beta_a_seg

    beta_a = np.full_like(x, np.nan, dtype=np.float64)
    alpha_a = np.full_like(x, np.nan, dtype=np.float64)
    beta_a[ref_index:] = beta_a_seg
    alpha_a[ref_index:] = alpha_a_seg

    return beta_a, alpha_a



def invert_profiles(
    x_profiles: np.ndarray,
    range_km: np.ndarray,
    beta_m_profile: np.ndarray,
    beta_a_ref_values: np.ndarray | float,
    sa_aer: float = SA_AER,
    sm_mol: float = SM_MOL,
    ref_index: int = 0,
):
    """
    Batch inversion for multiple profiles.
    beta_a_ref_values can be:
    - scalar
    - 1D array of length N_profile
    """
    x_profiles = np.asarray(x_profiles, dtype=np.float64)
    n_prof, n_gate = x_profiles.shape

    beta_a_ref_arr = np.asarray(beta_a_ref_values, dtype=np.float64)
    if beta_a_ref_arr.ndim == 0:
        beta_a_ref_arr = np.full((n_prof,), float(beta_a_ref_arr), dtype=np.float64)
    elif beta_a_ref_arr.shape[0] != n_prof:
        raise ValueError("beta_a_ref_values length must match number of profiles")

    beta_a_all = np.full((n_prof, n_gate), np.nan, dtype=np.float64)
    alpha_a_all = np.full((n_prof, n_gate), np.nan, dtype=np.float64)

    for i in range(n_prof):
        beta_i, alpha_i = fernald_forward_single(
            x_profile=x_profiles[i],
            range_km=range_km,
            beta_m_profile=beta_m_profile,
            beta_a_ref=float(beta_a_ref_arr[i]),
            sa_aer=sa_aer,
            sm_mol=sm_mol,
            ref_index=ref_index,
            clip_negative=True,
        )
        beta_a_all[i] = beta_i
        alpha_a_all[i] = alpha_i

    return beta_a_all, alpha_a_all


# =========================================================
# Gap handling for blank non-acquisition periods
# =========================================================
def suggest_gap_threshold_seconds(
    time_local: pd.DatetimeIndex,
    min_threshold_seconds: float = 30.0,
) -> float:
    if len(time_local) < 2:
        return min_threshold_seconds

    dt_seconds = np.diff(time_local.asi8) / 1e9
    dt_seconds = dt_seconds[np.isfinite(dt_seconds)]
    dt_seconds = dt_seconds[dt_seconds > 0]

    if dt_seconds.size == 0:
        return min_threshold_seconds

    median_dt = float(np.median(dt_seconds))
    return max(min_threshold_seconds, 5.0 * median_dt)



def insert_blank_columns_for_gaps(
    time_local: pd.DatetimeIndex,
    field_2d: np.ndarray,
    gap_threshold_seconds: float,
):
    time_local = pd.DatetimeIndex(time_local)
    field_2d = np.asarray(field_2d, dtype=np.float64)

    if field_2d.ndim != 2:
        raise ValueError("field_2d must be 2D with shape (N_time, N_height)")
    if field_2d.shape[0] != len(time_local):
        raise ValueError("time_local length must match field_2d.shape[0]")

    if len(time_local) <= 1:
        return time_local, field_2d

    out_times = []
    out_fields = []
    nan_profile = np.full((field_2d.shape[1],), np.nan, dtype=np.float64)

    for i in range(len(time_local) - 1):
        t0 = time_local[i]
        t1 = time_local[i + 1]
        f0 = field_2d[i]

        out_times.append(t0)
        out_fields.append(f0)

        gap_seconds = (t1 - t0).total_seconds()
        if gap_seconds > gap_threshold_seconds:
            left_blank_time = t0 + pd.Timedelta(seconds=1)
            right_blank_time = t1 - pd.Timedelta(seconds=1)

            if right_blank_time <= left_blank_time:
                mid_blank_time = t0 + (t1 - t0) / 2
                out_times.append(mid_blank_time)
                out_fields.append(nan_profile.copy())
            else:
                out_times.append(left_blank_time)
                out_fields.append(nan_profile.copy())
                out_times.append(right_blank_time)
                out_fields.append(nan_profile.copy())

    out_times.append(time_local[-1])
    out_fields.append(field_2d[-1])

    new_time_local = pd.DatetimeIndex(out_times)
    new_field_2d = np.vstack(out_fields)
    return new_time_local, new_field_2d



def make_masked_cmap(base_cmap: str = "viridis"):
    cmap = plt.get_cmap(base_cmap).copy()
    cmap.set_bad(color="white", alpha=0.0)
    return cmap


# =========================================================
# Plotting
# =========================================================
def make_output_dir(output_root: str | Path, date_key) -> Path:
    output_root = Path(output_root)
    out_dir = output_root / str(date_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir



def auto_vmax(data_2d: np.ndarray):
    valid = np.asarray(data_2d, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return 1.0
    vmax = np.nanpercentile(valid, 99.0)
    if vmax <= 0:
        vmax = np.nanmax(valid)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return vmax



def plot_time_height(
    time_local: pd.DatetimeIndex,
    height_km: np.ndarray,
    alpha_2d: np.ndarray,
    title: str,
    out_png: str | Path,
    vmin: float = VMIN_ALPHA,
    vmax: float | None = VMAX_ALPHA,
    gap_threshold_seconds: float | None = None,
):
    """
    Plot a time-height curtain with blank gaps for non-acquisition periods.
    """
    time_local = pd.DatetimeIndex(time_local)
    alpha_2d = np.asarray(alpha_2d, dtype=np.float64)

    if alpha_2d.ndim != 2:
        raise ValueError("alpha_2d must be 2D")
    if alpha_2d.shape[0] != len(time_local):
        raise ValueError("time_local length must match alpha_2d.shape[0]")

    if vmax is None:
        vmax = auto_vmax(alpha_2d)

    if gap_threshold_seconds is None:
        gap_threshold_seconds = suggest_gap_threshold_seconds(time_local)

    time_plot, alpha_plot = insert_blank_columns_for_gaps(
        time_local=time_local,
        field_2d=alpha_2d,
        gap_threshold_seconds=gap_threshold_seconds,
    )

    t_num = datetime_index_to_plot_num(time_plot)
    alpha_masked = np.ma.masked_invalid(alpha_plot.T)
    cmap = make_masked_cmap("viridis")

    plt.figure(figsize=(12, 6))
    mesh = plt.pcolormesh(
        t_num,
        height_km,
        alpha_masked,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(mesh, label=r"$\alpha_a$ (km$^{-1}$)")
    plt.ylabel("Height (km)")
    plt.xlabel("Local time (UTC+8)")
    plt.title(title)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()



def plot_hourly_visibility_series(
    hourly_weather: pd.DataFrame,
    out_png: str | Path,
    date_key,
):
    fig, ax1 = plt.subplots(figsize=(10, 4))

    t = hourly_weather.index.tz_localize(None)
    ax1.plot(t, hourly_weather["visibility_km"], marker="o", linewidth=1.5)
    ax1.set_ylabel("Visibility (km)")
    ax1.set_xlabel("Local time (UTC+8)")
    ax1.set_title(f"Hourly visibility series ({date_key}, UTC+8)")
    ax1.grid(True, alpha=0.3)

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# One-date processing
# =========================================================
def process_one_date_folder(preprocessed_date_dir: str | Path, output_root: str | Path):
    preprocessed_date_dir = Path(preprocessed_date_dir)
    date_key = get_date_from_folder_name(preprocessed_date_dir)
    out_dir = make_output_dir(output_root, date_key)

    # 1. Hourly weather table for the date
    hourly_weather = build_hourly_weather_for_date(VISIBILITY_XLSX_PATH, date_key)
    hourly_visibility_km = hourly_weather["visibility_km"]

    # Optional diagnostic figure of visibility itself
    vis_png = out_dir / f"{date_key}_hourly_visibility.png"
    plot_hourly_visibility_series(hourly_weather, vis_png, date_key)

    # 2. Whole-day single-radial dataset
    day_single = collect_one_day_single_radials(preprocessed_date_dir)

    range_m = day_single["range_m"]
    range_km = range_m / 1000.0
    height_km = slant_range_to_height_km(
        range_m=range_m,
        elev_deg=ELEV_DEG,
        lidar_alt_m=LIDAR_ALT_M,
    )

    x_total_single = day_single["x_total"]
    time_local_single = day_single["time_local"]
    timestamp_utc_s_single = day_single["timestamp_utc_s"]

    vis_single_km = interpolate_hourly_visibility_to_times(hourly_visibility_km, time_local_single)
    beta_ng_single, beta_a_ref_single, alpha_a_ref_single, q_single = \
        visibility_to_reference_beta_alpha_from_km(vis_single_km)

    # 3. Whole-day file-internal VAD-16 dataset
    day_vad = collect_one_day_file_internal_vad(preprocessed_date_dir)

    x_total_vad = day_vad["x_vad"]
    time_local_vad = day_vad["time_local_vad"]
    timestamp_utc_s_vad = day_vad["timestamp_utc_s_vad"]

    vis_vad_km = interpolate_hourly_visibility_to_times(hourly_visibility_km, time_local_vad)
    beta_ng_vad, beta_a_ref_vad, alpha_a_ref_vad, q_vad = \
        visibility_to_reference_beta_alpha_from_km(vis_vad_km)

    # 4. Molecular backscatter
    beta_m_profile = molecular_backscatter_km(
        height_km=height_km,
        lambda_nm=LAMBDA_NM,
    )

    # 5. Invert whole-day single-radial curtain
    beta_a_single, alpha_a_single = invert_profiles(
        x_profiles=x_total_single,
        range_km=range_km,
        beta_m_profile=beta_m_profile,
        beta_a_ref_values=beta_a_ref_single,
        sa_aer=SA_AER,
        sm_mol=SM_MOL,
        ref_index=0,
    )

    # 6. Invert whole-day file-internal VAD-16 curtain
    beta_a_vad, alpha_a_vad = invert_profiles(
        x_profiles=x_total_vad,
        range_km=range_km,
        beta_m_profile=beta_m_profile,
        beta_a_ref_values=beta_a_ref_vad,
        sa_aer=SA_AER,
        sm_mol=SM_MOL,
        ref_index=0,
    )

    # 7. Output curtains
    curtain_single_png = out_dir / f"{date_key}_time_height_single_hourly_vis.png"
    curtain_vad_png = out_dir / f"{date_key}_time_height_vad16_file_internal_hourly_vis.png"

    plot_time_height(
        time_local=time_local_single,
        height_km=height_km,
        alpha_2d=alpha_a_single,
        title=f"Single-radial aerosol extinction curtain ({date_key}, UTC+8)",
        out_png=curtain_single_png,
        gap_threshold_seconds=SINGLE_GAP_THRESHOLD_SECONDS,
    )

    plot_time_height(
        time_local=time_local_vad,
        height_km=height_km,
        alpha_2d=alpha_a_vad,
        title=f"File-internal VAD-16 aerosol extinction curtain ({date_key}, UTC+8)",
        out_png=curtain_vad_png,
        gap_threshold_seconds=VAD16_GAP_THRESHOLD_SECONDS,
    )

    # 8. Save results
    out_npz = out_dir / f"{date_key}_fernald_inversion_hourly_vis_only.npz"
    np.savez_compressed(
        out_npz,
        # basic info
        date_key=np.array(str(date_key)),
        hourly_visibility_local_time_str=hourly_weather.index.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype="U19"),
        hourly_visibility_km=hourly_weather["visibility_km"].to_numpy(dtype=np.float64),
        hourly_AQI=hourly_weather["AQI"].to_numpy(dtype=np.float64),
        hourly_PM25_index=hourly_weather["PM25_index"].to_numpy(dtype=np.float64),
        hourly_PM10_index=hourly_weather["PM10_index"].to_numpy(dtype=np.float64),

        # axes
        range_m=range_m,
        range_km=range_km,
        height_km=height_km,
        beta_m_profile=beta_m_profile,

        # whole-day single-radial
        timestamp_utc_s_single=timestamp_utc_s_single,
        time_local_single_str=time_local_single.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype="U19"),
        visibility_single_km=vis_single_km,
        beta_near_ground_single=beta_ng_single,
        beta_a_ref_single=beta_a_ref_single,
        alpha_a_ref_single=alpha_a_ref_single,
        q_single=q_single,
        x_total_single=x_total_single,
        beta_a_single=beta_a_single,
        alpha_a_single=alpha_a_single,

        # whole-day file-internal VAD-16
        timestamp_utc_s_vad16=timestamp_utc_s_vad,
        time_local_vad16_str=time_local_vad.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype="U19"),
        visibility_vad16_km=vis_vad_km,
        beta_near_ground_vad16=beta_ng_vad,
        beta_a_ref_vad16=beta_a_ref_vad,
        alpha_a_ref_vad16=alpha_a_ref_vad,
        q_vad16=q_vad,
        x_total_vad16=x_total_vad,
        beta_a_vad16=beta_a_vad,
        alpha_a_vad16=alpha_a_vad,

        # settings
        elev_deg=np.array(ELEV_DEG, dtype=np.float64),
        lambda_nm=np.array(LAMBDA_NM, dtype=np.float64),
        sa_aer=np.array(SA_AER, dtype=np.float64),
        sm_mol=np.array(SM_MOL, dtype=np.float64),
        k_beta=np.array(K_BETA, dtype=np.float64),
        n_azimuth_per_vad=np.array(N_AZIMUTH_PER_VAD, dtype=np.int32),
        n_discard_after_vad_mean=np.array(day_vad["n_discard_after_vad_mean"], dtype=np.int32),
    )

    # 9. Console output
    print("=" * 78)
    print("Hourly-visibility Fernald inversion finished")
    print(f"Date: {date_key}")
    print(f"Preprocessed folder: {preprocessed_date_dir}")
    print(f"Single-radial profiles: {alpha_a_single.shape[0]}")
    print(f"File-internal VAD-16 profiles: {alpha_a_vad.shape[0]}")
    print(f"Discarded radials after VAD-16 grouping: {day_vad['n_discard_after_vad_mean']}")
    print(f"Visibility figure: {vis_png}")
    print(f"Single-radial curtain: {curtain_single_png}")
    print(f"File-internal VAD-16 curtain: {curtain_vad_png}")
    print(f"Output NPZ: {out_npz}")
    print("=" * 78)


# =========================================================
# Batch entry
# =========================================================
def iter_date_folders(root: str | Path, whitelist: Optional[list[str]] = None):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"PREPROCESSED_ROOT does not exist: {root}")

    candidates = [p for p in root.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.name)

    if whitelist is not None:
        whitelist_set = set(whitelist)
        candidates = [p for p in candidates if p.name in whitelist_set]

    return candidates



def main():
    date_folders = iter_date_folders(PREPROCESSED_ROOT, DATE_FOLDER_WHITELIST)
    if len(date_folders) == 0:
        raise FileNotFoundError("No date subfolders found to process")

    failed = []
    for date_dir in date_folders:
        try:
            process_one_date_folder(date_dir, OUTPUT_ROOT)
        except Exception as exc:
            failed.append((str(date_dir), str(exc)))
            print(f"[FAILED] {date_dir}: {exc}")

    print("-" * 78)
    print(f"Total date folders: {len(date_folders)}")
    print(f"Failed: {len(failed)}")
    if failed:
        for item in failed:
            print(item)
    print("-" * 78)


if __name__ == "__main__":
    main()
