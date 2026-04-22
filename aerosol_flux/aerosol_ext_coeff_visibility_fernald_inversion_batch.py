# -*- coding: utf-8 -*-
"""
aerosol_visibility_fernald.py

Functions
---------
1. Read all *_preprocessed.npz files for one date
2. Read daily visibility from CSV
3. Convert UTC-second timestamps to UTC+8
4. Build X(R) = p_rcs + s_rcs
5. Compute molecular backscatter beta_m(z)
6. Perform Fernald inversion for:
   - all single-radial profiles of the day
   - file-internal VAD-16 mean profiles of the day
7. For a target local time:
   - locate the nearest acquisition file
   - find the nearest complete 16-radial group in that file
   - invert the 16 single profiles
   - invert the mean profile of the same 16-radial group
8. Save figures and output npz

All plot labels are in English.
All plot fonts are set to Times New Roman.
"""

from __future__ import annotations

from pathlib import Path
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
# Paths and user settings
# =========================================================
PREPROCESSED_DATE_DIR = Path(r"E:\测风组实验数据\气溶胶反演\预处理\2024_11_09")

# 能见度 CSV 文件
VISIBILITY_CSV_PATH = Path(r"E:\测风组实验数据\气溶胶反演\ncei_beijing\54511099999_2024_SI.csv")

# 输出根目录
OUTPUT_ROOT = Path(r"E:\测风组实验数据\气溶胶反演\fernald_output")

# 指定需要输出的“目标时刻”（UTC+8，本地时间）
TARGET_LOCAL_TIME_STR = "2024-11-09 20:00:00"


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

VMIN_ALPHA = 0.0
VMAX_ALPHA = None


# =========================================================
# Date / visibility
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
    folder_name = folder.name
    return parse_date_like_to_date(folder_name)


def load_daily_visibility_m(visibility_csv_path: str | Path, target_date) -> float:
    """
    Read VISIB_M for the target date from CSV.
    If multiple rows exist on the same date, use the daily mean.
    Required columns:
    - DATE
    - VISIB_M
    """
    visibility_csv_path = Path(visibility_csv_path)
    df = pd.read_csv(visibility_csv_path)

    required_cols = {"DATE", "VISIB_M"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in visibility CSV: {missing}")

    df = df.copy()
    df["DATE_PARSED"] = df["DATE"].apply(parse_date_like_to_date)
    df["VISIB_M"] = pd.to_numeric(df["VISIB_M"], errors="coerce")
    df = df.dropna(subset=["VISIB_M"])

    vis_daily = df.groupby("DATE_PARSED", as_index=True)["VISIB_M"].mean()

    if target_date not in vis_daily.index:
        raise KeyError(f"No VISIB_M found for date {target_date}")

    visibility_m = float(vis_daily.loc[target_date])
    if visibility_m <= 0:
        raise ValueError(f"Invalid visibility for {target_date}: {visibility_m}")

    return visibility_m


def q_from_visibility_km(u_km: float) -> float:
    if u_km > 50.0:
        return 1.6
    elif u_km > 6.0:
        return 1.3
    else:
        return 0.585 * (u_km ** (1.0 / 3.0))


def visibility_to_reference_beta_alpha(
    visibility_m: float,
    lambda_nm: float = LAMBDA_NM,
    sa_aer: float = SA_AER,
    k_beta: float = K_BETA,
):
    """
    Return:
    - beta_near_ground : km^-1 sr^-1
    - beta_a_ref       : km^-1 sr^-1
    - alpha_a_ref      : km^-1
    - q                : unitless
    """
    u_km = visibility_m / 1000.0
    if u_km <= 0:
        raise ValueError("Visibility must be positive")

    q = q_from_visibility_km(u_km)
    beta_near_ground = 3.91 / (sa_aer * u_km) * ((lambda_nm / 550.0) ** (-q))
    beta_a_ref = k_beta * beta_near_ground
    alpha_a_ref = sa_aer * beta_a_ref
    return beta_near_ground, beta_a_ref, alpha_a_ref, q


# =========================================================
# Time utilities
# =========================================================
def utc_seconds_to_local_datetime_index(utc_seconds: np.ndarray) -> pd.DatetimeIndex:
    utc_seconds = np.asarray(utc_seconds, dtype=np.float64)
    dt_utc = pd.to_datetime(utc_seconds, unit="s", utc=True)
    dt_local = dt_utc.tz_convert("Asia/Shanghai")
    return dt_local


def parse_target_local_time(target_local_time_str: str) -> pd.Timestamp:
    target_ts = pd.Timestamp(target_local_time_str)
    if target_ts.tzinfo is None:
        target_ts = target_ts.tz_localize("Asia/Shanghai")
    else:
        target_ts = target_ts.tz_convert("Asia/Shanghai")
    return target_ts


def datetime_index_to_plot_num(dt_index: pd.DatetimeIndex) -> np.ndarray:
    dt_naive = dt_index.tz_localize(None)
    return mdates.date2num(dt_naive.to_pydatetime())


def find_nearest_time_index(dt_index: pd.DatetimeIndex, target_local_time_str: str) -> int:
    target_ts = parse_target_local_time(target_local_time_str)
    diff_ns = np.abs(dt_index.asi8 - target_ts.value)
    return int(np.argmin(diff_ns))


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
# Collect single-radial results for the whole day
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


# =========================================================
# File-internal VAD-16 averaging
# =========================================================
def average_one_file_by_vad_cycle(
    x_total: np.ndarray,
    timestamp_utc_s: np.ndarray,
    azimuth_deg: np.ndarray,
    n_azimuth_per_vad: int = N_AZIMUTH_PER_VAD,
):
    """
    Do VAD-16 averaging within a single file only.
    No cross-file grouping.
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
    """
    Build day-long VAD-16 results by averaging within each file, then concatenate.
    """
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
# Target file and target VAD-16 group
# =========================================================
def find_target_file(preprocessed_date_dir: str | Path, target_local_time_str: str):
    """
    Priority:
    1. file whose local time range contains the target time
    2. otherwise, file whose midpoint is closest to the target time
    """
    preprocessed_date_dir = Path(preprocessed_date_dir)
    npz_files = sorted(preprocessed_date_dir.glob("*_preprocessed.npz"))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No *_preprocessed.npz found in {preprocessed_date_dir}")

    target_ts = parse_target_local_time(target_local_time_str)

    file_infos = []
    for npz_path in npz_files:
        d = load_one_preprocessed_npz(npz_path)
        t0 = d["time_local"][0]
        t1 = d["time_local"][-1]
        t_mid = t0 + (t1 - t0) / 2
        file_infos.append((d, t0, t1, t_mid))

    containing = [item for item in file_infos if item[1] <= target_ts <= item[2]]
    if len(containing) > 0:
        containing.sort(key=lambda x: abs(x[3].value - target_ts.value))
        return containing[0][0]

    file_infos.sort(key=lambda x: abs(x[3].value - target_ts.value))
    return file_infos[0][0]


def find_target_vad16_group_in_file(
    file_data: dict,
    target_local_time_str: str,
    n_azimuth_per_vad: int = N_AZIMUTH_PER_VAD,
):
    """
    In the target file, find the complete 16-radial group whose mean time
    is closest to the target local time.
    """
    x_total = np.asarray(file_data["x_total"], dtype=np.float64)
    timestamp = np.asarray(file_data["timestamp"], dtype=np.float64)
    time_local = file_data["time_local"]
    azimuth = np.asarray(file_data["azimuth"], dtype=np.float64)

    sort_idx = np.argsort(timestamp)
    x_total = x_total[sort_idx]
    timestamp = timestamp[sort_idx]
    azimuth = azimuth[sort_idx]
    time_local = time_local[sort_idx]

    n_radial, n_gate = x_total.shape
    n_cycle = n_radial // n_azimuth_per_vad
    n_used = n_cycle * n_azimuth_per_vad

    if n_cycle == 0:
        raise ValueError("The file does not contain a complete 16-radial VAD group")

    x_used = x_total[:n_used]
    t_used = timestamp[:n_used]
    az_used = azimuth[:n_used]
    time_used = time_local[:n_used]

    x_group = x_used.reshape(n_cycle, n_azimuth_per_vad, n_gate)
    t_group = t_used.reshape(n_cycle, n_azimuth_per_vad)
    az_group = az_used.reshape(n_cycle, n_azimuth_per_vad)
    time_group = time_used.to_numpy().reshape(n_cycle, n_azimuth_per_vad)

    t_group_mean = t_group.mean(axis=1)
    t_group_local = utc_seconds_to_local_datetime_index(t_group_mean)

    target_ts = parse_target_local_time(target_local_time_str)
    diff_ns = np.abs(t_group_local.asi8 - target_ts.value)
    group_idx = int(np.argmin(diff_ns))

    return {
        "group_idx": group_idx,
        "x_group_16": x_group[group_idx],
        "timestamp_group_16": t_group[group_idx],
        "time_local_group_16": pd.DatetimeIndex(time_group[group_idx]),
        "azimuth_group_16": az_group[group_idx],
        "x_group_mean": x_group[group_idx].mean(axis=0),
        "timestamp_group_mean": t_group_mean[group_idx],
        "time_local_group_mean": t_group_local[group_idx],
        "n_cycle_in_file": n_cycle,
        "n_used_in_file": n_used,
        "n_discard_in_file": n_radial - n_used,
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
    beta_a_ref: float,
    sa_aer: float = SA_AER,
    sm_mol: float = SM_MOL,
    ref_index: int = 0,
):
    x_profiles = np.asarray(x_profiles, dtype=np.float64)

    n_prof, n_gate = x_profiles.shape
    beta_a_all = np.full((n_prof, n_gate), np.nan, dtype=np.float64)
    alpha_a_all = np.full((n_prof, n_gate), np.nan, dtype=np.float64)

    for i in range(n_prof):
        beta_i, alpha_i = fernald_forward_single(
            x_profile=x_profiles[i],
            range_km=range_km,
            beta_m_profile=beta_m_profile,
            beta_a_ref=beta_a_ref,
            sa_aer=sa_aer,
            sm_mol=sm_mol,
            ref_index=ref_index,
            clip_negative=True,
        )
        beta_a_all[i] = beta_i
        alpha_a_all[i] = alpha_i

    return beta_a_all, alpha_a_all


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

def suggest_gap_threshold_seconds(time_local: pd.DatetimeIndex, min_threshold_seconds: float = 30.0) -> float:
    """
    根据时间序列的中位时间间隔，自动估计“采集间断”阈值。
    """
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
    """
    对时间序列中的长间隔插入 NaN 空列，使未采集时段在图中显示为空白。

    参数
    ----
    time_local : DatetimeIndex, shape (N_time,)
    field_2d   : ndarray, shape (N_time, N_height)
    gap_threshold_seconds : float
        当相邻两时刻间隔超过该阈值时，视为未采集时段

    返回
    ----
    new_time_local : DatetimeIndex
    new_field_2d   : ndarray
    """
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
            # 在长时间间隔两端各插入一个 NaN 时间列，形成空白带
            left_blank_time = t0 + pd.Timedelta(seconds=1)
            right_blank_time = t1 - pd.Timedelta(seconds=1)

            if right_blank_time <= left_blank_time:
                # 如果间隔非常短，退化为在中点插一个 NaN
                mid_blank_time = t0 + (t1 - t0) / 2
                out_times.append(mid_blank_time)
                out_fields.append(nan_profile.copy())
            else:
                out_times.append(left_blank_time)
                out_fields.append(nan_profile.copy())

                out_times.append(right_blank_time)
                out_fields.append(nan_profile.copy())

    # 最后一个时间点
    out_times.append(time_local[-1])
    out_fields.append(field_2d[-1])

    new_time_local = pd.DatetimeIndex(out_times)
    new_field_2d = np.vstack(out_fields)

    return new_time_local, new_field_2d

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

    alpha_2d shape: (N_time, N_height)
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

    # 在无采集时段插入 NaN 空列
    time_plot, alpha_plot = insert_blank_columns_for_gaps(
        time_local=time_local,
        field_2d=alpha_2d,
        gap_threshold_seconds=gap_threshold_seconds,
    )

    t_num = datetime_index_to_plot_num(time_plot)

    # 将 NaN 掩膜，这样空白区域不会被着色
    alpha_masked = np.ma.masked_invalid(alpha_plot.T)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white", alpha=0.0)

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


def plot_target_vad16_profiles(
    height_km: np.ndarray,
    alpha_group_16: np.ndarray,
    alpha_group_mean: np.ndarray,
    time_local_group_16: pd.DatetimeIndex,
    time_local_group_mean,
    out_png: str | Path,
):
    """
    Plot:
    - 16 individual profiles in light colors
    - mean profile in dark thick line
    """
    plt.figure(figsize=(7, 8))

    for i in range(alpha_group_16.shape[0]):
        label = "Single radial profiles" if i == 0 else None
        plt.plot(
            alpha_group_16[i],
            height_km,
            linewidth=1.0,
            alpha=0.35,
            label=label,
        )

    t_mean_str = pd.Timestamp(time_local_group_mean).strftime("%Y-%m-%d %H:%M:%S")
    plt.plot(
        alpha_group_mean,
        height_km,
        linewidth=2.5,
        label=f"Mean of target VAD-16 group  {t_mean_str}",
    )

    plt.xlabel(r"Aerosol extinction coefficient $\alpha_a$ (km$^{-1}$)")
    plt.ylabel("Height (km)")
    plt.title("Aerosol extinction profiles of target 16-radial group")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    # -----------------------------------------------------
    # 1. Date and output directory
    # -----------------------------------------------------
    date_key = get_date_from_folder_name(PREPROCESSED_DATE_DIR)
    out_dir = make_output_dir(OUTPUT_ROOT, date_key)

    # -----------------------------------------------------
    # 2. Daily visibility
    # -----------------------------------------------------
    visibility_m = load_daily_visibility_m(VISIBILITY_CSV_PATH, date_key)
    beta_ng, beta_a_ref, alpha_a_ref, q_val = visibility_to_reference_beta_alpha(
        visibility_m=visibility_m,
        lambda_nm=LAMBDA_NM,
        sa_aer=SA_AER,
        k_beta=K_BETA,
    )

    # -----------------------------------------------------
    # 3. Whole-day single-radial dataset
    # -----------------------------------------------------
    day_single = collect_one_day_single_radials(PREPROCESSED_DATE_DIR)

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

    # -----------------------------------------------------
    # 4. Whole-day file-internal VAD-16 dataset
    # -----------------------------------------------------
    day_vad = collect_one_day_file_internal_vad(PREPROCESSED_DATE_DIR)

    x_total_vad = day_vad["x_vad"]
    time_local_vad = day_vad["time_local_vad"]
    timestamp_utc_s_vad = day_vad["timestamp_utc_s_vad"]

    # -----------------------------------------------------
    # 5. Molecular backscatter
    # -----------------------------------------------------
    beta_m_profile = molecular_backscatter_km(
        height_km=height_km,
        lambda_nm=LAMBDA_NM,
    )

    # -----------------------------------------------------
    # 6. Invert whole-day single-radial curtain
    # -----------------------------------------------------
    beta_a_single, alpha_a_single = invert_profiles(
        x_profiles=x_total_single,
        range_km=range_km,
        beta_m_profile=beta_m_profile,
        beta_a_ref=beta_a_ref,
        sa_aer=SA_AER,
        sm_mol=SM_MOL,
        ref_index=0,
    )

    # -----------------------------------------------------
    # 7. Invert whole-day file-internal VAD-16 curtain
    # -----------------------------------------------------
    beta_a_vad, alpha_a_vad = invert_profiles(
        x_profiles=x_total_vad,
        range_km=range_km,
        beta_m_profile=beta_m_profile,
        beta_a_ref=beta_a_ref,
        sa_aer=SA_AER,
        sm_mol=SM_MOL,
        ref_index=0,
    )

    # -----------------------------------------------------
    # 8. Find target file and target 16-radial group
    # -----------------------------------------------------
    target_file_data = find_target_file(
        preprocessed_date_dir=PREPROCESSED_DATE_DIR,
        target_local_time_str=TARGET_LOCAL_TIME_STR,
    )

    target_group = find_target_vad16_group_in_file(
        file_data=target_file_data,
        target_local_time_str=TARGET_LOCAL_TIME_STR,
        n_azimuth_per_vad=N_AZIMUTH_PER_VAD,
    )

    x_group_16 = target_group["x_group_16"]
    x_group_mean = target_group["x_group_mean"]

    # 16 single profiles
    beta_group_16, alpha_group_16 = invert_profiles(
        x_profiles=x_group_16,
        range_km=range_km,
        beta_m_profile=beta_m_profile,
        beta_a_ref=beta_a_ref,
        sa_aer=SA_AER,
        sm_mol=SM_MOL,
        ref_index=0,
    )

    # mean profile of the same 16-radial group
    beta_group_mean, alpha_group_mean = fernald_forward_single(
        x_profile=x_group_mean,
        range_km=range_km,
        beta_m_profile=beta_m_profile,
        beta_a_ref=beta_a_ref,
        sa_aer=SA_AER,
        sm_mol=SM_MOL,
        ref_index=0,
        clip_negative=True,
    )

    # -----------------------------------------------------
    # 9. Output figures
    # -----------------------------------------------------
    target_group_png = out_dir / f"{date_key}_target_vad16_profiles.png"
    curtain_single_png = out_dir / f"{date_key}_time_height_single.png"
    curtain_vad_png = out_dir / f"{date_key}_time_height_vad16_file_internal.png"

    plot_target_vad16_profiles(
        height_km=height_km,
        alpha_group_16=alpha_group_16,
        alpha_group_mean=alpha_group_mean,
        time_local_group_16=target_group["time_local_group_16"],
        time_local_group_mean=target_group["time_local_group_mean"],
        out_png=target_group_png,
    )

    plot_time_height(
        time_local=time_local_single,
        height_km=height_km,
        alpha_2d=alpha_a_single,
        title=f"Single-radial aerosol extinction curtain ({date_key}, UTC+8)",
        out_png=curtain_single_png,
        gap_threshold_seconds=30.0,  # 单径向时间分辨率高，阈值可设小一些
    )

    plot_time_height(
        time_local=time_local_vad,
        height_km=height_km,
        alpha_2d=alpha_a_vad,
        title=f"File-internal VAD-16 aerosol extinction curtain ({date_key}, UTC+8)",
        out_png=curtain_vad_png,
        gap_threshold_seconds=120.0,  # 文件内16径向平均后的时间分辨率更低
    )

    # -----------------------------------------------------
    # 10. Save results
    # -----------------------------------------------------
    out_npz = out_dir / f"{date_key}_fernald_inversion.npz"
    np.savez_compressed(
        out_npz,
        # basic info
        date_key=np.array(str(date_key)),
        visibility_m=np.array(visibility_m, dtype=np.float64),
        q_visibility=np.array(q_val, dtype=np.float64),
        beta_near_ground=np.array(beta_ng, dtype=np.float64),
        beta_a_ref=np.array(beta_a_ref, dtype=np.float64),
        alpha_a_ref=np.array(alpha_a_ref, dtype=np.float64),

        # axes
        range_m=range_m,
        range_km=range_km,
        height_km=height_km,
        beta_m_profile=beta_m_profile,

        # whole-day single-radial
        timestamp_utc_s_single=timestamp_utc_s_single,
        time_local_single_str=time_local_single.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype="U19"),
        x_total_single=x_total_single,
        beta_a_single=beta_a_single,
        alpha_a_single=alpha_a_single,

        # whole-day file-internal VAD-16
        timestamp_utc_s_vad16=timestamp_utc_s_vad,
        time_local_vad16_str=time_local_vad.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype="U19"),
        x_total_vad16=x_total_vad,
        beta_a_vad16=beta_a_vad,
        alpha_a_vad16=alpha_a_vad,

        # target file / target group
        target_local_time_str=np.array(TARGET_LOCAL_TIME_STR),
        target_file_path=np.array(target_file_data["npz_path"]),
        target_group_index=np.array(target_group["group_idx"], dtype=np.int32),
        target_group_time_local_mean=np.array(
            pd.Timestamp(target_group["time_local_group_mean"]).strftime("%Y-%m-%d %H:%M:%S")
        ),
        target_group_time_local_16=np.array(
            target_group["time_local_group_16"].strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype="U19")
        ),
        x_group_16=x_group_16,
        x_group_mean=x_group_mean,
        beta_group_16=beta_group_16,
        alpha_group_16=alpha_group_16,
        beta_group_mean=beta_group_mean,
        alpha_group_mean=alpha_group_mean,

        # settings
        elev_deg=np.array(ELEV_DEG, dtype=np.float64),
        lambda_nm=np.array(LAMBDA_NM, dtype=np.float64),
        sa_aer=np.array(SA_AER, dtype=np.float64),
        sm_mol=np.array(SM_MOL, dtype=np.float64),
        k_beta=np.array(K_BETA, dtype=np.float64),
        n_azimuth_per_vad=np.array(N_AZIMUTH_PER_VAD, dtype=np.int32),
        n_discard_after_vad_mean=np.array(day_vad["n_discard_after_vad_mean"], dtype=np.int32),
        target_file_n_cycle=np.array(target_group["n_cycle_in_file"], dtype=np.int32),
        target_file_n_discard=np.array(target_group["n_discard_in_file"], dtype=np.int32),
    )

    # -----------------------------------------------------
    # 11. Console output
    # -----------------------------------------------------
    print("=" * 72)
    print("Fernald inversion finished")
    print(f"Date: {date_key}")
    print(f"Daily VISIB_M: {visibility_m:.3f} m")
    print(f"q: {q_val:.6f}")
    print(f"beta_near_ground: {beta_ng:.6e} km^-1 sr^-1")
    print(f"beta_a_ref: {beta_a_ref:.6e} km^-1 sr^-1")
    print(f"alpha_a_ref: {alpha_a_ref:.6e} km^-1")
    print(f"Single-radial profiles: {alpha_a_single.shape[0]}")
    print(f"File-internal VAD-16 profiles: {alpha_a_vad.shape[0]}")
    print(f"Discarded radials after file-internal VAD-16 averaging: {day_vad['n_discard_after_vad_mean']}")
    print(f"Target time: {TARGET_LOCAL_TIME_STR}")
    print(f"Matched file: {target_file_data['npz_path']}")
    print(f"Matched target-group mean time: {target_group['time_local_group_mean']}")
    print(f"Output NPZ: {out_npz}")
    print(f"Target-group figure: {target_group_png}")
    print(f"Single-radial curtain: {curtain_single_png}")
    print(f"File-internal VAD-16 curtain: {curtain_vad_png}")
    print("=" * 72)


if __name__ == "__main__":
    main()