# -*- coding: utf-8 -*-
"""
fernald_from_h5v2_npz.py

基于 h5_to_radial_v2.py 输出的 NPZ 数据，利用 aerosol_flux 中的 Fernald 前向反演方法，
结合 beijing_weather.xlsx 逐小时能见度数据，反演气溶胶消光系数，并进行 SNR 质量控制。

数据流
------
h5_to_radial_v2.py 输出 NPZ
    ├── peak_norm_P / peak_norm_S  →  RCS = peak_norm × r²  →  X_total = p_rcs + s_rcs
    ├── SNR_P / SNR_S              →  SNR QC (门级 + 廓线级)
    ├── time (本地时间字符串)        →  从 beijing_weather.xlsx 插值逐小时能见度
    └── azi_data                   →  VAD-16 分组平均

Fernald 反演输出
    ├── 单径向逐时消光系数幕帘图
    ├── VAD-16 文件内平均逐时消光系数幕帘图
    ├── 可选目标时刻 16 径向组图
    └── 结果 NPZ 文件

注意
----
- 本脚本假设 h5_to_radial_v2.py 中的 time 字段已经是 UTC+8 本地时间字符串。
- 距离门范围：物理第 4~60 门（57 个门），对应约 262.5 m ~ 4462.5 m。
- beijing_weather.xlsx 路径默认为 F:\\3220240787\\DataShareClub\\weather_data\\beijing_weather.xlsx。
"""

from __future__ import annotations

import re
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
# 用户设置 — 请按需修改
# =========================================================
# h5_to_radial_v2.py 输出 NPZ 所在目录（可处理单个 .npz 或整个目录）
INPUT_NPZ_DIR = Path(r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\2026_05_27_radial_wind_260514.npz")

# 逐小时气象数据表 (xlsx)
VISIBILITY_XLSX_PATH = Path(r"F:\3220240787\DataShareClub\weather_data\beijing_weather.xlsx")

# 输出根目录
OUTPUT_ROOT = Path(r"E:\测风组实验数据\气溶胶反演\fernald_from_h5v2")

# 可选：只处理指定的 NPZ 文件 stem（不含 _radial_wind_260514.npz 后缀）
# 设为 None 则自动扫描 INPUT_NPZ_DIR 下所有匹配的 npz 文件
NPZ_STEM_WHITELIST: Optional[list[str]] = None
# 示例：NPZ_STEM_WHITELIST = ["2026_05_27"]

# 可选目标本地时间 (UTC+8)，用于绘制目标 16 径向组图。
# 键为日期字符串 "YYYY-MM-DD"，值为目标时刻 "YYYY-MM-DD HH:MM:SS"。
TARGET_LOCAL_TIME_MAP: dict[str, str] = {
    "2026-05-27": "2026-05-27 21:00:00",
}

# 可选：指定某个时刻，绘制单条径向的气溶胶消光系数廓线。
# 格式同 TARGET_LOCAL_TIME_MAP，系统自动取该时刻最近的一条单径向反演结果。
SINGLE_PROFILE_TIME_MAP: dict[str, str] = {
    "2026-05-27": "2026-05-27 21:00:00",
}

# 幕帘图中非采集时段的空白间隔阈值 (秒)
SINGLE_GAP_THRESHOLD_SECONDS = 30.0
VAD16_GAP_THRESHOLD_SECONDS = 180.0


# =========================================================
# 系统参数
# =========================================================
C = 299792458.0                # 光速, m/s
TAU_PULSE_S = 500e-9           # 脉冲宽度, s
RANGE_RES_M = C * TAU_PULSE_S / 2.0   # ≈ 75 m

LAMBDA_NM = 1550.0             # 激光波长, nm
SA_AER = 29.978                # 气溶胶激光雷达比
SM_MOL = 8.0 * np.pi / 3.0     # 分子激光雷达比
K_BETA = 0.2165                # 地面到最低有效探测门的映射系数

ELEV_DEG = 72.0                # 仰角, 度
LIDAR_ALT_M = 0.0              # 仪器海拔, m
N_AZIMUTH_PER_VAD = 16         # 每圈 VAD 方位数

# h5_to_radial_v2.py 输出的距离门范围 (1-based 物理门号)
KEEP_GATE_START = 4            # 第 4 门, ~262.5 m
KEEP_GATE_END = 60             # 第 60 门, ~4462.5 m

# 绘图范围
VMIN_ALPHA = 0.0
VMAX_ALPHA = None


# =========================================================
# SNR 质量控制参数
# =========================================================
# 门级 SNR 阈值 (dB)：P 和 S 两通道均需大于此值
SNR_THRESHOLD_DB = -25.0

# 廓线级 QC：低层有效门数判定
# 检查距离门前 LOW_GATE_COUNT 个门中，有效门数不少于 MIN_VALID_LOW_GATES
LOW_GATE_COUNT = 20
MIN_VALID_LOW_GATES = 8

# 是否启用廓线级 QC（剔除低层信噪比过差的整条廓线）
ENABLE_PROFILE_QC = True


# =========================================================
# 1. 读取 h5_to_radial_v2.py 输出的 NPZ
# =========================================================
def load_h5v2_npz(npz_path: str | Path) -> dict:
    """
    读取 h5_to_radial_v2.py 输出的 NPZ 文件，构造 Fernald 反演所需的全部中间量。

    返回
    ----
    dict 包含：
        time_local       : pd.DatetimeIndex, UTC+8 本地时间
        timestamp_utc_s  : np.ndarray, UTC 秒
        azimuth_deg      : np.ndarray, 方位角 (度)
        range_m          : np.ndarray, 距离 (m), shape (57,)
        p_rcs            : np.ndarray, P 通道 RCS, shape (N, 57)
        s_rcs            : np.ndarray, S 通道 RCS, shape (N, 57)
        x_total          : np.ndarray, X_total = p_rcs + s_rcs, shape (N, 57)
        SNR_P            : np.ndarray, P 通道 SNR (dB), shape (N, 57)
        SNR_S            : np.ndarray, S 通道 SNR (dB), shape (N, 57)
        gate_valid       : np.ndarray, SNR QC 后的门有效掩膜 (bool), shape (N, 57)
        profile_valid    : np.ndarray, 廓线级 QC 有效掩膜 (bool), shape (N,)
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=False)

    # --- 基础字段 ---
    time_str = np.asarray(data["time"])          # shape (N,), dtype <U19
    azi = np.asarray(data["azi_data"], dtype=np.float64)  # shape (N,)
    peak_norm_P = np.asarray(data["peak_norm_P"], dtype=np.float64)  # (N, 57)
    peak_norm_S = np.asarray(data["peak_norm_S"], dtype=np.float64)  # (N, 57)
    SNR_P_raw = np.asarray(data["SNR_P"], dtype=np.float64)          # (N, 57)
    SNR_S_raw = np.asarray(data["SNR_S"], dtype=np.float64)          # (N, 57)

    n_radial, n_gates = peak_norm_P.shape  # n_gates = 57

    # --- 距离轴 ---
    kept_gate_numbers = np.arange(KEEP_GATE_START, KEEP_GATE_START + n_gates, dtype=np.float64)  # 4 ~ 60
    range_m = (kept_gate_numbers - 0.5) * RANGE_RES_M  # shape (57,)

    # --- 构造 RCS ---
    r2 = range_m[None, :] ** 2
    p_rcs = peak_norm_P * r2
    s_rcs = peak_norm_S * r2

    # p_rcs 和 s_rcs 中 NaN 位置在求和时视为 0
    p_rcs_filled = np.where(np.isfinite(p_rcs), p_rcs, 0.0)
    s_rcs_filled = np.where(np.isfinite(s_rcs), s_rcs, 0.0)
    x_total = np.maximum(p_rcs_filled + s_rcs_filled, 1e-12)

    # --- 时间解析 ---
    # time 字段为 "YYYY-MM-DD HH:MM:SS" 格式的 UTC+8 本地时间
    time_local = pd.to_datetime(time_str).tz_localize("Asia/Shanghai")
    time_utc = time_local.tz_convert("UTC")
    timestamp_utc_s = time_utc.astype("int64").to_numpy(dtype=np.float64) / 1e9

    # --- SNR QC ---
    gate_valid, profile_valid = apply_snr_qc(SNR_P_raw, SNR_S_raw)

    return {
        "time_local": time_local,
        "timestamp_utc_s": timestamp_utc_s,
        "azimuth_deg": azi,
        "range_m": range_m,
        "p_rcs": p_rcs,
        "s_rcs": s_rcs,
        "x_total": x_total,
        "SNR_P": SNR_P_raw,
        "SNR_S": SNR_S_raw,
        "gate_valid": gate_valid,
        "profile_valid": profile_valid,
        "n_gates": n_gates,
        "source_npz": str(npz_path),
    }


# =========================================================
# 2. SNR 质量控制
# =========================================================
def apply_snr_qc(
    SNR_P: np.ndarray,
    SNR_S: np.ndarray,
    snr_threshold_db: float = SNR_THRESHOLD_DB,
    low_gate_count: int = LOW_GATE_COUNT,
    min_valid_low_gates: int = MIN_VALID_LOW_GATES,
    enable_profile_qc: bool = ENABLE_PROFILE_QC,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对 P/S 两通道 SNR 进行质量控制。

    门级 QC：
        P 和 S 通道 SNR 均 > snr_threshold_db，且均为有限值。

    廓线级 QC：
        对于每条径向，检查前 low_gate_count 个门中，
        通过门级 QC 的门数不少于 min_valid_low_gates。

    返回
    ----
    gate_valid    : (N, G) bool 数组，门级有效掩膜
    profile_valid : (N,)   bool 数组，廓线级有效掩膜
    """
    SNR_P = np.asarray(SNR_P, dtype=np.float64)
    SNR_S = np.asarray(SNR_S, dtype=np.float64)

    # 门级：两通道 SNR 均有效且超过阈值
    gate_valid = (
        np.isfinite(SNR_P)
        & np.isfinite(SNR_S)
        & (SNR_P > snr_threshold_db)
        & (SNR_S > snr_threshold_db)
    )

    # 廓线级
    if enable_profile_qc:
        n_gates = gate_valid.shape[1]
        check_count = min(low_gate_count, n_gates)
        low_valid_count = np.sum(gate_valid[:, :check_count], axis=1)
        profile_valid = low_valid_count >= min_valid_low_gates
    else:
        profile_valid = np.ones(gate_valid.shape[0], dtype=bool)

    return gate_valid, profile_valid


# =========================================================
# 3. 几何关系与分子后向散射
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

    Units: z in km, lambda in nm, beta_m in km^-1 sr^-1
    """
    z_km = np.asarray(height_km, dtype=np.float64)
    beta_m = 1.54e-3 * (532.0 / lambda_nm) ** 4 * np.exp(-z_km / 7.0)
    return beta_m


# =========================================================
# 4. beijing_weather.xlsx 逐小时能见度读取
# =========================================================
def _parse_hourly_local_time_with_year(value, year: int) -> pd.Timestamp:
    """解析 'MM-DD HH:00' 格式的本地时间字符串。"""
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
    读取 beijing_weather.xlsx 逐小时气象表。

    表格布局 (0-based 列索引):
    - col 0  : 本地时间字符串 MM-DD HH:00
    - col 9  : 能见度, km
    - col 13 : AQI
    - col 14 : PM2.5 index
    - col 15 : PM10 index
    """
    xlsx_path = Path(xlsx_path)
    raw = pd.read_excel(xlsx_path, header=None)

    if raw.shape[0] < 3:
        raise ValueError("xlsx 表数据行不足")

    data = raw.iloc[2:].copy()
    required_max_col = 15
    if data.shape[1] <= required_max_col:
        raise ValueError(f"xlsx 表至少需要 {required_max_col + 1} 列")

    df = pd.DataFrame({
        "time_local": data.iloc[:, 0].apply(
            lambda x: _parse_hourly_local_time_with_year(x, year)
        ),
        "visibility_km": pd.to_numeric(data.iloc[:, 9], errors="coerce"),
        "AQI": pd.to_numeric(data.iloc[:, 13], errors="coerce"),
        "PM25_index": pd.to_numeric(data.iloc[:, 14], errors="coerce"),
        "PM10_index": pd.to_numeric(data.iloc[:, 15], errors="coerce"),
    })

    df = df.dropna(subset=["time_local"]).sort_values("time_local")
    df = df.drop_duplicates(subset=["time_local"])
    df = df.set_index("time_local")
    return df


def build_hourly_weather_for_date(
    xlsx_path: str | Path,
    target_date,
) -> pd.DataFrame:
    """
    为指定日期构建完整的 00:00~23:00 逐小时能见度表。
    缺失值在时间维线性插值。
    """
    target_date = pd.Timestamp(target_date).date()
    year = target_date.year

    df_year = load_hourly_weather_table_for_year(xlsx_path, year)

    day_start = pd.Timestamp(target_date).tz_localize("Asia/Shanghai")
    hourly_index = pd.date_range(
        start=day_start, periods=24, freq="1h", tz="Asia/Shanghai"
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
        raise ValueError(f"日期 {target_date} 无有效逐小时能见度数据")

    return df_day


def interpolate_hourly_visibility_to_times(
    hourly_visibility_km: pd.Series,
    target_times_local: pd.DatetimeIndex,
) -> np.ndarray:
    """
    将逐小时能见度插值到雷达各廓线时刻 (UTC+8)。
    """
    hourly_visibility_km = hourly_visibility_km.copy().sort_index()
    target_times_local = pd.DatetimeIndex(target_times_local).sort_values()

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
    能见度 (km) → β_near_ground, β_a_ref, α_a_ref, q
    """
    u_km = np.asarray(visibility_km, dtype=np.float64)
    if np.any(u_km <= 0):
        raise ValueError("能见度必须为正")

    q = q_from_visibility_km(u_km)
    beta_near_ground = 3.91 / (sa_aer * u_km) * np.power((lambda_nm / 550.0), -q)
    beta_a_ref = k_beta * beta_near_ground
    alpha_a_ref = sa_aer * beta_a_ref
    return beta_near_ground, beta_a_ref, alpha_a_ref, q


# =========================================================
# 5. Fernald 前向反演
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    单条径向的 Fernald 前向反演。

    Units: range_km in km, beta in km^-1 sr^-1, alpha in km^-1
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    批量 Fernald 反演。
    beta_a_ref_values 可为标量或长度为 N 的数组。
    """
    x_profiles = np.asarray(x_profiles, dtype=np.float64)
    n_prof, n_gate = x_profiles.shape

    beta_a_ref_arr = np.asarray(beta_a_ref_values, dtype=np.float64)
    if beta_a_ref_arr.ndim == 0:
        beta_a_ref_arr = np.full((n_prof,), float(beta_a_ref_arr), dtype=np.float64)
    elif beta_a_ref_arr.shape[0] != n_prof:
        raise ValueError("beta_a_ref_values 长度须与廓线数一致")

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
# 6. VAD-16 文件内平均
# =========================================================
def average_one_file_by_vad_cycle(
    x_total: np.ndarray,
    timestamp_utc_s: np.ndarray,
    azimuth_deg: np.ndarray,
    n_azimuth_per_vad: int = N_AZIMUTH_PER_VAD,
) -> dict:
    """
    对单个数据块（一个 NPZ 文件）内按时序排列的径向做 VAD-16 平均。
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


# =========================================================
# 7. 日期/时间工具
# =========================================================
def extract_date_from_time_local(time_local: pd.DatetimeIndex):
    """从本地时间索引提取日期，返回第一个有效日期。"""
    valid_times = time_local[~pd.isna(time_local)]
    if len(valid_times) == 0:
        raise ValueError("时间索引全为空")
    return valid_times[0].date()


def utc_seconds_to_local_datetime_index(utc_seconds: np.ndarray) -> pd.DatetimeIndex:
    utc_seconds = np.asarray(utc_seconds, dtype=np.float64)
    dt_utc = pd.to_datetime(utc_seconds, unit="s", utc=True)
    dt_local = dt_utc.tz_convert("Asia/Shanghai")
    return dt_local


def normalize_local_timestamp(value) -> pd.Timestamp:
    """
    Return an Asia/Shanghai timestamp from common timestamp representations.

    Numeric epoch values are treated as UTC and their unit is inferred from
    magnitude. This avoids pd.Timestamp interpreting epoch microseconds as
    nanoseconds, which produces dates near 1970 in plot legends.
    """
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, np.datetime64):
        ts = pd.Timestamp(value)
    elif np.isscalar(value) and np.issubdtype(np.asarray(value).dtype, np.number):
        numeric_value = float(value)
        abs_value = abs(numeric_value)
        if abs_value >= 1e17:
            unit = "ns"
        elif abs_value >= 1e14:
            unit = "us"
        elif abs_value >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        ts = pd.to_datetime(numeric_value, unit=unit, utc=True)
    else:
        ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("Asia/Shanghai")
    return ts.tz_convert("Asia/Shanghai")


def datetime_index_to_plot_num(dt_index: pd.DatetimeIndex) -> np.ndarray:
    dt_naive = dt_index.tz_localize(None)
    return mdates.date2num(dt_naive.to_pydatetime())


# =========================================================
# 8. 幕帘图空白间隔处理
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
# 9. 绘图
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
    time_local = pd.DatetimeIndex(time_local)
    alpha_2d = np.asarray(alpha_2d, dtype=np.float64)

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
        t_num, height_km, alpha_masked,
        shading="auto", cmap=cmap, vmin=vmin, vmax=vmax,
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
    plt.figure(figsize=(7, 8))

    for i in range(alpha_group_16.shape[0]):
        label = "Single radial profiles" if i == 0 else None
        plt.plot(
            alpha_group_16[i], height_km,
            linewidth=1.0, alpha=0.35, label=label,
        )

    t_mean = normalize_local_timestamp(time_local_group_mean)
    t_mean_str = t_mean.strftime("%Y-%m-%d %H:%M:%S")
    plt.plot(
        alpha_group_mean, height_km,
        linewidth=2.5, label=f"Mean of target VAD-16 group  {t_mean_str}",
    )

    plt.xlabel(r"Aerosol extinction coefficient $\alpha_a$ (km$^{-1}$)")
    plt.ylabel("Height (km)")
    plt.title("Aerosol extinction profiles of target 16-radial group")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def mean_alpha_profiles(alpha_group_16: np.ndarray) -> np.ndarray:
    """
    Average already-inverted single-radial extinction profiles gate by gate.

    This is the quantity shown as the mean in the target VAD-16 profile plot.
    It is intentionally different from running Fernald once on an averaged
    signal profile, because Fernald inversion is nonlinear.
    """
    alpha_group_16 = np.asarray(alpha_group_16, dtype=np.float64)
    return np.nanmean(alpha_group_16, axis=0)


def plot_single_extinction_profile(
    height_km: np.ndarray,
    alpha_profile: np.ndarray,
    time_local: pd.Timestamp,
    azimuth_deg: float,
    out_png: str | Path,
):
    """
    绘制单条径向的气溶胶消光系数廓线。
    """
    t_str = pd.Timestamp(time_local).strftime("%Y-%m-%d %H:%M:%S")

    plt.figure(figsize=(7, 8))
    plt.plot(alpha_profile, height_km, linewidth=2.0, color="#1f77b4")
    plt.xlabel(r"Aerosol extinction coefficient $\alpha_a$ (km$^{-1}$)")
    plt.ylabel("Height (km)")
    plt.title(f"Single radial aerosol extinction profile\n{t_str}  Azimuth={azimuth_deg:.1f}°")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 10. 单 NPZ 文件处理主流程
# =========================================================
def process_one_npz(npz_path: str | Path, output_root: str | Path):
    """
    处理 h5_to_radial_v2.py 输出的单个 NPZ 文件，完成：
    1. 读取并构造 RCS
    2. 读取 beijing_weather.xlsx 逐小时能见度
    3. Fernald 前向反演（单径向 + VAD-16）
    4. 绘制幕帘图与目标组图
    5. 保存结果 NPZ
    """
    npz_path = Path(npz_path)
    output_root = Path(output_root)

    print("=" * 78)
    print(f"处理 NPZ: {npz_path.name}")

    # --- 1. 加载数据 ---
    npz_data = load_h5v2_npz(npz_path)

    time_local = npz_data["time_local"]
    range_m = npz_data["range_m"]
    x_total = npz_data["x_total"]
    gate_valid = npz_data["gate_valid"]
    profile_valid = npz_data["profile_valid"]
    SNR_P = npz_data["SNR_P"]
    SNR_S = npz_data["SNR_S"]

    range_km = range_m / 1000.0
    height_km = slant_range_to_height_km(
        range_m=range_m, elev_deg=ELEV_DEG, lidar_alt_m=LIDAR_ALT_M
    )

    # 确定日期
    date_key = extract_date_from_time_local(time_local)
    date_str = str(date_key)
    out_dir = make_output_dir(output_root, date_str)

    print(f"日期: {date_str}")
    print(f"径向数: {x_total.shape[0]}, 距离门数: {x_total.shape[1]}")
    print(f"廓线级有效径向数: {np.sum(profile_valid)} / {len(profile_valid)}")

    # --- 2. 逐小时能见度 ---
    hourly_weather = build_hourly_weather_for_date(VISIBILITY_XLSX_PATH, date_key)
    hourly_visibility_km = hourly_weather["visibility_km"]

    # --- 3. 分子后向散射 ---
    beta_m_profile = molecular_backscatter_km(
        height_km=height_km, lambda_nm=LAMBDA_NM
    )

    # --- 4. 构造 QC 后的 X_total ---
    # 单径向：无效门置 NaN（反演时自动跳过）
    x_total_qc = x_total.copy()
    x_total_qc[~gate_valid] = np.nan

    # 插值逐时能见度到各廓线
    vis_single_km = interpolate_hourly_visibility_to_times(
        hourly_visibility_km, time_local
    )
    beta_ng_single, beta_a_ref_single, alpha_a_ref_single, q_single = \
        visibility_to_reference_beta_alpha_from_km(vis_single_km)

    # --- 5. 单径向 Fernald 反演 ---
    print("正在进行单径向 Fernald 反演 ...")
    beta_a_single, alpha_a_single = invert_profiles(
        x_profiles=x_total_qc,
        range_km=range_km,
        beta_m_profile=beta_m_profile,
        beta_a_ref_values=beta_a_ref_single,
        sa_aer=SA_AER,
        sm_mol=SM_MOL,
        ref_index=0,
    )

    # --- 5b. 可选单时刻廓线图 ---
    single_profile_time_str = SINGLE_PROFILE_TIME_MAP.get(date_str, None)
    single_profile_png = None
    if single_profile_time_str is not None:
        print(f"绘制单时刻廓线: {single_profile_time_str}")
        target_ts = pd.Timestamp(single_profile_time_str).tz_localize("Asia/Shanghai")
        diff_ns = np.abs(time_local.asi8 - target_ts.value)
        best_idx = int(np.argmin(diff_ns))

        alpha_best = alpha_a_single[best_idx]
        az_best = npz_data["azimuth_deg"][best_idx]
        time_best = time_local[best_idx]

        single_profile_png = out_dir / f"{date_str}_single_profile_{pd.Timestamp(time_best).strftime('%H%M%S')}.png"
        plot_single_extinction_profile(
            height_km=height_km,
            alpha_profile=alpha_best,
            time_local=time_best,
            azimuth_deg=float(az_best),
            out_png=single_profile_png,
        )

    # --- 6. VAD-16 文件内平均 ---
    print("正在进行 VAD-16 文件内平均 ...")
    vad_result = average_one_file_by_vad_cycle(
        x_total=x_total_qc,
        timestamp_utc_s=npz_data["timestamp_utc_s"],
        azimuth_deg=npz_data["azimuth_deg"],
        n_azimuth_per_vad=N_AZIMUTH_PER_VAD,
    )

    x_total_vad = vad_result["x_vad"]
    timestamp_utc_s_vad = vad_result["timestamp_utc_s_vad"]
    time_local_vad = utc_seconds_to_local_datetime_index(timestamp_utc_s_vad)

    if x_total_vad.shape[0] > 0:
        vis_vad_km = interpolate_hourly_visibility_to_times(
            hourly_visibility_km, time_local_vad
        )
        beta_ng_vad, beta_a_ref_vad, alpha_a_ref_vad, q_vad = \
            visibility_to_reference_beta_alpha_from_km(vis_vad_km)

        beta_a_vad, alpha_a_vad = invert_profiles(
            x_profiles=x_total_vad,
            range_km=range_km,
            beta_m_profile=beta_m_profile,
            beta_a_ref_values=beta_a_ref_vad,
            sa_aer=SA_AER,
            sm_mol=SM_MOL,
            ref_index=0,
        )
    else:
        beta_a_vad = np.empty((0, x_total.shape[1]), dtype=np.float64)
        alpha_a_vad = np.empty((0, x_total.shape[1]), dtype=np.float64)
        vis_vad_km = np.array([])
        beta_ng_vad = np.array([])
        beta_a_ref_vad = np.array([])
        alpha_a_ref_vad = np.array([])
        q_vad = np.array([])
        print("[警告] 无完整 VAD-16 周期，跳过 VAD-16 反演。")

    # --- 7. 幕帘图 ---
    curtain_single_png = out_dir / f"{date_str}_time_height_single_h5v2.png"
    plot_time_height(
        time_local=time_local,
        height_km=height_km,
        alpha_2d=alpha_a_single,
        title=f"Single-radial aerosol extinction curtain ({date_str}, UTC+8)",
        out_png=curtain_single_png,
        gap_threshold_seconds=SINGLE_GAP_THRESHOLD_SECONDS,
    )

    if alpha_a_vad.shape[0] > 0:
        curtain_vad_png = out_dir / f"{date_str}_time_height_vad16_file_internal_h5v2.png"
        plot_time_height(
            time_local=time_local_vad,
            height_km=height_km,
            alpha_2d=alpha_a_vad,
            title=f"File-internal VAD-16 aerosol extinction curtain ({date_str}, UTC+8)",
            out_png=curtain_vad_png,
            gap_threshold_seconds=VAD16_GAP_THRESHOLD_SECONDS,
        )

    # --- 8. 可选目标 16 径向组图 ---
    target_local_time_str = TARGET_LOCAL_TIME_MAP.get(date_str, None)
    target_group_png = None

    if target_local_time_str is not None:
        print(f"查找目标时刻: {target_local_time_str}")
        target_ts = pd.Timestamp(target_local_time_str).tz_localize("Asia/Shanghai")

        # 在单径向中找到最接近目标时刻的完整 VAD-16 组
        sort_idx = np.argsort(npz_data["timestamp_utc_s"])
        x_sorted = x_total_qc[sort_idx]
        t_sorted = npz_data["timestamp_utc_s"][sort_idx]
        az_sorted = npz_data["azimuth_deg"][sort_idx]
        time_sorted = time_local[sort_idx]

        n_radial, n_gate = x_sorted.shape
        n_cycle = n_radial // N_AZIMUTH_PER_VAD
        n_used = n_cycle * N_AZIMUTH_PER_VAD

        if n_cycle > 0:
            x_used = x_sorted[:n_used]
            t_used = t_sorted[:n_used]
            time_used = time_sorted[:n_used]

            x_group = x_used.reshape(n_cycle, N_AZIMUTH_PER_VAD, n_gate)
            t_group = t_used.reshape(n_cycle, N_AZIMUTH_PER_VAD)
            time_group = time_used.to_numpy().reshape(n_cycle, N_AZIMUTH_PER_VAD)

            t_group_mean = t_group.mean(axis=1)
            t_group_local = utc_seconds_to_local_datetime_index(t_group_mean)

            diff_ns = np.abs(t_group_local.asi8 - target_ts.value)
            group_idx = int(np.argmin(diff_ns))

            x_group_16 = x_group[group_idx]
            time_group_16 = pd.DatetimeIndex(time_group[group_idx])
            time_group_mean = t_group_local[group_idx]

            # 逐时能见度插值
            vis_group_16_km = interpolate_hourly_visibility_to_times(
                hourly_visibility_km, time_group_16
            )
            _, beta_a_ref_group_16, _, _ = \
                visibility_to_reference_beta_alpha_from_km(vis_group_16_km)

            beta_group_16, alpha_group_16 = invert_profiles(
                x_profiles=x_group_16,
                range_km=range_km,
                beta_m_profile=beta_m_profile,
                beta_a_ref_values=beta_a_ref_group_16,
            )

            alpha_group_mean = mean_alpha_profiles(alpha_group_16)

            target_group_png = out_dir / f"{date_str}_target_vad16_profiles_h5v2.png"
            plot_target_vad16_profiles(
                height_km=height_km,
                alpha_group_16=alpha_group_16,
                alpha_group_mean=alpha_group_mean,
                time_local_group_16=time_group_16,
                time_local_group_mean=time_group_mean,
                out_png=target_group_png,
            )
        else:
            print("[警告] 无完整 VAD-16 组可用于目标时刻反演。")

    # --- 9. 保存结果 NPZ ---
    out_npz = out_dir / f"{date_str}_fernald_inversion_h5v2.npz"

    save_dict = {
        # 基本信息
        "date_key": np.array(date_str),
        "source_npz": np.array(str(npz_path)),
        "hourly_visibility_local_time_str": hourly_weather.index.strftime(
            "%Y-%m-%d %H:%M:%S"
        ).to_numpy(dtype="U19"),
        "hourly_visibility_km": hourly_weather["visibility_km"].to_numpy(dtype=np.float64),
        "hourly_AQI": hourly_weather["AQI"].to_numpy(dtype=np.float64),
        "hourly_PM25_index": hourly_weather["PM25_index"].to_numpy(dtype=np.float64),
        "hourly_PM10_index": hourly_weather["PM10_index"].to_numpy(dtype=np.float64),

        # 坐标轴
        "range_m": range_m,
        "range_km": range_km,
        "height_km": height_km,
        "beta_m_profile": beta_m_profile,

        # 单径向
        "timestamp_utc_s_single": npz_data["timestamp_utc_s"],
        "time_local_single_str": time_local.strftime(
            "%Y-%m-%d %H:%M:%S"
        ).to_numpy(dtype="U19"),
        "azimuth_deg_single": npz_data["azimuth_deg"],
        "visibility_single_km": vis_single_km,
        "beta_near_ground_single": beta_ng_single,
        "beta_a_ref_single": beta_a_ref_single,
        "alpha_a_ref_single": alpha_a_ref_single,
        "q_single": q_single,
        "x_total_single": x_total_qc,
        "beta_a_single": beta_a_single,
        "alpha_a_single": alpha_a_single,

        # SNR QC 信息
        "SNR_P": SNR_P,
        "SNR_S": SNR_S,
        "gate_valid": gate_valid,
        "profile_valid": profile_valid,

        # VAD-16
        "timestamp_utc_s_vad16": timestamp_utc_s_vad,
        "time_local_vad16_str": (
            time_local_vad.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype="U19")
            if len(time_local_vad) > 0
            else np.array([], dtype="U19")
        ),
        "visibility_vad16_km": vis_vad_km,
        "beta_near_ground_vad16": beta_ng_vad,
        "beta_a_ref_vad16": beta_a_ref_vad,
        "alpha_a_ref_vad16": alpha_a_ref_vad,
        "q_vad16": q_vad,
        "x_total_vad16": x_total_vad,
        "beta_a_vad16": beta_a_vad,
        "alpha_a_vad16": alpha_a_vad,

        # 参数
        "elev_deg": np.array(ELEV_DEG, dtype=np.float64),
        "lambda_nm": np.array(LAMBDA_NM, dtype=np.float64),
        "sa_aer": np.array(SA_AER, dtype=np.float64),
        "sm_mol": np.array(SM_MOL, dtype=np.float64),
        "k_beta": np.array(K_BETA, dtype=np.float64),
        "n_azimuth_per_vad": np.array(N_AZIMUTH_PER_VAD, dtype=np.int32),
        "n_discard_after_vad_mean": np.array(vad_result["n_discard"], dtype=np.int32),
        "snr_threshold_db": np.array(SNR_THRESHOLD_DB, dtype=np.float64),
        "low_gate_count": np.array(LOW_GATE_COUNT, dtype=np.int32),
        "min_valid_low_gates": np.array(MIN_VALID_LOW_GATES, dtype=np.int32),
    }

    np.savez_compressed(out_npz, **save_dict)

    # --- 10. 控制台输出 ---
    print("-" * 78)
    print("Fernald 反演完成 (h5_to_radial_v2.py 数据源)")
    print(f"日期: {date_str}")
    print(f"输入 NPZ: {npz_path}")
    print(f"单径向廓线数: {alpha_a_single.shape[0]}")
    print(f"VAD-16 廓线数: {alpha_a_vad.shape[0]}")
    print(f"VAD-16 分组舍弃径向数: {vad_result['n_discard']}")
    print(f"单径向幕帘图: {curtain_single_png}")
    if alpha_a_vad.shape[0] > 0:
        print(f"VAD-16 幕帘图: {curtain_vad_png}")
    if single_profile_png is not None:
        print(f"单时刻廓线图: {single_profile_png}")
    if target_group_png is not None:
        print(f"目标组图: {target_group_png}")
    print(f"输出 NPZ: {out_npz}")
    print("=" * 78)

    return out_npz


# =========================================================
# 11. 批量入口
# =========================================================
def _extract_date_from_npz_name(npz_path: Path) -> Optional[str]:
    """
    从 NPZ 文件名中提取日期字符串，如 '2026_05_27'。
    命名规则: {stem}_radial_wind_260514.npz
    """
    name = npz_path.stem  # e.g., "2026_05_27_radial_wind_260514"
    # 尝试匹配 YYYY_MM_DD 开头的文件名
    m = re.match(r"(\d{4}_\d{2}_\d{2})", name)
    if m:
        return m.group(1)
    return None


def collect_npz_jobs(input_dir: str | Path, whitelist: Optional[list[str]] = None):
    """
    扫描目录下 h5_to_radial_v2.py 输出的 NPZ 文件。
    返回 [(date_str, npz_path), ...]。
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_dir}")

    if input_dir.is_file() and input_dir.suffix.lower() == ".npz":
        date_str = _extract_date_from_npz_name(input_dir)
        return [(date_str or input_dir.stem, input_dir)]

    npz_files = sorted(input_dir.glob("*_radial_wind_260514.npz"))
    if not npz_files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到 *_radial_wind_260514.npz 文件")

    jobs = []
    for npz_path in npz_files:
        date_str = _extract_date_from_npz_name(npz_path)
        stem = date_str or npz_path.stem
        if whitelist is not None and stem not in whitelist:
            continue
        jobs.append((stem, npz_path))

    return jobs


def main():
    jobs = collect_npz_jobs(INPUT_NPZ_DIR, NPZ_STEM_WHITELIST)

    if len(jobs) == 0:
        print("未找到可处理的 NPZ 文件。")
        return

    print(f"共发现 {len(jobs)} 个 NPZ 文件待处理。")

    failed = []
    for stem, npz_path in jobs:
        try:
            process_one_npz(npz_path, OUTPUT_ROOT)
        except Exception as exc:
            failed.append((str(npz_path), str(exc)))
            print(f"[FAILED] {npz_path.name}: {exc}")
            import traceback
            traceback.print_exc()

    print("-" * 78)
    print(f"总计: {len(jobs)}, 成功: {len(jobs) - len(failed)}, 失败: {len(failed)}")
    if failed:
        for item in failed:
            print(f"  {item[0]}: {item[1]}")
    print("-" * 78)


if __name__ == "__main__":
    main()
