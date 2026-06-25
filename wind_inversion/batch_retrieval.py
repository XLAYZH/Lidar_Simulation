# -*- coding: utf-8 -*-
"""
batch_retrieval.py — 矢量风批量反演调度器

功能
----
读取 h5_to_radial_v2.py 输出的径向风速 NPZ 文件，对 P/S 双通道分别进行
矢量风反演，输出包含水平风速、垂直风速、风向、U/V 分量的 Excel 文件。

数据处理流程
------------
1. 读取 NPZ   → 加载径向风速 (radial_v_P/S) + SNR (SNR_P/S) + 方位角 + 时间
2. 方位角校正 → 四舍五入 0.5°、360°→0°、减去系统偏置 105° 后映射 0-360°
3. SNR QC     → ① 逐点阈值剔除（SNR < -25 dB）
                ② 整条径向 SNR 突降检测（与局部中位数差 > 6 dB）
4. 时间切分   → 按相邻径向时间间隔 > 10 s 切分为连续数据片段
5. 矢量反演   → 逐片段调用 vector_wind_solver.SWF()，P/S 独立反演
6. 输出 Excel → 每片段一个 xlsx，含 P/S 通道的风速/风向/U/V 分量

可配置参数
----------
RADIAL_NPZ_INPUT    — 输入 NPZ 文件路径（单个文件或文件夹）
OUTPUT_DIR          — 输出目录
CHANNEL_METHODS     — P/S 通道反演算法选择（SVD / AIR / cooks / FSWF）
USE_SNR_QC          — 是否启用 SNR 质量控制
SNR_POINT_THRESHOLD_DB       — 单点 SNR 剔除阈值 (dB)
USE_RADIAL_SNR_JUMP_QC       — 是否启用整条径向异常剔除
RADIAL_DROP_THRESHOLD_DB     — 径向突降阈值 (dB)
NUM_RADIALS          — 每次合成所需径向数（默认 16）
ELEVATION_ANGLE      — 激光雷达俯仰角（默认 72°）
AZIMUTH_OFFSET_DEG   — 系统方位偏置（默认 105°）

输出文件命名
------------
wind_results_{日期}_{片段序号}_P-{算法}_S-{算法}.xlsx
   - P Wind Speed / P Vertical Speed / P Wind Direction / P U  / P V
   - S Wind Speed / S Vertical Speed / S Wind Direction / S U  / S V

使用示例
--------
1. 修改上面的 RADIAL_NPZ_INPUT 指向目标 NPZ 文件或文件夹
2. 修改 OUTPUT_DIR 指定输出目录
3. 直接运行：python batch_retrieval.py
"""
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from vector_wind_solver import SWF

# ============================= 常改动参数区 =============================
# 输入路径列表：可以是单个文件、单个文件夹，或多个文件夹的列表。
# 程序会收集所有路径下的 *_radial_wind_260514.npz 文件统一处理。
RADIAL_NPZ_INPUT = [
    r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\year_2024",
    r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\year_2025",
    r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\year_2026",
]

# 输出根目录。程序会自动按年份创建子目录（如 year_2024/、year_2025/、year_2026/）。
OUTPUT_DIR = r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\wind_vector_results\FSWF"

# P、S 通道分别采用的矢量风反演算法。只改这里即可完成通道算法切换。
# 可选：'SVD'、'AIR'、'cooks'、'FSWF'。其中 'DSWF' 仅在一维分支保留。
CHANNEL_METHODS = {
    'P': 'FSWF',   # P 通道默认沿用原代码中的 FSWF
    'S': 'FSWF',    # S 通道默认沿用原代码中的 SVD
}

# =========================
# SNR质量控制参数
# =========================
USE_SNR_QC = True

# 逐点SNR阈值，低于该值的“单个高度门径向风速”置为NaN
SNR_POINT_THRESHOLD_DB = -25

# 是否启用整条径向异常剔除
USE_RADIAL_SNR_JUMP_QC = True

# 用哪些高度门判断整条径向是否异常
# 例如前5个有效高度门；避免远距离无信号高度门影响判断
QC_GATE_START = 0
QC_GATE_END = 5

# 与局部中位数相比，低于多少dB认为该径向异常
RADIAL_DROP_THRESHOLD_DB = 6

# 局部比较窗口，前后各2条径向
LOCAL_WINDOW = 2

# 扫描和几何参数。保持原代码默认值。
NUM_RADIALS = 16
ELEVATION_ANGLE = 72
AZIMUTH_OFFSET_DEG = 105
TIME_GAP_SECONDS = 10
AZIMUTH_TOLERANCE_DEG = 2.0
MIN_VALID_AZI = 13

# 并行处理 npz 文件数量。单文件处理时该值不起作用。
N_JOBS = 30
# =======================================================================


def collect_npz_files(input_paths):
    """
    收集径向风速 npz 文件。
    input_paths 可以是：
      - 单个字符串（文件或文件夹路径）
      - 列表/元组（多个文件或文件夹路径）
    """
    if isinstance(input_paths, (str, Path)):
        input_paths = [input_paths]

    all_npz_files = []
    for input_path in input_paths:
        path = Path(input_path)
        if not path.is_absolute():
            raise ValueError(f"RADIAL_NPZ_INPUT 必须是绝对路径：{input_path}")
        if not path.exists():
            raise FileNotFoundError(f"路径不存在：{input_path}")

        if path.is_file():
            if path.suffix.lower() != '.npz':
                raise ValueError(f"输入文件必须是 .npz 文件：{path}")
            all_npz_files.append(path)
        else:
            npz_files = sorted(path.glob('*_radial_wind_260514.npz'))
            if not npz_files:
                print(f"警告：未在路径中找到 *_radial_wind_260514.npz：{path}")
            all_npz_files.extend(npz_files)

    if not all_npz_files:
        raise FileNotFoundError(f"所有输入路径中均未找到 *_radial_wind_260514.npz 文件")
    return all_npz_files


def infer_date_str(npz_path):
    """
    从文件名中恢复日期/样本标识。
    例如 2025_06_25_radial_wind_0325.npz -> 2025_06_25。
    """
    name = Path(npz_path).stem
    suffix = '_radial_wind_260514'
    return name[:-len(suffix)] if name.endswith(suffix) else name


def prepare_azimuth(azi_data):
    """
    保持原代码方位角校正方式：
    1. 四舍五入到 0.5°；
    2. 360° 改为 0°；
    3. 减去系统方位偏置 105° 后映射至 0–360°。
    """
    azi_data = np.round(azi_data * 2) / 2
    azi_data[azi_data == 360] = 0
    azi_data = (azi_data - AZIMUTH_OFFSET_DEG) % 360
    return azi_data


def split_by_time(radial_v_P, radial_v_S, azi_data, time_data, SNR_P, SNR_S):
    """
    按相邻径向时间间隔切分数据，避免断采前后的径向混合反演。
    """
    time_diff = np.diff(pd.to_datetime(time_data))
    time_diff_seconds = time_diff / np.timedelta64(1, 's')
    split_indices = np.where(time_diff_seconds > TIME_GAP_SECONDS)[0] + 1

    return (
        np.split(radial_v_P, split_indices, axis=0),
        np.split(radial_v_S, split_indices, axis=0),
        np.split(azi_data, split_indices, axis=0),
        np.split(time_data, split_indices, axis=0),
        np.split(SNR_P, split_indices, axis=0),
        np.split(SNR_S, split_indices, axis=0),
    )


def calculate_height():
    """
    沿用原代码高度计算方式。
    """
    range_bin = 512 * 0.15 * np.sin(np.radians(ELEVATION_ANGLE))
    height = np.arange(0.5, 57.5, 1) * range_bin
    return height


def invert_channel(radial_velocity, azi_data, time_data, method):
    """
    调用 SWF 完成单通道矢量风反演。
    """
    return SWF(
        radial_velocity,
        azi_data,
        time_data,
        NUM_RADIALS,
        method,
        elevationangle=ELEVATION_ANGLE,
        azi_tolerance=AZIMUTH_TOLERANCE_DEG,
        min_valid_azi=MIN_VALID_AZI,
        max_time_gap_seconds=TIME_GAP_SECONDS,
    )


def write_wind_excel(output_path, time_index, height,
                     hor_speed_P, ver_speed_P, wind_dir_P, V_all_P,
                     hor_speed_S, ver_speed_S, wind_dir_S, V_all_S):
    """
    保存 P/S 两通道矢量风结果。保留原有 Excel 输出结构，并增加 U/V 分量。
    """
    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame(hor_speed_P, index=time_index, columns=height).to_excel(writer, sheet_name='P Wind Speed')
        pd.DataFrame(ver_speed_P, index=time_index, columns=height).to_excel(writer, sheet_name='P Vertical Speed')
        pd.DataFrame(wind_dir_P, index=time_index, columns=height).to_excel(writer, sheet_name='P Wind Direction')
        pd.DataFrame(V_all_P[:, :, 1], index=time_index, columns=height).to_excel(writer, sheet_name='P U Component')
        pd.DataFrame(V_all_P[:, :, 2], index=time_index, columns=height).to_excel(writer, sheet_name='P V Component')

        pd.DataFrame(hor_speed_S, index=time_index, columns=height).to_excel(writer, sheet_name='S Wind Speed')
        pd.DataFrame(ver_speed_S, index=time_index, columns=height).to_excel(writer, sheet_name='S Vertical Speed')
        pd.DataFrame(wind_dir_S, index=time_index, columns=height).to_excel(writer, sheet_name='S Wind Direction')
        pd.DataFrame(V_all_S[:, :, 1], index=time_index, columns=height).to_excel(writer, sheet_name='S U Component')
        pd.DataFrame(V_all_S[:, :, 2], index=time_index, columns=height).to_excel(writer, sheet_name='S V Component')

def local_median_excluding_self(x, window=2):
    """
    对每条径向，计算其相邻径向的局部中位数，不包含自身。
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)

    n = len(x)
    for i in range(n):
        left = max(0, i - window)
        right = min(n, i + window + 1)

        neighbor_idx = [j for j in range(left, right) if j != i]
        neighbor_values = x[neighbor_idx]

        if np.sum(np.isfinite(neighbor_values)) >= 2:
            out[i] = np.nanmedian(neighbor_values)

    return out


def detect_bad_snr_radials(SNR, gate_start=0, gate_end=20,
                           local_window=2, drop_threshold_db=6):
    """
    根据每条径向的SNR剖面中位数，检测整条径向SNR突降异常。

    返回：
        bad_radial_mask: shape = (n_radials,)
    """
    snr_used = SNR[:, gate_start:gate_end]

    # 每条径向的代表性SNR
    radial_snr_median = np.nanmedian(snr_used, axis=1)

    # 相邻径向的局部背景SNR
    local_background = local_median_excluding_self(
        radial_snr_median,
        window=local_window
    )

    # 当前径向比相邻径向低太多，则判为异常
    bad_radial_mask = (
        np.isfinite(radial_snr_median)
        & np.isfinite(local_background)
        & ((local_background - radial_snr_median) >= drop_threshold_db)
    )

    return bad_radial_mask, radial_snr_median, local_background


def apply_snr_qc(radial_v_P, radial_v_S, SNR_P, SNR_S):
    """
    根据SNR对径向风速进行质量控制。
    注意：不直接修改SNR，只把异常SNR对应的径向风速置为NaN。
    """
    radial_v_P = radial_v_P.copy()
    radial_v_S = radial_v_S.copy()

    if not USE_SNR_QC:
        return radial_v_P, radial_v_S

    # 1. 单点低SNR剔除
    radial_v_P[SNR_P < SNR_POINT_THRESHOLD_DB] = np.nan
    radial_v_S[SNR_S < SNR_POINT_THRESHOLD_DB] = np.nan

    # 2. 整条径向SNR突降剔除
    if USE_RADIAL_SNR_JUMP_QC:
        bad_P, p_med, p_bg = detect_bad_snr_radials(
            SNR=SNR_P,
            gate_start=QC_GATE_START,
            gate_end=QC_GATE_END,
            local_window=LOCAL_WINDOW,
            drop_threshold_db=RADIAL_DROP_THRESHOLD_DB
        )

        bad_S, s_med, s_bg = detect_bad_snr_radials(
            SNR=SNR_S,
            gate_start=QC_GATE_START,
            gate_end=QC_GATE_END,
            local_window=LOCAL_WINDOW,
            drop_threshold_db=RADIAL_DROP_THRESHOLD_DB
        )

        # 可选策略A：P、S各自剔除
        radial_v_P[bad_P, :] = np.nan
        radial_v_S[bad_S, :] = np.nan

        # 可选策略B：如果认为P/S同步异常才是真正系统异常，可改成：
        # bad_both = bad_P & bad_S
        # radial_v_P[bad_both, :] = np.nan
        # radial_v_S[bad_both, :] = np.nan

        print(f"P-channel bad SNR radials: {np.sum(bad_P)}")
        print(f"S-channel bad SNR radials: {np.sum(bad_S)}")

    return radial_v_P, radial_v_S

def process_npz_file(npz_path, output_root):
    """
    处理一个径向风速 npz 文件，进行 P/S 通道矢量风反演并保存结果。
    输出自动按年份子目录（year_2024 / year_2025 / year_2026）组织。
    """
    npz_path = Path(npz_path)
    date_str = infer_date_str(npz_path)

    # 自动从父目录名提取年份子目录（如 year_2024），若无法提取则用 "misc"
    parent_name = npz_path.parent.name
    if parent_name.startswith('year_'):
        year_subdir = parent_name
    else:
        year_subdir = 'misc'

    output_dir = Path(output_root) / year_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在处理：{npz_path}  →  输出到 {output_dir}")

    data = np.load(npz_path)
    radial_v_P = data['radial_v_P']
    radial_v_S = data['radial_v_S']
    azi_data = prepare_azimuth(data['azi_data'])
    time_data = data['time']
    SNR_P = data['SNR_P']
    SNR_S = data['SNR_S']

    # SNR QC
    radial_v_P, radial_v_S = apply_snr_qc(
        radial_v_P,
        radial_v_S,
        SNR_P,
        SNR_S
    )

    (radial_v_P_splits,
     radial_v_S_splits,
     azi_data_splits,
     time_data_splits,
     SNR_P_splits,
     SNR_S_splits) = split_by_time(radial_v_P, radial_v_S, azi_data, time_data, SNR_P, SNR_S)

    height = calculate_height()

    p_method = CHANNEL_METHODS['P']
    s_method = CHANNEL_METHODS['S']

    for i in tqdm(range(len(radial_v_P_splits)), desc=f"Processing {date_str}"):
        azi_data_sub = azi_data_splits[i]
        radial_v_P_sub = radial_v_P_splits[i]
        radial_v_S_sub = radial_v_S_splits[i]
        time_data_sub = time_data_splits[i]

        if len(time_data_sub) <= NUM_RADIALS:
            print(f"片段 {i} 长度不足 {NUM_RADIALS + 1} 个径向，跳过。")
            continue

        start_time = time.time()
        V_all_P, hor_speed_P, ver_speed_P, wind_dir_P = invert_channel(
            radial_v_P_sub, azi_data_sub, time_data_sub, p_method
        )
        print(f"P通道（{p_method}）计算耗时：{time.time() - start_time:.2f} 秒，有效反演点数：{np.sum(np.all(np.isfinite(V_all_P), axis=2))}/{V_all_P.shape[0] * V_all_P.shape[1]}")

        start_time = time.time()
        V_all_S, hor_speed_S, ver_speed_S, wind_dir_S = invert_channel(
            radial_v_S_sub, azi_data_sub, time_data_sub, s_method
        )
        print(f"S通道（{s_method}）计算耗时：{time.time() - start_time:.2f} 秒，有效反演点数：{np.sum(np.all(np.isfinite(V_all_S), axis=2))}/{V_all_S.shape[0] * V_all_S.shape[1]}")

        time_index = time_data_sub[:-NUM_RADIALS]
        output_path = output_dir / f'wind_results_{date_str}_{i}_P-{p_method}_S-{s_method}.xlsx'
        write_wind_excel(
            output_path,
            time_index,
            height,
            hor_speed_P, ver_speed_P, wind_dir_P, V_all_P,
            hor_speed_S, ver_speed_S, wind_dir_S, V_all_S,
        )
        print(f"已保存：{output_path}")


def main():
    npz_files = collect_npz_files(RADIAL_NPZ_INPUT)
    if N_JOBS == 1 or len(npz_files) == 1:
        for npz_file in npz_files:
            process_npz_file(npz_file, OUTPUT_DIR)
    else:
        Parallel(n_jobs=N_JOBS)(delayed(process_npz_file)(npz_file, OUTPUT_DIR) for npz_file in npz_files)


if __name__ == '__main__':
    main()
