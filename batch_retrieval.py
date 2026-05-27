import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from vector_wind_solver import SWF

# ============================= 常改动参数区 =============================
# 绝对路径；既可以是单个 *_radial_wind_0325.npz 文件，也可以是包含该类 npz 的文件夹。
RADIAL_NPZ_INPUT = r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\22deg5_radial_wind_260514.npz"

# 输出目录。每个连续数据片段输出一个 Excel。
OUTPUT_DIR = r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\wind_vector_results"

# P、S 通道分别采用的矢量风反演算法。只改这里即可完成通道算法切换。
# 可选：'SVD'、'AIR'、'cooks'、'FSWF'。其中 'DSWF' 仅在一维分支保留。
CHANNEL_METHODS = {
    'P': 'cooks',   # P 通道默认沿用原代码中的 FSWF
    'S': 'cooks',    # S 通道默认沿用原代码中的 SVD
}

# =========================
# SNR质量控制参数
# =========================
USE_SNR_QC = True

# 逐点SNR阈值，低于该值的“单个高度门径向风速”置为NaN
SNR_POINT_THRESHOLD_DB = -23

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
N_JOBS = 1
# =======================================================================


def collect_npz_files(input_path):
    """
    收集径向风速 npz 文件。input_path 可以是单个 npz 文件或文件夹。
    """
    path = Path(input_path)
    if not path.is_absolute():
        raise ValueError(f"RADIAL_NPZ_INPUT 必须是绝对路径：{input_path}")
    if not path.exists():
        raise FileNotFoundError(f"路径不存在：{input_path}")

    if path.is_file():
        if path.suffix.lower() != '.npz':
            raise ValueError(f"输入文件必须是 .npz 文件：{path}")
        return [path]

    npz_files = sorted(path.glob('*_radial_wind_260514.npz'))
    if not npz_files:
        raise FileNotFoundError(f"未在路径中找到 *_radial_wind_260514.npz：{path}")
    return npz_files


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

def process_npz_file(npz_path, output_dir):
    """
    处理一个径向风速 npz 文件，进行 P/S 通道矢量风反演并保存结果。
    """
    npz_path = Path(npz_path)
    date_str = infer_date_str(npz_path)
    print(f"正在处理：{npz_path}")

    data = np.load(npz_path)
    radial_v_P = data['radial_v_P']
    radial_v_S = data['radial_v_S']
    azi_data = prepare_azimuth(data['azi_data'])
    time_data = data['time']
    SNR_P = data['SNR_P']
    SNR_S = data['SNR_S']

    # SNR QC：这里加入
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
