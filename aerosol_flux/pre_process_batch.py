# -*- coding: utf-8 -*-
"""
pre_process.py

功能：
1. 读取单个或某一天文件夹下的全部 CDWL 原始 h5 文件
2. 将 specData 重排为 [径向, 通道, 距离门, 频率]
3. 裁剪 80~160 MHz 频谱 ROI
4. 以首门为基底噪声，保留第 4~30 门
5. 识别多普勒峰范围并计算谱质心
6. 对峰区功率进行积分
7. 按本振功率、脉冲能量、累积脉冲数归一化
8. 进行 R^2 距离平方修正，输出 p_rcs、s_rcs
9. 将结果保存为 npz，供后续 Fernald 反演使用

当前版本特点：
- 不需要 weather_type
- 不需要 manifest.csv
- 只需手动修改 main() 中的 input_date_dir
"""

from __future__ import annotations

import csv
import traceback
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 基本参数
# =========================================================
N_BINS = 512
N_GATES_PER_CH = 60
N_CHANNELS = 2

F_MAX_MHZ = 500.0
ROI_LOW = 80.0
ROI_HIGH = 160.0

# True: 假设原始展平顺序为 [P60门, S60门] × 512bin
# 若未来发现通道/门顺序异常，可改为 False 再检查
USE_DEFAULT_LAYOUT = True


# =========================================================
# 系统参数
# =========================================================
C = 299792458.0                # 光速, m/s
LAMBDA0_M = 1550e-9            # 波长, m
F_AOM_HZ = 120e6               # AOM 频移, Hz
TAU_PULSE_S = 500e-9           # 脉冲宽度, s
E_PULSE_J = 40e-6              # 单脉冲能量, J
P_LO_P_W = 1.85e-3             # P 通道本振功率, W
P_LO_S_W = 1.64e-3             # S 通道本振功率, W
PRF_HZ = 1.0e4                 # 重频, Hz
N_ACC = 10000                  # 累积脉冲数

RANGE_RES_M = C * TAU_PULSE_S / 2.0   # 约 75 m


# =========================================================
# 数据读写与基础处理
# =========================================================
def get_valid_mask(spec_data: np.ndarray) -> np.ndarray:
    """
    根据 specData 判断有效径向。
    数据说明中最后一行通常为 0 表示结束；特殊情况下主动停止时也可能不足 101 行。
    """
    if spec_data.ndim != 2:
        raise ValueError("spec_data 必须为二维数组 (N_radial, 61440)")
    return np.any(spec_data != 0, axis=1)


def reshape_spec(spec_data: np.ndarray, use_default_layout: bool = True) -> np.ndarray:
    """
    将 specData 从 (N, 61440) 重排为 (N, 2, 60, 512)
    返回维度含义：[径向, 通道(P/S), 距离门, 频率bin]
    """
    if spec_data.ndim != 2:
        raise ValueError("spec_data 必须为二维数组")

    n_radials = spec_data.shape[0]
    expected_len = N_CHANNELS * N_GATES_PER_CH * N_BINS
    if spec_data.shape[1] != expected_len:
        raise ValueError(
            f"spec_data 第二维应为 {expected_len}，当前为 {spec_data.shape[1]}"
        )

    if use_default_layout:
        # [P1 512bin, P2 512bin, ..., P60 512bin, S1 512bin, ..., S60 512bin]
        spec_4d = spec_data.reshape(n_radials, N_CHANNELS, N_GATES_PER_CH, N_BINS)
    else:
        # 备用布局
        spec_4d = (
            spec_data.reshape(n_radials, N_BINS, N_CHANNELS, N_GATES_PER_CH)
            .transpose(0, 2, 3, 1)
        )

    return spec_4d


def build_freq_axis() -> np.ndarray:
    """
    构造频率轴，单位 MHz
    """
    df = F_MAX_MHZ / N_BINS
    freq_axis = np.arange(N_BINS, dtype=np.float64) * df
    return freq_axis


def crop_roi(
    spec_4d: np.ndarray,
    freq_axis: np.ndarray,
    roi_low: float,
    roi_high: float,
):
    """
    在频率维裁剪 ROI
    """
    if spec_4d.ndim != 4:
        raise ValueError("spec_4d 必须为四维数组 (N, 2, 60, 512)")

    roi_mask = (freq_axis >= roi_low) & (freq_axis <= roi_high)
    freq_roi = freq_axis[roi_mask]

    spec_p = spec_4d[:, 0, :, :]   # (N, 60, 512)
    spec_s = spec_4d[:, 1, :, :]   # (N, 60, 512)

    spec_p_roi = spec_p[:, :, roi_mask]
    spec_s_roi = spec_s[:, :, roi_mask]

    return spec_p_roi, spec_s_roi, freq_roi, roi_mask


# =========================================================
# 频谱矫正与峰区识别
# =========================================================
def subtract_baseline_noise_and_keep_gates(
    spec_roi: np.ndarray,
    noise_gate_idx: int = 0,
    keep_gate_start: int = 3,
    keep_gate_end: int = 29,
    clip_negative: bool = True,
):
    """
    对单通道 ROI 频谱执行：
    1) 以首个距离门作为基底噪声直接相减
    2) 仅保留第 4~30 个距离门（Python 索引 3~29）

    返回
    ----
    spec_corr : ndarray, shape (N_radial, 27, N_freq)
    kept_gate_numbers : ndarray, 1-based, 即 [4, 5, ..., 30]
    """
    if spec_roi.ndim != 3:
        raise ValueError("spec_roi 必须为三维数组 (N_radial, N_gate, N_freq)")

    if noise_gate_idx < 0 or noise_gate_idx >= spec_roi.shape[1]:
        raise IndexError("noise_gate_idx 超出距离门范围")

    # 关键：转为浮点型，避免无符号整型减法问题
    spec_roi = spec_roi.astype(np.float32, copy=False)

    noise_baseline = spec_roi[:, noise_gate_idx:noise_gate_idx + 1, :]
    spec_corr_all = spec_roi - noise_baseline

    if clip_negative:
        spec_corr_all = np.maximum(spec_corr_all, 0.0)

    spec_corr = spec_corr_all[:, keep_gate_start:keep_gate_end + 1, :]
    kept_gate_numbers = np.arange(keep_gate_start + 1, keep_gate_end + 2, dtype=np.int32)

    return spec_corr, kept_gate_numbers


def find_peak_bounds_by_trend_change(spectrum_1d: np.ndarray):
    """
    以谱峰最大值位置为中心：
    - 向左寻找第一个趋势变化点（局部谷点）作为左边界
    - 向右寻找第一个趋势变化点（局部谷点）作为右边界
    """
    y = np.asarray(spectrum_1d, dtype=np.float64)

    if y.ndim != 1:
        raise ValueError("spectrum_1d 必须为一维数组")

    if np.all(~np.isfinite(y)) or np.nanmax(y) <= 0:
        return -1, -1, -1

    peak_idx = int(np.nanargmax(y))
    n = y.size

    left_idx = 0
    found_left = False
    for i in range(peak_idx - 1, 0, -1):
        if y[i - 1] > y[i]:
            left_idx = i
            found_left = True
            break
    if not found_left:
        left_idx = 0

    right_idx = n - 1
    found_right = False
    for i in range(peak_idx + 1, n - 1):
        if y[i + 1] > y[i]:
            right_idx = i
            found_right = True
            break
    if not found_right:
        right_idx = n - 1

    if left_idx > peak_idx:
        left_idx = peak_idx
    if right_idx < peak_idx:
        right_idx = peak_idx

    return peak_idx, left_idx, right_idx


def compute_spectral_centroid(
    freq_axis: np.ndarray,
    spectrum_1d: np.ndarray,
    left_idx: int,
    right_idx: int,
):
    """
    在多普勒峰范围内计算谱质心频率
    """
    if left_idx < 0 or right_idx < 0 or right_idx < left_idx:
        return np.nan

    f_seg = freq_axis[left_idx:right_idx + 1]
    p_seg = spectrum_1d[left_idx:right_idx + 1]

    power_sum = np.sum(p_seg, dtype=np.float64)
    if power_sum <= 0:
        return np.nan

    centroid = np.sum(f_seg * p_seg, dtype=np.float64) / power_sum
    return centroid


def detect_doppler_peak_ranges(spec_corr: np.ndarray, freq_axis: np.ndarray):
    """
    对单通道去噪后的频谱批量检测多普勒峰范围，并计算谱质心
    """
    n_radial, n_gate, n_freq = spec_corr.shape

    peak_idx = np.full((n_radial, n_gate), -1, dtype=np.int32)
    left_idx = np.full((n_radial, n_gate), -1, dtype=np.int32)
    right_idx = np.full((n_radial, n_gate), -1, dtype=np.int32)
    centroid_freq = np.full((n_radial, n_gate), np.nan, dtype=np.float32)
    peak_mask = np.zeros((n_radial, n_gate, n_freq), dtype=bool)

    for i in range(n_radial):
        for j in range(n_gate):
            y = spec_corr[i, j, :]
            p_idx, l_idx, r_idx = find_peak_bounds_by_trend_change(y)

            peak_idx[i, j] = p_idx
            left_idx[i, j] = l_idx
            right_idx[i, j] = r_idx

            if p_idx >= 0 and l_idx >= 0 and r_idx >= 0 and r_idx >= l_idx:
                peak_mask[i, j, l_idx:r_idx + 1] = True
                centroid_freq[i, j] = compute_spectral_centroid(freq_axis, y, l_idx, r_idx)

    return {
        "peak_idx": peak_idx,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "centroid_freq_mhz": centroid_freq,
        "peak_mask": peak_mask,
    }


def plot_one_doppler_peak_example(
    spec_corr: np.ndarray,
    freq_axis: np.ndarray,
    peak_result: dict,
    radial_idx: int,
    gate_idx: int,
    channel_name: str,
    kept_gate_numbers: np.ndarray,
    save_dir: str | Path,
) -> None:
    """
    临时可视化某一条功率谱及其多普勒峰范围。
    批处理阶段通常关闭。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_radial, n_gate, _ = spec_corr.shape
    radial_idx = int(np.clip(radial_idx, 0, n_radial - 1))
    gate_idx = int(np.clip(gate_idx, 0, n_gate - 1))

    y = spec_corr[radial_idx, gate_idx, :]
    p_idx = int(peak_result["peak_idx"][radial_idx, gate_idx])
    l_idx = int(peak_result["left_idx"][radial_idx, gate_idx])
    r_idx = int(peak_result["right_idx"][radial_idx, gate_idx])
    c_freq = peak_result["centroid_freq_mhz"][radial_idx, gate_idx]

    if p_idx < 0 or l_idx < 0 or r_idx < 0:
        print(f"[警告] {channel_name} 通道无有效峰，未绘图")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(freq_axis, y, linewidth=1.5, label="Corrected spectrum")
    plt.axvline(freq_axis[p_idx], linestyle="--", linewidth=1.2, label=f"Peak @ {freq_axis[p_idx]:.3f} MHz")
    plt.axvline(freq_axis[l_idx], linestyle=":", linewidth=1.2, label=f"Left @ {freq_axis[l_idx]:.3f} MHz")
    plt.axvline(freq_axis[r_idx], linestyle=":", linewidth=1.2, label=f"Right @ {freq_axis[r_idx]:.3f} MHz")
    plt.axvspan(freq_axis[l_idx], freq_axis[r_idx], alpha=0.25, label="Peak range")

    if np.isfinite(c_freq):
        plt.axvline(c_freq, linestyle="-.", linewidth=1.2, label=f"Centroid @ {c_freq:.3f} MHz")

    physical_gate = int(kept_gate_numbers[gate_idx])
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power Spectral Density (a.u.)")
    plt.title(f"{channel_name}-channel Doppler peak\nRadial={radial_idx}, Gate={physical_gate}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = save_dir / f"doppler_peak_{channel_name}_radial_{radial_idx}_gate_{physical_gate}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def spectrum_correction_pipeline(
    spec_p_roi: np.ndarray,
    spec_s_roi: np.ndarray,
    freq_roi: np.ndarray,
    plot_example: bool = False,
    example_channel: str = "P",
    example_radial_idx: int = 10,
    example_gate_idx: int = 0,
    save_dir: str | Path | None = None,
):
    """
    频谱矫正总流程：
    1) 各通道以首门为基底噪声直接相减
    2) 各通道保留第 4~30 门
    3) 批量确定多普勒峰范围
    4) 可选绘制一个示例峰区用于验证
    """
    spec_p_corr, kept_gate_numbers = subtract_baseline_noise_and_keep_gates(
        spec_p_roi,
        noise_gate_idx=0,
        keep_gate_start=3,
        keep_gate_end=29,
        clip_negative=True,
    )
    peak_result_p = detect_doppler_peak_ranges(spec_p_corr, freq_roi)

    spec_s_corr, kept_gate_numbers_s = subtract_baseline_noise_and_keep_gates(
        spec_s_roi,
        noise_gate_idx=0,
        keep_gate_start=3,
        keep_gate_end=29,
        clip_negative=True,
    )
    peak_result_s = detect_doppler_peak_ranges(spec_s_corr, freq_roi)

    if not np.array_equal(kept_gate_numbers, kept_gate_numbers_s):
        raise RuntimeError("P/S 两通道保留门号不一致")

    if plot_example:
        if save_dir is None:
            save_dir = Path.cwd() / "debug_figs"

        if example_channel.upper() == "P":
            plot_one_doppler_peak_example(
                spec_corr=spec_p_corr,
                freq_axis=freq_roi,
                peak_result=peak_result_p,
                radial_idx=example_radial_idx,
                gate_idx=example_gate_idx,
                channel_name="P",
                kept_gate_numbers=kept_gate_numbers,
                save_dir=save_dir,
            )
        elif example_channel.upper() == "S":
            plot_one_doppler_peak_example(
                spec_corr=spec_s_corr,
                freq_axis=freq_roi,
                peak_result=peak_result_s,
                radial_idx=example_radial_idx,
                gate_idx=example_gate_idx,
                channel_name="S",
                kept_gate_numbers=kept_gate_numbers,
                save_dir=save_dir,
            )
        else:
            raise ValueError("example_channel 只能是 'P' 或 'S'")

    return {
        "kept_gate_numbers": kept_gate_numbers,
        "spec_p_corr": spec_p_corr,
        "spec_s_corr": spec_s_corr,

        "p_peak_idx": peak_result_p["peak_idx"],
        "p_left_idx": peak_result_p["left_idx"],
        "p_right_idx": peak_result_p["right_idx"],
        "p_centroid_freq_mhz": peak_result_p["centroid_freq_mhz"],
        "p_peak_mask": peak_result_p["peak_mask"],

        "s_peak_idx": peak_result_s["peak_idx"],
        "s_left_idx": peak_result_s["left_idx"],
        "s_right_idx": peak_result_s["right_idx"],
        "s_centroid_freq_mhz": peak_result_s["centroid_freq_mhz"],
        "s_peak_mask": peak_result_s["peak_mask"],
    }


# =========================================================
# 峰区积分、归一化、距离平方修正
# =========================================================
def integrate_peak_area(
    spec_corr: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    freq_axis_mhz: np.ndarray,
):
    """
    在多普勒峰范围内对峰区谱值积分

    返回：
    - peak_sum      : 不乘频宽的离散求和
    - peak_int_mhz  : 乘以 MHz 频宽后的积分
    - peak_int_hz   : 乘以 Hz 频宽后的积分
    """
    spec_corr = np.asarray(spec_corr, dtype=np.float32)
    n_radial, n_gate, _ = spec_corr.shape

    df_mhz = float(np.mean(np.diff(freq_axis_mhz)))
    df_hz = df_mhz * 1e6

    peak_sum = np.zeros((n_radial, n_gate), dtype=np.float32)
    peak_int_mhz = np.zeros((n_radial, n_gate), dtype=np.float32)
    peak_int_hz = np.zeros((n_radial, n_gate), dtype=np.float32)

    for i in range(n_radial):
        for j in range(n_gate):
            l_idx = int(left_idx[i, j])
            r_idx = int(right_idx[i, j])

            if l_idx < 0 or r_idx < 0 or r_idx < l_idx:
                peak_sum[i, j] = np.nan
                peak_int_mhz[i, j] = np.nan
                peak_int_hz[i, j] = np.nan
                continue

            seg = spec_corr[i, j, l_idx:r_idx + 1]
            s = np.sum(seg, dtype=np.float64)

            peak_sum[i, j] = s
            peak_int_mhz[i, j] = s * df_mhz
            peak_int_hz[i, j] = s * df_hz

    return {
        "peak_sum": peak_sum,
        "peak_int_mhz": peak_int_mhz,
        "peak_int_hz": peak_int_hz,
        "df_mhz": df_mhz,
        "df_hz": df_hz,
    }


def normalize_peak_area(
    peak_area: np.ndarray,
    lo_power_w: float,
    pulse_energy_j: float = E_PULSE_J,
    n_acc: int = N_ACC,
):
    """
    对峰区积分量进行系统参数归一化
    """
    peak_area = np.asarray(peak_area, dtype=np.float32)
    norm_area = peak_area / (lo_power_w * pulse_energy_j * float(n_acc))
    return norm_area


def gate_numbers_to_range_m(
    kept_gate_numbers: np.ndarray,
    range_res_m: float = RANGE_RES_M,
    range_offset_m: float = 0.0,
):
    """
    将物理距离门号转换为名义门中心距离
    """
    kept_gate_numbers = np.asarray(kept_gate_numbers, dtype=np.float32)
    ranges_m = (kept_gate_numbers - 0.5) * range_res_m + range_offset_m
    return ranges_m


def make_range_corrected_signal(
    norm_peak_area: np.ndarray,
    kept_gate_numbers: np.ndarray,
    range_res_m: float = RANGE_RES_M,
    range_offset_m: float = 0.0,
):
    """
    构造距离平方修正信号:
        RCS = norm_peak_area * r^2
    """
    ranges_m = gate_numbers_to_range_m(
        kept_gate_numbers=kept_gate_numbers,
        range_res_m=range_res_m,
        range_offset_m=range_offset_m,
    )  # (N_gate,)

    rcs = norm_peak_area * (ranges_m[None, :] ** 2)
    return rcs, ranges_m


# =========================================================
# 单文件处理
# =========================================================
def process_one_file(
    h5_path: str | Path,
    output_npz_path: str | Path,
    plot_example: bool = False,
    example_channel: str = "P",
    example_radial_idx: int = 10,
    example_gate_idx: int = 0,
    plot_dir: str | Path | None = None,
):
    """
    处理单个 h5 文件，并输出对应的 npz 文件
    """
    h5_path = Path(h5_path)
    output_npz_path = Path(output_npz_path)
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        azimuth = f["azimuthAngle"][...]
        los_velo = f["losVeloData"][...]
        spec_data = f["specData"][...]
        timestamp = f["timeStamp"][...]

    valid_mask = get_valid_mask(spec_data)
    azimuth = azimuth[valid_mask]
    los_velo = los_velo[valid_mask]
    spec_data = spec_data[valid_mask]
    timestamp = timestamp[valid_mask]

    if spec_data.shape[0] == 0:
        raise RuntimeError(f"文件中无有效径向: {h5_path}")

    spec_4d = reshape_spec(spec_data, use_default_layout=USE_DEFAULT_LAYOUT)

    freq_axis = build_freq_axis()
    spec_p_roi, spec_s_roi, freq_roi, roi_mask = crop_roi(
        spec_4d, freq_axis, ROI_LOW, ROI_HIGH
    )

    # 转浮点，避免整型相减问题
    spec_p_roi = spec_p_roi.astype(np.float32, copy=False)
    spec_s_roi = spec_s_roi.astype(np.float32, copy=False)

    correction_results = spectrum_correction_pipeline(
        spec_p_roi=spec_p_roi,
        spec_s_roi=spec_s_roi,
        freq_roi=freq_roi,
        plot_example=plot_example,
        example_channel=example_channel,
        example_radial_idx=example_radial_idx,
        example_gate_idx=example_gate_idx,
        save_dir=plot_dir,
    )

    p_peak_area_results = integrate_peak_area(
        spec_corr=correction_results["spec_p_corr"],
        left_idx=correction_results["p_left_idx"],
        right_idx=correction_results["p_right_idx"],
        freq_axis_mhz=freq_roi,
    )

    s_peak_area_results = integrate_peak_area(
        spec_corr=correction_results["spec_s_corr"],
        left_idx=correction_results["s_left_idx"],
        right_idx=correction_results["s_right_idx"],
        freq_axis_mhz=freq_roi,
    )

    p_peak_norm = normalize_peak_area(
        peak_area=p_peak_area_results["peak_sum"],
        lo_power_w=P_LO_P_W,
        pulse_energy_j=E_PULSE_J,
        n_acc=N_ACC,
    )

    s_peak_norm = normalize_peak_area(
        peak_area=s_peak_area_results["peak_sum"],
        lo_power_w=P_LO_S_W,
        pulse_energy_j=E_PULSE_J,
        n_acc=N_ACC,
    )

    p_rcs, range_m = make_range_corrected_signal(
        norm_peak_area=p_peak_norm,
        kept_gate_numbers=correction_results["kept_gate_numbers"],
        range_res_m=RANGE_RES_M,
        range_offset_m=0.0,
    )

    s_rcs, _ = make_range_corrected_signal(
        norm_peak_area=s_peak_norm,
        kept_gate_numbers=correction_results["kept_gate_numbers"],
        range_res_m=RANGE_RES_M,
        range_offset_m=0.0,
    )

    np.savez_compressed(
        output_npz_path,
        azimuth=azimuth,
        timestamp=timestamp,
        los_velo=los_velo,
        freq_axis_full=freq_axis,
        freq_axis_roi=freq_roi,
        roi_mask=roi_mask,
        spec_p_roi=spec_p_roi,
        spec_s_roi=spec_s_roi,

        **correction_results,

        p_peak_sum=p_peak_area_results["peak_sum"],
        p_peak_int_mhz=p_peak_area_results["peak_int_mhz"],
        p_peak_int_hz=p_peak_area_results["peak_int_hz"],

        s_peak_sum=s_peak_area_results["peak_sum"],
        s_peak_int_mhz=s_peak_area_results["peak_int_mhz"],
        s_peak_int_hz=s_peak_area_results["peak_int_hz"],

        p_peak_norm=p_peak_norm,
        s_peak_norm=s_peak_norm,

        range_m=range_m,
        p_rcs=p_rcs,
        s_rcs=s_rcs,

        source_h5_path=str(h5_path),
    )

    return output_npz_path


# =========================================================
# 批量处理：某一天的文件夹
# =========================================================
def build_output_path(
    h5_path: str | Path,
    input_date_dir: str | Path,
    output_root: str | Path,
) -> Path:
    """
    将某个 h5 文件映射为输出 npz 路径。
    输出目录中自动建立与日期文件夹同名的子目录。
    """
    h5_path = Path(h5_path)
    input_date_dir = Path(input_date_dir)
    output_root = Path(output_root)

    date_folder_name = input_date_dir.name
    out_dir = output_root / date_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir / f"{h5_path.stem}_preprocessed.npz"


def batch_process_one_date_folder(
    input_date_dir: str | Path,
    output_root: str | Path,
    failure_csv_path: str | Path | None = None,
    recursive: bool = False,
    example_plot_for_first_file: bool = False,
):
    """
    对某一个日期文件夹下的所有 h5 文件进行批处理
    """
    input_date_dir = Path(input_date_dir)
    output_root = Path(output_root)

    if not input_date_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_date_dir}")

    if recursive:
        h5_files = sorted(input_date_dir.rglob("*.h5"))
    else:
        h5_files = sorted(input_date_dir.glob("*.h5"))

    if len(h5_files) == 0:
        raise FileNotFoundError(f"在 {input_date_dir} 中未找到任何 .h5 文件")

    success_count = 0
    failed_records = []
    first_success_done = False

    for h5_path in h5_files:
        output_npz_path = build_output_path(
            h5_path=h5_path,
            input_date_dir=input_date_dir,
            output_root=output_root,
        )

        plot_example = False
        plot_dir = None
        if example_plot_for_first_file and (not first_success_done):
            plot_example = True
            plot_dir = output_npz_path.parent / "debug_figs"

        try:
            process_one_file(
                h5_path=h5_path,
                output_npz_path=output_npz_path,
                plot_example=plot_example,
                example_channel="P",
                example_radial_idx=10,
                example_gate_idx=0,
                plot_dir=plot_dir,
            )
            success_count += 1
            first_success_done = True
            print(f"[OK] {h5_path} -> {output_npz_path}")

        except Exception as e:
            failed_records.append(
                {
                    "h5_path": str(h5_path),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[FAILED] {h5_path}")
            print(traceback.format_exc())

    if failure_csv_path is not None:
        failure_csv_path = Path(failure_csv_path)
        failure_csv_path.parent.mkdir(parents=True, exist_ok=True)

        with failure_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            fieldnames = ["h5_path", "error", "traceback"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in failed_records:
                writer.writerow(record)

    print("=" * 60)
    print(f"日期文件夹: {input_date_dir}")
    print(f"总文件数: {len(h5_files)}")
    print(f"成功数: {success_count}")
    print(f"失败数: {len(failed_records)}")
    print("=" * 60)

    return failed_records


# =========================================================
# 主程序
# =========================================================
def main():
    """
    当前阶段只需要手动修改 input_date_dir 即可切换不同天气对应的日期数据
    """
    input_date_dir = r"E:\测风组实验数据\RawData\2024\2024_11_09"
    output_root = r"E:\测风组实验数据\气溶胶反演\预处理"
    failure_csv_path = r"E:\测风组实验数据\气溶胶反演\预处理\failed_files_2024-11-09.csv"

    failed_records = batch_process_one_date_folder(
        input_date_dir=input_date_dir,
        output_root=output_root,
        failure_csv_path=failure_csv_path,
        recursive=False,
        example_plot_for_first_file=False,
    )

    if failed_records:
        print("该日期批处理结束，但存在失败文件。")
    else:
        print("该日期批处理全部完成。")


if __name__ == "__main__":
    main()