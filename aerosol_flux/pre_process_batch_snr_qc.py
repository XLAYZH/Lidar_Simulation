# -*- coding: utf-8 -*-
"""
pre_process_batch_snr_qc.py

功能
----
1. 读取某一天文件夹下的全部 CDWL 原始 h5 文件
2. 将 specData 重排为 [径向, 通道, 距离门, 频率]
3. 裁剪 80~160 MHz ROI
4. 以首门为噪声基底直接相减，仅保留第 4~30 门
5. 识别多普勒峰区并计算谱质心
6. 计算峰区积分
7. 基于“矫正后峰区积分 / 未矫正首门 ROI 噪声”计算峰区 SNR
8. 输出门级/廓线级 QC 掩膜，并同时保存：
   - 原始 RCS 产品（raw）
   - 门级/廓线级 QC 后的 RCS 产品（zero / nan 两种）
9. 保持与后续反演脚本的字段兼容：timestamp, azimuth, range_m, p_rcs, s_rcs

说明
----
- 为保证与现有反演脚本兼容，字段 p_rcs / s_rcs 仍保留为原始未 QC 的 RCS。
- 推荐后续在反演脚本中优先使用：
    p_rcs_qc_nan, s_rcs_qc_nan
  或使用 gate/profile QC 掩膜进行质量控制。
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
USE_DEFAULT_LAYOUT = True


# =========================================================
# 系统参数
# =========================================================
C = 299792458.0
LAMBDA0_M = 1550e-9
F_AOM_HZ = 120e6
TAU_PULSE_S = 500e-9
E_PULSE_J = 40e-6
P_LO_P_W = 1.85e-3
P_LO_S_W = 1.64e-3
PRF_HZ = 1.0e4
N_ACC = 10000

RANGE_RES_M = C * TAU_PULSE_S / 2.0


# =========================================================
# SNR / QC 参数（建议起始值）
# =========================================================
# 单通道门级 SNR 阈值（dB）
GATE_MIN_SNR_DB = -6.0

# 单通道峰积分线性 SNR 阈值（可选辅助条件）
GATE_MIN_SNR_LINEAR = 1.0

# 廓线级 QC：要求低层前5个保留门（物理第4~8门）中，
# 至少有 MIN_VALID_LOW_GATES 个门在 P 或 S 任一通道通过门级阈值
LOW_GATE_SLICE = slice(0, 15)
MIN_VALID_LOW_GATES = 8

# 若峰区 bin 数过窄，则视为不稳定峰区
MIN_PEAK_BINS = 2

# 若峰区 bin 数过宽，则通常说明边界识别失真
MAX_PEAK_BINS = 40


# =========================================================
# 基础处理
# =========================================================
def get_valid_mask(spec_data: np.ndarray) -> np.ndarray:
    if spec_data.ndim != 2:
        raise ValueError("spec_data 必须为二维数组")
    return np.any(spec_data != 0, axis=1)


def reshape_spec(spec_data: np.ndarray, use_default_layout: bool = True) -> np.ndarray:
    n_radials = spec_data.shape[0]
    expected_len = N_CHANNELS * N_GATES_PER_CH * N_BINS
    if spec_data.shape[1] != expected_len:
        raise ValueError(f"spec_data 第二维应为 {expected_len}，当前为 {spec_data.shape[1]}")

    if use_default_layout:
        spec_4d = spec_data.reshape(n_radials, N_CHANNELS, N_GATES_PER_CH, N_BINS)
    else:
        spec_4d = spec_data.reshape(n_radials, N_BINS, N_CHANNELS, N_GATES_PER_CH).transpose(0, 2, 3, 1)
    return spec_4d


def build_freq_axis() -> np.ndarray:
    df = F_MAX_MHZ / N_BINS
    return np.arange(N_BINS, dtype=np.float64) * df


def crop_roi(spec_4d: np.ndarray, freq_axis: np.ndarray, roi_low: float, roi_high: float):
    roi_mask = (freq_axis >= roi_low) & (freq_axis <= roi_high)
    freq_roi = freq_axis[roi_mask]

    spec_p = spec_4d[:, 0, :, :]
    spec_s = spec_4d[:, 1, :, :]

    spec_p_roi = spec_p[:, :, roi_mask]
    spec_s_roi = spec_s[:, :, roi_mask]

    return spec_p_roi, spec_s_roi, freq_roi, roi_mask


# =========================================================
# 峰区识别
# =========================================================
def subtract_baseline_noise_and_keep_gates(
    spec_roi: np.ndarray,
    noise_gate_idx: int = 0,
    keep_gate_start: int = 3,
    keep_gate_end: int = 29,
    clip_negative: bool = True,
):
    spec_roi = np.asarray(spec_roi, dtype=np.float32)
    noise_baseline = spec_roi[:, noise_gate_idx:noise_gate_idx + 1, :]
    spec_corr_all = spec_roi - noise_baseline
    if clip_negative:
        spec_corr_all = np.maximum(spec_corr_all, 0.0)

    spec_corr = spec_corr_all[:, keep_gate_start:keep_gate_end + 1, :]
    kept_gate_numbers = np.arange(keep_gate_start + 1, keep_gate_end + 2, dtype=np.int32)
    return spec_corr, kept_gate_numbers


def find_peak_bounds_by_trend_change(spectrum_1d: np.ndarray):
    y = np.asarray(spectrum_1d, dtype=np.float64)
    if np.all(~np.isfinite(y)) or np.nanmax(y) <= 0:
        return -1, -1, -1

    peak_idx = int(np.nanargmax(y))
    n = y.size

    left_idx = 0
    for i in range(peak_idx - 1, 0, -1):
        if y[i - 1] > y[i]:
            left_idx = i
            break

    right_idx = n - 1
    for i in range(peak_idx + 1, n - 1):
        if y[i + 1] > y[i]:
            right_idx = i
            break

    if left_idx > peak_idx:
        left_idx = peak_idx
    if right_idx < peak_idx:
        right_idx = peak_idx

    return peak_idx, left_idx, right_idx


def compute_spectral_centroid(freq_axis: np.ndarray, spectrum_1d: np.ndarray, left_idx: int, right_idx: int):
    if left_idx < 0 or right_idx < 0 or right_idx < left_idx:
        return np.nan
    f_seg = freq_axis[left_idx:right_idx + 1]
    p_seg = spectrum_1d[left_idx:right_idx + 1]
    power_sum = np.sum(p_seg, dtype=np.float64)
    if power_sum <= 0:
        return np.nan
    return np.sum(f_seg * p_seg, dtype=np.float64) / power_sum


def detect_doppler_peak_ranges(spec_corr: np.ndarray, freq_axis: np.ndarray):
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
):
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
    spec_p_corr, kept_gate_numbers = subtract_baseline_noise_and_keep_gates(spec_p_roi)
    peak_result_p = detect_doppler_peak_ranges(spec_p_corr, freq_roi)

    spec_s_corr, kept_gate_numbers_s = subtract_baseline_noise_and_keep_gates(spec_s_roi)
    peak_result_s = detect_doppler_peak_ranges(spec_s_corr, freq_roi)

    if not np.array_equal(kept_gate_numbers, kept_gate_numbers_s):
        raise RuntimeError("P/S 两通道保留门号不一致")

    if plot_example:
        if save_dir is None:
            save_dir = Path.cwd() / "debug_figs"
        if example_channel.upper() == "P":
            plot_one_doppler_peak_example(spec_p_corr, freq_roi, peak_result_p, example_radial_idx, example_gate_idx, "P", kept_gate_numbers, save_dir)
        else:
            plot_one_doppler_peak_example(spec_s_corr, freq_roi, peak_result_s, example_radial_idx, example_gate_idx, "S", kept_gate_numbers, save_dir)

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
# 峰区积分 / 归一化 / 距离平方修正
# =========================================================
def integrate_peak_area(spec_corr: np.ndarray, left_idx: np.ndarray, right_idx: np.ndarray, freq_axis_mhz: np.ndarray):
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


def normalize_peak_area(peak_area: np.ndarray, lo_power_w: float, pulse_energy_j: float = E_PULSE_J, n_acc: int = N_ACC):
    peak_area = np.asarray(peak_area, dtype=np.float32)
    return peak_area / (lo_power_w * pulse_energy_j * float(n_acc))


def gate_numbers_to_range_m(kept_gate_numbers: np.ndarray, range_res_m: float = RANGE_RES_M, range_offset_m: float = 0.0):
    kept_gate_numbers = np.asarray(kept_gate_numbers, dtype=np.float32)
    return (kept_gate_numbers - 0.5) * range_res_m + range_offset_m


def make_range_corrected_signal(norm_peak_area: np.ndarray, kept_gate_numbers: np.ndarray, range_res_m: float = RANGE_RES_M, range_offset_m: float = 0.0):
    ranges_m = gate_numbers_to_range_m(kept_gate_numbers, range_res_m, range_offset_m)
    rcs = norm_peak_area * (ranges_m[None, :] ** 2)
    return rcs, ranges_m


# =========================================================
# SNR 计算
# =========================================================
def compute_noise_stats_from_raw_roi(spec_roi_raw: np.ndarray, noise_gate_idx: int = 0):
    """
    使用未矫正的第1门 ROI 频谱估计噪声统计量。
    返回每个径向的每-bin 平均噪声功率与标准差。
    """
    spec_roi_raw = np.asarray(spec_roi_raw, dtype=np.float64)
    noise_roi = spec_roi_raw[:, noise_gate_idx, :]  # (N_radial, N_freq_roi)

    noise_mean_per_bin = np.nanmean(noise_roi, axis=1)
    noise_std_per_bin = np.nanstd(noise_roi, axis=1, ddof=0)

    return {
        "noise_mean_per_bin": noise_mean_per_bin.astype(np.float32),
        "noise_std_per_bin": noise_std_per_bin.astype(np.float32),
    }


def compute_peak_snr_from_peaksum(
    peak_sum: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    noise_mean_per_bin: np.ndarray,
    noise_std_per_bin: np.ndarray,
    eps: float = 1e-12,
):
    """
    基于“矫正后峰区积分 / 未矫正首门 ROI 噪声”计算峰区 SNR。

    返回：
    - n_peak_bins
    - noise_power_equiv
    - peak_snr_linear
    - peak_snr_db
    - peak_sigma_score
    """
    peak_sum = np.asarray(peak_sum, dtype=np.float64)
    left_idx = np.asarray(left_idx, dtype=np.int64)
    right_idx = np.asarray(right_idx, dtype=np.int64)
    noise_mean_per_bin = np.asarray(noise_mean_per_bin, dtype=np.float64)
    noise_std_per_bin = np.asarray(noise_std_per_bin, dtype=np.float64)

    n_peak_bins = np.where(
        (left_idx >= 0) & (right_idx >= left_idx),
        right_idx - left_idx + 1,
        0,
    ).astype(np.int32)

    noise_power_equiv = noise_mean_per_bin[:, None] * n_peak_bins
    peak_snr_linear = peak_sum / np.maximum(noise_power_equiv, eps)
    peak_snr_db = 10.0 * np.log10(np.maximum(peak_snr_linear, eps))

    peak_sigma_score = peak_sum / np.maximum(
        noise_std_per_bin[:, None] * np.sqrt(np.maximum(n_peak_bins, 1)),
        eps,
    )

    invalid = (n_peak_bins <= 0) | (~np.isfinite(peak_sum))
    peak_snr_linear[invalid] = np.nan
    peak_snr_db[invalid] = np.nan
    peak_sigma_score[invalid] = np.nan
    noise_power_equiv[invalid] = np.nan

    return {
        "n_peak_bins": n_peak_bins,
        "noise_power_equiv": noise_power_equiv.astype(np.float32),
        "peak_snr_linear": peak_snr_linear.astype(np.float32),
        "peak_snr_db": peak_snr_db.astype(np.float32),
        "peak_sigma_score": peak_sigma_score.astype(np.float32),
    }


# =========================================================
# QC 掩膜与推荐产品
# =========================================================
def build_channel_gate_qc_mask(
    peak_snr_linear: np.ndarray,
    peak_snr_db: np.ndarray,
    n_peak_bins: np.ndarray,
    min_snr_db: float = GATE_MIN_SNR_DB,
    min_snr_linear: float = GATE_MIN_SNR_LINEAR,
    min_peak_bins: int = MIN_PEAK_BINS,
    max_peak_bins: int = MAX_PEAK_BINS,
):
    peak_snr_linear = np.asarray(peak_snr_linear, dtype=np.float64)
    peak_snr_db = np.asarray(peak_snr_db, dtype=np.float64)
    n_peak_bins = np.asarray(n_peak_bins, dtype=np.int32)

    gate_valid = (
        np.isfinite(peak_snr_db) &
        np.isfinite(peak_snr_linear) &
        (peak_snr_db >= min_snr_db) &
        (peak_snr_linear >= min_snr_linear) &
        (n_peak_bins >= min_peak_bins) &
        (n_peak_bins <= max_peak_bins)
    )
    return gate_valid


def build_combined_profile_qc_mask(
    p_gate_valid: np.ndarray,
    s_gate_valid: np.ndarray,
    low_gate_slice: slice = LOW_GATE_SLICE,
    min_valid_low_gates: int = MIN_VALID_LOW_GATES,
):
    p_gate_valid = np.asarray(p_gate_valid, dtype=bool)
    s_gate_valid = np.asarray(s_gate_valid, dtype=bool)
    combined_gate_valid = p_gate_valid | s_gate_valid
    low_gate_valid_count = np.sum(combined_gate_valid[:, low_gate_slice], axis=1)
    profile_valid = low_gate_valid_count >= min_valid_low_gates
    return {
        "combined_gate_valid": combined_gate_valid,
        "low_gate_valid_count": low_gate_valid_count.astype(np.int32),
        "profile_valid": profile_valid,
    }


def apply_gate_and_profile_qc(data_2d: np.ndarray, gate_valid: np.ndarray, profile_valid: np.ndarray, fill_value: float):
    data_2d = np.asarray(data_2d, dtype=np.float64)
    gate_valid = np.asarray(gate_valid, dtype=bool)
    profile_valid = np.asarray(profile_valid, dtype=bool)

    out = np.array(data_2d, copy=True)
    out[~gate_valid] = fill_value
    out[~profile_valid, :] = fill_value
    return out.astype(np.float32)


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
    spec_p_roi, spec_s_roi, freq_roi, roi_mask = crop_roi(spec_4d, freq_axis, ROI_LOW, ROI_HIGH)

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

    # 峰区积分（raw）
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

    # 归一化 / RCS（raw）
    p_peak_norm = normalize_peak_area(p_peak_area_results["peak_sum"], P_LO_P_W, E_PULSE_J, N_ACC)
    s_peak_norm = normalize_peak_area(s_peak_area_results["peak_sum"], P_LO_S_W, E_PULSE_J, N_ACC)

    p_rcs_raw, range_m = make_range_corrected_signal(
        norm_peak_area=p_peak_norm,
        kept_gate_numbers=correction_results["kept_gate_numbers"],
        range_res_m=RANGE_RES_M,
        range_offset_m=0.0,
    )
    s_rcs_raw, _ = make_range_corrected_signal(
        norm_peak_area=s_peak_norm,
        kept_gate_numbers=correction_results["kept_gate_numbers"],
        range_res_m=RANGE_RES_M,
        range_offset_m=0.0,
    )

    # 噪声统计（未矫正首门）
    p_noise_stats = compute_noise_stats_from_raw_roi(spec_p_roi, noise_gate_idx=0)
    s_noise_stats = compute_noise_stats_from_raw_roi(spec_s_roi, noise_gate_idx=0)

    # 峰区 SNR
    p_snr_results = compute_peak_snr_from_peaksum(
        peak_sum=p_peak_area_results["peak_sum"],
        left_idx=correction_results["p_left_idx"],
        right_idx=correction_results["p_right_idx"],
        noise_mean_per_bin=p_noise_stats["noise_mean_per_bin"],
        noise_std_per_bin=p_noise_stats["noise_std_per_bin"],
    )
    s_snr_results = compute_peak_snr_from_peaksum(
        peak_sum=s_peak_area_results["peak_sum"],
        left_idx=correction_results["s_left_idx"],
        right_idx=correction_results["s_right_idx"],
        noise_mean_per_bin=s_noise_stats["noise_mean_per_bin"],
        noise_std_per_bin=s_noise_stats["noise_std_per_bin"],
    )

    # 通道级门位 QC
    p_gate_valid = build_channel_gate_qc_mask(
        p_snr_results["peak_snr_linear"],
        p_snr_results["peak_snr_db"],
        p_snr_results["n_peak_bins"],
    )
    s_gate_valid = build_channel_gate_qc_mask(
        s_snr_results["peak_snr_linear"],
        s_snr_results["peak_snr_db"],
        s_snr_results["n_peak_bins"],
    )

    # 联合廓线级 QC（P 或 S 任一通道通过，则该门视为联合有效）
    profile_qc = build_combined_profile_qc_mask(
        p_gate_valid=p_gate_valid,
        s_gate_valid=s_gate_valid,
        low_gate_slice=LOW_GATE_SLICE,
        min_valid_low_gates=MIN_VALID_LOW_GATES,
    )

    combined_gate_valid = profile_qc["combined_gate_valid"]
    profile_valid = profile_qc["profile_valid"]
    low_gate_valid_count = profile_qc["low_gate_valid_count"]

    # QC 后产品：zero 版本（便于现有反演脚本继续运行）
    p_peak_norm_qc_zero = apply_gate_and_profile_qc(p_peak_norm, p_gate_valid, profile_valid, fill_value=0.0)
    s_peak_norm_qc_zero = apply_gate_and_profile_qc(s_peak_norm, s_gate_valid, profile_valid, fill_value=0.0)

    p_rcs_qc_zero, _ = make_range_corrected_signal(
        norm_peak_area=p_peak_norm_qc_zero,
        kept_gate_numbers=correction_results["kept_gate_numbers"],
        range_res_m=RANGE_RES_M,
        range_offset_m=0.0,
    )
    s_rcs_qc_zero, _ = make_range_corrected_signal(
        norm_peak_area=s_peak_norm_qc_zero,
        kept_gate_numbers=correction_results["kept_gate_numbers"],
        range_res_m=RANGE_RES_M,
        range_offset_m=0.0,
    )

    # QC 后产品：nan 版本（推荐后续绘图/质量控制使用）
    p_peak_norm_qc_nan = apply_gate_and_profile_qc(p_peak_norm, p_gate_valid, profile_valid, fill_value=np.nan)
    s_peak_norm_qc_nan = apply_gate_and_profile_qc(s_peak_norm, s_gate_valid, profile_valid, fill_value=np.nan)

    p_rcs_qc_nan, _ = make_range_corrected_signal(
        norm_peak_area=np.nan_to_num(p_peak_norm_qc_nan, nan=np.nan),
        kept_gate_numbers=correction_results["kept_gate_numbers"],
        range_res_m=RANGE_RES_M,
        range_offset_m=0.0,
    )
    s_rcs_qc_nan, _ = make_range_corrected_signal(
        norm_peak_area=np.nan_to_num(s_peak_norm_qc_nan, nan=np.nan),
        kept_gate_numbers=correction_results["kept_gate_numbers"],
        range_res_m=RANGE_RES_M,
        range_offset_m=0.0,
    )
    # 恢复 NaN 掩膜
    p_rcs_qc_nan[np.isnan(p_peak_norm_qc_nan)] = np.nan
    s_rcs_qc_nan[np.isnan(s_peak_norm_qc_nan)] = np.nan

    # 为兼容现有反演脚本，字段 p_rcs / s_rcs 保持原始产品；
    # 同时提供推荐使用的 QC 版本。
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
        # raw peak area
        p_peak_sum=p_peak_area_results["peak_sum"],
        p_peak_int_mhz=p_peak_area_results["peak_int_mhz"],
        p_peak_int_hz=p_peak_area_results["peak_int_hz"],
        s_peak_sum=s_peak_area_results["peak_sum"],
        s_peak_int_mhz=s_peak_area_results["peak_int_mhz"],
        s_peak_int_hz=s_peak_area_results["peak_int_hz"],
        # raw normalized / RCS
        p_peak_norm=p_peak_norm,
        s_peak_norm=s_peak_norm,
        range_m=range_m,
        p_rcs=p_rcs_raw,
        s_rcs=s_rcs_raw,
        p_rcs_raw=p_rcs_raw,
        s_rcs_raw=s_rcs_raw,
        # noise statistics
        p_noise_mean_per_bin=p_noise_stats["noise_mean_per_bin"],
        p_noise_std_per_bin=p_noise_stats["noise_std_per_bin"],
        s_noise_mean_per_bin=s_noise_stats["noise_mean_per_bin"],
        s_noise_std_per_bin=s_noise_stats["noise_std_per_bin"],
        # SNR
        p_n_peak_bins=p_snr_results["n_peak_bins"],
        p_noise_power_equiv=p_snr_results["noise_power_equiv"],
        p_peak_snr_linear=p_snr_results["peak_snr_linear"],
        p_peak_snr_db=p_snr_results["peak_snr_db"],
        p_peak_sigma_score=p_snr_results["peak_sigma_score"],
        s_n_peak_bins=s_snr_results["n_peak_bins"],
        s_noise_power_equiv=s_snr_results["noise_power_equiv"],
        s_peak_snr_linear=s_snr_results["peak_snr_linear"],
        s_peak_snr_db=s_snr_results["peak_snr_db"],
        s_peak_sigma_score=s_snr_results["peak_sigma_score"],
        # QC masks
        p_gate_valid=p_gate_valid,
        s_gate_valid=s_gate_valid,
        combined_gate_valid=combined_gate_valid,
        low_gate_valid_count=low_gate_valid_count,
        profile_valid=profile_valid,
        # recommended QC products
        p_peak_norm_qc_zero=p_peak_norm_qc_zero,
        s_peak_norm_qc_zero=s_peak_norm_qc_zero,
        p_peak_norm_qc_nan=p_peak_norm_qc_nan,
        s_peak_norm_qc_nan=s_peak_norm_qc_nan,
        p_rcs_qc_zero=p_rcs_qc_zero,
        s_rcs_qc_zero=s_rcs_qc_zero,
        p_rcs_qc_nan=p_rcs_qc_nan,
        s_rcs_qc_nan=s_rcs_qc_nan,
        source_h5_path=str(h5_path),
        gate_min_snr_db=np.array(GATE_MIN_SNR_DB, dtype=np.float32),
        gate_min_snr_linear=np.array(GATE_MIN_SNR_LINEAR, dtype=np.float32),
        min_peak_bins=np.array(MIN_PEAK_BINS, dtype=np.int32),
        max_peak_bins=np.array(MAX_PEAK_BINS, dtype=np.int32),
        low_gate_start=np.array(LOW_GATE_SLICE.start, dtype=np.int32),
        low_gate_stop=np.array(LOW_GATE_SLICE.stop, dtype=np.int32),
        min_valid_low_gates=np.array(MIN_VALID_LOW_GATES, dtype=np.int32),
    )

    return output_npz_path


# =========================================================
# 批量处理：某一天的文件夹
# =========================================================
def build_output_path(h5_path: str | Path, input_date_dir: str | Path, output_root: str | Path) -> Path:
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
        output_npz_path = build_output_path(h5_path, input_date_dir, output_root)

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
    input_date_dir = r"E:\测风组实验数据\RawData\2024\2024_10_26"
    output_root = r"E:\测风组实验数据\气溶胶反演\预处理_SNR_QC"
    failure_csv_path = r"E:\测风组实验数据\气溶胶反演\预处理_SNR_QC\failed_files_2024_10_26.csv"

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
