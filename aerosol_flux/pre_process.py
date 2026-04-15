"""
功率谱数据预处理：
2026/4/8 Ver.1：
将工控机输出的.h5功率谱数据处理后格式为：频谱范围ROI为80~160MHz、距离门范围4~30（对应280~2000m探测范围），单数据文件扫描径向数不变。已经过直接去噪、谱质心法确定多普勒峰区、峰区积分、功率归一化。
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# =========================
# 需要修改的参数
# =========================
H5_PATH = r"E:\GraduateStu6428\Codes\MATLAB\Codes_Data\data_19_03_02.h5"
OUT_PATH = r"E:\GraduateStu6428\Codes\MATLAB\Codes_Data"


N_BINS = 512                 # 频率bin数
N_GATES_PER_CH = 60          # 每个通道的距离门数
N_CHANNELS = 2               # P、S 两通道
F_MAX_MHZ = 500.0            # 频率范围 0~500 MHz
ROI_LOW = 80.0               # ROI下限
ROI_HIGH = 160.0             # ROI上限

C = 299792458.0
# ===== 系统参数 =====
LAMBDA0_M = 1550e-9          # 波长
F_AOM_HZ = 120e6             # AOM频移
TAU_PULSE_S = 500e-9         # 脉冲宽度
E_PULSE_J = 40e-6            # 单脉冲能量
P_LO_P_W = 1.85e-3           # P通道本振功率
P_LO_S_W = 1.64e-3           # S通道本振功率
PRF_HZ = 1.0e4               # 重复频率
N_ACC = 10000                # 累积脉冲数

RANGE_RES_M = C * TAU_PULSE_S / 2.0   # 75 m

"""
预处理功率谱数据
"""
# 是否启用“默认展平顺序”
# True: 假设原始顺序为 [P60门, S60门] × 512bin
# 如果结果检查异常，再尝试改为 False
USE_DEFAULT_LAYOUT = True


def get_valid_mask(spec_data: np.ndarray) -> np.ndarray:
    """
    根据 specData 判断有效径向。
    数据说明指出各 dataset 共有 101 行，最后一行为 0 表示结束；
    特殊情况下主动停止系统时，数据也可能不足 101 行。
    """
    return np.any(spec_data != 0, axis=1)


def reshape_spec(spec_data: np.ndarray, use_default_layout: bool = True) -> np.ndarray:
    """
    将 specData 从 (N, 61440) 重排为 (N, 2, 60, 512)
    返回维度含义: [径向, 通道(P/S), 距离门, 频率bin]
    """
    n_radials = spec_data.shape[0]

    if use_default_layout:
        # 假设展平顺序为:
        # [P1的512bin, P2的512bin, ..., P60的512bin, S1的512bin, ..., S60的512bin]
        spec_4d = spec_data.reshape(n_radials, N_CHANNELS, N_GATES_PER_CH, N_BINS)
    else:
        # 备用布局：
        # 若原数据是先按频率，再按通道/距离门存储，可尝试这一种
        spec_4d = spec_data.reshape(n_radials, N_BINS, N_CHANNELS, N_GATES_PER_CH).transpose(0, 2, 3, 1)

    return spec_4d


def build_freq_axis() -> np.ndarray:
    """
    构造频率轴。这里采用更常见的数字频率bin定义：
    bin宽度 = 500 / 512 MHz
    """
    df = F_MAX_MHZ / N_BINS
    freq_axis = np.arange(N_BINS) * df
    return freq_axis


def crop_roi(spec_4d: np.ndarray, freq_axis: np.ndarray,
             roi_low: float, roi_high: float):
    """
    在频率维裁剪 ROI
    """
    roi_mask = (freq_axis >= roi_low) & (freq_axis <= roi_high)
    freq_roi = freq_axis[roi_mask]

    # 分离两个通道
    spec_p = spec_4d[:, 0, :, :]          # (N, 60, 512)
    spec_s = spec_4d[:, 1, :, :]          # (N, 60, 512)

    # 裁剪 ROI
    spec_p_roi = spec_p[:, :, roi_mask]   # (N, 60, N_roi)
    spec_s_roi = spec_s[:, :, roi_mask]   # (N, 60, N_roi)

    return spec_p_roi, spec_s_roi, freq_roi, roi_mask

"""
直接相减去噪，并去除没有物理意义的距离门（镜面信号门和超出系统探测能力的门）
"""
def subtract_baseline_noise_and_keep_gates(
    spec_roi: np.ndarray,
    noise_gate_idx: int = 0,
    keep_gate_start: int = 3,
    keep_gate_end: int = 29,
    clip_negative: bool = True
):
    """
    对单通道 ROI 频谱执行：
    1) 以首个距离门作为基底噪声直接相减
    2) 仅保留第4~30个距离门（Python索引即 3~29）

    参数
    ----
    spec_roi : ndarray, shape (N_radial, 60, N_freq)
        单通道 ROI 频谱
    noise_gate_idx : int
        基底噪声门索引，默认第1门（Python索引0）
    keep_gate_start : int
        保留起始门，默认第4门（Python索引3）
    keep_gate_end : int
        保留终止门，默认第30门（Python索引29）
    clip_negative : bool
        去噪后是否将负值截断为0。为保证后续峰值检测与质心计算稳定，默认截断。

    返回
    ----
    spec_corr : ndarray, shape (N_radial, 27, N_freq)
        去噪并裁剪距离门后的频谱
    kept_gate_numbers : ndarray, shape (27,)
        保留的物理距离门编号（1-based），即 [4, 5, ..., 30]
    """
    if spec_roi.ndim != 3:
        raise ValueError("spec_roi 必须为三维数组 (N_radial, N_gate, N_freq)")

    if noise_gate_idx < 0 or noise_gate_idx >= spec_roi.shape[1]:
        raise IndexError("noise_gate_idx 超出距离门范围")
    spec_roi = spec_roi.astype(np.float32, copy=False)

    noise_baseline = spec_roi[:, noise_gate_idx:noise_gate_idx + 1, :]   # (N, 1, Nf)
    spec_corr_all = spec_roi - noise_baseline

    if clip_negative:
        spec_corr_all = np.maximum(spec_corr_all, 0.0)

    spec_corr = spec_corr_all[:, keep_gate_start:keep_gate_end + 1, :]
    kept_gate_numbers = np.arange(keep_gate_start + 1, keep_gate_end + 2)  # 1-based

    return spec_corr, kept_gate_numbers


def find_peak_bounds_by_trend_change(spectrum_1d: np.ndarray):
    """
    对单条功率谱确定多普勒峰左右边界。

    定义说明
    --------
    以谱峰最大值位置 peak_idx 为中心：
    - 向左搜索时，若远离峰顶过程中原本下降的趋势首次转为上升，
      则把该转折前一点视为左边界；
    - 向右搜索时，若远离峰顶过程中原本下降的趋势首次转为上升，
      则把该转折前一点视为右边界。

    这等价于把“第一个趋势变化点”实现为：
    峰两侧从峰顶向外扫描时遇到的第一个局部谷点。

    返回
    ----
    peak_idx : int
        峰值位置
    left_idx : int
        左边界索引
    right_idx : int
        右边界索引
    """
    y = np.asarray(spectrum_1d, dtype=float)

    if y.ndim != 1:
        raise ValueError("spectrum_1d 必须为一维数组")

    if np.all(~np.isfinite(y)) or np.nanmax(y) <= 0:
        return -1, -1, -1

    peak_idx = int(np.nanargmax(y))
    n = y.size

    # ---------- 向左搜索 ----------
    left_idx = 0
    found_left = False
    for i in range(peak_idx - 1, 0, -1):
        # 远离峰顶向左走时，若下一个点重新升高，则当前点视为局部谷点
        if y[i - 1] > y[i]:
            left_idx = i
            found_left = True
            break
    if not found_left:
        left_idx = 0

    # ---------- 向右搜索 ----------
    right_idx = n - 1
    found_right = False
    for i in range(peak_idx + 1, n - 1):
        # 远离峰顶向右走时，若下一个点重新升高，则当前点视为局部谷点
        if y[i + 1] > y[i]:
            right_idx = i
            found_right = True
            break
    if not found_right:
        right_idx = n - 1

    # 保底修正
    if left_idx > peak_idx:
        left_idx = peak_idx
    if right_idx < peak_idx:
        right_idx = peak_idx

    return peak_idx, left_idx, right_idx


def compute_spectral_centroid(freq_axis: np.ndarray, spectrum_1d: np.ndarray, left_idx: int, right_idx: int):
    """
    在多普勒峰范围内计算谱质心频率。
    """
    if left_idx < 0 or right_idx < 0 or right_idx < left_idx:
        return np.nan

    f_seg = freq_axis[left_idx:right_idx + 1]
    p_seg = spectrum_1d[left_idx:right_idx + 1]

    power_sum = np.sum(p_seg)
    if power_sum <= 0:
        return np.nan

    centroid = np.sum(f_seg * p_seg) / power_sum
    return centroid


def detect_doppler_peak_ranges(spec_corr: np.ndarray, freq_axis: np.ndarray):
    """
    对单通道去噪后的频谱批量检测多普勒峰范围，并计算谱质心。

    参数
    ----
    spec_corr : ndarray, shape (N_radial, N_gate_kept, N_freq)
        去噪并保留指定距离门后的频谱
    freq_axis : ndarray, shape (N_freq,)
        ROI 频率轴（MHz）

    返回
    ----
    result : dict
        包含峰位、左右边界、质心频率和峰区掩膜
    """
    n_radial, n_gate, n_freq = spec_corr.shape

    peak_idx = np.full((n_radial, n_gate), -1, dtype=int)
    left_idx = np.full((n_radial, n_gate), -1, dtype=int)
    right_idx = np.full((n_radial, n_gate), -1, dtype=int)
    centroid_freq = np.full((n_radial, n_gate), np.nan, dtype=float)
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

    result = {
        "peak_idx": peak_idx,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "centroid_freq_mhz": centroid_freq,
        "peak_mask": peak_mask
    }
    return result


def plot_one_doppler_peak_example(
    spec_corr: np.ndarray,
    freq_axis: np.ndarray,
    peak_result: dict,
    radial_idx: int,
    gate_idx: int,
    channel_name: str,
    kept_gate_numbers: np.ndarray,
    # save_dir: str
):
    """
    临时可视化某一条功率谱及其多普勒峰范围。
    验证通过后可删除本函数，或在主程序中关闭调用。
    """
    # os.makedirs(save_dir, exist_ok=True)

    y = spec_corr[radial_idx, gate_idx, :]
    p_idx = peak_result["peak_idx"][radial_idx, gate_idx]
    l_idx = peak_result["left_idx"][radial_idx, gate_idx]
    r_idx = peak_result["right_idx"][radial_idx, gate_idx]
    c_freq = peak_result["centroid_freq_mhz"][radial_idx, gate_idx]

    if p_idx < 0 or l_idx < 0 or r_idx < 0:
        print(f"[警告] {channel_name} 通道 radial_idx={radial_idx}, gate_idx={gate_idx} 无有效峰，未绘图")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(freq_axis, y, linewidth=1.5, label="Corrected spectrum")
    plt.axvline(freq_axis[p_idx], linestyle="--", linewidth=1.2, label=f"Peak @ {freq_axis[p_idx]:.3f} MHz")
    plt.axvline(freq_axis[l_idx], linestyle=":", linewidth=1.2, label=f"Left boundary @ {freq_axis[l_idx]:.3f} MHz")
    plt.axvline(freq_axis[r_idx], linestyle=":", linewidth=1.2, label=f"Right boundary @ {freq_axis[r_idx]:.3f} MHz")
    plt.axvspan(freq_axis[l_idx], freq_axis[r_idx], alpha=0.25, label="Doppler peak range")

    if np.isfinite(c_freq):
        plt.axvline(c_freq, linestyle="-.", linewidth=1.2, label=f"Centroid @ {c_freq:.3f} MHz")

    physical_gate = kept_gate_numbers[gate_idx]
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power Spectral Density (a.u.)")
    plt.title(f"{channel_name}-channel Doppler peak example\nRadial={radial_idx}, Gate={physical_gate}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def spectrum_correction_pipeline(
    spec_p_roi: np.ndarray,
    spec_s_roi: np.ndarray,
    freq_roi: np.ndarray,
    # save_dir: str,
    plot_example: bool = True,
    example_channel: str = "P",
    example_radial_idx: int = 10,
    example_gate_idx: int = 0
):
    """
    频谱矫正总流程：
    1) 各通道以首门为基底噪声直接相减
    2) 各通道保留第4~30门
    3) 批量确定多普勒峰范围
    4) 临时绘制一个示例峰区用于验证

    返回
    ----
    result : dict
        便于直接写入 npz
    """
    # ---------- P通道 ----------
    spec_p_corr, kept_gate_numbers = subtract_baseline_noise_and_keep_gates(
        spec_p_roi,
        noise_gate_idx=0,
        keep_gate_start=3,
        keep_gate_end=29,
        clip_negative=True
    )
    peak_result_p = detect_doppler_peak_ranges(spec_p_corr, freq_roi)

    # ---------- S通道 ----------
    spec_s_corr, kept_gate_numbers_s = subtract_baseline_noise_and_keep_gates(
        spec_s_roi,
        noise_gate_idx=0,
        keep_gate_start=3,
        keep_gate_end=29,
        clip_negative=True
    )
    peak_result_s = detect_doppler_peak_ranges(spec_s_corr, freq_roi)

    # 两通道保留门号应一致
    if not np.array_equal(kept_gate_numbers, kept_gate_numbers_s):
        raise RuntimeError("P/S 两通道保留门号不一致")

    if plot_example:
        if example_channel.upper() == "P":
            plot_one_doppler_peak_example(
                spec_corr=spec_p_corr,
                freq_axis=freq_roi,
                peak_result=peak_result_p,
                radial_idx=example_radial_idx,
                gate_idx=example_gate_idx,
                channel_name="P",
                kept_gate_numbers=kept_gate_numbers,
                # save_dir=save_dir
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
                # save_dir=save_dir
            )
        else:
            raise ValueError("example_channel 只能是 'P' 或 'S'")

    result = {
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
    return result

def integrate_peak_area(spec_corr: np.ndarray,
                        left_idx: np.ndarray,
                        right_idx: np.ndarray,
                        freq_axis_mhz: np.ndarray):
    """
    在多普勒峰范围内对峰区谱值积分。

    参数
    ----
    spec_corr : ndarray, shape (N_radial, N_gate, N_freq)
        去噪后的频谱
    left_idx, right_idx : ndarray, shape (N_radial, N_gate)
        峰区左右边界索引
    freq_axis_mhz : ndarray, shape (N_freq,)
        ROI频率轴，单位 MHz

    返回
    ----
    result : dict
        peak_sum        : 不乘频宽的离散求和
        peak_int_mhz    : 乘以 MHz 频宽后的积分
        peak_int_hz     : 乘以 Hz 频宽后的积分
    """
    spec_corr = np.asarray(spec_corr, dtype=np.float32)
    n_radial, n_gate, n_freq = spec_corr.shape

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
        "df_hz": df_hz
    }

def normalize_peak_area(peak_area: np.ndarray,
                        lo_power_w: float,
                        pulse_energy_j: float = E_PULSE_J,
                        n_acc: int = N_ACC):
    """
    对峰区积分量进行系统参数归一化。对本振功率、脉冲能量、累积脉冲数进行归一化。
    """
    peak_area = np.asarray(peak_area, dtype=np.float32)
    norm_area = peak_area / (lo_power_w * pulse_energy_j * float(n_acc))
    return norm_area

def gate_numbers_to_range_m(kept_gate_numbers: np.ndarray,
                            range_res_m: float = RANGE_RES_M,
                            range_offset_m: float = 0.0):
    """
    将物理距离门号转换为名义门中心距离。
    若后续完成系统时延标定，可把 range_offset_m 改为对应修正值。
    """
    kept_gate_numbers = np.asarray(kept_gate_numbers, dtype=np.float32)
    ranges_m = (kept_gate_numbers - 0.5) * range_res_m + range_offset_m
    return ranges_m

def make_range_corrected_signal(norm_peak_area: np.ndarray,
                                kept_gate_numbers: np.ndarray,
                                range_res_m: float = RANGE_RES_M,
                                range_offset_m: float = 0.0):
    """
    构造距离平方修正信号 RCS = norm_peak_area * r^2
    """
    ranges_m = gate_numbers_to_range_m(
        kept_gate_numbers,
        range_res_m=range_res_m,
        range_offset_m=range_offset_m
    )  # shape (N_gate,)

    rcs = norm_peak_area * (ranges_m[None, :] ** 2)
    return rcs, ranges_m

def main():
    with h5py.File(H5_PATH, "r") as f:
        azimuth = f["azimuthAngle"][...]
        los_velo = f["losVeloData"][...]
        spec_data = f["specData"][...]
        timestamp = f["timeStamp"][...]

    # 1. 去除无效径向
    valid_mask = get_valid_mask(spec_data)
    azimuth = azimuth[valid_mask]
    los_velo = los_velo[valid_mask]
    spec_data = spec_data[valid_mask]
    timestamp = timestamp[valid_mask]

    # 2. 重排为 [径向, 通道, 距离门, 频率]
    spec_4d = reshape_spec(spec_data, use_default_layout=USE_DEFAULT_LAYOUT)

    # 3. 构造频率轴并裁剪 80~160 MHz
    freq_axis = build_freq_axis()
    spec_p_roi, spec_s_roi, freq_roi, roi_mask = crop_roi(
        spec_4d, freq_axis, ROI_LOW, ROI_HIGH
    )

    spec_p_roi = spec_p_roi.astype(np.float32, copy=False)
    spec_s_roi = spec_s_roi.astype(np.float32, copy=False)

    correction_results = spectrum_correction_pipeline(
        spec_p_roi=spec_p_roi,
        spec_s_roi=spec_s_roi,
        freq_roi=freq_roi,
        # save_dir=FIG_DIR,
        plot_example=True,  # 验证完成后改为 False
        example_channel="P",  # 也可改成 "S"
        example_radial_idx=10,  # 示例时刻
        example_gate_idx=0  # 保留门中的第1个，即物理第4门
    )

    # 4. 如果后续要与 losVeloData 对齐，可将 P 通道去掉前 3 个距离门
    #    这样得到 57 + 60 = 117 个门，与 losVeloData 一致
    spec_p_roi_for_los = spec_p_roi[:, 3:, :]   # P通道第4~60门
    spec_for_los = np.concatenate([spec_p_roi_for_los, spec_s_roi], axis=1)

    # 5. ===== 峰区积分 =====
    p_peak_area_results = integrate_peak_area(
        spec_corr=correction_results["spec_p_corr"],
        left_idx=correction_results["p_left_idx"],
        right_idx=correction_results["p_right_idx"],
        freq_axis_mhz=freq_roi
    )

    s_peak_area_results = integrate_peak_area(
        spec_corr=correction_results["spec_s_corr"],
        left_idx=correction_results["s_left_idx"],
        right_idx=correction_results["s_right_idx"],
        freq_axis_mhz=freq_roi
    )

    # ===== 本振功率归一化 =====
    p_peak_norm = normalize_peak_area(
        peak_area=p_peak_area_results["peak_sum"],
        lo_power_w=P_LO_P_W,
        pulse_energy_j=E_PULSE_J,
        n_acc=N_ACC
    )

    s_peak_norm = normalize_peak_area(
        peak_area=s_peak_area_results["peak_sum"],
        lo_power_w=P_LO_S_W,
        pulse_energy_j=E_PULSE_J,
        n_acc=N_ACC
    )

    # ===== 距离平方修正 =====
    p_rcs, range_m = make_range_corrected_signal(
        norm_peak_area=p_peak_norm,
        kept_gate_numbers=correction_results["kept_gate_numbers"],
        range_res_m=RANGE_RES_M,
        range_offset_m=0.0
    )

    s_rcs, _ = make_range_corrected_signal(
        norm_peak_area=s_peak_norm,
        kept_gate_numbers=correction_results["kept_gate_numbers"],
        range_res_m=RANGE_RES_M,
        range_offset_m=0.0
    )

    # 6. 保存
    np.savez_compressed(
        OUT_PATH,
        azimuth=azimuth,
        timestamp=timestamp,
        los_velo=los_velo,
        freq_axis_full=freq_axis,
        freq_axis_roi=freq_roi,
        roi_mask=roi_mask,
        spec_p_roi=spec_p_roi,
        spec_s_roi=spec_s_roi,
        **correction_results,

        # p_peak_sum和s_peak_sum: 峰区离散求和结果，即原始的峰区积分量
        p_peak_sum=p_peak_area_results["peak_sum"],
        p_peak_int_mhz=p_peak_area_results["peak_int_mhz"],
        p_peak_int_hz=p_peak_area_results["peak_int_hz"],

        s_peak_sum=s_peak_area_results["peak_sum"],
        s_peak_int_mhz=s_peak_area_results["peak_int_mhz"],
        s_peak_int_hz=s_peak_area_results["peak_int_hz"],

        # p_peak_norm和s_peak_norm：本振与发射条件归一化。适合做P/S两通道比较，也可以做跨时刻、跨数据的比较（以防系统参数变化）
        p_peak_norm=p_peak_norm,
        s_peak_norm=s_peak_norm,

        # p_rcs和s_rcs：距离平方修正，构造相对后向散射系数廓线
        range_m=range_m,
        p_rcs=p_rcs,
        s_rcs=s_rcs,
    )

    print("处理完成")
    print(f"spec_p_corr shape: {correction_results['spec_p_corr'].shape}")
    print(f"spec_s_corr shape: {correction_results['spec_s_corr'].shape}")
    print(f"保留距离门: {correction_results['kept_gate_numbers'][0]} ~ {correction_results['kept_gate_numbers'][-1]}")


if __name__ == "__main__":
    main()
