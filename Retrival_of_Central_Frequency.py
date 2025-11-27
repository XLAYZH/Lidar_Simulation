import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
from datetime import datetime, timezone, timedelta
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from numpy.polynomial.polynomial import Polynomial


def read_h5_file(file_path):
    """读取HDF5文件并返回文件对象"""
    try:
        return h5py.File(file_path, 'r')
    except Exception as e:
        print(f"文件读取错误: {str(e)}")
        return None

def load_h5_datasets(h5_file):
    """加载HDF5文件中的核心数据集并进行维度转换"""
    data_dict = {}
    try:
        # 加载基本数据集
        data_dict['azimuth'] = np.array(h5_file['azimuthAngle'][:]).squeeze()
        data_dict['velocity'] = np.array(h5_file['losVeloData'][:], dtype=np.float64)

        # 频谱数据转换：二维(101, 61440) → 三维(101, 120, 512)
        spec_2d = np.array(h5_file['specData'][:], dtype=np.uint64)
        data_dict['spectrum_3d'] = spec_2d.reshape((spec_2d.shape[0], 120, 512))

        data_dict['timestamp'] = np.array(h5_file['timeStamp'][:]).squeeze().astype(np.uint64)

        # 删除第 101 个元素/行（索引为 100）
        if data_dict['azimuth'].shape[0] > 100:
            data_dict['azimuth'] = np.delete(data_dict['azimuth'], 100, axis=0)

        if data_dict['velocity'].shape[0] > 100:
            data_dict['velocity'] = np.delete(data_dict['velocity'], 100, axis=0)

        if data_dict['spectrum_3d'].shape[0] > 100:
            data_dict['spectrum_3d'] = np.delete(data_dict['spectrum_3d'], 100, axis=0)

        if data_dict['timestamp'].shape[0] > 100:
            data_dict['timestamp'] = np.delete(data_dict['timestamp'], 100, axis=0)

        # 维度验证
        print("\n数据集加载成功:")
        print(f"方位角: {data_dict['azimuth'].shape}")
        print(f"速度场: {data_dict['velocity'].shape}")
        print(f"频谱数据(三维): {data_dict['spectrum_3d'].shape}")
        print(f"时间戳: {data_dict['timestamp'].shape}")

        return data_dict

    except KeyError as ke:
        print(f"错误: 未找到数据集 - {str(ke)}")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return None

def parse_range_input(input_str):
    """解析用户输入的距离门范围，支持单个值、逗号分隔列表和连续范围"""
    if not input_str.strip():
        print("错误: 输入为空")
        return []

    if '~' in input_str:  # 连续范围格式 (如 "4~8")
        try:
            start, end = map(int, input_str.split('~'))
            if start > end:
                print("错误: 起始值不能大于结束值")
                return []
            return list(range(start, end + 1))
        except ValueError:
            print("错误: 连续范围格式不正确，应为类似 '4~8'")
            return []

    elif ',' in input_str:  # 离散值格式 (如 "4,7,11")
        try:
            values = list(map(int, input_str.split(',')))
            valid_values = [v for v in values if 1 <= v <= 120]
            invalid_values = [v for v in values if v < 1 or v > 120]
            if invalid_values:
                print(f"警告: 以下距离门超出有效范围(1-120)，已跳过: {invalid_values}")
            return valid_values
        except ValueError:
            print("错误: 离散值格式不正确，应为类似 '4,7,11'")
            return []

    else:  # 单个值 (如 "5")
        try:
            value = int(input_str)
            if 1 <= value <= 120:
                return [value]
            else:
                print(f"错误: 距离门必须在 1 到 120 之间")
                return []
        except ValueError:
            print("错误: 请输入有效的整数、逗号分隔的多个整数，或类似 '4~8' 的范围")
            return []

def split_channels(spectrum):
    """
    将三维频谱数据划分为 P 通道和 S 通道
    :param spectrum: shape=(100, 120, 512) 的三维数组
    :return: (p_channel, s_channel)
    """
    p_channel = spectrum[:, :60, :]  # 索引 0~59，前60个距离门为P通道
    s_channel = spectrum[:, 60:, :]  # 索引 60~119，后60个距离门为S通道
    return p_channel, s_channel


def denoise_spectrum(p_channel, s_channel):
    """
    对 P 和 S 通道执行去噪处理
    :param p_channel: P 通道数据 shape=(60, 512)
    :param s_channel: S 通道数据 shape=(60, 512)
    :return: (p_cleaned, s_cleaned)
    """
    p_channel = p_channel.astype(np.float64)
    s_channel = s_channel.astype(np.float64)
    # 使用第1个距离门作为基底噪声
    p_cleaned = p_channel[:, 3:, :] - p_channel[:, [0], :]  # 第4个开始减去第一个
    s_cleaned = s_channel[:, 3:, :] - s_channel[:, [0], :]
    return p_cleaned, s_cleaned


def plot_spectrum(data_dict, time_index, range_gates):
    """
    分别在两个图窗中绘制指定时间点和距离门范围的频谱图：
    - 图窗1：原始信号
    - 图窗2：去噪后信号
    频率范围：80-160MHz (对应512个点中的82-164个点)
    """

    # 获取当前时间戳并转换为UTC+8时间
    timestamp = data_dict['timestamp'][time_index]
    utc_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)  # 转换为UTC时间
    utc_plus_8 = utc_time + timedelta(hours=8)  # 加上8小时得到UTC+8时间
    time_str = 'Beijing Time: ' + utc_plus_8.strftime('%Y-%m-%d %H:%M:%S')  # 格式化时间字符串

    # 频率参数设置 (整个频谱0-500MHz)
    total_bandwidth = 500.0  # MHz
    n_points = 512
    df = total_bandwidth / n_points  # 频率分辨率 (约0.9766 MHz/点)

    # 计算80-160MHz对应的索引范围
    start_idx = int(80 / df)  # ~82
    end_idx = int(160 / df) + 1  # ~164
    freqs = np.arange(start_idx, end_idx) * df  # 实际频率值

    # 获取当前时间点的频谱数据（三维数组：shape=(100, 120, 512)）
    spectrum_3d = data_dict['spectrum_3d']

    # 模块化调用：分通道
    p_channel, s_channel = split_channels(spectrum_3d)

    # 模块化调用：去噪
    p_cleaned, s_cleaned = denoise_spectrum(p_channel, s_channel)

    # 创建第一个图窗：原始信号
    plt.figure(figsize=(12, 6))
    for gate in range_gates:
        gate_idx = gate - 1
        if 0 <= gate_idx < 120:
            if gate_idx < 60:
                channel_name = "P通道"
            else:
                channel_name = "S通道"

            spec_slice = spectrum_3d[time_index, gate_idx, start_idx:end_idx]
            plt.plot(freqs, spec_slice, label=f'{channel_name} 距离门 {gate}')
        else:
            print(f"警告: 距离门 {gate} 超出有效范围(1~120)，已跳过")

    plt.title(f"时间点 {time_index} ({time_str}) 原始频谱图\n(方位角: {data_dict['azimuth'][time_index]:.2f}°)")
    plt.xlabel('频率 (MHz)')
    plt.ylabel('信号强度')
    plt.xlim(80, 160)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 创建第二个图窗：去噪后信号
    plt.figure(figsize=(12, 6))
    for gate in range_gates:
        gate_idx = gate - 1
        if 0 <= gate_idx < 120:
            is_valid_gate = False
            cleaned_data = None

            if gate_idx < 60:
                channel_name = "P通道"
                if gate >=4:
                    cleaned_data = p_cleaned[time_index, gate_idx - 3, :]  # 第4个距离门开始
                    is_valid_gate = True
                else:
                    print(f"警告: 距离门 {gate} 不在P通道有效范围内（需 >=4），已跳过")
            else:
                channel_name = "S通道"
                if gate >= 64:
                    cleaned_data = s_cleaned[time_index, gate_idx - 63, :]  # 第63个距离门开始
                    is_valid_gate = True
                else:
                    print(f"警告: 距离门 {gate} 不在S通道有效范围内（需 >=64），已跳过")

            if is_valid_gate:
                spec_slice = cleaned_data[start_idx:end_idx]
                plt.plot(freqs, spec_slice, label=f'{channel_name} 距离门 {gate}', linestyle='--')
        else:
            print(f"警告: 距离门 {gate} 超出有效范围(1-120)，已跳过")

    plt.title(f"时间点 {time_index} ({time_str}) 去噪后频谱图\n(方位角: {data_dict['azimuth'][time_index]:.2f}°)")
    plt.xlabel('频率 (MHz)')
    plt.ylabel('信号强度')
    plt.xlim(80, 160)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 显示图形
    plt.show()

def find_center_frequency_max(spec_slice, freqs):
    """最大值法求中心频率"""
    return round(freqs[np.argmax(spec_slice)], 3)

def find_center_frequency_xdb(spec_slice, freqs, xdb=3):
    """
    使用 X-dB 带宽法查找中心频率（沿主瓣搜索阈值点）
    返回中心频率、左右阈值边界频率（f_low, f_high）
    """
    if np.max(spec_slice) <= 0:
        return np.nan, None, None

    p_max = np.max(spec_slice)
    p_th = p_max / 10 ** (xdb / 10)
    idx_peak = np.argmax(spec_slice)

    # 向左搜索
    left_idx = idx_peak
    while left_idx > 0 and spec_slice[left_idx] >= p_th:
        left_idx -= 1
    f_low = freqs[left_idx]

    # 向右搜索
    right_idx = idx_peak
    while right_idx < len(spec_slice) - 1 and spec_slice[right_idx] >= p_th:
        right_idx += 1
    f_high = freqs[right_idx]

    fc = round((f_low + f_high) / 2, 3)
    return fc, round(f_low, 3), round(f_high, 3)

def find_center_frequency_centroid(spec_slice, freqs):
    """
    使用质心法查找中心频率
    步骤：从峰值向两侧寻找功率开始上升的位置，形成搜索区间，再计算该区间的功率质心
    返回：fc, f_start, f_end
    """
    idx_peak = np.argmax(spec_slice)
    left = idx_peak
    while left > 1 and spec_slice[left - 1] <= spec_slice[left]:
        left -= 1
    f_start = freqs[left]

    right = idx_peak
    while right < len(spec_slice) - 2 and spec_slice[right + 1] <= spec_slice[right]:
        right += 1
    f_end = freqs[right]

    sub_freqs = freqs[left:right+1]
    sub_power = spec_slice[left:right+1]

    numerator = np.sum(sub_freqs * sub_power)
    denominator = np.sum(sub_power)
    fc = numerator / denominator if denominator > 0 else np.nan

    return round(fc, 3), round(f_start, 3), round(f_end, 3)

def  gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

def find_center_frequency_fitting(spec_slice, freqs):
    """
    拟合法：分别使用三种拟合方式寻找峰值（中心频率）
    返回：三个中心频率（poly2, gaussian, spline）
    """
    idx_peak = np.argmax(spec_slice)
    left = max(0, idx_peak - 5)
    right = min(len(freqs), idx_peak + 6)
    x_fit = freqs[left:right]
    y_fit = spec_slice[left:right]

    # ① 二次多项式拟合
    coeffs = np.polyfit(x_fit, y_fit, deg=2)
    p2_fc = -coeffs[1] / (2 * coeffs[0])

    # ② 高斯拟合
    try:
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=[np.max(y_fit), freqs[idx_peak], 1.0])
        g_fc = popt[1]
    except:
        g_fc = np.nan

    # ③ 分段三次样条插值
    spline = CubicSpline(x_fit, y_fit)
    x_dense = np.linspace(x_fit[0], x_fit[-1], 500)
    y_dense = spline(x_dense)
    s_fc = x_dense[np.argmax(y_dense)]

    return round(p2_fc, 3), round(g_fc, 3), round(s_fc, 3), x_dense, y_dense, spline

def process_all_valid_gates(p_cleaned, s_cleaned, start_idx, end_idx):
    """
    对所有有效距离门进行中心频率检索
    :param p_cleaned: 去噪后的P通道数据
    :param s_cleaned: 去噪后的S通道数据
    :param start_idx: 频率范围起始索引
    :param end_idx: 频率范围结束索引
    :return: 包含所有有效距离门中心频率的字典
    """
    results = {}
    # P通道的有效距离门
    for gate in range(4, 61):
        gate_idx = gate - 1
        if 0 <= gate_idx < 60:
            spec_slice = p_cleaned[gate_idx - 3, start_idx:end_idx]
            freqs = np.arange(start_idx, end_idx) * (500.0 / 512)
            center_freqs = []
            for t in range (spec_slice.shape[0]):
                center_freq = find_center_frequency_max(spec_slice[t], freqs)
                center_freqs.append(center_freq)
            results[f"P通道_距离门_{gate}"] = center_freqs

    # S通道的有效距离门
    for gate in range(64, 121):
        gate_idx = gate - 1
        if 60 <= gate_idx < 120:
            spec_slice = s_cleaned[gate_idx - 63, start_idx:end_idx]
            freqs = np.arange(start_idx, end_idx) * (500.0 / 512)
            center_freqs = []
            for t in range (spec_slice.shape[0]):
                center_freq = find_center_frequency_max(spec_slice[t], freqs)
                center_freqs.append(center_freq)
            results[f"S通道_距离门_{gate}"] = center_freqs

    return  results


def plot_cleaned_spectrum_with_center_freq(p_cleaned, s_cleaned, time_index, gate, start_idx, end_idx, method="max"):
    """
    绘制去噪频谱及中心频率标注（支持最大值法、3dB带宽法、质心法）
    """
    total_bandwidth = 500.0  # MHz
    df = total_bandwidth / 512
    freqs = np.arange(start_idx, end_idx) * df

    gate_idx = gate - 1
    if gate_idx < 60 and gate >= 4:
        spec_slice = p_cleaned[time_index, gate_idx - 3, start_idx:end_idx]
        channel_name = "P通道"
    elif gate_idx >= 60 and gate >= 64:
        spec_slice = s_cleaned[time_index, gate_idx - 63, start_idx:end_idx]
        channel_name = "S通道"
    else:
        print(f"无效距离门: {gate}")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, spec_slice, label=f"{channel_name} 距离门 {gate}")

    if method == "max":
        fc = find_center_frequency_max(spec_slice, freqs)
        label = "最大值法"
        plt.axvline(fc, color='red', linestyle='--', label=f"{label}中心频率: {fc:.3f} MHz")

    elif method == "xdb":
        fc, f_low, f_high = find_center_frequency_xdb(spec_slice, freqs, xdb=3)
        label = "3dB带宽法"
        plt.axvline(fc, color='red', linestyle='--', label=f"{label}中心频率: {fc:.3f} MHz")
        plt.axvline(f_low, color='green', linestyle=':', label=f"min f_th: {f_low:.3f} MHz")
        plt.axvline(f_high, color='blue', linestyle=':', label=f"max f_th: {f_high:.3f} MHz")
        plt.axhline(np.max(spec_slice) / 10**(3/10), color='gray', linestyle='--', label='3dB阈值水平')

    elif method == "centroid":
        fc, f_start, f_end = find_center_frequency_centroid(spec_slice, freqs)
        label = "质心法"
        plt.axvline(fc, color='red', linestyle='--', label=f"{label}中心频率: {fc:.3f} MHz")
        plt.axvspan(f_start, f_end, color='lightgreen', alpha=0.3, label="质心搜索带宽")

    elif method == "fit":
        p2_fc, g_fc, s_fc, x_dense, y_dense, spline = find_center_frequency_fitting(spec_slice, freqs)
        label = "拟合法"
        plt.plot(x_dense, y_dense, color='orange', linestyle='-', alpha=0.6, label='三次样条')
        plt.axvline(p2_fc, color='purple', linestyle='--', label=f"多项式拟合fc: {p2_fc:.3f} MHz")
        plt.axvline(g_fc, color='teal', linestyle='--', label=f"高斯拟合fc: {g_fc:.3f} MHz")
        plt.axvline(s_fc, color='brown', linestyle='--', label=f"样条拟合fc: {s_fc:.3f} MHz")

    else:
        raise ValueError("无效的中心频率检索方法")

    plt.title(f"时间点 {time_index} 去噪后频谱（{label}）")
    plt.xlabel("频率 (MHz)")
    plt.ylabel("信号强度")
    plt.xlim(80, 160)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if method == "fit":
        print(f"多项式拟合中心频率: {p2_fc:.3f} MHz")
        print(f"高斯拟合中心频率: {g_fc:.3f} MHz")
        print(f"样条拟合中心频率: {s_fc:.3f} MHz")
    else:
        print(f"{label}中心频率: {fc:.3f} MHz")


def main():
    """主程序：文件处理和数据可视化"""
    # 获取用户输入路径
    user_path = input("请输入HDF5文件路径或包含HDF5文件的文件夹路径: ").strip()
    cleaned_path = user_path.replace('"', '').replace("'", '')

    # 处理文件或文件夹
    if os.path.isfile(cleaned_path) and cleaned_path.endswith('.h5'):
        files = [cleaned_path]
    elif os.path.isdir(cleaned_path):
        files = glob.glob(os.path.join(cleaned_path, '**', '*.h5'), recursive=True)
        if not files:
            print("错误: 未找到HDF5文件")
            return
        print(f"找到 {len(files)} 个HDF5文件")
    else:
        print("错误: 无效路径")
        return

        # 处理每个文件
    for file_path in files:
        print(f"\n处理文件: {os.path.basename(file_path)}")

        # 读取文件
        h5_file = read_h5_file(file_path)
        if h5_file is None:
            continue

            # 加载数据集
        data_dict = load_h5_datasets(h5_file)
        h5_file.close()  # 关闭文件

        if data_dict is None:
            continue

        # 获取用户输入的时间点和距离门
        time_input = input(f"请输入时间点索引 (0-{data_dict['spectrum_3d'].shape[0] - 1}): ")
        range_input = input("请输入距离门 (单个:5, 多个:4,7,11, 范围:4~8): ").strip()

        #解析时间点输入
        try:
            time_index = int(float(time_input.strip()))
            if not (0 <= time_index < data_dict['spectrum_3d'].shape[0]):
                print(f"错误: 时间点索引必须在 0 到 {data_dict['spectrum_3d'].shape[0] - 1} 之间")
                continue
        except ValueError:
            print("错误: 时间点必须为有效数字")
            continue

        # 解析距离门输入
        range_gates = parse_range_input(range_input)
        # 绘制频谱图
        plot_spectrum(data_dict, time_index, range_gates)

        gate_to_plot = input("请输入需要展示中心频率检索算法的距离门示例（范围：4~60；64~120）：")
        gate_to_plot = int(gate_to_plot)
        if not (4 <= gate_to_plot <= 60) or (64 <= gate_to_plot <= 120):
            print("错误: 请输入有效的距离门")
            continue
        total_bandwidth = 500.0  # MHz
        n_points = 512
        df = total_bandwidth / n_points
        start_idx = int(80 / df)  # ~82
        end_idx = int(160 / df) + 1  # ~164

        # 获取当前时间点的频谱数据
        spectrum_3d = data_dict['spectrum_3d']

        # 模块化调用：分通道
        p_channel, s_channel = split_channels(spectrum_3d)

        # 模块化调用：去噪
        p_cleaned, s_cleaned = denoise_spectrum(p_channel, s_channel)

        # 绘制带有中心频率标记的去噪频谱图
        plot_cleaned_spectrum_with_center_freq(p_cleaned, s_cleaned, time_index, gate_to_plot, start_idx, end_idx, "xdb")
        plot_cleaned_spectrum_with_center_freq(p_cleaned, s_cleaned, time_index, gate_to_plot, start_idx, end_idx, "max")
        plot_cleaned_spectrum_with_center_freq(p_cleaned, s_cleaned, time_index, gate_to_plot, start_idx, end_idx, "centroid")
        plot_cleaned_spectrum_with_center_freq(p_cleaned, s_cleaned, time_index, gate_to_plot, start_idx, end_idx, "fit")


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SIMSUN']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
    main()