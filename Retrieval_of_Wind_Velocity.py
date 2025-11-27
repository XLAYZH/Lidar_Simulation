import os
import glob
import h5py
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
import time as time_module


def gaussian(x, a, mu, sigma):
    """高斯函数"""
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def find_center_frequency_max(spec_slice, freqs):
    """最大值法求中心频率"""
    return freqs[np.argmax(spec_slice)]


def find_center_frequency_xdb(spec_slice, freqs, xdb=3):
    """
    使用 X-dB 带宽法查找中心频率
    """
    if np.max(spec_slice) <= 0:
        return np.nan

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

    return (f_low + f_high) / 2


def find_center_frequency_centroid(spec_slice, freqs):
    """
    使用质心法查找中心频率
    """
    idx_peak = np.argmax(spec_slice)
    left = idx_peak
    while left > 1 and spec_slice[left - 1] <= spec_slice[left]:
        left -= 1

    right = idx_peak
    while right < len(spec_slice) - 2 and spec_slice[right + 1] <= spec_slice[right]:
        right += 1

    sub_freqs = freqs[left:right + 1]
    sub_power = spec_slice[left:right + 1]

    numerator = np.sum(sub_freqs * sub_power)
    denominator = np.sum(sub_power)
    return numerator / denominator if denominator > 0 else np.nan


def find_center_frequency_fitting(spec_slice, freqs):
    """
    拟合法：使用二次多项式拟合寻找峰值（中心频率）
    """
    idx_peak = np.argmax(spec_slice)
    left = max(0, idx_peak - 5)
    right = min(len(freqs), idx_peak + 6)
    x_fit = freqs[left:right]
    y_fit = spec_slice[left:right]

    # 二次多项式拟合
    coeffs = np.polyfit(x_fit, y_fit, deg=2)
    return -coeffs[1] / (2 * coeffs[0])


def split_channels(spectrum):
    """
    将三维频谱数据划分为 P 通道和 S 通道
    """
    p_channel = spectrum[:, :60, :]  # 索引 0~59，前60个距离门为P通道
    s_channel = spectrum[:, 60:, :]  # 索引 60~119，后60个距离门为S通道
    return p_channel, s_channel


def denoise_spectrum(p_channel, s_channel):
    """
    对 P 和 S 通道执行去噪处理
    """
    p_channel = p_channel.astype(np.float64)
    s_channel = s_channel.astype(np.float64)
    # 使用第1个距离门作为基底噪声
    p_cleaned = p_channel[:, 3:, :] - p_channel[:, [0], :]  # 第4个开始减去第一个
    s_cleaned = s_channel[:, 3:, :] - s_channel[:, [0], :]
    return p_cleaned, s_cleaned


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

        return data_dict

    except KeyError as ke:
        print(f"错误: 未找到数据集 - {str(ke)}")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return None


def calculate_radial_wind_speed(file_path, method="max"):
    """
    读取HDF5文件，计算每个时间点、每个距离门的径向风速

    参数:
    file_path: HDF5文件路径
    method: 中心频率计算方法 ("max", "xdb", "centroid", "fit")

    返回:
    radial_wind_speed: 径向风速数组 (时间点数, 距离门数)
    timestamps: 时间戳数组
    azimuths: 方位角数组
    """

    # 常量定义
    f0 = 120.0  # MHz
    wavelength = 1550e-9  # 米

    # 读取文件
    h5_file = read_h5_file(file_path)
    if h5_file is None:
        return None, None, None

    # 加载数据
    data_dict = load_h5_datasets(h5_file)
    h5_file.close()

    if data_dict is None:
        return None, None, None

    # 获取频谱数据
    spectrum_3d = data_dict['spectrum_3d']

    # 分通道和去噪
    p_channel, s_channel = split_channels(spectrum_3d)
    p_cleaned, s_cleaned = denoise_spectrum(p_channel, s_channel)

    # 频率参数
    total_bandwidth = 500.0  # MHz
    n_points = 512
    df = total_bandwidth / n_points
    start_idx = int(80 / df)  # ~82
    end_idx = int(160 / df) + 1  # ~164
    freqs = np.arange(start_idx, end_idx) * df

    # 初始化径向风速数组
    n_times = spectrum_3d.shape[0]
    n_gates = 120  # 总距离门数
    radial_wind_speed = np.full((n_times, n_gates), np.nan)

    # 计算每个时间点和距离门的径向风速
    for t in range(n_times):
        for gate in range(n_gates):
            # 确定通道和有效性
            if gate < 60:  # P通道
                if gate < 3:  # 前3个距离门用于去噪参考，跳过
                    continue
                spec_slice = p_cleaned[t, gate - 3, start_idx:end_idx]
                channel_type = "P"
            else:  # S通道
                if gate < 63:  # 前3个距离门用于去噪参考，跳过
                    continue
                spec_slice = s_cleaned[t, gate - 63, start_idx:end_idx]
                channel_type = "S"

            # 计算中心频率
            try:
                if method == "max":
                    fc = find_center_frequency_max(spec_slice, freqs)
                elif method == "xdb":
                    fc = find_center_frequency_xdb(spec_slice, freqs, xdb=3)
                elif method == "centroid":
                    fc = find_center_frequency_centroid(spec_slice, freqs)
                elif method == "fit":
                    fc = find_center_frequency_fitting(spec_slice, freqs)
                else:
                    raise ValueError("无效的中心频率检索方法")

                # 计算径向风速: V_r = (f_r - f_0) * lambda / 2
                if not np.isnan(fc):
                    vr = - (fc - f0) * wavelength / 2 * 1e6  # 转换为 m/s
                    radial_wind_speed[t, gate] = vr

            except Exception as e:
                print(f"时间点 {t}, 距离门 {gate} 计算出错: {str(e)}")
                continue

    return radial_wind_speed, data_dict['timestamp'], data_dict['azimuth']


def calculate_vector_wind_speed(radial_wind_speed, azimuths, timestamps, channel='P'):
    """
    使用VAD方法中的DSWF法计算矢量风速
    参数:
    radial_wind_speed: 径向风速数组 (时间点数, 距离门数)
    azimuths: 方位角数组 (时间点数,)
    timestamps: 时间戳数组 (时间点数,)
    channel: 选择使用的通道 ('P' 或 'S')
    
    返回:
    vector_wind_speed: 矢量风速估计 (时间窗口数, 距离门数, 3)
    window_timestamps: 时间戳数组 (时间窗口数,)
    heights: 高度数组 (距离门数,)
    """
    # 常量定义
    phi = np.radians(72)  # 仰角72度
    c = 299792458  # 光速 m/s
    n_points = 512  # 距离门数
    window_size = 16  # 滑动窗口大小
    step_size = 1  # 步长
    
    # 确定通道范围
    if channel == 'P':
        gate_start, gate_end = 4, 60  # P通道有效距离门(索引3-59，对应距离门4-60)
    elif channel == 'S':
        gate_start, gate_end = 64, 120  # S通道有效距离门(索引63-119，对应距离门64-120)
    else:
        raise ValueError("通道必须是 'P' 或 'S'")
    
    # 计算高度
    # 高度范围是基于1-60个距离门计算的
    heights = np.arange(3, 59) * 0.5 * n_points * 1e-9 * c * np.sin(phi)
    
    # 初始化结果数组
    n_times = radial_wind_speed.shape[0]
    n_gates = gate_end - gate_start
    n_windows = n_times - window_size + 1  # 滑动窗口数量
    
    vector_wind_speed = np.full((n_windows, n_gates, 3), np.nan)
    window_timestamps = np.full(n_windows, np.nan)
    
    # 计算每个距离门的矢量风速
    for gate_idx, gate in enumerate(range(gate_start, gate_end)):
        # 获取当前距离门的所有时间点数据
        vr_data = radial_wind_speed[:, gate]
        
        # 对每个滑动窗口计算矢量风速
        for i in range(n_windows):
            # 获取窗口数据
            window_vr = vr_data[i:i+window_size]
            window_azimuths = azimuths[i:i+window_size]
            
            # 检查是否有足够的有效数据
            valid_mask = ~np.isnan(window_vr)
            if np.sum(valid_mask) < window_size // 2:  # 至少需要一半有效数据
                continue
            
            # 计算S_i向量 [sin(phi), cos(phi)cos(theta), cos(phi)sin(theta)]
            thetas = np.radians(window_azimuths[valid_mask])
            S_matrix = np.column_stack([
                np.full(np.sum(valid_mask), np.sin(phi)),  # z分量
                np.cos(phi) * np.cos(thetas),              # U分量
                np.cos(phi) * np.sin(thetas)               # Y分量
            ])
            
            # 构建方程组并求解
            # V_vector = (Σ S_i*(S_i)')^(-1) * Σ vr(i)*S_i
            A = S_matrix.T @ S_matrix
            b = S_matrix.T @ window_vr[valid_mask]
            
            try:
                # 求解线性方程组
                wind_vector = np.linalg.solve(A, b)
                vector_wind_speed[i, gate_idx, :] = wind_vector
            except np.linalg.LinAlgError:
                # 矩阵奇异，无法求解
                continue
            
            # 计算窗口平均时间戳
            window_timestamps[i] = np.mean(timestamps[i:i+window_size])
    
    return vector_wind_speed, window_timestamps, heights


def calculate_horizontal_wind(vector_wind_speed):
    """
    计算水平风速和风向
    参数:
    vector_wind_speed: 矢量风速估计 (时间窗口数, 距离门数, 3)
    
    返回:
    horizontal_speed: 水平风速 (时间窗口数, 距离门数)
    horizontal_direction: 水平风向 (时间窗口数, 距离门数)
    """
    # U分量 (索引1) 和 Y分量 (索引2)
    U = vector_wind_speed[:, :, 1]
    Y = vector_wind_speed[:, :, 2]
    
    # 水平风速 = sqrt(U^2 + Y^2)
    horizontal_speed = np.sqrt(U**2 + Y**2)
    
    # 水平风向 = mod(270 - arctan(U/Y), 360)
    horizontal_direction = np.mod(270 - np.degrees(np.arctan2(U, Y)), 360)
    
    return horizontal_speed, horizontal_direction


def plot_wind_data(horizontal_speed, horizontal_direction, window_timestamps, heights, channel, output_dir='.'):
    """
    绘制水平风速和风向图像
    参数:
    horizontal_speed: 水平风速 (时间窗口数, 距离门数)
    horizontal_direction: 水平风向 (时间窗口数, 距离门数)
    window_timestamps: 时间戳数组 (时间窗口数,)
    heights: 高度数组 (距离门数,)
    channel: 通道名称 ('P' 或 'S')
    output_dir: 输出目录
    """
    # 将时间戳转换为UTC+8时间（时间戳单位是秒）
    times_utc8 = []
    for ts in window_timestamps:
        # 时间戳是秒单位
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) + timedelta(hours=8)
        times_utc8.append(dt)
    
    # 确保时间数据是单调递增的
    times_utc8 = sorted(times_utc8)
    
    # 创建时间网格和高度网格
    time_grid, height_grid = np.meshgrid(times_utc8, heights, indexing='ij')
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制水平风速图
    # 限制风速范围在0-30 m/s
    speed_clipped = np.clip(horizontal_speed, 0, 30)
    im1 = ax1.pcolormesh(time_grid, height_grid, speed_clipped, cmap='hsv', vmin=0, vmax=30, shading='auto')
    ax1.set_ylabel('Height (m)')
    ax1.set_title(f'{channel} Channel - Horizontal Wind Speed')
    ax1.set_ylim([None, 1500])  # 设置y轴上限为1500米
    
    # 根据时间跨度自适应设置时间轴刻度，限制最大刻度数
    if times_utc8:
        time_span_hours = (times_utc8[-1] - times_utc8[0]).total_seconds() / 3600
        if time_span_hours <= 1:  # 1小时内
            # 限制最多显示12个刻度
            interval_minutes = max(1, int(60 * time_span_hours / 12))
            ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval_minutes))
            time_format = '%H:%M'
        elif time_span_hours <= 6:  # 6小时内
            interval_minutes = max(5, int(60 * time_span_hours / 12))
            ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval_minutes))
            time_format = '%H:%M'
        elif time_span_hours <= 24:  # 24小时内
            interval_hours = max(1, int(time_span_hours / 12))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hours))
            time_format = '%H:%M'
        else:  # 超过24小时
            interval_hours = max(2, int(time_span_hours / 12))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hours))
            time_format = '%m-%d %H:%M'
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Wind Speed (m/s)')
    
    # 绘制水平风向图
    # 对风向数据进行处理
    im2 = ax2.pcolormesh(time_grid, height_grid, horizontal_direction, cmap='twilight', vmin=0, vmax=360, shading='auto')
    ax2.set_xlabel('Time (Beijing)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title(f'{channel} Channel - Horizontal Wind Direction')
    ax2.set_ylim([None, 1500])  # 设置y轴上限为1500米
    
    # 根据时间跨度自适应设置时间轴刻度，限制最大刻度数
    if times_utc8:
        time_span_hours = (times_utc8[-1] - times_utc8[0]).total_seconds() / 3600
        if time_span_hours <= 1:  # 1小时内
            # 限制最多显示12个刻度
            interval_minutes = max(1, int(60 * time_span_hours / 12))
            ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval_minutes))
            time_format = '%H:%M'
        elif time_span_hours <= 6:  # 6小时内
            interval_minutes = max(5, int(60 * time_span_hours / 12))
            ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval_minutes))
            time_format = '%H:%M'
        elif time_span_hours <= 24:  # 24小时内
            interval_hours = max(1, int(time_span_hours / 12))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hours))
            time_format = '%H:%M'
        else:  # 超过24小时
            interval_hours = max(2, int(time_span_hours / 12))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hours))
            time_format = '%m-%d %H:%M'
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Wind Direction')
    cbar2.set_ticks([0, 90, 180, 270, 360])
    cbar2.set_ticklabels(['North', 'West', 'South', 'East', 'North'])

    # 添加时间标注
    if times_utc8:
        date_str = times_utc8[0].strftime('%Y-%m-%d')
        fig.text(0.70, 0.02, f' {date_str}', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # 为底部的日期标注留出空间

    # 保存图像
    plt.savefig(f'{output_dir}/wind_data_{channel}_channel.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_wind_profile_animation(horizontal_speed, window_timestamps, heights, channel, output_dir='.'):
    """
    绘制风廓线随时间变化的动态图，横轴为风速，纵轴为高度
    
    参数:
    horizontal_speed: 水平风速 (时间窗口数, 距离门数)
    window_timestamps: 时间戳数组 (时间窗口数,)
    heights: 高度数组 (距离门数,)
    channel: 通道名称 ('P' 或 'S')
    output_dir: 输出目录
    """
    # 将时间戳转换为UTC+8时间
    times_utc8 = []
    for ts in window_timestamps:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) + timedelta(hours=8)
        times_utc8.append(dt)
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 初始化线条和填充
    line, = ax.plot([], [], 'b-', linewidth=2)
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('Horizontal Wind Speed (m/s)')
    ax.set_ylabel('Height (m)')
    ax.set_title(f'{channel} Channel - Wind Profile Animation')
    ax.set_xlim(0, 30)  # 风速范围0-30 m/s
    ax.set_ylim(0, 1500)  # 限制高度在1500米以内
    ax.grid(True, alpha=0.3)
    
    # 添加时间文本
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    def animate(frame):
        # 清除之前的填充
        for collection in ax.collections:
            collection.remove()
            
        # 获取当前帧的数据
        speed_data = horizontal_speed[frame, :]
        valid_mask = ~np.isnan(speed_data)
        
        # 只显示1500米以下的数据
        if len(heights) > 0:
            height_mask = heights <= 1500
            valid_mask = valid_mask & height_mask
        
        if np.sum(valid_mask) > 0:
            # 绘制风廓线
            line.set_data(speed_data[valid_mask], heights[valid_mask])
            # 填充风廓线与y轴之间的区域
            ax.fill_betweenx(heights[valid_mask], 0, speed_data[valid_mask], 
                             color='blue', alpha=0.3)
        
        # 更新时间文本
        if times_utc8:
            time_str = times_utc8[frame].strftime('%Y-%m-%d %H:%M:%S')
            time_text.set_text(f'Time: {time_str}')
        
        return line, time_text
    
    # 创建动画
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, animate, frames=len(window_timestamps), 
                         interval=200, blit=False, repeat=True)
    
    # 保存动画
    anim.save(f'{output_dir}/wind_profile_{channel}_channel.gif', 
              writer='pillow', fps=5)
    
    # 显示动画
    plt.tight_layout()
    plt.show()
    
    print(f"风廓线动态图已保存为: {output_dir}/wind_profile_{channel}_channel.gif")


def plot_wind_shear_animation(horizontal_direction, window_timestamps, heights, channel, output_dir='.'):
    """
    绘制风切变情况随时间的动态图，横轴为风向，纵轴为高度
    
    参数:
    horizontal_direction: 水平风向 (时间窗口数, 距离门数)
    window_timestamps: 时间戳数组 (时间窗口数,)
    heights: 高度数组 (距离门数,)
    channel: 通道名称 ('P' 或 'S')
    output_dir: 输出目录
    """
    # 将时间戳转换为UTC+8时间
    times_utc8 = []
    for ts in window_timestamps:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) + timedelta(hours=8)
        times_utc8.append(dt)
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 初始化线条
    line, = ax.plot([], [], 'r-', linewidth=2)
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('Wind Direction (degrees)')
    ax.set_ylabel('Height (m)')
    ax.set_title(f'{channel} Channel - Wind Shear Animation')
    ax.set_xlim(0, 360)  # 风向范围0-360度
    ax.set_ylim(0, 1500)  # 限制高度在1500米以内
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xticklabels(['North', 'East', 'South', 'West', 'North'])
    ax.grid(True, alpha=0.3)
    
    # 添加时间文本
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    def animate(frame):
        # 获取当前帧的数据
        direction_data = horizontal_direction[frame, :]
        valid_mask = ~np.isnan(direction_data)
        
        # 只显示1500米以下的数据
        if len(heights) > 0:
            height_mask = heights <= 1500
            valid_mask = valid_mask & height_mask
        
        if np.sum(valid_mask) > 0:
            line.set_data(direction_data[valid_mask], heights[valid_mask])
        
        # 更新时间文本
        if times_utc8:
            time_str = times_utc8[frame].strftime('%Y-%m-%d %H:%M:%S')
            time_text.set_text(f'Time: {time_str}')
        
        return line, time_text
    
    # 创建动画
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, animate, frames=len(window_timestamps),
                         interval=200, blit=False, repeat=True)
    
    # 保存动画
    anim.save(f'{output_dir}/wind_shear_{channel}_channel.gif',
              writer='pillow', fps=5)
    
    # 显示动画
    plt.tight_layout()
    plt.show()
    
    print(f"风切变动态图已保存为: {output_dir}/wind_shear_{channel}_channel.gif")


def analyze_wind_profile(horizontal_speed, heights, time_index, channel, output_dir='.'):
    """
    分析特定时刻的风廓线，验证其是否符合指数率经验模型
    
    参数:
    horizontal_speed: 水平风速 (时间窗口数, 距离门数)
    heights: 高度数组 (距离门数,)
    time_index: 时间索引
    channel: 通道名称 ('P' 或 'S')
    output_dir: 输出目录
    """
    # 获取指定时间的风廓线数据
    speed_data = horizontal_speed[time_index, :]
    valid_mask = ~np.isnan(speed_data)
    
    if np.sum(valid_mask) == 0:
        print("指定时间点没有有效数据")
        return
    
    # 过滤高度在1500米以内的数据
    height_mask = heights <= 1500
    combined_mask = valid_mask & height_mask
    
    if np.sum(combined_mask) == 0:
        print("指定时间点在1500米高度内没有有效数据")
        return
    
    current_speeds = speed_data[combined_mask]
    current_heights = heights[combined_mask]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制实际风廓线
    ax.scatter(current_speeds, current_heights, color='blue', label='Actual Data', s=20)
    
    # 选择300m和800m处的风速进行分析
    # 找到最接近这两个高度的数据点
    idx_300 = np.argmin(np.abs(current_heights - 300))
    idx_800 = np.argmin(np.abs(current_heights - 800))
    
    z1 = current_heights[idx_300]
    z2 = current_heights[idx_800]
    v1 = current_speeds[idx_300]
    v2 = current_speeds[idx_800]
    
    # 计算p值
    if v1 > 0 and v2 > 0 and z1 != z2:
        p = np.log(v1/v2) / np.log(z1/z2)
        p = np.clip(p, 0, 1)  # 限制p值在[0,1]之间
        
        # 使用计算得到的p值绘制仿真风廓线
        # V = V_ref * (Z/Z_ref)^p
        # 以300m处为参考点
        z_ref = z1
        v_ref = v1
        z_model = np.linspace(np.min(current_heights), np.max(current_heights), 100)
        v_model = v_ref * np.power(z_model / z_ref, p)
        
        # 绘制模型曲线
        ax.plot(v_model, z_model, 'r-', linewidth=2, 
                label=f'Model (p={p:.3f})\nV₁={v1:.2f}m/s@{z1:.0f}m\nV₂={v2:.2f}m/s@{z2:.0f}m')
        
        # 标记用于计算的两个点
        ax.scatter([v1, v2], [z1, z2], color='red', s=100, zorder=5, marker='x')
        
        print(f"风廓线分析结果:")
        print(f"  参考点1: 高度={z1:.0f}m, 风速={v1:.2f}m/s")
        print(f"  参考点2: 高度={z2:.0f}m, 风速={v2:.2f}m/s")
        print(f"  计算得到的p值: {p:.3f}")
        print(f"  p值在[0,1]范围内: {'是' if 0 <= p <= 1 else '否'}")
    else:
        print("无法计算p值，因为风速为0或高度相同")
        p = None
    
    # 设置图形属性
    ax.set_xlabel('Horizontal Wind Speed (m/s)')
    ax.set_ylabel('Height (m)')
    ax.set_title(f'{channel} Channel - Wind Profile Analysis at Time Index {time_index}')
    ax.set_ylim(0, 1500)
    ax.set_xlim(0, np.max(current_speeds) * 1.1 if len(current_speeds) > 0 else 30)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f'{output_dir}/wind_profile_analysis_{channel}_time{time_index}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"风廓线分析图已保存为: {output_dir}/wind_profile_analysis_{channel}_time{time_index}.png")
    
    return p


def process_multiple_files(directory_path, method="centroid"):
    """
    处理目录下所有HDF5文件，计算径向风速

    参数:
    directory_path: 包含HDF5文件的目录路径
    method: 中心频率计算方法

    返回:
    所有文件的径向风速数据列表
    """
    # 查找所有HDF5文件
    files = glob.glob(os.path.join(directory_path, '**', '*.h5'), recursive=True)
    if not files:
        print("未找到HDF5文件")
        return []

    results = []
    for file_path in files:
        print(f"处理文件: {os.path.basename(file_path)}")
        wind_speed, timestamps, azimuths = calculate_radial_wind_speed(file_path, method)
        if wind_speed is not None:
            results.append({
                'file': file_path,
                'wind_speed': wind_speed,
                'timestamps': timestamps,
                'azimuths': azimuths
            })

    return results


# 使用示例
if __name__ == "__main__":
    # 处理单个文件或文件夹
    input_path = input("请输入HDF5文件路径或文件夹路径: ").strip()
    input_path = input_path.replace('"', '').replace("'", '')

    if os.path.isfile(input_path):
        # 处理单个文件
        wind_speed, timestamps, azimuths = calculate_radial_wind_speed(input_path, method="centroid")
        if wind_speed is not None:
            print(f"径向风速数据形状: {wind_speed.shape}")
            print(f"时间戳数量: {len(timestamps)}")
            print(f"方位角数量: {len(azimuths)}")
            
            # 使用DSWF法计算矢量风速
            print("正在计算P通道矢量风速...")
            vector_wind_p, window_times_p, heights_p = calculate_vector_wind_speed(wind_speed, azimuths, timestamps, channel='P')
            horizontal_speed_p, horizontal_direction_p = calculate_horizontal_wind(vector_wind_p)
            
            print("正在计算S通道矢量风速...")
            vector_wind_s, window_times_s, heights_s = calculate_vector_wind_speed(wind_speed, azimuths, timestamps, channel='S')
            horizontal_speed_s, horizontal_direction_s = calculate_horizontal_wind(vector_wind_s)
            
            print(f"P通道水平风速形状: {horizontal_speed_p.shape}")
            print(f"P通道水平风向形状: {horizontal_direction_p.shape}")
            print(f"S通道水平风速形状: {horizontal_speed_s.shape}")
            print(f"S通道水平风向形状: {horizontal_direction_s.shape}")
            print(f"P通道高度数组形状: {heights_p.shape}")
            print(f"S通道高度数组形状: {heights_s.shape}")
            
            # 绘制风速和风向图像
            print("正在绘制P通道风速和风向图像...")
            plot_wind_data(horizontal_speed_p, horizontal_direction_p, window_times_p, heights_p, 'P')
            
            print("正在绘制S通道风速和风向图像...")
            plot_wind_data(horizontal_speed_s, horizontal_direction_s, window_times_s, heights_s, 'S')
            
            # 绘制风廓线动态图
            print("正在绘制P通道风廓线动态图...")
            plot_wind_profile_animation(horizontal_speed_p, window_times_p, heights_p, 'P')
            
            print("正在绘制S通道风廓线动态图...")
            plot_wind_profile_animation(horizontal_speed_s, window_times_s, heights_s, 'S')
            
            # 绘制风切变动态图
            print("正在绘制P通道风切变动态图...")
            plot_wind_shear_animation(horizontal_direction_p, window_times_p, heights_p, 'P')
            
            print("正在绘制S通道风切变动态图...")
            plot_wind_shear_animation(horizontal_direction_s, window_times_s, heights_s, 'S')
            
            # 分析特定时刻的风廓线
            mid_time_index =  2 #len(window_times_p) // 2
            print(f"正在分析P通道第{mid_time_index}时刻的风廓线...")
            analyze_wind_profile(horizontal_speed_p, heights_p, mid_time_index, 'P')
            
            print(f"正在分析S通道第{mid_time_index}时刻的风廓线...")
            analyze_wind_profile(horizontal_speed_s, heights_s, mid_time_index, 'S')
            
    elif os.path.isdir(input_path):
        # 处理文件夹中的所有HDF5文件
        h5_files = glob.glob(os.path.join(input_path, "*.h5"))
        if not h5_files:
            print("指定文件夹中未找到HDF5文件")
        else:
            print(f"找到 {len(h5_files)} 个HDF5文件，开始处理...")
            
            # 按文件名排序以确保时间顺序
            h5_files.sort()
            
            # 存储所有文件的数据
            all_p_speeds = []
            all_p_directions = []
            all_p_times = []
            all_s_speeds = []
            all_s_directions = []
            all_s_times = []
            heights_p = None
            heights_s = None
            
            # 依次处理每个文件，显示进度条
            for i, file_path in enumerate(h5_files):
                # 显示进度条
                progress = (i + 1) / len(h5_files)
                bar_length = 40
                block = int(round(bar_length * progress))
                progress_str = f"[{'#' * block}{'-' * (bar_length - block)}] {int(progress * 100)}%"
                print(f"\r处理进度: {progress_str} ({i+1}/{len(h5_files)}) {os.path.basename(file_path)}", end='')
                
                wind_speed, timestamps, azimuths = calculate_radial_wind_speed(file_path, method="centroid")
                if wind_speed is not None:
                    # 使用DSWF法计算矢量风速
                    vector_wind_p, window_times_p, heights_p_temp = calculate_vector_wind_speed(wind_speed, azimuths, timestamps, channel='P')
                    horizontal_speed_p, horizontal_direction_p = calculate_horizontal_wind(vector_wind_p)
                    
                    vector_wind_s, window_times_s, heights_s_temp = calculate_vector_wind_speed(wind_speed, azimuths, timestamps, channel='S')
                    horizontal_speed_s, horizontal_direction_s = calculate_horizontal_wind(vector_wind_s)
                    
                    # 保存高度信息（只需要保存一次）
                    if heights_p is None:
                        heights_p = heights_p_temp
                    if heights_s is None:
                        heights_s = heights_s_temp
                    
                    # 收集数据
                    all_p_speeds.append(horizontal_speed_p)
                    all_p_directions.append(horizontal_direction_p)
                    all_p_times.append(window_times_p)
                    all_s_speeds.append(horizontal_speed_s)
                    all_s_directions.append(horizontal_direction_s)
                    all_s_times.append(window_times_s)
            
            # 完成进度条
            print(f"\r处理进度: [{'#' * 40}] 100% ({len(h5_files)}/{len(h5_files)}) 完成!")
            
            # 合并所有数据
            if all_p_speeds:
                # 合并时间维度的数据
                combined_p_speed = np.concatenate(all_p_speeds, axis=0)
                combined_p_direction = np.concatenate(all_p_directions, axis=0)
                combined_p_times = np.concatenate(all_p_times, axis=0)
                
                combined_s_speed = np.concatenate(all_s_speeds, axis=0)
                combined_s_direction = np.concatenate(all_s_directions, axis=0)
                combined_s_times = np.concatenate(all_s_times, axis=0)
                
                print(f"\n合并后P通道数据形状: {combined_p_speed.shape}")
                print(f"合并后S通道数据形状: {combined_s_speed.shape}")
                
                # 绘制整个文件夹的风速和风向图像
                print("正在绘制整个文件夹的P通道风速和风向图像...")
                plot_wind_data(combined_p_speed, combined_p_direction, combined_p_times, heights_p, 'P_combined')
                
                print("正在绘制整个文件夹的S通道风速和风向图像...")
                plot_wind_data(combined_s_speed, combined_s_direction, combined_s_times, heights_s, 'S_combined')
                
                # # 绘制风廓线动态图
                # print("正在绘制整个文件夹的P通道风廓线动态图...")
                # plot_wind_profile_animation(combined_p_speed, combined_p_times, heights_p, 'P_combined')
                #
                # print("正在绘制整个文件夹的S通道风廓线动态图...")
                # plot_wind_profile_animation(combined_s_speed, combined_s_times, heights_s, 'S_combined')
                #
                # # 绘制风切变动态图
                # print("正在绘制整个文件夹的P通道风切变动态图...")
                # plot_wind_shear_animation(combined_p_direction, combined_p_times, heights_p, 'P_combined')
                #
                # print("正在绘制整个文件夹的S通道风切变动态图...")
                # plot_wind_shear_animation(combined_s_direction, combined_s_times, heights_s, 'S_combined')

                # 分析特定时刻的风廓线（选择中间时刻）
                mid_time_index = len(combined_p_times) // 2
                print(f"正在分析合并数据P通道第{mid_time_index}时刻的风廓线...")
                analyze_wind_profile(combined_p_speed, heights_p, mid_time_index, 'P_combined')
                
                print(f"正在分析合并数据S通道第{mid_time_index}时刻的风廓线...")
                analyze_wind_profile(combined_s_speed, heights_s, mid_time_index, 'S_combined')
            else:
                print("没有成功处理任何文件")
    else:
        print("无效的文件或文件夹路径")
