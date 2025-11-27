 import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
import os

# --- Matplotlib 设置 ---
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'


# --- 复制 new_simulation_fryj.py 中的核心类 (LidarParams, LidarData) ---

class LidarParams:
    def __init__(self):
        self.pulse_acc = 1000
        self.direction_los = 16
        self.n_circle = 1
        self.direction_num = self.direction_los * self.n_circle

        self.c = 3 * 10 ** 8
        self.wavelength = 1550e-9
        self.pulse_engrgy = 50e-6
        self.pulse_repeat = 10e3
        self.pulse_width = 500e-9
        self.local_power = 3e-3
        self.frequency_aom = 120e6
        self.sample_rate = 1e9

        self.electric = 1.6e-19
        self.responsitivity = 1
        self.band_width = 200e6
        self.K = 1.380649e-23
        self.temperature = 273.15 + 20
        self.impedance = 50
        self.elevation_angle = 72
        self.max_detect_z = 3840
        self.delta_T = 1 / self.sample_rate
        self.delta_R = 1
        self.detect_R = np.arange(self.delta_R, self.max_detect_z + self.delta_R, self.delta_R)

        self.real_height = self.detect_R * np.sin(self.elevation_angle / 180 * np.pi)
        self.length_height = len(self.real_height)

        self.time = np.arange(self.delta_T, (np.ceil(2 * self.max_detect_z / self.c / self.delta_T) + 1) * self.delta_T,
                              self.delta_T)
        self.length_time = len(self.time)
        self.Points_per_bin = 512
        self.Range_bin = int(np.floor(self.length_time / self.Points_per_bin))
        self.FFT_points = 1024
        self.freqs = np.fft.fftfreq(self.FFT_points, 1 / self.sample_rate)[:512]
        self.freqs[0] = 1e-10
        self.system_efficency = 0.1
        self.telescope_D = 120e-3
        self.telescope_S = np.pi * (self.telescope_D / 2) ** 2


# 激光器脉冲传输功率
def transmit_power(t, pulse_width):
    return (2 * np.sqrt(np.log(2)) / (np.sqrt(np.pi) * pulse_width)) * np.exp(
        -4 * np.log(2) * t ** 2 / pulse_width ** 2)


# 计算视线风速
def calculate_v_los(V_m, azimuthangle_record, elevationangle_record, params):
    V_los_all = np.zeros((params.length_height, params.direction_num))
    for j in range(params.direction_num):
        Si = np.array([
            np.sin(np.radians(elevationangle_record[j])),
            np.cos(np.radians(elevationangle_record[j])) * np.sin(np.radians(azimuthangle_record[j])),
            np.cos(np.radians(elevationangle_record[j])) * np.cos(np.radians(azimuthangle_record[j]))
        ])
        V_los = np.dot(V_m, Si)
        V_los_all[:, j] = V_los
    return V_los_all


class LidarData:
    def __init__(self, params):
        self.params = params

        # 激活 LidarData 中的恒定风场 (w=0, v=3, u=-3)
        V_m = np.array([[0, 3, -3] for _ in self.params.detect_R])
        self.V_m_interp = V_m

        azimuthangle = np.linspace(0, 360, self.params.direction_los, endpoint=False)
        self.azimuthangle_record = np.repeat(azimuthangle, self.params.n_circle)
        self.elevationangle_record = np.full(self.params.direction_num, self.params.elevation_angle)

        self.V_los_all = calculate_v_los(self.V_m_interp, self.azimuthangle_record, self.elevationangle_record,
                                         self.params)

        self.backscatter_profile, self.transmittance = self.calculate_backscatter_transmittance()

        self.freqshift_aom = np.exp(-2j * np.pi * self.params.frequency_aom * self.params.time)

        # 我们只模拟第一个方向 (idx=0)
        self.freqshift_wind = -2 * self.V_los_all[:, 0] / self.params.wavelength
        self.b = np.exp(-2j * np.pi * np.outer(self.freqshift_wind, self.params.time))

        aaa = 2 * self.params.detect_R / self.params.c
        time_matrix = self.params.time[:, np.newaxis] - aaa[np.newaxis, :]
        self.power_matrix = np.sqrt(
            self.params.pulse_engrgy * self.params.pulse_repeat * transmit_power(time_matrix, self.params.pulse_width))

        self.K_m_squared = (
                                       self.params.telescope_S / self.params.detect_R ** 2) * self.params.system_efficency * self.params.delta_R

    def calculate_backscatter_transmittance(self):
        # (此函数与 new_simulation_fryj.py 相同)
        height_back = self.params.real_height / 1000
        atmo_extinction = (8 * np.pi / 3) * 1.54e-3 * (532e-9 / self.params.wavelength) ** 4 * np.exp(-height_back / 7)
        aero_extinction = 50 * (2.47e-3 * np.exp(-height_back / 2) + 5.13e-6 * np.exp(
            -(height_back - 20) ** 2 / 36)) * 532e-9 / self.params.wavelength
        extinction_all = atmo_extinction + aero_extinction
        backscatter_rayleigh = atmo_extinction / (8 * np.pi / 3)
        backscatter_aerosol = aero_extinction * 0.02
        backscatter_profile = backscatter_rayleigh + backscatter_aerosol

        # 修正：积分应沿斜距 R (detect_R)，而不是垂直高度 H (real_height)
        def extinction_integral(R_km, extinction_profile_vs_H_km):
            # 我们需要将 R 映射回 H 来查找 extinction
            H_km_interp = R_km * np.sin(self.params.elevation_angle / 180 * np.pi)
            ext_val = np.interp(H_km_interp, height_back, extinction_profile_vs_H_km)
            return ext_val

        optical_depth = np.array(
            [quad(lambda r_km: extinction_integral(r_km, extinction_all), 0, R_km)[0]
             for R_km in self.params.detect_R / 1000])

        transmittance = np.exp(-2 * optical_depth)

        # 修正：K_m_squared 也需要 T^2 和 beta
        # (在 new_simulation_fryj.py 中，K_m_squared 没包含 T 和 beta，但在 process_pulse 中补上了)
        # (在 lidar_sim.py 中， K_m_squared 包含了 T 和 beta)
        # 我们遵循 new_simulation_fryj.py 的逻辑， K_m_squared 不包含 T 和 beta
        # 并在计算 i_h 时再乘上 T 和 beta

        self.K_m_squared = (
                                       self.params.telescope_S / self.params.detect_R ** 2) * self.params.system_efficency * self.params.delta_R

        return backscatter_profile, transmittance


# --- 主执行程序：复现 图2.6 ---
if __name__ == '__main__':
    print("初始化参数和LidarData...")
    params = LidarParams()
    lidar_data = LidarData(params)

    # --- 模拟 'process_pulse' 的核心信号部分 ---
    print("模拟散斑效应 (Km)...")

    # 结合 backscatter 和 transmittance 到 K_m_squared
    # K_m_squared_eff = T^2(R_m) * beta(R_m) * (A/R_m^2) * n_eff * dR
    # (注意: beta 是 H 的函数, T 是 R 的函数)
    # 我们需要将 beta(H) 插值到 R
    beta_interp_R = np.interp(params.detect_R / 1000, params.real_height / 1000, lidar_data.backscatter_profile)

    K_m_squared_effective = lidar_data.transmittance * beta_interp_R * lidar_data.K_m_squared

    # 模拟 Km (随机振幅和相位)
    phase = np.random.uniform(0, 2 * np.pi, len(K_m_squared_effective))
    amplitude = np.random.rayleigh(np.sqrt(K_m_squared_effective))
    Km = amplitude * np.exp(1j * phase)

    print("计算时域外差信号 i_h ...")
    # 计算 i_h (公式 2.17)
    i_h = 2 * np.sqrt(params.local_power) * lidar_data.freqshift_aom * \
          np.sum(lidar_data.power_matrix * Km[np.newaxis, :] * lidar_data.b.T, axis=1)

    # --- 绘图 ---
    print("开始绘图...")

    # X轴：将时间转换为距离
    distance_axis_m = params.time * params.c / 2

    # Y轴：将电流 A 转换为 mA
    i_h_real_mA = np.real(i_h) * 1000

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(distance_axis_m, i_h_real_mA, color='mediumblue', linewidth=0.8)

    # 设置主图
    zh_font = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 指定中文字体路径
    ax.set_xlabel("距离 (m)", fontproperties=zh_font, fontsize=14)
    ax.set_ylabel("外差信号 (mA)", fontproperties=zh_font, fontsize=14)
    ax.set_title("图2.6 (复现): 大气分层模型仿真时域外差电流", fontproperties=zh_font, fontsize=16)
    ax.set_xlim(0, 4000)
    ax.grid(True, alpha=0.3)

    # --- 添加内嵌图 (Inset) ---
    # 对应论文  的 960m - 1140m 范围
    ax_inset = ax.inset_axes([0.5, 0.6, 0.45, 0.35])  # [x, y, width, height]

    # 找到对应的索引
    idx_start = np.argmin(np.abs(distance_axis_m - 960))
    idx_end = np.argmin(np.abs(distance_axis_m - 1140))

    ax_inset.plot(distance_axis_m[idx_start:idx_end], i_h_real_mA[idx_start:idx_end], color='mediumblue', linewidth=1.0)
    ax_inset.set_title("960m - 1140m 放大", fontproperties=zh_font, fontsize=10)
    ax_inset.grid(True, alpha=0.3)

    plt.savefig("fig_2_6_reproduction.png")
    plt.show()