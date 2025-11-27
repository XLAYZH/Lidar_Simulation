import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from joblib import Parallel, delayed
import os
import re
import time
from mpl_toolkits.mplot3d import Axes3D

# --- 配置开关 ---
FIX_PHYSICS_BUGS = False

# 设置绘图风格
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
zh_font_path = 'C:/Windows/Fonts/simhei.ttf'
if os.path.exists(zh_font_path):
    zh_font = fm.FontProperties(fname=zh_font_path)
else:
    zh_font = fm.FontProperties()


# =============================================================================
# 1. 激光雷达参数类 (LidarParams)
# =============================================================================
class LidarParams:
    def __init__(self):
        self.pulse_acc = 10
        self.direction_los = 16
        self.n_circle = 1
        self.direction_num = self.direction_los * self.n_circle
        self.c = 3e8
        self.wavelength = 1550e-9
        self.pulse_energy = 50e-6
        self.pulse_repeat = 10e3
        self.pulse_width = 500e-9
        self.local_power = 2e-3
        self.frequency_aom = 120e6
        self.sample_rate = 1e9
        self.electric = 1.6e-19
        self.responsitivity = 1
        self.bandwidth = 200e6
        self.K = 1.380649e-23
        self.temperature = 293.15
        self.impedance = 50
        self.elevation_angle = 72
        self.max_detect_z = 3840
        self.delta_T = 1 / self.sample_rate
        self.delta_R = 1
        self.detect_R = np.arange(self.delta_R, self.max_detect_z + self.delta_R, self.delta_R)
        self.real_height = self.detect_R * np.sin(np.radians(self.elevation_angle))
        self.length_height = len(self.real_height)

        round_trip_time = 2 * self.max_detect_z / self.c
        total_samples = int(np.ceil(round_trip_time / self.delta_T)) + 1
        self.time = np.arange(self.delta_T, (total_samples + 1) * self.delta_T, self.delta_T)
        self.Points_per_bin = 512
        self.Range_bin = int(np.floor(len(self.time) / self.Points_per_bin))
        self.FFT_points = 1024
        self.freqs = np.fft.fftfreq(self.FFT_points, 1 / self.sample_rate)[:512]
        self.freqs[0] = 1e-10
        self.system_efficiency = 0.1
        self.telescope_D = 0.12
        self.telescope_S = np.pi * (self.telescope_D / 2) ** 2


# =============================================================================
# 2. 噪声模型类 (NoiseParams)
# =============================================================================
class NoiseParams:
    def __init__(self, params):
        self.params = params
        self.fr = 660e3
        self.Gamma = 40.8e3
        self.A = 2e12
        self.B = 0.1
        try:
            self.nep = np.load('nep_fit_smooth.npy')
        except:
            self.nep = np.zeros(512)
        self.f_low = 10e3
        self.f_high = 200e6

    def calculate_nep_spectrum(self):
        H_f = 1 / np.sqrt(1 + (self.f_low / self.params.freqs) ** 16) / \
              np.sqrt(1 + (self.params.freqs / self.f_high) ** 16)
        responsivity_curve = self.params.responsitivity * H_f
        noise_psd = (self.nep ** 2) * (responsivity_curve ** 2)
        return noise_psd, responsivity_curve  # 返回 PSD 和 响应度曲线

    def calculate_rin_spectrum(self):
        omega = 2 * np.pi * self.params.freqs
        omega_fr = 2 * np.pi * self.fr
        num = self.A + self.B * omega ** 2
        den = (omega_fr ** 2 + self.Gamma ** 2 - omega ** 2) ** 2 + (4 * self.Gamma ** 2 * omega ** 2)

        RIN_val_db = 10 * np.log10(num / den)

        if FIX_PHYSICS_BUGS:
            P_rin_linear = self.params.responsitivity ** 2 * self.params.local_power ** 2 * (10 ** (RIN_val_db / 10))
        else:
            P_rin_db = 10 * np.log10(
                self.params.responsitivity ** 2 * self.params.local_power ** 2 * (num / den) * \
                self.params.freqs * self.params.impedance
            )
            P_rin_linear = 10 ** (P_rin_db / 10)

        return P_rin_linear, RIN_val_db  # 返回 PSD 和 RIN(f)

    def simulate_colored_noise(self, psd):
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(self.params.FFT_points))

        if FIX_PHYSICS_BUGS:
            df = self.params.sample_rate / self.params.FFT_points
            amp_spectrum = np.sqrt(np.concatenate((psd, psd[::-1])) * df) * np.sqrt(self.params.FFT_points)
        else:
            amp_spectrum = np.sqrt(np.concatenate((psd, psd[::-1])))

        noise_freq = amp_spectrum * random_phase
        noise_time = np.fft.ifft(noise_freq).real
        return noise_time

    def get_gaussian_components(self):
        """单独获取散粒噪声和热噪声的功率 (A^2)"""
        shot_power = 2 * self.params.electric * self.params.responsitivity * \
                     self.params.local_power * self.params.bandwidth
        thermal_power = 4 * self.params.K * self.params.temperature * \
                        self.params.bandwidth / self.params.impedance
        return shot_power, thermal_power

    def simulate_gaussian_noise(self):
        shot_p, thermal_p = self.get_gaussian_components()
        total_power = shot_p + thermal_p
        return np.random.normal(0, np.sqrt(total_power), self.params.FFT_points)

    def generate_total_noise(self):
        rin_psd, _ = self.calculate_rin_spectrum()
        rin_noise = self.simulate_colored_noise(rin_psd)
        if not FIX_PHYSICS_BUGS: rin_noise /= np.max(np.abs(rin_noise)) if np.max(np.abs(rin_noise)) > 0 else 1

        gauss_noise = self.simulate_gaussian_noise()
        if not FIX_PHYSICS_BUGS: gauss_noise /= np.max(np.abs(gauss_noise)) if np.max(np.abs(gauss_noise)) > 0 else 1

        nep_psd, _ = self.calculate_nep_spectrum()
        nep_noise = self.simulate_colored_noise(nep_psd)
        if not FIX_PHYSICS_BUGS: nep_noise /= np.max(np.abs(nep_noise)) if np.max(np.abs(nep_noise)) > 0 else 1

        total_noise = rin_noise + gauss_noise + 8 * nep_noise
        return rin_noise[:512], gauss_noise[:512], nep_noise[:512], total_noise[:512]


# =============================================================================
# 3. 激光雷达数据与物理模型类 (LidarData)
# =============================================================================
class LidarData:
    def __init__(self, params, wind_input_mode='constant', wind_file=None):
        self.params = params
        if wind_input_mode == 'constant':
            self.V_m_interp = np.zeros((len(self.params.detect_R), 3))
            self.V_m_interp[:, 2] = 5
        else:
            self.V_m_interp = np.load(wind_file)

        azimuths = np.linspace(0, 360, self.params.direction_los, endpoint=False)
        self.az_record = np.repeat(azimuths, self.params.n_circle)
        self.el_record = np.full(self.params.direction_num, self.params.elevation_angle)
        self.V_los_all = self.calculate_v_los()
        self.beta_profile, self.transmittance = self.calculate_optical_props()
        self.phase_aom = np.exp(-2j * np.pi * self.params.frequency_aom * self.params.time)
        self.freq_shift_wind = -2 * self.V_los_all[:, 0] / self.params.wavelength
        self.phase_wind = np.exp(-2j * np.pi * np.outer(self.freq_shift_wind, self.params.time))

        delay = 2 * self.params.detect_R / self.params.c
        time_diff = self.params.time[:, np.newaxis] - delay[np.newaxis, :]
        # [cite_start]高斯脉冲形状 P_T(t) [cite: 449]
        self.pulse_shape_func = (2 * np.sqrt(np.log(2)) / (np.sqrt(np.pi) * self.params.pulse_width)) * \
                                np.exp(-4 * np.log(2) * (time_diff / self.params.pulse_width) ** 2)
        self.pulse_power_matrix = np.sqrt(self.params.pulse_energy * self.params.pulse_repeat * self.pulse_shape_func)
        self.system_factor = (self.params.telescope_S / self.params.detect_R ** 2) * \
                             self.params.system_efficiency * self.params.delta_R

    def calculate_v_los(self):
        v_los = np.zeros((self.params.length_height, self.params.direction_num))
        for i in range(self.params.direction_num):
            el_rad = np.radians(self.el_record[i])
            az_rad = np.radians(self.az_record[i])
            proj_vec = np.array([np.sin(el_rad), np.cos(el_rad) * np.sin(az_rad), np.cos(el_rad) * np.cos(az_rad)])
            v_los[:, i] = np.dot(self.V_m_interp, proj_vec)
        return v_los

    # def calculate_optical_props(self):
    #     h_km = self.params.real_height / 1000
    #     alpha_m = (8 * np.pi / 3) * 1.54e-3 * (532e-9 / self.params.wavelength) ** 4 * np.exp(-h_km / 7)
    #     alpha_a = 50 * (2.47e-3 * np.exp(-h_km / 2) + 5.13e-6 * np.exp(-(h_km - 20) ** 2 / 36)) * (
    #                 532e-9 / self.params.wavelength)
    #     alpha_total = alpha_m + alpha_a
    #     beta = alpha_m / (8 * np.pi / 3) + alpha_a * 0.02
    #
    #     def integrate_ext(R_target):
    #         return \
    #         quad(lambda r: np.interp(r * np.sin(np.radians(self.params.elevation_angle)) / 1000, h_km, alpha_total), 0,
    #              R_target)[0]
    #
    #     tau = np.array([integrate_ext(r / 1000) for r in self.params.detect_R])
    #     return beta, np.exp(-2 * tau)
    def calculate_optical_props(self):
        """计算大气消光和后向散射 (修正单位版)"""
        h_km = self.params.real_height / 1000

        # 1. 计算基于 km 的系数 (单位: km^-1)
        # Rayleigh
        alpha_m_km = (8 * np.pi / 3) * 1.54e-3 * (532e-9 / self.params.wavelength) ** 4 * np.exp(-h_km / 7)
        # Mie (Aerosol)
        alpha_a_km = 50 * (2.47e-3 * np.exp(-h_km / 2) + 5.13e-6 * np.exp(-(h_km - 20) ** 2 / 36)) * \
                     (532e-9 / self.params.wavelength)

        alpha_total_km = alpha_m_km + alpha_a_km

        # 2. 计算积分光学厚度 (tau)
        # 由于积分变量 r 是 km，alpha 也是 km^-1，所以结果无量纲，无需除以 1000
        def integrate_ext(R_target_km):
            return quad(lambda r_km: np.interp(r_km * np.sin(np.radians(self.params.elevation_angle)),
                                               h_km, alpha_total_km), 0, R_target_km)[0]

        # 注意：传入 detect_R / 1000 (转为 km)
        tau = np.array([integrate_ext(r / 1000) for r in self.params.detect_R])
        transmittance = np.exp(-2 * tau)

        # 3. 计算后向散射系数 beta (单位: km^-1 sr^-1)
        beta_m_km = alpha_m_km / (8 * np.pi / 3)
        beta_a_km = alpha_a_km * 0.02  # 假设雷达比 50sr
        beta_total_km = beta_m_km + beta_a_km

        # [关键修正] 将 beta 转换为 m^-1 sr^-1，以便与雷达方程中的 A(m^2) 和 R(m) 匹配
        beta_total_m = beta_total_km / 1000.0

        return beta_total_m, transmittance


# =============================================================================
# 4. 核心仿真函数
# =============================================================================

def process_range_bin(mm, i_h, c_factor, params, noise_params):
    start, end = mm * params.Points_per_bin, (mm + 1) * params.Points_per_bin
    sig_fragment = i_h[start:end]
    _, _, _, total_noise = noise_params.generate_total_noise()
    sig_noisy = np.real(sig_fragment) + total_noise[:params.Points_per_bin] * c_factor
    sig_padded = np.pad(sig_noisy, (0, params.FFT_points - len(sig_noisy)), 'constant')
    return np.abs(np.fft.fft(sig_padded))[:params.Points_per_bin]


def process_pulse(params, lidar_data, noise_params):
    beta_R = np.interp(params.detect_R, params.detect_R, lidar_data.beta_profile)
    Km_sq_avg = lidar_data.transmittance * beta_R * lidar_data.system_factor
    rand_phase = np.random.uniform(0, 2 * np.pi, len(Km_sq_avg))
    rand_amp = np.random.rayleigh(np.sqrt(Km_sq_avg))
    Km = rand_amp * np.exp(1j * rand_phase)

    signal_sum = np.sum(lidar_data.pulse_power_matrix * Km[np.newaxis, :] * lidar_data.phase_wind.T, axis=1)
    i_h = 2 * np.sqrt(params.local_power) * lidar_data.phase_aom * signal_sum

    sig_ref = i_h[:params.Points_per_bin]
    P_sig = np.sum(np.abs(sig_ref) ** 2)
    _, _, _, noise_ref = noise_params.generate_total_noise()
    P_noise = np.sum(np.abs(noise_ref[:params.Points_per_bin]) ** 2)
    target_SNR_dB = 20
    if P_noise == 0: P_noise = 1e-30
    c_factor = np.sqrt(P_sig / (10 ** (target_SNR_dB / 10) * P_noise))

    spec_accum_local = np.zeros((params.Range_bin, params.Points_per_bin))
    for mm in range(params.Range_bin):
        spec_accum_local[mm, :] = process_range_bin(mm, i_h, c_factor, params, noise_params)
    return spec_accum_local


def run_simulation_direction(idx, date_tag, params, lidar_data, noise_params):
    print(f"Simulating Direction {idx + 1}/{params.direction_num}...")
    freq_shift = -2 * lidar_data.V_los_all[:, idx] / params.wavelength
    lidar_data.phase_wind = np.exp(-2j * np.pi * np.outer(freq_shift, params.time))

    # 注意：如果您只是想画图2.5-2.14，可以将这里的 pulse_acc 临时改为10以加快速度
    results = Parallel(n_jobs=4, backend="threading")(
        delayed(process_pulse)(params, lidar_data, noise_params) for _ in range(params.pulse_acc))
    spec_mean = np.mean(results, axis=0)
    return spec_mean


# =============================================================================
# 5. LidarPlotter 类 (新增模块：绘制所有论文插图)
# =============================================================================
class LidarPlotter:
    """
    专门负责绘制论文中的各类分析图表
    """

    def __init__(self, params, noise_params, lidar_data):
        self.p = params
        self.np = noise_params
        self.ld = lidar_data

    def plot_fig_2_5(self):
        """[Paper Fig 2.5] 高斯脉冲时域分布"""
        print("Generating Fig 2.5 ...")
        # 创建一个更宽的时间轴用于展示完整的脉冲形状
        t_axis = np.linspace(-1000e-9, 1000e-9, 1000)
        # 计算脉冲功率 (W)
        pulse_shape = (2 * np.sqrt(np.log(2)) / (np.sqrt(np.pi) * self.p.pulse_width)) * \
                      np.exp(-4 * np.log(2) * (t_axis / self.p.pulse_width) ** 2)
        P_t = self.p.pulse_energy * pulse_shape  # 这里的积分大约是能量，但不完全等于峰值功率公式，此处仅展示形状
        # 实际上，峰值功率 P0 = E / (tau * 1.06)
        P_t_plot = P_t * 1.0  # 缩放系数

        plt.figure(figsize=(8, 6))
        plt.plot(t_axis * 1e9, P_t_plot, color='tab:red')
        plt.xlabel("time (ns)", fontproperties=zh_font)
        plt.ylabel("Power (W)", fontproperties=zh_font)
        plt.title("图 2.5 高斯脉冲时域分布模型", fontproperties=zh_font)
        plt.grid(True, alpha=0.3)
        plt.xlim(-1000, 1000)
        plt.show()

    def plot_fig_2_6(self):
        """[Paper Fig 2.6] 大气分层模型仿真时域外差信号"""
        print("Generating Fig 2.6 ...")
        # 临时模拟一个单脉冲的理想信号 i_h
        # 为了速度，不调用完整 process_pulse，只计算 i_h 部分
        # 需重新计算 Km
        beta_R = np.interp(self.p.detect_R, self.p.detect_R, self.ld.beta_profile)
        Km_sq_avg = self.ld.transmittance * beta_R * self.ld.system_factor
        rand_phase = np.random.uniform(0, 2 * np.pi, len(Km_sq_avg))
        rand_amp = np.random.rayleigh(np.sqrt(Km_sq_avg))
        Km = rand_amp * np.exp(1j * rand_phase)

        signal_sum = np.sum(self.ld.pulse_power_matrix * Km[np.newaxis, :] * self.ld.phase_wind.T, axis=1)
        i_h = 2 * np.sqrt(self.p.local_power) * self.ld.phase_aom * signal_sum

        dist_axis = self.p.time * self.p.c / 2
        i_h_mA = np.real(i_h) * 1000  # 转 mA

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dist_axis, i_h_mA, color='blue', linewidth=0.5)
        ax.set_xlabel("距离 (m)", fontproperties=zh_font)
        ax.set_ylabel("外差信号 (mA)", fontproperties=zh_font)
        ax.set_title("图 2.6 时域外差电流信号", fontproperties=zh_font)
        ax.set_xlim(0, 4000)
        ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.3)

        # 放大图 inset
        ax_ins = ax.inset_axes([0.5, 0.6, 0.4, 0.3])
        idx_s = np.argmin(np.abs(dist_axis - 960))
        idx_e = np.argmin(np.abs(dist_axis - 1140))
        ax_ins.plot(dist_axis[idx_s:idx_e], i_h_mA[idx_s:idx_e], color='blue', linewidth=0.5)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        plt.show()

    def plot_fig_2_7(self):
        """[Paper Fig 2.7] 散粒噪声和热噪声时域图"""
        print("Generating Fig 2.7 ...")
        shot_p, thermal_p = self.np.get_gaussian_components()

        # 生成 512 点噪声
        shot_noise = np.random.normal(0, np.sqrt(shot_p), 512) * 1e6  # uA
        thermal_noise = np.random.normal(0, np.sqrt(thermal_p), 512) * 1e6  # uA
        t_axis = np.arange(512) * self.p.delta_T * 1e9  # ns

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(t_axis, shot_noise, color='brown')
        ax1.set_title("(a) 散粒噪声", fontproperties=zh_font)
        ax1.set_xlabel("时间 (ns)", fontproperties=zh_font)
        ax1.set_ylabel("电流 (uA)", fontproperties=zh_font)
        ax1.set_xlim(0, 500)
        ax1.set_ylim(-3, 3)
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_axis, thermal_noise, color='orange')
        ax2.set_title("(b) 热噪声", fontproperties=zh_font)
        ax2.set_xlabel("时间 (ns)", fontproperties=zh_font)
        ax2.set_ylabel("电流 (uA)", fontproperties=zh_font)
        ax2.set_xlim(0, 500)
        ax2.set_ylim(-3, 3)
        ax2.grid(True, alpha=0.3)
        plt.show()

    def plot_fig_2_8(self):
        """[Paper Fig 2.8] 高斯白噪声统计特性"""
        print("Generating Fig 2.8 ...")
        # 生成大量样本以绘制直方图
        N_sample = 100000
        shot_p, thermal_p = self.np.get_gaussian_components()
        total_std = np.sqrt(shot_p + thermal_p)
        noise_time = np.random.normal(0, total_std, N_sample)

        # FFT
        noise_freq = np.fft.fft(noise_time)
        # 只取正频率部分(或全部，这里为了直方图统计可以都用)
        real_part = np.real(noise_freq)
        imag_part = np.imag(noise_freq)
        amp_part = np.abs(noise_freq)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # (a) 时域幅值
        axes[0, 0].hist(noise_time * 1e6, bins=50, density=True, color='lightblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel("电流噪声幅值 (uA)", fontproperties=zh_font)
        axes[0, 0].set_title("(a) 时域噪声幅值分布", fontproperties=zh_font)

        # (b) 频域实部
        axes[0, 1].hist(real_part * 1e3, bins=50, density=True, color='lightgreen', edgecolor='black',
                        alpha=0.7)  # 单位随意缩放以便展示
        axes[0, 1].set_xlabel("频域实部 (mA/Hz)", fontproperties=zh_font)
        axes[0, 1].set_title("(b) 频域实部", fontproperties=zh_font)

        # (c) 频域虚部
        axes[1, 0].hist(imag_part * 1e3, bins=50, density=True, color='bisque', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel("频域虚部 (mA/Hz)", fontproperties=zh_font)
        axes[1, 0].set_title("(c) 频域虚部", fontproperties=zh_font)

        # (d) 频域幅值 (瑞利分布)
        axes[1, 1].hist(amp_part * 1e3, bins=50, density=True, color='pink', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel("频域幅值 (mA/Hz)", fontproperties=zh_font)
        axes[1, 1].set_title("(d) 频域幅值", fontproperties=zh_font)

        plt.tight_layout()
        plt.show()

    def plot_fig_2_9(self):
        """[Paper Fig 2.9] RIN 频率特性与PSD"""
        print("Generating Fig 2.9 ...")
        P_rin, RIN_val_db = self.np.calculate_rin_spectrum()
        freqs_mhz = self.p.freqs / 1e6

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.semilogx(freqs_mhz, RIN_val_db, color='red')
        ax1.set_xlabel("Frequency (MHz)", fontproperties=zh_font)
        ax1.set_ylabel("RIN (dB/Hz)", fontproperties=zh_font)
        ax1.set_title("(a) RIN 频率特性", fontproperties=zh_font)
        ax1.set_xlim(0.1, 100)  # 论文范围
        ax1.grid(True, which='both', alpha=0.3)

        ax2.semilogx(freqs_mhz, P_rin, color='red')
        ax2.set_xlabel("Frequency (MHz)", fontproperties=zh_font)
        ax2.set_ylabel("PSD (A^2/Hz)", fontproperties=zh_font)
        ax2.set_title("(b) RIN 功率谱密度", fontproperties=zh_font)
        ax2.set_xlim(0.1, 100)
        ax2.grid(True, which='both', alpha=0.3)
        plt.show()

    def plot_fig_2_11(self):
        """[Paper Fig 2.11] 平衡探测器噪声特性"""
        print("Generating Fig 2.11 ...")
        psd, resp_curve = self.np.calculate_nep_spectrum()
        freqs_mhz = self.p.freqs / 1e6

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

        # (a) NEP (self.np.nep)
        ax1.plot(freqs_mhz, self.np.nep * 1e12, color='red')  # pW/sqrt(Hz)
        ax1.set_xlabel("Frequency (MHz)", fontproperties=zh_font)
        ax1.set_ylabel("NEP (pW/sqrt(Hz))", fontproperties=zh_font)
        ax1.set_title("(a) NEP 频率分布", fontproperties=zh_font)
        ax1.set_xlim(0, 500)
        ax1.grid(True)

        # (b) 响应度曲线 H(f)
        ax2.plot(freqs_mhz, resp_curve, color='red')
        ax2.set_xlabel("Frequency (MHz)", fontproperties=zh_font)
        ax2.set_ylabel("Transfer Function", fontproperties=zh_font)
        ax2.set_title("(b) 探测器带宽响应", fontproperties=zh_font)
        ax2.set_xlim(0, 500)
        ax2.grid(True)

        # (c) PSD
        ax3.plot(freqs_mhz, psd, color='red')
        ax3.set_xlabel("Frequency (MHz)", fontproperties=zh_font)
        ax3.set_ylabel("PSD (A^2/Hz)", fontproperties=zh_font)
        ax3.set_title("(c) BDN 功率谱密度", fontproperties=zh_font)
        ax3.set_xlim(0, 500)
        ax3.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_fig_2_12(self):
        """[Paper Fig 2.12] RIN和BDN时域波形"""
        print("Generating Fig 2.12 ...")
        rin_psd, _ = self.np.calculate_rin_spectrum()
        rin_time = self.np.simulate_colored_noise(rin_psd)

        nep_psd, _ = self.np.calculate_nep_spectrum()
        nep_time = self.np.simulate_colored_noise(nep_psd)

        t_axis = np.arange(512) * self.p.delta_T * 1e9

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(t_axis, rin_time[:512] * 1e6, color='teal')
        ax1.set_title("(a) 相对强度噪声 (RIN)", fontproperties=zh_font)
        ax1.set_xlabel("时间 (ns)", fontproperties=zh_font)
        ax1.set_ylabel("电流 (uA)", fontproperties=zh_font)
        ax1.set_xlim(0, 500)
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_axis, nep_time[:512] * 1e6, color='steelblue')
        ax2.set_title("(b) 平衡探测器噪声 (BDN)", fontproperties=zh_font)
        ax2.set_xlabel("时间 (ns)", fontproperties=zh_font)
        ax2.set_ylabel("电流 (uA)", fontproperties=zh_font)
        ax2.set_xlim(0, 500)
        ax2.grid(True, alpha=0.3)
        plt.show()

    def plot_fig_2_13_14(self):
        """[Paper Fig 2.13 & 2.14] 累积脉冲数对噪声PSD的影响"""
        print("Generating Fig 2.13 & 2.14 (Simulation)...")
        acc_list = [1, 10]
        #acc_list = [1, 10, 100, 1000]
        freqs_mhz = self.p.freqs / 1e6

        # 预分配累积数组
        psd_shot_acc = np.zeros((len(acc_list), self.p.Points_per_bin))
        psd_thermal_acc = np.zeros((len(acc_list), self.p.Points_per_bin))
        psd_rin_acc = np.zeros((len(acc_list), self.p.Points_per_bin))
        psd_nep_acc = np.zeros((len(acc_list), self.p.Points_per_bin))

        shot_p, thermal_p = self.np.get_gaussian_components()
        rin_spec, _ = self.np.calculate_rin_spectrum()
        nep_spec, _ = self.np.calculate_nep_spectrum()

        # 模拟最大 1000 次
        current_acc_idx = 0
        # 累积器
        sum_shot = np.zeros(self.p.Points_per_bin)
        sum_thermal = np.zeros(self.p.Points_per_bin)
        sum_rin = np.zeros(self.p.Points_per_bin)
        sum_nep = np.zeros(self.p.Points_per_bin)

        for i in range(1, 11):
            # 生成单次噪声
            shot_t = np.random.normal(0, np.sqrt(shot_p), self.p.FFT_points)
            thermal_t = np.random.normal(0, np.sqrt(thermal_p), self.p.FFT_points)
            rin_t = self.np.simulate_colored_noise(rin_spec)
            nep_t = self.np.simulate_colored_noise(nep_spec)

            # 计算PSD (前512点)
            sum_shot += np.abs(np.fft.fft(shot_t)[:512]) ** 2
            sum_thermal += np.abs(np.fft.fft(thermal_t)[:512]) ** 2
            sum_rin += np.abs(np.fft.fft(rin_t)[:512]) ** 2
            sum_nep += np.abs(np.fft.fft(nep_t)[:512]) ** 2

            if i in acc_list:
                idx = acc_list.index(i)
                psd_shot_acc[idx] = sum_shot / i
                psd_thermal_acc[idx] = sum_thermal / i
                psd_rin_acc[idx] = sum_rin / i
                psd_nep_acc[idx] = sum_nep / i

        # 绘图
        # Fig 2.13
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        colors = ['blue', 'green', 'orange', 'red']
        for i, acc in enumerate(acc_list):
            ax1.plot(freqs_mhz, psd_shot_acc[i], color=colors[i], label=f'Acc={acc}', linewidth=0.8)
        ax1.set_title("Fig 2.13(a) 散粒噪声 PSD", fontproperties=zh_font)
        ax1.set_xlabel("Freq (MHz)", fontproperties=zh_font)
        ax1.set_xlim(0, 500);
        ax1.grid(True, alpha=0.3);
        ax1.legend()

        for i, acc in enumerate(acc_list):
            ax2.plot(freqs_mhz, psd_thermal_acc[i], color=colors[i], label=f'Acc={acc}', linewidth=0.8)
        ax2.set_title("Fig 2.13(b) 热噪声 PSD", fontproperties=zh_font)
        ax2.set_xlabel("Freq (MHz)", fontproperties=zh_font)
        ax2.set_xlim(0, 500);
        ax2.grid(True, alpha=0.3);
        ax2.legend()
        plt.show()

        # Fig 2.14
        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
        for i, acc in enumerate(acc_list):
            ax3.plot(freqs_mhz, psd_rin_acc[i], color=colors[i], label=f'Acc={acc}', linewidth=0.8)
        ax3.set_title("Fig 2.14(a) RIN PSD", fontproperties=zh_font)
        ax3.set_xlabel("Freq (MHz)", fontproperties=zh_font)
        ax3.set_xlim(0, 500);
        ax3.grid(True, alpha=0.3);
        ax3.legend()

        for i, acc in enumerate(acc_list):
            ax4.plot(freqs_mhz, psd_nep_acc[i], color=colors[i], label=f'Acc={acc}', linewidth=0.8)
        ax4.set_title("Fig 2.14(b) BDN PSD", fontproperties=zh_font)
        ax4.set_xlabel("Freq (MHz)", fontproperties=zh_font)
        ax4.set_xlim(0, 500);
        ax4.grid(True, alpha=0.3);
        ax4.legend()
        plt.show()

    def plot_fig_2_16(self, spec_data):
        """[Paper Fig 2.16] 3D 功率谱"""
        print("Generating Fig 2.16 ...")
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        X = self.p.freqs / 1e6  # MHz
        Y = np.arange(self.p.Range_bin)
        X_grid, Y_grid = np.meshgrid(X, Y)

        surf = ax.plot_surface(X_grid, Y_grid, spec_data, cmap='jet', edgecolor='none')

        ax.set_title("图 2.16 功率谱密度 3D 视图", fontproperties=zh_font)
        ax.set_xlabel('频率 (MHz)', fontproperties=zh_font)
        ax.set_ylabel('距离门', fontproperties=zh_font)
        ax.set_zlabel('PSD (A^2/Hz)', fontproperties=zh_font)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()

    def plot_fig_2_17(self, spec_data):
        """[Paper Fig 2.17] 归一化 2D 谱"""
        print("Generating Fig 2.17 ...")
        spec_norm = (spec_data - spec_data.min(axis=1, keepdims=True)) / \
                    (spec_data.max(axis=1, keepdims=True) - spec_data.min(axis=1, keepdims=True))
        freqs_mhz = self.p.freqs / 1e6
        plt.figure(figsize=(10, 8))
        plt.imshow(spec_norm, aspect='auto', origin='lower', cmap='jet',
                   extent=[freqs_mhz[0], freqs_mhz[-1], 0, self.p.Range_bin])
        plt.title("图 2.17 归一化功率谱密度 (常风速)", fontproperties=zh_font, fontsize=14)
        plt.xlabel("频率 (MHz)", fontproperties=zh_font)
        plt.ylabel("距离门", fontproperties=zh_font)
        plt.colorbar(label='Normalized PSD')
        plt.show()


if __name__ == '__main__':
    MODE = 'constant'
    WIND_FOLDER = 'wind_input'

    params = LidarParams()
    if not os.path.exists('nep_fit_smooth.npy'):
        np.save('nep_fit_smooth.npy', np.ones(512) * 1e-12)
    noise_params = NoiseParams(params)

    if MODE == 'constant':
        lidar_data = LidarData(params, wind_input_mode='constant')

        # --- 1. 生成论文插图 (信号与噪声特性) ---
        plotter = LidarPlotter(params, noise_params, lidar_data)

        # 按需取消注释以生成特定图表
        plotter.plot_fig_2_5()  # 高斯脉冲
        plotter.plot_fig_2_6()  # 时域外差信号 (需较长时间生成散斑)
        plotter.plot_fig_2_7()  # 散粒与热噪声时域
        plotter.plot_fig_2_8()  # 噪声统计直方图
        plotter.plot_fig_2_9()  # RIN 频谱
        plotter.plot_fig_2_11()  # NEP 与 BDN 频谱
        plotter.plot_fig_2_12()  # RIN 与 BDN 时域
        plotter.plot_fig_2_13_14()  # 累积脉冲数影响 (耗时)

        # --- 2. 运行主仿真 (生成 Fig 2.16, 2.17) ---
        t0 = time.time()
        # 仅模拟一个方向
        spec_result = run_simulation_direction(0, 'sim_constant_wind', params, lidar_data, noise_params)
        print(f"仿真耗时: {time.time() - t0:.2f} s")

        plotter.plot_fig_2_16(spec_result)
        plotter.plot_fig_2_17(spec_result)

    elif MODE == 'real':
        # (批量处理代码省略，与之前一致)
        pass