import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from joblib import Parallel, delayed
import os
import re

# --- Matplotlib 设置 ---
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'SIMSUN'


chinese_font = fm.FontProperties(fname='C:/WINDOWS/Fonts/simsun.ttc') # 确保路径正确

# --- LidarParams (基本不变, 为噪声模块提供参数) ---
class LidarParams:
    def __init__(self):
        self.c = 3e8
        self.wavelength = 1550e-9
        self.local_oscillator_power = 3e-3
        self.aom_frequency = 120e6
        self.sampling_rate = 1e9
        self.time_step = 1 / self.sampling_rate
        self.electron_charge = 1.6e-19
        self.detector_responsivity = 1
        self.bandwidth = 200e6
        self.Boltzmann = 1.380649e-23
        self.temperature = 273.15 + 20
        self.impedance = 50
        self.points_per_bin = 512
        self.fft_length = 1024
        self.freq_bins = np.fft.fftfreq(self.fft_length, d=self.time_step)[:self.points_per_bin]
        self.freq_bins[0] = 1e-10

        # 扫描和范围设置 (为后续模块做准备)
        self.pulse_count = 1000
        self.beams_per_scan = 16
        self.scan_circles = 1
        self.total_beams = self.beams_per_scan * self.scan_circles
        self.elevation_angle = 72
        self.max_range = 3840
        self.range_resolution = 1
        self.range_bins = np.arange(self.range_resolution, self.max_range + self.range_resolution,
                                    self.range_resolution)

        round_trip_time = 2 * self.max_range / self.c
        total_samples = int(np.ceil(round_trip_time / self.time_step)) + 1
        self.time_vector = np.arange(self.time_step, (total_samples + 1) * self.time_step, self.time_step)
        self.time_length = len(self.time_vector)
        self.range_bin_count = int(np.floor(self.time_length / self.points_per_bin))

        self.system_efficiency = 0.1
        self.telescope_diameter = 0.12
        self.telescope_area = np.pi * (self.telescope_diameter / 2) ** 2


# --- NoiseParams (核心修改点) ---
class NoiseParams:
    """
    噪声模型参数和生成功能 (已重构为可配置)
    """

    def __init__(self, lidar_params: LidarParams, noise_config: dict):
        self.lp = lidar_params

        # 存储噪声配置
        self.config = noise_config

        # RIN参数
        self.relaxation_freq = 660e3
        self.damping_rate = 40.8e3
        self.A_param = 2e12
        self.B_param = 0.1

        # NEP噪声参数
        try:
            self.nep_spectrum = np.load('nep_fit_smooth.npy')
        except FileNotFoundError:
            print("警告: 'nep_fit_smooth.npy' 未找到。NEP噪声将不可用。")
            self.nep_spectrum = np.zeros_like(self.lp.freq_bins)
            self.config['include_nep'] = False  # 自动禁用

        self.f_low = 10e3
        self.f_high = 200e6

    def compute_nep_noise_spectrum(self):
        """计算NEP对应的噪声电流功率谱"""
        resp_curve = (self.lp.detector_responsivity /
                      np.sqrt(1 + (self.f_low / self.lp.freq_bins) ** 16) /
                      np.sqrt(1 + (self.lp.freq_bins / self.f_high) ** 16))
        noise_psd = (self.nep_spectrum ** 2) * (resp_curve ** 2)
        return noise_psd

    def generate_nep_noise_time(self):
        """生成NEP带宽内时域噪声电流"""
        psd = self.compute_nep_noise_spectrum()
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(self.lp.fft_length))
        noise_freq = np.sqrt(np.concatenate((psd, psd[::-1]))) * random_phase
        nep_time = np.fft.ifft(noise_freq).real
        return nep_time

    def compute_rin_spectrum(self):
        """计算RIN引起的功率谱"""
        omega = 2 * np.pi * self.lp.freq_bins
        omega_fr = 2 * np.pi * self.relaxation_freq
        num = self.A_param + self.B_param * omega ** 2
        den = (omega_fr ** 2 + self.damping_rate ** 2 - omega ** 2) ** 2 + (4 * self.damping_rate ** 2 * omega ** 2)
        RIN_f = 10 * np.log10(num / den)

        P_rin_power_db = 10 * np.log10(
            (
                        self.lp.detector_responsivity ** 2 * self.lp.local_oscillator_power ** 2 * num / den) * self.lp.freq_bins * self.lp.impedance
        )
        return RIN_f, P_rin_power_db

    def generate_rin_noise_time(self):
        """生成RIN时域噪声电流"""
        _, P_rin_db = self.compute_rin_spectrum()
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(self.lp.fft_length))
        noise_freq = np.sqrt(10 ** (np.concatenate((P_rin_db, P_rin_db[::-1])) / 10)) * random_phase
        return np.fft.ifft(noise_freq).real

    def compute_shot_thermal_noise_current(self):
        """计算高斯（散粒+热）噪声电流标准差"""
        term1 = 2 * self.lp.electron_charge * self.lp.detector_responsivity * self.lp.local_oscillator_power * self.lp.bandwidth
        term2 = 4 * self.lp.Boltzmann * self.lp.temperature * self.lp.bandwidth / self.lp.impedance
        return np.sqrt(term1 + term2)

    def generate_shot_thermal_noise_time(self):
        """生成高斯（散粒+热）噪声时域电流"""
        sigma = self.compute_shot_thermal_noise_current()
        return np.random.normal(0, sigma, self.lp.fft_length)

    def generate_selected_noise(self):
        """
        [新功能] 根据 self.config 灵活生成和组合噪声。
        返回一个包含所选时域噪声分量的字典和组合后的总噪声。
        """
        components = {}
        total_noise = np.zeros(self.lp.fft_length)

        # 1. 生成散粒+热噪声 (高斯白噪声)
        if self.config.get('include_shot_thermal', False):
            shot = self.generate_shot_thermal_noise_time()
            shot_norm = shot / np.max(np.abs(shot))  # 归一化
            components['shot_thermal'] = shot_norm
            total_noise += shot_norm
            print("已生成: 散粒+热噪声")

        # 2. 生成相对强度噪声 (RIN)
        if self.config.get('include_rin', False):
            rin = self.generate_rin_noise_time()
            rin_norm = rin / np.max(np.abs(rin))  # 归一化
            components['rin'] = rin_norm
            total_noise += rin_norm
            print("已生成: 相对强度噪声 (RIN)")

        # 3. 生成平衡探测器噪声 (NEP)
        if self.config.get('include_nep', False):
            nep = self.generate_nep_noise_time()
            nep_norm = nep / np.max(np.abs(nep))  # 归一化
            weight = self.config.get('nep_weight', 8.0)  # 获取权重
            components['nep'] = nep_norm * weight
            total_noise += nep_norm * weight
            print(f"已生成: 平衡探测器噪声 (NEP)，权重: {weight}")

        # 返回时域分量字典和时域总噪声
        # 我们只关心第一个距离门长度（512）的噪声
        return {k: v[:self.lp.points_per_bin] for k, v in components.items()}, total_noise[:self.lp.points_per_bin]


# --- 辅助函数 (为后续模块做准备) ---
def pulse_shape(t, width):
    return (2 * np.sqrt(np.log(2)) / (np.sqrt(np.pi) * width)) * np.exp(-4 * np.log(2) * t ** 2 / width ** 2)


def compute_line_of_sight_velocity(V_field, azimuths, elevations, params):
    # (与上一版相同)
    los = np.zeros((len(params.real_heights), params.total_beams))
    for idx in range(params.total_beams):
        sv = np.array([
            np.sin(np.radians(elevations[idx])),  # 对应 w
            np.cos(np.radians(elevations[idx])) * np.sin(np.radians(azimuths[idx])),  # 对应 v
            np.cos(np.radians(elevations[idx])) * np.cos(np.radians(azimuths[idx]))  # 对应 u
        ])
        v_los = np.dot(V_field, sv)
        los[:, idx] = v_los
    return los


# --- 信号处理 (为后续模块做准备) ---
def process_single_pulse(params: LidarParams, simulator: 'LidarSimulator', noise_model: NoiseParams,
                         wind_phase: np.ndarray):
    """
    处理单个脉冲 (已更新，以使用 noise_model 中的灵活配置)
    """
    # 随机相位与振幅 (模拟散斑效应)
    rand_phase = np.random.uniform(0, 2 * np.pi, len(simulator.scatter_power_factor))
    rand_amp = np.random.rayleigh(np.sqrt(simulator.scatter_power_factor))
    K_complex = rand_amp * np.exp(1j * rand_phase)

    # 快速计算电流 (i_h)
    ih = (2 * params.detector_responsivity * np.sqrt(
        params.local_oscillator_power) * simulator.aom_phase_shift * np.sum(
        simulator.pulse_power_matrix * K_complex[np.newaxis, :] * wind_phase.T, axis=1))

    spec_fft_accumulated_local = np.zeros((params.range_bin_count, params.points_per_bin))

    # [核心修改]
    # 在脉冲级别生成一次噪声（包含所有点，512*N个），而不是在循环内（N次，每次512个）
    # 这更高效，因为FFT和IFFT在长序列上更高效
    # 但为了模块清晰，我们遵循参考代码的逻辑：在每个距离门循环中生成噪声

    # 计算第一个距离门的 c_factor (SNR 归一化因子)
    signal_power_ref = np.sum(np.abs(ih[:params.points_per_bin]) ** 2)
    # 临时生成一次噪声以获取功率
    _, total_noise_ref = noise_model.generate_selected_noise()
    noise_power_ref = np.sum(np.abs(total_noise_ref) ** 2)
    target_k_noise = 20  # 目标信噪比 (dB)
    if noise_power_ref < 1e-30: noise_power_ref = 1e-30
    c_factor = np.sqrt(signal_power_ref / (10 ** (target_k_noise / 10) * noise_power_ref))

    for mm in range(params.range_bin_count):
        start_mm = mm * params.points_per_bin
        end_mm = (mm + 1) * params.points_per_bin

        signal_fragment = ih[start_mm:end_mm]

        # [核心修改]
        # 根据配置动态生成噪声
        noise_components, total_noise = noise_model.generate_selected_noise()

        # 加入噪声并FFT
        combined = np.real(signal_fragment) + total_noise * c_factor

        zero_padding = params.fft_length - len(combined)
        padded = np.pad(combined, (0, zero_padding), 'constant')

        spec_fft = np.abs(np.fft.fft(padded, n=params.fft_length))[:params.points_per_bin]
        spec_fft_accumulated_local[mm, :] += spec_fft

    return spec_fft_accumulated_local


# --- LidarSimulator (基本不变, 为后续模块做准备) ---
class LidarSimulator:
    def __init__(self, params: LidarParams, noise_config: dict, wind_data_file: str):
        self.params = params
        # [核心修改] 将 noise_config 传递给 NoiseParams
        self.noise = NoiseParams(params, noise_config)

        try:
            self.wind_velocity_field = np.load(wind_data_file)
        except FileNotFoundError:
            print(f"错误: 风场文件 '{wind_data_file}' 未找到。")
            print("将使用零风场进行替代。")
            # 假设 (w, v, u)
            self.wind_velocity_field = np.zeros((len(self.params.range_bins), 3))

        azimuths = np.linspace(0, 360, self.params.beams_per_scan, endpoint=False)
        self.azimuth_angle_record = np.repeat(azimuths, self.params.scan_circles)
        self.elevation_angle_record = np.full(self.params.total_beams, self.params.elevation_angle)

        self.v_los_all = compute_line_of_sight_velocity(self.wind_velocity_field, self.azimuth_angle_record,
                                                        self.elevation_angle_record, self.params)
        self.backscatter_profile, self.transmittance = self._compute_backscatter_and_transmittance()
        self.aom_phase_shift = np.exp(-2j * np.pi * self.params.aom_frequency * self.params.time_vector)

        delay = 2 * self.params.range_bins / self.params.c
        time_diff = self.params.time_vector[:, np.newaxis] - delay[np.newaxis, :]
        transmit = pulse_shape(time_diff, self.params.pulse_duration)
        self.pulse_power_matrix = np.sqrt(self.params.pulse_energy * self.params.pulse_repetition * transmit)

        self.scatter_power_factor = (
                self.transmittance * self.backscatter_profile *
                (
                            self.params.telescope_area / self.params.range_bins ** 2) * self.params.system_efficiency * self.params.range_resolution
        )

    def _compute_backscatter_and_transmittance(self):
        # (与上一版相同)
        height_km = self.params.real_heights / 1000
        ray_ext = (8 * np.pi / 3) * 1.54e-3 * (532e-9 / self.params.wavelength) ** 4 * np.exp(-height_km / 7)
        aero_ext = 50 * (2.47e-3 * np.exp(-height_km / 2) + 5.13e-6 * np.exp(-(height_km - 20) ** 2 / 36)) * (
                    532e-9 / self.params.wavelength)
        extinction_total = ray_ext + aero_ext

        beta_ray = ray_ext / (8 * np.pi / 3)
        beta_aero = aero_ext * 0.02
        backscatter_profile = beta_ray + beta_aero

        def integrate_ext(h):
            # 沿斜路径积分
            return quad(lambda x: np.interp(x, self.params.range_bins / 1000, extinction_total), 0, h)[0]

        tau = np.array([integrate_ext(h) for h in self.params.range_bins / 1000])
        trans = np.exp(-2 * tau)
        return backscatter_profile, trans

    def simulate_direction(self, beam_index: int, date_tag: str):
        # (与上一版相同)
        print(f"开始处理: 日期 {date_tag}, 方向 {beam_index + 1}/{self.params.total_beams}")
        freq_shift = -2 * self.v_los_all[:, beam_index] / self.params.wavelength
        wind_phase = np.exp(-2j * np.pi * np.outer(freq_shift, self.params.time_vector))

        spectra = Parallel(n_jobs=4)(delayed(process_single_pulse)(
            self.params, self, self.noise, wind_phase
        ) for _ in range(self.params.pulse_count))

        mean_spectrum = np.mean(spectra, axis=0)

        out_folder = os.path.join('result_data', f'{date_tag}')
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f'spectrum_beam_{beam_index}.npy')
        np.save(out_path, mean_spectrum)
        print(f"完成并保存: {out_path}")


# --- 模块一：演示主程序 ---
if __name__ == '__main__':

    print("--- 模块一：灵活的噪声生成与展示 ---")

    # 1. 定义噪声配置
    # 您可以修改这里来选择不同的噪声组合
    noise_config = {
        'include_shot_thermal': True,  # 模拟散粒+热噪声
        'include_rin': True,  # 模拟相对强度噪声
        'include_nep': True,  # 模拟平衡探测器噪声
        'nep_weight': 8.0  # NEP 噪声的权重 (来自 new_simulation_fryj.py)
    }

    # 仅模拟 高斯白噪声 (散粒+热)
    # noise_config = { 'include_shot_thermal': True }

    # 仅模拟 RIN 和 NEP
    # noise_config = { 'include_rin': True, 'include_nep': True, 'nep_weight': 8.0 }

    # 2. 初始化参数和噪声模型
    params = LidarParams()
    noise_model = NoiseParams(params, noise_config)

    # 3. 生成选定的噪声
    print(f"正在根据配置生成噪声: {noise_config}")
    noise_components, total_noise_time = noise_model.generate_selected_noise()

    # 4. 绘制结果
    print("绘制时域和频域的噪声分量...")

    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"噪声模块仿真 (配置: {noise_config})", fontsize=16)

    # --- 时域图 ---
    ax_time = axes[0, 0]
    time_axis = np.arange(params.points_per_bin) * params.time_step * 1e9  # ns

    if 'shot_thermal' in noise_components:
        ax_time.plot(time_axis, noise_components['shot_thermal'], label='散粒/热噪声 (归一化)', alpha=0.7)
    if 'rin' in noise_components:
        ax_time.plot(time_axis, noise_components['rin'], label='RIN (归一化)', alpha=0.7)
    if 'nep' in noise_components:
        ax_time.plot(time_axis, noise_components['nep'], label=f"NEP (归一化 * {noise_config.get('nep_weight', 1.0)})",
                     alpha=0.7)

    ax_time.set_xlabel("时间 (ns)")
    ax_time.set_ylabel("归一化电流 (任意单位)")
    ax_time.set_title("时域噪声分量")
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)

    # --- 频域图 (PSD) ---
    ax_freq = axes[0, 1]
    freq_axis_mhz = params.freq_bins / 1e6  # MHz


    # 计算PSD (功率谱密度)
    def calculate_psd(time_signal):
        padded = np.pad(time_signal, (0, params.fft_length - len(time_signal)))
        fft_result = np.fft.fft(padded)
        psd = np.abs(fft_result[:params.points_per_bin]) ** 2 / (params.fft_length)  # MATLAB aperiodogram
        return 10 * np.log10(psd + 1e-20)  # 转为 dB


    if 'shot_thermal' in noise_components:
        ax_freq.plot(freq_axis_mhz, calculate_psd(noise_components['shot_thermal']), label='散粒/热噪声 PSD', alpha=0.7)
    if 'rin' in noise_components:
        ax_freq.plot(freq_axis_mhz, calculate_psd(noise_components['rin']), label='RIN PSD', alpha=0.7)
    if 'nep' in noise_components:
        ax_freq.plot(freq_axis_mhz, calculate_psd(noise_components['nep']), label='NEP PSD', alpha=0.7)

    ax_freq.set_xlabel("频率 (MHz)")
    ax_freq.set_ylabel("功率谱密度 (dB, 任意单位)")
    ax_freq.set_title("频域噪声分量 (PSD)")
    ax_freq.legend()
    ax_freq.grid(True, alpha=0.3)
    ax_freq.set_ylim(bottom=-100)  # 调整Y轴范围以便观察

    # --- 总噪声 时域 ---
    ax_total_time = axes[1, 0]
    ax_total_time.plot(time_axis, total_noise_time, label='总噪声 (时域)', color='black')
    ax_total_time.set_xlabel("时间 (ns)")
    ax_total_time.set_ylabel("归一化电流 (任意单位)")
    ax_total_time.set_title("组合总噪声 (时域)")
    ax_total_time.legend()
    ax_total_time.grid(True, alpha=0.3)

    # --- 总噪声 频域 (PSD) ---
    ax_total_freq = axes[1, 1]
    ax_total_freq.plot(freq_axis_mhz, calculate_psd(total_noise_time), label='总噪声 (PSD)', color='black')
    ax_total_freq.set_xlabel("频率 (MHz)")
    ax_total_freq.set_ylabel("功率谱密度 (dB, 任意单位)")
    ax_total_freq.set_title("组合总噪声 (PSD)")
    ax_total_freq.legend()
    ax_total_freq.grid(True, alpha=0.3)
    ax_total_freq.set_ylim(bottom=-100)  # 调整Y轴范围以便观察

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()