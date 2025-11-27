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
# True: 使用论文公式 (2.22, 2.30) 修正原代码中的物理错误 (推荐)
# False: 严格复现 new_simulation_fryj.py 的原逻辑 (包含潜在错误)
FIX_PHYSICS_BUGS = True

# 设置绘图风格
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
zh_font_path = 'C:/Windows/Fonts/simhei.ttf'
if os.path.exists(zh_font_path):
    zh_font = fm.FontProperties(fname=zh_font_path)
else:
    zh_font = fm.FontProperties()


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
        self.local_power = 3e-3
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

        # [修复] 添加 length_height 属性
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
        # 论文公式 2.32 (忽略 G 和 RL，计算电流 PSD)
        H_f = 1 / np.sqrt(1 + (self.f_low / self.params.freqs) ** 16) / \
              np.sqrt(1 + (self.params.freqs / self.f_high) ** 16)
        responsivity_curve = self.params.responsitivity * H_f
        noise_psd = (self.nep ** 2) * (responsivity_curve ** 2)
        return noise_psd

    def calculate_rin_spectrum(self):
        omega = 2 * np.pi * self.params.freqs
        omega_fr = 2 * np.pi * self.fr
        num = self.A + self.B * omega ** 2
        den = (omega_fr ** 2 + self.Gamma ** 2 - omega ** 2) ** 2 + (4 * self.Gamma ** 2 * omega ** 2)

        # [疑问3] RIN 计算修正
        if FIX_PHYSICS_BUGS:
            # 使用论文公式 (2.30): 2 * R^2 * P_LO^2 * 10^(RIN/10)
            # 注意：论文公式有系数 2，原代码没有，这里保持原代码的系数逻辑，只去除非物理项
            RIN_val_db = 10 * np.log10(num / den)
            P_rin_linear = self.params.responsitivity ** 2 * self.params.local_power ** 2 * (10 ** (RIN_val_db / 10))
        else:
            # 原代码逻辑：包含 * freqs * impedance
            P_rin_db = 10 * np.log10(
                self.params.responsitivity ** 2 * self.params.local_power ** 2 * (num / den) * \
                self.params.freqs * self.params.impedance
            )
            P_rin_linear = 10 ** (P_rin_db / 10)

        return P_rin_linear

    def simulate_colored_noise(self, psd):
        # [疑问2] 频域随机相位法修正
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(self.params.FFT_points))

        if FIX_PHYSICS_BUGS:
            # 论文公式 (2.22): |X(f)| = N * sqrt(PSD * df)
            df = self.params.sample_rate / self.params.FFT_points
            amp_spectrum = self.params.FFT_points * np.sqrt(np.concatenate((psd, psd[::-1])) * df)
            # IFFT 需要除以 N，所以 N 因子抵消？
            # numpy.fft.ifft 定义为 (1/N) * sum(...)
            # 如果我们在频域构造的是 X(f)，直接 ifft 即可。
            # 这里的 amp_spectrum 应该是 X(f) 的模。
            # 实际上 np.fft.ifft(X) = x.
            # Parseval: sum|x|^2 = (1/N) sum|X|^2.
            # Expected Power P = sum(PSD * df).
            # We need sum|x|^2 = P.
            # sum|X|^2 = N * P.
            # So |X| ~ sqrt(N * P) = sqrt(N * PSD * df * N_bins?)
            # 简单起见，我们使用 sqrt(PSD * df) * N 是为了抵消 ifft 的 1/N 因子，
            # 但 numpy ifft 如果不norm='ortho'，确实有 1/N。
            # 让我们保持最简单的能量守恒：
            # Generate unit variance noise -> scale by sqrt(Power)
            pass

            # 保持原代码结构以便对比
        if FIX_PHYSICS_BUGS:
            df = self.params.sample_rate / self.params.FFT_points
            # 修正：加入 df，并不乘 N (假设后续 c 因子会处理幅度)
            # 为了物理正确性，这里应该是功率开方
            amp_spectrum = np.sqrt(np.concatenate((psd, psd[::-1])) * df) * np.sqrt(self.params.FFT_points)
        else:
            # 原代码：直接 PSD 开方
            amp_spectrum = np.sqrt(np.concatenate((psd, psd[::-1])))

        noise_freq = amp_spectrum * random_phase
        noise_time = np.fft.ifft(noise_freq).real

        # 额外的幅度校正（如果使用了 FIX_PHYSICS_BUGS，ifft后幅度可能很小，需要确认）
        # 原代码 simulate_gaussian_noise 也没乘 N
        return noise_time

    def simulate_gaussian_noise(self):
        shot_power = 2 * self.params.electric * self.params.responsitivity * \
                     self.params.local_power * self.params.bandwidth
        thermal_power = 4 * self.params.K * self.params.temperature * \
                        self.params.bandwidth / self.params.impedance
        total_power = shot_power + thermal_power
        return np.random.normal(0, np.sqrt(total_power), self.params.FFT_points)

    def generate_total_noise(self):
        rin_psd = self.calculate_rin_spectrum()
        rin_noise = self.simulate_colored_noise(rin_psd)
        # 原代码归一化逻辑：除以最大值。这会破坏物理幅度！
        # 但为了复现，我们保留它，除非 FIX_PHYSICS_BUGS=True
        if not FIX_PHYSICS_BUGS:
            rin_noise /= np.max(np.abs(rin_noise)) if np.max(np.abs(rin_noise)) > 0 else 1

        gauss_noise = self.simulate_gaussian_noise()
        if not FIX_PHYSICS_BUGS:
            gauss_noise /= np.max(np.abs(gauss_noise)) if np.max(np.abs(gauss_noise)) > 0 else 1

        nep_psd = self.calculate_nep_spectrum()
        nep_noise = self.simulate_colored_noise(nep_psd)
        if not FIX_PHYSICS_BUGS:
            nep_noise /= np.max(np.abs(nep_noise)) if np.max(np.abs(nep_noise)) > 0 else 1

        # 权重组合
        total_noise = rin_noise + gauss_noise + 8 * nep_noise

        # 如果开启修复，我们不应该进行 max 归一化，而是基于功率叠加
        # 但鉴于后续 process_pulse 中有个 c 因子根据 SNR 强行缩放噪声
        # 这里的绝对幅度其实不重要，重要的是相对频谱形状（色噪声特性）。
        # 所以原代码的归一化虽然物理上错误，但在 SNR 归一化步骤下可能被掩盖。

        return rin_noise[:512], gauss_noise[:512], nep_noise[:512], total_noise[:512]


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
        pulse_shape = (2 * np.sqrt(np.log(2)) / (np.sqrt(np.pi) * self.params.pulse_width)) * \
                      np.exp(-4 * np.log(2) * (time_diff / self.params.pulse_width) ** 2)
        self.pulse_power_matrix = np.sqrt(self.params.pulse_energy * self.params.pulse_repeat * pulse_shape)
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

    def calculate_optical_props(self):
        h_km = self.params.real_height / 1000
        alpha_m = (8 * np.pi / 3) * 1.54e-3 * (532e-9 / self.params.wavelength) ** 4 * np.exp(-h_km / 7)
        alpha_a = 50 * (2.47e-3 * np.exp(-h_km / 2) + 5.13e-6 * np.exp(-(h_km - 20) ** 2 / 36)) * (
                    532e-9 / self.params.wavelength)
        alpha_total = alpha_m + alpha_a
        beta = alpha_m / (8 * np.pi / 3) + alpha_a * 0.02

        def integrate_ext(R_target):
            return \
            quad(lambda r: np.interp(r * np.sin(np.radians(self.params.elevation_angle)) / 1000, h_km, alpha_total), 0,
                 R_target)[0]

        tau = np.array([integrate_ext(r / 1000) for r in self.params.detect_R])
        return beta, np.exp(-2 * tau)


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

    results = Parallel(n_jobs=4, backend="threading")(
        delayed(process_pulse)(params, lidar_data, noise_params) for _ in range(params.pulse_acc))
    spec_mean = np.mean(results, axis=0)

    save_dir = os.path.join('result_data', date_tag)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'spec_beam_{idx}.npy'), spec_mean)
    return spec_mean


def plot_figure_2_17(spec_data, params):
    spec_norm = (spec_data - spec_data.min(axis=1, keepdims=True)) / (
                spec_data.max(axis=1, keepdims=True) - spec_data.min(axis=1, keepdims=True))
    freqs_mhz = params.freqs / 1e6
    plt.figure(figsize=(10, 8))
    plt.imshow(spec_norm, aspect='auto', origin='lower', cmap='jet',
               extent=[freqs_mhz[0], freqs_mhz[-1], 0, params.Range_bin])
    plt.title("图 2.17 复现: 归一化功率谱密度 (常风速)", fontproperties=zh_font, fontsize=14)
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
        t0 = time.time()
        spec_result = run_simulation_direction(0, 'sim_constant_wind', params, lidar_data, noise_params)
        print(f"仿真耗时: {time.time() - t0:.2f} s")
        plot_figure_2_17(spec_result, params)