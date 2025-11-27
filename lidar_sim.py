import numpy as np  # 数值计算库
from scipy.interpolate import PchipInterpolator  # 分段Hermite插值
from scipy.integrate import quad  # 数值积分
import matplotlib.pyplot as plt  # 绘图库
import matplotlib.font_manager as fm  # 字体管理
from concurrent.futures import ThreadPoolExecutor  # 线程池并行
from joblib import Parallel, delayed  # 并行计算
import os  # 操作系统接口
import re  # 正则表达式

# 设置中文字体和数学字体
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体
plt.rcParams['font.family'] = 'serif'  # 字体族
plt.rcParams['font.serif'] = 'Times New Roman'  # 西文字体
chinese_font = fm.FontProperties(fname='Front/simsun.ttc')  # 中文字体

class LidarParams:
    """
    激光雷达系统参数集中管理类
    """
    def __init__(self):
        # 扫描设置
        self.pulse_count = 1000  # 每个方向处理的脉冲数
        self.beams_per_scan = 16  # 每圈光束数
        self.scan_circles = 1  # 扫描圈数
        self.total_beams = self.beams_per_scan * self.scan_circles  # 总光束数

        # 基本常数
        self.c = 3e8  # 光速 (m/s)
        self.wavelength = 1550e-9  # 激光波长 (m)

        # 激光脉冲参数
        self.pulse_energy = 50e-6  # 脉冲能量 (J)
        self.pulse_repetition = 10e3  # 脉冲重复率 (Hz)
        self.pulse_duration = 500e-9  # 脉冲宽度 (s)

        # 本振功率与AOM频率
        self.local_oscillator_power = 3e-3  # 本振功率 (W)
        self.aom_frequency = 120e6  # AOM移频 (Hz)

        # 采样设置
        self.sampling_rate = 1e9  # 采样率 (Hz)
        self.time_step = 1 / self.sampling_rate  # 采样间隔 (s)

        # 光电探测器与热噪声参数
        self.electron_charge = 1.6e-19  # 电子电荷 (C)
        self.detector_responsivity = 1  # 探测器响应度 (A/W)
        self.bandwidth = 200e6  # 接收带宽 (Hz)
        self.Boltzmann = 1.380649e-23  # 玻尔兹曼常数 (J/K)
        self.temperature = 273.15 + 20  # 温度 (K)
        self.impedance = 50  # 系统阻抗 (Ω)

        # 仪器几何与效率
        self.elevation_angle = 72  # 望远镜仰角 (°)
        self.max_range = 3840  # 最大探测距离 (m)
        self.range_resolution = 1  # 距离分辨 (m)
        self.range_bins = np.arange(self.range_resolution, self.max_range + self.range_resolution, self.range_resolution)
        self.real_heights = self.range_bins * np.sin(np.radians(self.elevation_angle))  # 考虑仰角的实际高度 (m)

        # 时间向量
        round_trip_time = 2 * self.max_range / self.c
        total_samples = int(np.ceil(round_trip_time / self.time_step)) + 1
        self.time_vector = np.arange(self.time_step, (total_samples + 1) * self.time_step, self.time_step)

        # FFT设置
        self.points_per_bin = 512  # 每个距离门样本数
        self.fft_length = 1024  # FFT点数
        self.freq_bins = np.fft.fftfreq(self.fft_length, d=self.time_step)[:self.points_per_bin]
        self.freq_bins[0] = 1e-10  # 避免零频率除零

        # 系统光学参数
        self.system_efficiency = 0.1  # 系统总体效率
        self.telescope_diameter = 0.12  # 望远镜直径 (m)
        self.telescope_area = np.pi * (self.telescope_diameter / 2) ** 2  # 收集面积 (m^2)

class NoiseParams:
    """
    噪声模型参数和生成功能
    """
    def __init__(self, lidar_params: LidarParams):
        self.lp = lidar_params  # 引用Lidar参数
        # RIN参数
        self.relaxation_freq = 660e3  # 弛豫振荡频率 (Hz)
        self.damping_rate = 40.8e3  # 阻尼因子 (1/s)
        self.A_param = 2e12  # RIN模型参数A
        self.B_param = 0.1  # RIN模型参数B
        # NEP噪声参数
        self.nep_spectrum = np.load('nep_fit_smooth.npy')  # NEP(Hz)频谱数据
        self.f_low = 10e3  # 低频截止 (Hz)
        self.f_high = 200e6  # 高频截止 (Hz)

    def compute_nep_noise_spectrum(self):
        """
        计算NEP对应的噪声电流功率谱
        """
        resp_curve = (self.lp.detector_responsivity /
                      np.sqrt(1 + (self.f_low / self.lp.freq_bins) ** 16) /
                      np.sqrt(1 + (self.lp.freq_bins / self.f_high) ** 16))  # 响应度滤波曲线
        ## 绘制响应度曲线
        # plt.figure(figsize=(8, 8))
        # plt.plot(self.lp.freq_bins / 1e6, resp_curve, label='Responsivity Curve', color='blue')
        # plt.xlabel('Frequency (MHz)')
        # plt.ylabel('Responsivity (A/W)')
        # plt.title('Detector Responsivity vs Frequency')
        # plt.xlim(0, 500)
        # plt.ylim(0)
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        ## 以上绘制响应度曲线

        noise_psd = (self.nep_spectrum ** 2) * (resp_curve ** 2)  # 噪声功率谱
        ## 绘制噪声功率谱，于杰论文第27页图2.11(c)，根据式(2.32)计算，单位是A^2× 1e-24/Hz，但公式中NEP(f)的值貌似没有计算？
        # plt.figure(figsize=(8, 8))
        # plt.plot(self.lp.freq_bins / 1e6, noise_psd * 1e-24, label='Noise Spectrum', color='red')
        # plt.xlabel('Frequency (MHz)')
        # plt.ylabel(r'Power Spectral Density (A$^{2}$/Hz)')
        # plt.title('Noise Spectrum')
        # plt.xlim(0, 500)
        # plt.ylim(0)
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        ## 以上绘制噪声功率谱

        return noise_psd

    def generate_nep_noise_time(self):
        """生成NEP带宽内时域噪声电流"""
        psd = self.compute_nep_noise_spectrum()  # 噪声谱
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(self.lp.fft_length))  # 随机相位
        # 构造双边谱
        noise_freq = np.sqrt(np.concatenate((psd, psd[::-1]))) * random_phase
        nep_time = np.fft.ifft(noise_freq).real  # IFFT到时域

        return nep_time

    def compute_rin_spectrum(self):
        """计算RIN引起的功率谱"""
        omega = 2 * np.pi * self.lp.freq_bins
        omega_fr = 2 * np.pi * self.relaxation_freq
        num = self.A_param + self.B_param * omega ** 2
        den = (omega_fr ** 2 + self.damping_rate ** 2 - omega ** 2) ** 2 + (4 * self.damping_rate ** 2 * omega ** 2)
        RIN_f = 10 * np.log10(num / den)
        P_rin = 10 * np.log10((self.lp.detector_responsivity ** 2 * self.lp.local_oscillator_power ** 2 * num / den) * self.lp.freq_bins * self.lp.impedance)
        # 绘制RIN频谱
        plt.figure(figsize=(8, 8))
        plt.semilogx(self.lp.freq_bins / 1e6, P_rin, label='RIN Spectrum', color='green')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (dB)')
        plt.title('RIN Spectrum')
        plt.xlim(0.1)
        plt.ylim(-200, -100)
        plt.legend()
        plt.show()
        # 绘制RIN频谱

        return RIN_f, P_rin

    def generate_rin_noise_time(self):
        """生成RIN时域噪声电流"""
        PSD_rin = self.compute_rin_spectrum()
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(self.lp.fft_length))
        noise_freq = np.sqrt(10 ** (np.concatenate((PSD_rin, PSD_rin[::-1])) / 10)) * random_phase
        return np.fft.ifft(noise_freq).real

    def compute_shot_thermal_noise_current(self):
        """计算高斯（散粒+热）噪声电流标准差"""
        term1 = 2 * self.lp.electron_charge * self.lp.detector_responsivity * self.lp.local_oscillator_power * self.lp.bandwidth
        term2 = 4 * self.lp.Boltzmann * self.lp.temperature * self.lp.bandwidth / self.lp.impedance

        ##绘制散粒噪声和热噪声时域电流
        # plot_sample_time = np.arange(0, 512) * self.lp.time_step * 1e9
        # plt.figure(figsize=(8, 8))
        # shot_noise = np.random.normal(0, np.sqrt(term1), 512) * 1e6
        # plt.plot(plot_sample_time, shot_noise, label='Shot Noise', color='green')
        # plt.xlabel('Time (ns)')
        # plt.ylabel(r'Current (${\mu}$A)')
        # plt.title('Shot Noise')
        # plt.xlim(0, 512)
        # plt.ylim(-2, 2)
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        #
        # plt.figure(figsize=(8, 8))
        # thermal_noise = np.random.normal(0, np.sqrt(term2), 512) * 1e6
        # plt.plot(plot_sample_time, thermal_noise, label='Thermal Noise', color='orange')
        # plt.xlabel('Time (ns)')
        # plt.ylabel(r'Current (${\mu}$A)')
        # plt.title('Thermal Noise')
        # plt.xlim(0, 512)
        # plt.ylim(-2, 2)
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        ##以上绘制散粒噪声和热噪声时域电流
        return np.sqrt(term1 + term2)

    def generate_shot_thermal_noise_time(self):
        """生成高斯（散粒+热）噪声时域电流"""
        sigma = self.compute_shot_thermal_noise_current()
        # ##绘制高斯（散粒+热）噪声时域电流和频域分布特征
        #
        # # 生成10^5个采样点的高斯白噪声
        # sample_count = 100000  # 10^5个采样点
        # time_domain_noise = np.random.normal(0, sigma, sample_count)
        #
        # # 绘制时域电流幅值
        # plt.figure(figsize=(10, 8))
        # time_axis = np.arange(sample_count) * self.lp.time_step * 1e9  # 转换为ns
        # plt.plot(time_axis, time_domain_noise * 1e6, color='blue', linewidth=0.5)
        # plt.xlabel('Time (ns)')
        # plt.ylabel('Current (μA)')
        # plt.title('Gaussian White Noise - Time Domain')
        # plt.xlim(0, 512) #仅绘制前512个采样点
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.show()
        #
        # # 绘制时域电流幅值的直方图
        # plt.figure(figsize=(10, 8))
        # hist_data, bin_edges, _ = plt.hist(time_domain_noise * 1e6, bins=100, density=True, alpha=0.7, color='lightblue', edgecolor='black', linewidth=0.5)
        #
        # # 绘制理论包络线（高斯分布）
        # x = np.linspace(time_domain_noise.min() * 1e6, time_domain_noise.max() * 1e6, 1000)
        # gaussian_envelope = (1/(sigma * 1e6 * np.sqrt(2*np.pi))) * np.exp(-0.5 * (x/(sigma * 1e6))**2)
        # plt.plot(x, gaussian_envelope, 'r-', linewidth=2, label='Gaussian Envelope')
        #
        # plt.xlabel('Current (μA/Hz)')
        # plt.ylabel('Probability Density')
        # plt.title('Histogram of Time Domain Noise Amplitude')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.show()
        #
        # # 进行FFT变换
        # freq_domain_noise = np.fft.fft(time_domain_noise, n=sample_count)
        # freq_bins = np.fft.fftfreq(sample_count, d=self.lp.time_step)
        #
        # # 只取正频率部分
        # positive_freq_idx = freq_bins >= 0
        # freq_bins_positive = freq_bins[positive_freq_idx]
        # freq_domain_noise_positive = freq_domain_noise[positive_freq_idx]
        #
        # # 绘制频域实部幅值的直方图
        # plt.figure(figsize=(10, 8))
        # real_part = np.real(freq_domain_noise_positive)
        # hist_data, bin_edges, _ = plt.hist(real_part * 1e6, bins=100, density=True, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=0.5)
        #
        # # 绘制理论包络线（高斯分布）
        # real_std = np.std(real_part)
        # x = np.linspace(real_part.min() * 1e6, real_part.max() * 1e6, 1000)
        # gaussian_envelope = (1/(real_std * 1e6 * np.sqrt(2*np.pi))) * np.exp(-0.5 * (x/(real_std * 1e6))**2)
        # plt.plot(x, gaussian_envelope, 'r-', linewidth=2, label='Gaussian Envelope')
        #
        # plt.xlabel('Real Part (μA/Hz)')
        # plt.ylabel('Probability Density')
        # plt.title('Histogram of Frequency Domain Real Part')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.show()
        #
        # # 绘制频域虚部幅值的直方图
        # plt.figure(figsize=(10, 8))
        # imag_part = np.imag(freq_domain_noise_positive)
        # hist_data, bin_edges, _ = plt.hist(imag_part * 1e6, bins=100, density=True, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.5)
        #
        # # 绘制理论包络线（高斯分布）
        # imag_std = np.std(imag_part)
        # x = np.linspace(imag_part.min() * 1e6, imag_part.max() * 1e6, 1000)
        # gaussian_envelope = (1/(imag_std * 1e6 * np.sqrt(2*np.pi))) * np.exp(-0.5 * (x/(imag_std * 1e6))**2)
        # plt.plot(x, gaussian_envelope, 'r-', linewidth=2, label='Gaussian Envelope')
        #
        # plt.xlabel('Imaginary Part (μA/Hz)')
        # plt.ylabel('Probability Density')
        # plt.title('Histogram of Frequency Domain Imaginary Part')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.show()
        #
        # # 绘制频域幅值的直方图
        # plt.figure(figsize=(10, 8))
        # magnitude = np.abs(freq_domain_noise_positive)
        # hist_data, bin_edges, _ = plt.hist(magnitude * 1e6, bins=100, density=True, alpha=0.7, color='lightyellow', edgecolor='black', linewidth=0.5)
        #
        # # # 计算瑞利分布的尺度参数
        # # sigma_sample = np.std(magnitude)
        # # rayleigh_sigma = sigma_sample
        # #
        # # # 生成瑞利分布的包络线
        # # x = np.linspace(0, magnitude.max() * 1e6, 1000)
        # # rayleigh_envelope = (x / (rayleigh_sigma * 1e6) ** 2) * np.exp(-x ** 2 / (2 * (rayleigh_sigma * 1e6) ** 2))
        # # plt.plot(x, rayleigh_envelope, 'r-', linewidth=2, label='Rayleigh Envelope')
        #
        # plt.xlabel('Magnitude (μA/Hz)')
        # plt.ylabel('Probability Density')
        # plt.title('Histogram of Frequency Domain Magnitude')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.show()
        # ##以上绘制高斯（散粒+热）噪声时域电流和频域分布特征
        # return np.random.normal(0, sigma, self.lp.fft_length)

    def generate_total_noise_time(self):
        """组合归一化所有噪声分量并输出"""
        rin = self.generate_rin_noise_time()
        rin /= np.max(np.abs(rin))
        shot = self.generate_shot_thermal_noise_time()
        shot /= np.max(np.abs(shot))
        nep = self.generate_nep_noise_time()
        nep /= np.max(np.abs(nep))
        total = rin + shot + 8 * nep
        return rin[:self.lp.points_per_bin], shot[:self.lp.points_per_bin], nep[:self.lp.points_per_bin], total[:self.lp.points_per_bin]

class LidarSimulator:
    """激光雷达仿真主类，基于真实风场数据"""
    def __init__(self, params: LidarParams, wind_data_file: str):
        self.params = params  # 参数引用
        self.noise = NoiseParams(params)  # 噪声模块实例
        # 加载风场速度矩阵 (shape: range_bins × 3)
        self.wind_velocity_field = np.load(wind_data_file)
        # 方向角记录
        azimuths = np.linspace(0, 360, self.params.beams_per_scan, endpoint=False)
        self.azimuth_angle_record = np.repeat(azimuths, self.params.scan_circles)
        self.elevation_angle_record = np.full(self.params.total_beams, self.params.elevation_angle)

        # 计算所有光束的视线速度 (m/s)
        self.v_los_all = compute_line_of_sight_velocity(self.wind_velocity_field, self.azimuth_angle_record, self.elevation_angle_record, self.params)

        # 后向散射和透过率
        self.backscatter_profile, self.transmittance = self._compute_backscatter_and_transmittance()

        # AOM移频相位矩阵
        self.aom_phase_shift = np.exp(-2j * np.pi * self.params.aom_frequency * self.params.time_vector)

        # 脉冲功率矩阵
        delay = 2 * self.params.range_bins / self.params.c
        time_diff = self.params.time_vector[:, np.newaxis] - delay[np.newaxis, :]
        transmit = pulse_shape(time_diff, self.params.pulse_duration)
        self.pulse_power_matrix = np.sqrt(self.params.pulse_energy * self.params.pulse_repetition * transmit)

        # 散射光功率因子
        self.scatter_power_factor = (self.params.telescope_area / self.params.range_bins**2) * self.params.system_efficiency * self.params.range_resolution

    def _compute_backscatter_and_transmittance(self):
        # 高度换算 (公里)
        height_km = self.params.real_heights / 1000
        # 分子和气溶胶消光系数
        ray_ext = (8*np.pi/3)*1.54e-3*(532e-9/self.params.wavelength)**4 * np.exp(-height_km/7)
        aero_ext = 50*(2.47e-3*np.exp(-height_km/2) + 5.13e-6*np.exp(-(height_km-20)**2/36))*532e-9/self.params.wavelength
        extinction_total = ray_ext + aero_ext
        # 后向散射系数
        beta_ray = ray_ext/(8*np.pi/3)
        beta_aero = aero_ext*0.02
        backscatter_prof = beta_ray + beta_aero
        # 光学厚度积分

        def integrate_ext(h): return quad(lambda x: np.interp(x, height_km, extinction_total), 0, h)[0]
        tau = np.array([integrate_ext(h) for h in self.params.real_heights/1000])
        trans = np.exp(-2*tau)
        return backscatter_prof, trans

    def simulate_direction(self, beam_index: int, date_tag: str):
        """对单一光束方向进行全脉冲仿真并输出频谱平均"""
        # 更新风速移频相位
        freq_shift = -2 * self.v_los_all[:, beam_index] / self.params.wavelength
        wind_phase = np.exp(-2j * np.pi * np.outer(freq_shift, self.params.time_vector))

        # 并行每个脉冲FFT累积
        spectra = Parallel(n_jobs=4)(delayed(process_single_pulse)(
            beam_index, wind_phase, self.pulse_power_matrix, self.aom_phase_shift,
            self.scatter_power_factor, self.noise) for _ in range(self.params.pulse_count))
        mean_spectrum = np.mean(spectra, axis=0)
        # 保存结果
        out_folder = os.path.join('result_data', f'{date_tag}')
        os.makedirs(out_folder, exist_ok=True)
        np.save(os.path.join(out_folder, f'spectrum_beam_{beam_index}.npy'), mean_spectrum)

# 脉冲形状函数 (高斯)
    def pulse_shape(t, width): return (2*np.sqrt(np.log(2))/(np.sqrt(np.pi)*width))*np.exp(-4*np.log(2)*t**2/width**2)

# 计算视线风速
    def compute_line_of_sight_velocity(V_field, azimuths, elevations, params):
        los = np.zeros((len(params.real_heights), params.total_beams))
        for idx in range(params.total_beams):
            sv = np.array([
                np.sin(np.radians(elevations[idx])),
                np.cos(np.radians(elevations[idx]))*np.sin(np.radians(azimuths[idx])),
                np.cos(np.radians(elevations[idx]))*np.cos(np.radians(azimuths[idx]))
            ])
            v_los = np.dot(V_field, sv)
            los[:, idx] = v_los
        return los

# 单脉冲处理并FFT
    def process_single_pulse(beam_idx, wind_phase, power_mat, aom_phase, scatter_factor, noise_model):
        # 随机相位与振幅
        rand_phase = np.random.uniform(0, 2*np.pi, len(scatter_factor))
        rand_amp = np.random.rayleigh(np.sqrt(scatter_factor))
        K_complex = rand_amp * np.exp(1j*rand_phase)
        # 快速计算电流
        ih = 2*np.sqrt(power_mat)*aom_phase*np.sum(power_mat*K_complex[np.newaxis,:]*wind_phase.T, axis=1)
        # 加入噪声并FFT
        sig = np.real(ih[:512])
        _,_,_,tot_noise = noise_model.generate_total_noise_time()
        # 计算归一化因子
        c_factor = np.sqrt(np.sum(np.abs(sig)**2)/(10**(20/10)*np.sum(np.abs(tot_noise)**2)))
        combined = sig + tot_noise * c_factor
        padded = np.pad(combined, (0, 1024 - len(combined)))
        return np.abs(np.fft.fft(padded)[:512])

if __name__ == '__main__':
    # data_folder = 'wind_input'  # 风场数据文件夹
    # files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]  # 所有.npy文件
    # for fname in files:
    #     date_tag = re.search(r'\d+', fname).group()  # 文件名中的日期标签
    #     params = LidarParams()  # 初始化参数
    #     simulator = LidarSimulator(params, os.path.join(data_folder, fname))  # 初始化仿真器
    #     # 并行处理各个光束方向
    #     Parallel(n_jobs=4)(delayed(simulator.simulate_direction)(idx, date_tag)
    #                       for idx in range(params.total_beams))  # 结束处理

    # 创建参数实例
    params = LidarParams()

    # 创建噪声参数实例
    noise_params = NoiseParams(params)

    # 调用计算NEP噪声谱的方法，这将触发响应度曲线的绘制
    _ = noise_params.compute_nep_noise_spectrum()
    noise_params.compute_shot_thermal_noise_current()
    noise_params.generate_shot_thermal_noise_time()
    noise_params.compute_rin_spectrum()
