import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import re
from joblib import Parallel, delayed
import time
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
zh = fm.FontProperties(fname='Front/simsun.ttc')

# 参数类，用于集中管理参数
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

        self.electric = 1.6e-19  # 电荷常量 单位C
        self.responsitivity = 1  # 探测器响应度(光电转换系数) 单位A/W
        self.band_width = 200e6
        self.K = 1.380649e-23
        self.temperature = 273.15 + 20
        self.impedance = 50
        self.elevation_angle = 72
        self.max_detect_z = 3840
        self.delta_T = 1 / self.sample_rate
        self.delta_R = 1
        self.detect_R = np.arange(self.delta_R, self.max_detect_z + self.delta_R, self.delta_R)

        self.real_height = self.detect_R * np.sin(self.elevation_angle / 180 * np.pi)  # 实际高度考虑激光倾斜角度
        self.length_height = len(self.real_height)

        self.time = np.arange(self.delta_T, (np.ceil(2 * self.max_detect_z / self.c / self.delta_T) + 1) * self.delta_T,
                              self.delta_T)
        self.length_time = len(self.time)
        self.Points_per_bin = 512
        self.Range_bin = int(np.floor(self.length_time / self.Points_per_bin))
        self.FFT_points = 1024
        self.freqs = np.fft.fftfreq(self.FFT_points, 1 / self.sample_rate)[:512]
        self.freqs[0] = 1e-10  # 避免频率为0导致的除零问题
        self.system_efficency = 0.1
        self.telescope_D = 120e-3
        self.telescope_S = np.pi * (self.telescope_D / 2) ** 2



class NoiseParams:
    def __init__(self, params):
        self.params = params
        # 信噪比
        self.snr_start = -10  # 起始SNR值
        self.snr_end = -35  # 结束SNR值
        self.snr = np.linspace(self.snr_start, self.snr_end, self.params.Range_bin)
        # 相对强度噪声参数 (RIN)
        self.fr = 660e3  # 弛豫频率，单位为Hz
        self.Gamma = 40.8e3  # 阻尼因子，单位为s^-1
        self.A = 2e12  # 参数A
        self.B = 0.1
        # NEP噪声参数
        self.nep = np.load('nep_fit_smooth.npy')
        self.f_low = 10e3  # 10 kHz
        self.f_high = 200e6  # 200 MHz

    def caculate_nep_noise(self):
        # 使用公式进行响应度曲线仿真
        responsivity = self.params.responsivity / np.sqrt(1 + (self.f_low / self.params.freqs) ** 16) / np.sqrt(1 + (self.params.freqs / self.f_high) ** 16)
        # # 绘制响应度曲线
        # plt.figure(figsize=(10, 6))
        # plt.plot(self.params.freqs, responsivity, label='Responsivity Curve (Bandpass-like)', color='green')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Responsivity (A/W)')
        # plt.title('Responsivity Curve Simulation using Bandpass-like Formula')
        # plt.legend()
        # plt.grid()
        # plt.show()
        # 计算噪声功率谱
        # 使用NEP拟合结果与响应度曲线计算噪声功率谱
        noise_power_spectrum = (self.nep**2) * (responsivity**2)
        # # 绘制噪声功率谱
        # plt.figure(figsize=(10, 6))
        # plt.plot(self.params.freqs, noise_power_spectrum, label='Noise Power Spectrum', color='blue')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Noise Power Spectrum (W^2/Hz)')
        # plt.title('Noise Power Spectrum Calculation')
        # plt.legend()
        # plt.grid()
        # plt.show()
        return noise_power_spectrum
    def simulate_nep_noise_current(self):
        P_nep_f = self.caculate_nep_noise()
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(1024))
        # 生成具有噪声功率谱的频域噪声信号
        noise_frequency_domain = np.sqrt(
            np.concatenate((P_nep_f, P_nep_f[::-1]))) * random_phase
        # 对频域噪声信号进行快速傅里叶逆变换得到时域噪声电流
        nep_noise_time_domain = np.fft.ifft(noise_frequency_domain).real
        return nep_noise_time_domain
    def calculate_rin(self):
        # 计算RIN(f)的值
        omega_f = 2 * np.pi * self.params.freqs
        omega_fr = 2 * np.pi * self.fr

        numerator = self.A + self.B * omega_f ** 2
        denominator = (omega_fr ** 2 + self.Gamma ** 2 - omega_f ** 2) ** 2 + (4 * self.Gamma ** 2 * omega_f ** 2)
        RIN_f = 10 * np.log10(numerator / denominator)

        # 计算RIN引起的功率谱密度
        P_RIN_f = 10 * np.log10(
            self.params.responsivity ** 2 * self.params.local_power ** 2 * (numerator / denominator) * self.params.freqs * self.params.impedance)

        return RIN_f, P_RIN_f
    def simulate_rin_noise_current(self):
        # 计算RIN引起的功率谱密度
        _, P_RIN_f = self.calculate_rin()
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(1024))
        # 生成具有噪声功率谱的频域噪声信号
        noise_frequency_domain = np.sqrt(10 ** (np.concatenate((P_RIN_f, P_RIN_f[::-1])) / 10)) * random_phase
        # 完成频域到时域的转换 (逆FFT)
        rin_noise_current_time_domain = np.fft.ifft(noise_frequency_domain, n=self.params.FFT_points)
        rin_noise_current_time_domain = np.real(rin_noise_current_time_domain)

        # 返回RIN噪声电流的时域结果
        return rin_noise_current_time_domain
    def calculate_gaussian_noise(self):
        # 计算高斯噪声的电流值
        gaussian_noise_current = np.sqrt(2 * self.params.electron_charge * self.params.responsivity *
                                         self.params.local_power * self.params.bandwidth +
                                         4 * self.params.K * self.params.temperature *
                                         self.params.bandwidth / self.params.impedance)
        return gaussian_noise_current
    def simulate_gaussian_noise_current(self):
        # 计算高斯噪声的功率谱密度
        gaussian_noise_power = self.calculate_gaussian_noise()
        # 生成高斯噪声电流
        gaussian_noise_current_time_domain = np.random.normal(0, np.sqrt(gaussian_noise_power), self.params.FFT_points)

        return gaussian_noise_current_time_domain
    def simulate_total_noise_current(self):
        # 计算RIN噪声和高斯噪声
        rin_noise = self.simulate_rin_noise_current()
        rin_noise = rin_noise / np.max(np.abs(rin_noise))
        gaussian_noise = self.simulate_gaussian_noise_current()
        gaussian_noise = gaussian_noise / np.max(np.abs(gaussian_noise))
        nep_noise = self.simulate_nep_noise_current()
        nep_noise = nep_noise / np.max(np.abs(nep_noise))
        # 归一化噪声
        total_noise = rin_noise + gaussian_noise + 8 * nep_noise

        return rin_noise[:512], gaussian_noise[:512], nep_noise[:512], total_noise[:512]
class LidarData:
    def __init__(self, params, wind_file):
        # 初始化参数
        self.params = params

        # 假设风速模型和插值
        # V_m = np.array([[0, 3, -3] for _ in self.params.detect_R])
        # self.V_m_interp = PchipInterpolator(self.params.detect_R, V_m, axis=0)(self.params.detect_R)
        # 导入风速模型
        self.V_m_interp = np.load(wind_file)
        # 方位角记录
        azimuthangle = np.linspace(0, 360, self.params.directions_per_circle, endpoint=False)
        self.azimuthangle_record = np.repeat(azimuthangle, self.params.num_circles)
        self.elevationangle_record = np.full(self.params.direction_num, self.params.elevation_angle)

        # 计算视线风速
        self.V_los_all = calculate_v_los(self.V_m_interp, self.azimuthangle_record, self.elevationangle_record, self.params)

        # 模拟每个方向的功率谱
        self.spec_fft_single = np.zeros((int(self.params.Range_bin), self.params.Points_per_bin))
        self.spec_fft_accumulated = np.zeros((int(self.params.Range_bin), self.params.Points_per_bin))

        # 计算后向散射和透过率
        self.backscatter_profile, self.transmittance = self.calculate_backscatter_transmittance()

        # AOM移频
        self.freqshift_aom = np.exp(-2j * np.pi * self.params.frequency_aom * self.params.time)

        # 计算风速引起的频移
        self.freqshift_wind = -2 * self.V_los_all[:, 0] / self.params.wavelength
        self.b = np.exp(-2j * np.pi * np.outer(self.freqshift_wind, self.params.time))

        # 计算功率矩阵
        aaa = 2 * self.params.detect_R / self.params.c
        time_matrix = self.params.time[:, np.newaxis] - aaa[np.newaxis, :]
        self.power_matrix = np.sqrt(self.params.pulse_energy * self.params.pulse_repeat * transmit_power(time_matrix, self.params.pulse_width))

        # 系统常数
        self.K_m_squared = (self.params.telescope_S / self.params.detect_R ** 2) * self.params.system_efficiency * self.params.delta_R

    def calculate_backscatter_transmittance(self):
        height_back = self.params.real_height / 1000  # 将高度转换为公里单位

        # 计算分子消光系数 (Rayleigh 散射)
        atmo_extinction = (8 * np.pi / 3) * 1.54e-3 * (532e-9 / self.params.wavelength) ** 4 * np.exp(-height_back / 7)
        # 计算气溶胶消光系数 (Mie 散射)
        aero_extinction = 50 * (2.47e-3 * np.exp(-height_back / 2) + 5.13e-6 * np.exp(
            -(height_back - 20) ** 2 / 36)) * 532e-9 / self.params.wavelength
        # 总消光系数为分子消光和气溶胶消光之和
        extinction_all = atmo_extinction + aero_extinction

        # 分子后向散射系数 (Rayleigh 散射)
        backscatter_rayleigh = atmo_extinction / (8 * np.pi / 3)
        # 气溶胶后向散射系数 (Mie 散射)
        backscatter_aerosol = aero_extinction * 0.02  # 假设后向散射比为 0.02
        # 总后向散射系数为分子和气溶胶的贡献之和
        backscatter_profile = backscatter_rayleigh + backscatter_aerosol

        # 计算光学厚度，积分从地面到当前高度的消光系数（沿着倾斜路径和垂直路径分别进行积分）
        def extinction_integral(height, extinction_all):
            return quad(lambda h: np.interp(h, height_back, extinction_all), 0, height)[0]

        optical_depth = np.array(
            [extinction_integral(height, extinction_all) for height in self.params.detect_R / 1000])
        optical_depth_r = np.array(
            [extinction_integral(height, extinction_all) for height in self.params.real_height / 1000])
        # 计算大气透过率，考虑激光往返经过大气层的能量损失
        transmittance = np.exp(-2 * optical_depth)
        transmittance_r = np.exp(-2 * optical_depth_r)
        return backscatter_profile, transmittance

# 激光器脉冲传输功率
def transmit_power(t, pulse_width):
    return (2 * np.sqrt(np.log(2)) / (np.sqrt(np.pi) * pulse_width)) * np.exp(-4 * np.log(2) * t ** 2 / pulse_width ** 2)
def transmit_power_flat_top(t, pulse_width):
    pulse_duration = pulse_width  # 脉冲宽度
    pulse_power = 1  # 平顶脉冲的功率（可以调整）
    # 判断时间t是否在脉冲宽度内
    power = np.where((t >= -pulse_duration / 2) & (t <= pulse_duration / 2), pulse_power, 0)
    return power
# 计算大气后向散射和透过率


# 计算视线风速
def calculate_v_los(V_m, azimuthangle_record, elevationangle_record, params):
    V_los_all = np.zeros((params.length_height, params.direction_num))
    for j in range(params.direction_num):
        # 计算每个方向上的单位矢量分量
        Si = np.array([
            np.sin(np.radians(elevationangle_record[j])),
            np.cos(np.radians(elevationangle_record[j])) * np.sin(np.radians(azimuthangle_record[j])),
            np.cos(np.radians(elevationangle_record[j])) * np.cos(np.radians(azimuthangle_record[j]))
        ])
        # 计算视线方向的风速，通过单位矢量与风速向量点乘得到
        V_los = np.dot(V_m, Si)
        V_los_all[:, j] = V_los
    return V_los_all


def process_range_bin(mm, i_h, c, params, noise_params):
    start_mm, end_mm = mm * params.Points_per_bin, (mm + 1) * params.Points_per_bin
    signal_fragment = i_h[start_mm:end_mm]
    rin_noise, gaussian_noise, nep_noise, total_noise = noise_params.simulate_total_noise_current()
    # # 噪声功率谱累积绘图
    # num_simulations = 10000
    # accumulated_power_spectrum = np.zeros(1024)
    # for _ in range(num_simulations):
    #     rin_noise, gaussian_noise, nep_noise, total_noise = noise_params.simulate_total_noise_current()
    #
    #     # 计算时域噪声电流的功率谱
    #     signal_fragment1 = np.real(signal_fragment) + total_noise * c
    #     noise_frequency_spectrum = np.fft.fft(np.pad(signal_fragment1, ((0, 512)), 'constant'), n=1024)
    #     power_spectrum = np.abs(noise_frequency_spectrum) ** 2 / 1024
    #     # 累积功率谱
    #     accumulated_power_spectrum += power_spectrum
    # # 计算平均功率谱
    # average_power_spectrum = accumulated_power_spectrum / num_simulations
    # # 绘制累积平均后的噪声功率谱
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.linspace(0, 1e9, 1024), average_power_spectrum, label='Accumulated Average Noise Power Spectrum',
    #          color='purple')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Average Noise Power Spectrum (W^2/Hz)')
    # plt.title('Accumulated Average Noise Power Spectrum from 1000 Simulations')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # plt.figure()
    # plt.plot(signal_fragment)
    # plt.plot(nep_noise * c, linestyle=':')
    # plt.plot(rin_noise * c, linestyle='--')
    # plt.plot(gaussian_noise * c, linestyle='-.')
    # plt.plot(signal_fragment + rin_noise * c + gaussian_noise * c + nep_noise * c)
    # plt.show()
    # 将噪声叠加到原始电流信号 i_h 上
    signal_fragment1 = np.real(signal_fragment) + total_noise * c
    # 对信号进行傅里叶变换，并进行零填充
    zero_padding = params.FFT_points - signal_fragment1.shape[0]
    signal_padded = np.pad(signal_fragment1, ((0, zero_padding)), 'constant')
    # 计算FFT并累积功率谱
    spec_fft = np.abs(np.fft.fft(signal_padded, n=params.FFT_points))[:512]
    return mm, spec_fft


def process_pulse(params, lidar_data, noise_params):

    # 随机生成相位和振幅
    phase = np.random.uniform(0, 2 * np.pi, len(lidar_data.K_m_squared))
    amplitude = np.random.rayleigh(np.sqrt(lidar_data.K_m_squared))
    Km = amplitude * np.exp(1j * phase)

    # 计算 i_h
    i_h = 2 * np.sqrt(params.local_power) * lidar_data.freqshift_aom * \
          np.sum(lidar_data.power_matrix * Km[np.newaxis, :] * lidar_data.b.T, axis=1)


    signal_fragment = i_h[:512]
    signal_power = np.sum(np.abs(signal_fragment) ** 2)
    rin_noise, gaussian_noise, nep_noise, total_noise = noise_params.simulate_total_noise_current()
    noise_power = np.sum(np.abs(total_noise) ** 2)
    # 计算调整系数 c
    target_k_noise = 20  # 目标的信噪比
    c = np.sqrt(signal_power / (10 ** (target_k_noise / 10) * noise_power))

    spec_fft_accumulated_local = np.zeros((int(params.Range_bin), params.Points_per_bin))

    # 并行处理每个距离门
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(process_range_bin, mm, i_h, c, params, noise_params) for mm in range(int(params.Range_bin))]
    #     for future in concurrent.futures.as_completed(futures):
    #         mm, spec_fft = future.result()
    #         spec_fft_accumulated_local[mm, :] += spec_fft
    for mm in range(int(params.Range_bin)):
        mm, spec_fft = process_range_bin(mm, i_h, c, params, noise_params)
        spec_fft_accumulated_local[mm, :] += spec_fft
    # 返回当前脉冲索引和累积的频谱结果
    return spec_fft_accumulated_local

def simulate_single_pulse(params, lidar_data, noise_params):

    # 处理单脉冲数据
    pulse_index, accumulated_result = process_pulse(0, lidar_data.K_m_squared, lidar_data.power_matrix,
                                                    lidar_data.b, lidar_data.freqshift_aom, params, noise_params)
    # 输出单脉冲结果
    print(f"Processed single pulse index: {pulse_index}")
    return accumulated_result
    # 并行外层循环（使用 joblib）
def process_direction(idx, date):
    # 更新方向的频移
    if idx != 0:
        lidar_data.freqshift_wind = -2 * lidar_data.V_los_all[:, idx] / lidar_data.params.wavelength
        lidar_data.b = np.exp(-2j * np.pi * np.outer(lidar_data.freqshift_wind, lidar_data.params.time))

    # 使用 ThreadPoolExecutor 替代内层的 multiprocessing Pool
    spec_fft_accumulated = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_pulse, params, lidar_data, noise_params)
            for pulse_index in range(params.pulse_acc)
        ]
        for i, future in enumerate(futures):
            print(f"Processing pulse {i + 1}/{params.pulse_acc} for direction index {idx} in date {date}")
            accumulated_result = future.result()
            spec_fft_accumulated.append(accumulated_result)

    # 计算 spec_fft_accumulated 的平均矩阵
    spec_fft_accumulated_mean = np.mean(spec_fft_accumulated, axis=0)

    output_folder = os.path.join('result_data', f'{date}_1000')
    os.makedirs(output_folder, exist_ok=True)
    # 保存结果
    output_file_path = os.path.join(output_folder, f'spec_fft_{idx}_{params.pulse_acc}.npy')
    np.save(output_file_path, spec_fft_accumulated_mean)

if __name__ == '__main__':

    # ----------------------------------------------------------------------------------------------------------------
    # # 噪声功率谱累积绘图
    # num_simulations = 10000
    # accumulated_power_spectrum = np.zeros(1024)
    # for _ in range(num_simulations):
    #     total_noise = noise_params.simulate_total_noise_current()
    #     # 计算时域噪声电流的功率谱
    #     noise_frequency_spectrum = np.fft.fft(np.pad(total_noise, ((0, 512)), 'constant'), n=1024)
    #     power_spectrum = np.abs(noise_frequency_spectrum) ** 2 / 1024
    #     # 累积功率谱
    #     accumulated_power_spectrum += power_spectrum
    # # 计算平均功率谱
    # average_power_spectrum = accumulated_power_spectrum
    # # 绘制累积平均后的噪声功率谱
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.linspace(0, 1e9, 1024), average_power_spectrum, label='Accumulated Average Noise Power Spectrum',
    #          color='purple')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Average Noise Power Spectrum (W^2/Hz)')
    # plt.title('Accumulated Average Noise Power Spectrum from 1000 Simulations')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # ---------------------------------------------------------------------------------------------------------------
    # 初始化参数
    main_folder = r'wind_input'
    # 获取所有子文件夹名
    npy_files = [f for f in os.listdir(main_folder) if f.endswith('.npy')]
    for npy_file in npy_files:
        folder_date = re.search(r'\d+', npy_file).group()
        file_path = os.path.join(main_folder, npy_file)

        params = LidarParams()
        noise_params = NoiseParams(params)
        lidar_data = LidarData(params, file_path)
        # for idx in range(0, lidar_data.params.direction_num):
        #     process_direction(idx, folder_date)
        # 设置临时文件目录，确保不包含中文字符
        # os.environ['JOBLIB_TEMP_FOLDER'] = 'C:\\Temp'
        # os.makedirs("C:\\Temp", exist_ok=True)
        # 使用 joblib 对最外层循环进行并行化
        Parallel(n_jobs=4)(delayed(process_direction)(idx, folder_date) for idx in range(0, lidar_data.params.direction_num))
    # # 初始化参数并统计耗时
    # start_time = time.time()
    # params = LidarParams()
    # noise_params = NoiseParams(params)
    # # 初始化LidarData类
    # lidar_data = LidarData(params)
    # init_time = time.time() - start_time
    # print(f"Parameter initialization time: {init_time:.2f} seconds")
    #
    # for idx in range(0, lidar_data.params.direction_num):
    #     if idx != 0:
    #         lidar_data.freqshift_wind = -2 * lidar_data.V_los_all[:, idx] / lidar_data.params.wavelength
    #         lidar_data.b = np.exp(-2j * np.pi * np.outer(lidar_data.freqshift_wind, lidar_data.params.time))
    #
    #     # 并行外部循环使用多进程
    #     start_time = time.time()
    #     spec_fft_accumulated = []
    #     with mp.Pool(processes=8) as pool:
    #         outer_futures = [
    #             pool.apply_async(process_pulse, (params, lidar_data, noise_params))
    #             for pulse_index in range(params.pulse_accumulation)]
    #         # 监视进度
    #         for i, outer_future in enumerate(outer_futures):
    #             print(f"Processing pulse {i + 1}/{params.pulse_accumulation} for direction index {idx}")
    #             accumulated_result = outer_future.get()
    #             # 累加到全局的 spec_fft_accumulated
    #             spec_fft_accumulated.append(accumulated_result)
    #     multi_process_time = time.time() - start_time
    #     print(f"Multi-process computation time for direction index {idx}: {multi_process_time:.2f} seconds")
    #     # 计算 spec_fft_accumulated 的平均矩阵
    #     spec_fft_accumulated_mean = np.mean(spec_fft_accumulated, axis=0)
    #     # 保存 spec_fft_accumulated_mean 结果
    #     np.save(f'spec_fft_{idx}_{params.pulse_accumulation}.npy', spec_fft_accumulated_mean)
    # lidar_data.spec_fft_accumulated = process_pulse(params, lidar_data, noise_params)
    # # 并行外部循环使用多进程
    # start_time = time.time()
    # spec_fft_accumulated = []
    # with mp.Pool(processes=8) as pool:
    #     outer_futures = [
    #         pool.apply_async(process_pulse, (params, lidar_data, noise_params))
    #         for pulse_index in range(params.pulse_accumulation)]
    #
    #     for i, outer_future in enumerate(outer_futures):
    #         print(f"Processing pulse {i + 1}/{params.pulse_accumulation}")
    #         accumulated_result = outer_future.get()
    #         # 累加到全局的 spec_fft_accumulated
    #         spec_fft_accumulated.append(accumulated_result)
    # spec_fft_accumulated_mean = np.mean(spec_fft_accumulated, axis=0)
    # multi_process_time = time.time() - start_time
    # print(f"Multi-process computation time: {multi_process_time:.2f} seconds")
    # ------------------------------------------------------------------------------------------------------------------
    # # 绘制功率谱仿真结果
    # # 归一化 lidar_data.spec_fft_accumulated 每一行
    # spec_normalized = (lidar_data.spec_fft_accumulated - lidar_data.spec_fft_accumulated.min(axis=1, keepdims=True)) / \
    #                   (lidar_data.spec_fft_accumulated.max(axis=1, keepdims=True) - lidar_data.spec_fft_accumulated.min(
    #                       axis=1, keepdims=True))
    # # 三维图像
    # X = np.linspace(0, lidar_data.params.Points_per_bin, spec_normalized.shape[1])
    # Y = np.linspace(1, lidar_data.params.Range_bin, spec_normalized.shape[0] - 1)
    # X, Y = np.meshgrid(X, Y)
    # spec_fft_accumulated_trimmed = lidar_data.spec_fft_accumulated[1:, :]
    # # 创建图形
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # 绘制3D表面图
    # surf = ax.plot_surface(X, Y, spec_fft_accumulated_trimmed, cmap='viridis')
    # # 添加颜色条
    # fig.colorbar(surf, ax=ax, label='Normalized Amplitude')
    # # 添加标签和标题
    # ax.set_xlabel('Frequency Bin')
    # ax.set_ylabel('Range Bin')
    # ax.set_zlabel('Normalized Amplitude')
    # ax.set_title('Accumulated Power Spectrum (Normalized)')
    # # 二维图像
    # # 绘图
    # plt.figure()
    # plt.imshow(spec_normalized, aspect='auto',
    #            extent=[0, lidar_data.params.Points_per_bin, 0, lidar_data.params.Range_bin], origin='lower')
    # plt.xlabel('Frequency Bin')
    # plt.ylabel('Range Bin')
    # plt.title('Accumulated Power Spectrum (Normalized)')
    # plt.colorbar(label='Normalized Amplitude')
    # plt.show()
    # ------------------------------------------------------------------------------------------------------------------







