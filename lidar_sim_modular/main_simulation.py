import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 导入核心模块
from A_lidar_params import params
from C_lidar_physics import LidarPhysics
from D_noise_model import NoiseModel
from E_wind_field import WindField
import S_plot_style as plot_style


class LidarMainSimulator:
    def __init__(self):
        print(">>> 初始化雷达仿真系统...")
        self.p = params
        self.physics = LidarPhysics()
        self.noise_model = NoiseModel()
        self.wind_field = WindField()

        # 尝试加载探空数据
        sounding_file = r"E:\GraduateStu6428\Codes\ObservationData54511\12Z\2025-12-01_12.csv"
        try:
            self.wind_field.load_sounding_data(sounding_file)
        except:
            pass

    def run_simulation(self, n_accum=1000, scan_mode='PPI'):
        """
        执行主仿真
        """
        # 1. 扫描参数配置
        if scan_mode == 'PPI':
            azimuths = np.linspace(0, 360, self.p.azimuth_count, endpoint=False)
        else:  # Staring (凝视模式，用于调试或画图2.16)
            azimuths = [0.0]
            print(">>> 进入凝视模式 (Azimuth=0°)")

        # 2. 距离门与FFT参数
        # 表2.1: 距离门采样点数 512, FFT点数 1024
        gate_len = 512
        n_fft = self.p.fft_points  # 1024
        n_freq_bins = n_fft // 2  # 只取正频率 (512点)

        # PSD 归一化系数 (转换为 A^2/Hz)
        psd_norm = 2.0 / (self.p.sample_rate * n_fft)

        # 3. 初始化结果容器 (方位角 x 距离门 x 频率)
        # 此时还不知道距离门数量，先设为 None
        sim_data = None
        range_gates = None

        print(f">>> 开始仿真 (模式: {scan_mode}, 累积: {n_accum})")

        # --- 方位角循环 ---
        for i, azi in enumerate(azimuths):
            if scan_mode == 'PPI':
                print(f"  -> 扫描方位角 {azi:.1f}° ({i + 1}/{len(azimuths)})")

            # A. 获取当前方向的径向风速
            v_los, _ = self.wind_field.get_radial_velocity(
                self.p.range_axis, azi, self.p.elevation_angle_deg, wind_type='constant'
            )

            # 临时累积器 (用于存放当前方位的平均谱)
            accum_spectrogram = None

            # --- 脉冲累积循环 ---
            for p_idx in tqdm(range(n_accum), desc="     脉冲累积", leave=False):
                # 1. 生成物理信号 (长时域)
                sig_complex = self.physics.simulate_ideal_signal(v_los_profile=v_los)

                # 2. 生成噪声 (动态长度)
                _, _, _, noise_total = self.noise_model.generate_total_noise(n_samples=len(sig_complex))

                # 3. 信噪比增强 (复现论文 2.4.3)
                # 仅在第一个脉冲计算缩放系数，后续保持一致以节省计算
                if p_idx == 0:
                    noise_p = np.var(noise_total)
                    sig_p_peak = np.max(np.abs(sig_complex) ** 2)
                    target_snr = 100.0  # 20dB

                    if sig_p_peak > 0:
                        scale_k = np.sqrt((noise_p * target_snr) / sig_p_peak)
                    else:
                        scale_k = 1.0

                total_signal = (sig_complex * scale_k) + noise_total

                # 4. 距离门切片与 FFT
                # 首次运行时确定维度
                if accum_spectrogram is None:
                    num_gates = len(total_signal) // gate_len
                    accum_spectrogram = np.zeros((num_gates, n_freq_bins))

                    # 计算距离轴
                    gate_time = gate_len / self.p.sample_rate
                    gate_res_m = (self.p.c * gate_time) / 2
                    range_gates = np.arange(num_gates) * gate_res_m + gate_res_m / 2

                # 对每个距离门做 FFT
                for gate_idx in range(num_gates):
                    seg = total_signal[gate_idx * gate_len: (gate_idx + 1) * gate_len]
                    # 补零 FFT (512 -> 1024)
                    fft_val = np.fft.fft(seg, n=n_fft)
                    # 累加功率谱
                    accum_spectrogram[gate_idx, :] += np.abs(fft_val[:n_freq_bins]) ** 2

            # C. 平均与存储
            avg_spectrogram = (accum_spectrogram / n_accum) * psd_norm

            # 初始化总数据矩阵
            if sim_data is None:
                sim_data = np.zeros((len(azimuths), num_gates, n_freq_bins))

            sim_data[i, :, :] = avg_spectrogram

        # 频率轴 (MHz)
        freq_axis = np.fft.fftfreq(n_fft, 1 / self.p.sample_rate)[:n_freq_bins] / 1e6

        return {
            'azimuths': azimuths,
            'range_gates': range_gates,
            'freq_axis': freq_axis,
            'data': sim_data  # Shape: [Azi, Gate, Freq]
        }

    def plot_figure_2_16(self, res):
        """
        复现图 2.16 (3D 瀑布图)
        展示: 第0个方位角, 距离 vs 频率 vs 功率(dB)
        关键: 必须切除低频 DC/RIN，否则看不见信号
        """
        print("正在绘制图 2.16 (3D)...")
        data = res['data'][0]  # 取第 0 个方位角 [Gates, Freqs]

        # [关键步骤] 切除前 5 MHz (去除 RIN 噪声墙)
        # 频率分辨率 ~1MHz, 切掉前 5 个点
        cut_idx = 0

        data_cut = data[:, cut_idx:]
        freq_axis_cut = res['freq_axis'][cut_idx:]
        range_axis = res['range_gates'] / 1000.0  # km

        # 转换为 dB (对数坐标才能看清弱信号)
        # 加 1e-30 防止 log(0)
        Z_dB = 10 * np.log10(data_cut + 1e-30)
        # [关键] 计算噪声的平均水平，用于设定色标
        # 取远处无信号区域的平均值
        noise_mean_db = np.mean(Z_dB[-10:, :])
        print(f"噪声基底平均水平: {noise_mean_db:.2f} dB")

        # 网格化
        X, Y = np.meshgrid(freq_axis_cut, range_axis)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        vmin = noise_mean_db - 5
        vmax = noise_mean_db + 10
        # 绘制
        surf = ax.plot_surface(X, Y, Z_dB, cmap='jet', linewidth=0, antialiased=False, vmin=vmin, vmax=vmax)

        ax.set_xlabel('Frequency (MHz)', fontproperties=plot_style.style.zh_font)
        ax.set_ylabel('Distance (km)', fontproperties=plot_style.style.zh_font)
        ax.set_zlabel('PSD (dB)', fontproperties=plot_style.style.zh_font)
        ax.set_title("3D Spectrogram of CDWL, Costant Wind Speed", fontproperties=plot_style.style.zh_font)

        # 视角调整
        ax.view_init(elev=35, azim=-70)
        plt.show()

    def plot_figure_2_17(self, res):
        """
        复现图 2.17 (2D 伪彩图)
        展示: 距离 vs 频率
        """
        print("正在绘制图 2.17 (2D)...")
        data = res['data'][0]

        cut_idx = 0
        data_cut = data[:, cut_idx:]
        freq_axis_cut = res['freq_axis'][cut_idx:]
        range_gates = res['range_gates']

        # 归一化 (让每个距离门的峰值都亮起来，方便观察频移)
        # axis=1 表示沿频率轴归一化
        data_norm = data_cut / np.max(data_cut, axis=1, keepdims=True)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(freq_axis_cut, range_gates, data_norm, cmap='jet', shading='auto')
        plt.colorbar(label='Normalized PSD')

        plot_style.style.apply_standard_layout(plt.gcf(), plt.gca(),
                                               title="Normalized PSD",
                                               xlabel="Frequency (MHz)", ylabel="Distance (m)")

        # 聚焦信号区域 (AOM 120MHz 附近)
        plt.xlim(0, 500)
        plt.ylim(0, 4000)
        plt.show()


if __name__ == "__main__":
    sim = LidarMainSimulator()

    # 1. 运行 "凝视模式" (Staring)
    # 目的: 生成单一方向的距离-频率图 (图 2.16/2.17)
    # n_accum=100 足够看清信号，若需极高质量可设为 1000
    results = sim.run_simulation(n_accum=50, scan_mode='Staring')

    # 2. 绘图
    sim.plot_figure_2_16(results)
    sim.plot_figure_2_17(results)

    print("\n提示: 如果您想看 PPI VAD 图 (S形曲线)，请将 scan_mode 改为 'PPI'")