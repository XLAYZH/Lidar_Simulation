import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 导入所有模块
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

        # 尝试加载探空数据 (可选)
        sounding_file = r"E:\GraduateStu6428\Codes\ObservationData54511\12Z\2025-12-01_12.csv"
        try:
            self.wind_field.load_sounding_data(sounding_file)
        except:
            pass

    def run_simulation(self, n_accum=1000, scan_mode='PPI', wind_type='real'):
        """
        执行仿真
        参数:
            n_accum: 累积脉冲数
            scan_mode: 'PPI' (圆锥扫描) 或 'Staring' (单径向凝视)
            wind_type: 'real' (探空), 'power_law' (指数律), 'hybrid' (混合), 'constant' (常风速)
        """
        if scan_mode == 'PPI':
            azimuths = np.linspace(0, 360, self.p.azimuth_count, endpoint=False)
        else:
            azimuths = [0.0]
            print(f">>> 进入凝视模式 (Staring Mode), Azimuth=0°, Wind Type={wind_type}")

        # 定义距离门参数
        gate_len = 512
        n_freq_bins = self.p.fft_points // 2
        psd_norm = 2.0 / (self.p.sample_rate * self.p.fft_points)

        sim_data = None
        range_gates = None
        freq_axis = None

        print(f">>> 开始仿真 (模式: {scan_mode}, 风场: {wind_type}, 累积: {n_accum})")

        for i, azi in enumerate(azimuths):
            if scan_mode == 'PPI':
                print(f"  -> 扫描方位角 {azi:.1f}° ({i + 1}/{len(azimuths)})")

            # [关键] 这里传入 wind_type
            v_los, _ = self.wind_field.get_radial_velocity(
                self.p.range_axis, azi, self.p.elevation_angle_deg, wind_type=wind_type
            )

            accum_spectrogram = None

            for p_idx in tqdm(range(n_accum), desc="     脉冲累积", leave=False):
                # 1. 生成物理信号
                sig_complex = self.physics.simulate_ideal_signal(v_los_profile=v_los)

                # 2. 生成噪声 (动态长度)
                _, _, _, noise_total = self.noise_model.generate_total_noise(n_samples=len(sig_complex))

                # 3. 信噪比增强 (强制首个距离门 SNR=20dB)
                if p_idx == 0:
                    noise_power = np.var(noise_total)
                    sig_power_peak = np.max(np.abs(sig_complex) ** 2)
                    target_snr_linear = 100.0  # 20dB

                    if sig_power_peak > 0:
                        scale_factor = np.sqrt((noise_power * target_snr_linear) / sig_power_peak)
                    else:
                        scale_factor = 1.0

                total_signal = (sig_complex * scale_factor) + noise_total

                # 4. 距离门切片与 FFT
                if accum_spectrogram is None:
                    num_gates = len(total_signal) // gate_len
                    accum_spectrogram = np.zeros((num_gates, n_freq_bins))
                    gate_res_m = (self.p.c * (gate_len / self.p.sample_rate)) / 2
                    range_gates = np.arange(num_gates) * gate_res_m + gate_res_m / 2

                for gate_idx in range(num_gates):
                    start = gate_idx * gate_len
                    end = start + gate_len
                    segment = total_signal[start:end]

                    fft_res = np.fft.fft(segment, n=self.p.fft_points)
                    psd = np.abs(fft_res[:n_freq_bins]) ** 2
                    accum_spectrogram[gate_idx, :] += psd

            avg_spectrogram = (accum_spectrogram / n_accum) * psd_norm

            if sim_data is None:
                sim_data = np.zeros((len(azimuths), num_gates, n_freq_bins))

            sim_data[i, :, :] = avg_spectrogram

        freq_axis = np.fft.fftfreq(self.p.fft_points, 1 / self.p.sample_rate)[:n_freq_bins] / 1e6

        return {
            'azimuths': azimuths,
            'range_gates': range_gates,
            'freq_axis': freq_axis,
            'data': sim_data
        }

    def plot_figure_2_16(self, res):
        print("绘制图 2.16 (3D瀑布图)...")
        data = res['data'][0]
        start_idx = 5
        data_cut = data[:, start_idx:]
        range_axis = res['range_gates'] / 1000.0
        freq_axis_cut = res['freq_axis'][start_idx:]

        X, Y = np.meshgrid(freq_axis_cut, range_axis)
        # 使用 dB 标度
        Z_log = 10 * np.log10(data_cut + 1e-30)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z_log, cmap='jet', linewidth=0, antialiased=False)

        ax.set_xlabel('频率 (MHz)', fontproperties=plot_style.style.zh_font)
        ax.set_ylabel('距离 (km)', fontproperties=plot_style.style.zh_font)
        ax.set_zlabel('PSD (dB)', fontproperties=plot_style.style.zh_font)
        ax.set_title("图 2.16 仿真复现 (常风速 10m/s)", fontproperties=plot_style.style.zh_font)

        ax.view_init(elev=40, azim=-60)
        plt.show()

    def plot_figure_2_17(self, res):
        print("绘制图 2.17 (2D伪彩图)...")
        data = res['data'][0]
        start_idx = 5
        data_cut = data[:, start_idx:]
        freq_axis_cut = res['freq_axis'][start_idx:]
        range_gates = res['range_gates']

        data_norm = data_cut / np.max(data_cut, axis=1, keepdims=True)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(freq_axis_cut, range_gates, data_norm, cmap='jet', shading='auto')
        plt.colorbar(label='归一化 PSD')

        plot_style.style.apply_standard_layout(plt.gcf(), plt.gca(),
                                               title="图 2.17 仿真复现 (常风速)",
                                               xlabel="频率 (MHz)",
                                               ylabel="距离 (m)")
        plt.xlim(50, 200)
        plt.show()


if __name__ == "__main__":
    sim = LidarMainSimulator()

    # [修改] 使用 'constant' 风场进行测试
    # 理论上应该看到一条笔直的信号线
    results = sim.run_simulation(n_accum=100, scan_mode='Staring', wind_type='constant')

    sim.plot_figure_2_16(results)
    sim.plot_figure_2_17(results)