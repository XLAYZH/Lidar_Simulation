import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import os

# 导入底层模块
from A_lidar_params import params
from C_lidar_physics import LidarPhysics
from D_noise_model import NoiseModel
from E_wind_field import WindField
import S_plot_style as plot_style


class LidarSimulator:
    def __init__(self):
        print(">>> 初始化激光雷达仿真器...")
        self.p = params
        self.physics = LidarPhysics()
        self.noise_model = NoiseModel()
        self.wind_field = WindField()

        # 尝试加载探空数据
        self.sounding_path = r"E:\GraduateStu6428\Codes\ObservationData54511\12Z\2025-12-01_12.csv"
        try:
            self.wind_field.load_sounding_data(self.sounding_path)
        except:
            print("注意: 未找到探空数据，将仅使用常风速或模型风场。")

    def _get_snr_profile_linear(self, num_gates):
        """
        生成沿距离门衰减的信噪比曲线 (-10dB -> -25dB)
        返回: 线性信噪比 (Signal_Power / Noise_Power)
        """
        snr_start_db = -10.0
        snr_end_db = -25.0

        # 生成线性的 dB 衰减序列
        snr_db_axis = np.linspace(snr_start_db, snr_end_db, num_gates)

        # 转换为线性功率比: SNR_lin = 10^(SNR_db / 10)
        snr_linear = 10 ** (snr_db_axis / 10.0)
        return snr_linear

    def simulate_single_radial(self, azimuth=0.0, wind_mode='constant', n_accum=50):
        """
        核心功能：仿真单个径向（单个方位角）的数据
        用于快速验证波形和频谱是否正确
        """
        # --- 1. 参数准备 ---
        gate_len = 512  # 距离门长度
        n_fft = self.p.fft_points  # 1024点
        n_freq = n_fft // 2

        # 计算距离分辨率
        # Range = c * t / 2 = c * (gate_len / fs) / 2
        range_res = (self.p.c * (gate_len / self.p.sample_rate)) / 2

        # 跑一次空的物理仿真，获取信号总长度，从而确定距离门数量
        dummy_sig = self.physics.simulate_ideal_signal(np.zeros_like(self.p.range_axis))
        total_points = len(dummy_sig)
        num_gates = total_points // gate_len

        # 生成距离轴 (取门中心)
        range_axis = (np.arange(num_gates) + 0.5) * range_res

        # 生成目标 SNR 曲线 (-10dB 到 -25dB)
        snr_profile = self._get_snr_profile_linear(num_gates)

        # PSD 归一化系数 (从 FFT模平方 -> A^2/Hz)
        psd_norm_factor = 2.0 / (self.p.sample_rate * n_fft)

        print(f">>> 开始单径向仿真: Azimuth={azimuth}°, Mode={wind_mode}, Accum={n_accum}")
        print(f"    距离门数: {num_gates}, 范围: {range_axis[0]:.1f}m - {range_axis[-1]:.1f}m")

        # 获取该方向的径向风速
        v_los, _ = self.wind_field.get_radial_velocity(
            self.p.range_axis, azimuth, self.p.elevation_angle_deg, wind_type=wind_mode
        )

        # 初始化累积容器
        psd_accum = np.zeros((num_gates, n_freq))

        # 用于波形验证的临时存储 (只存第0个脉冲)
        debug_waveforms = {
            'sig_components': [],  # 存放归一化后的信号分量
            'noise_components': [],  # 存放噪声分量
            'snr_targets': []  # 存放目标SNR
        }

        # --- 2. 脉冲累积循环 ---
        for p_idx in tqdm(range(n_accum), desc="Processing Pulses"):

            # A. 生成全长物理外差信号
            i_h_full = self.physics.simulate_ideal_signal(v_los_profile=v_los)

            # B. 逐距离门处理
            for g in range(num_gates):
                # 切片索引
                idx_start = g * gate_len
                idx_end = idx_start + gate_len

                # 1. 获取信号切片并归一化
                # 为了严格控制 SNR，我们需要消除物理信号自带的 1/R^2 衰减影响，
                # 将其视为标准波形 i_h(t)，然后乘上 sqrt(SNR) 系数
                i_h_slice = i_h_full[idx_start:idx_end]
                p_h = np.var(i_h_slice)

                if p_h > 1e-20:
                    i_h_norm = i_h_slice / np.sqrt(p_h)  # 归一化为单位功率
                else:
                    i_h_norm = np.zeros_like(i_h_slice)

                # 2. 生成该门的独立噪声
                _, _, _, i_n_slice = self.noise_model.generate_total_noise(n_samples=gate_len)
                p_n = np.var(i_n_slice)

                # 3. 计算叠加系数 (式 2.36)
                # i(t) = sqrt(SNR) * i_h + i_n
                # 目标: Signal_Power = SNR_linear * Noise_Power
                snr_lin = snr_profile[g]
                sig_coeff = np.sqrt(snr_lin * p_n)

                i_total = (sig_coeff * i_h_norm) + i_n_slice

                # (仅针对第0个脉冲) 保存波形数据用于后续验证1
                if p_idx == 0:
                    debug_waveforms['sig_components'].append(sig_coeff * i_h_norm)
                    debug_waveforms['noise_components'].append(i_n_slice)
                    debug_waveforms['snr_targets'].append(snr_lin)

                # 4. FFT 变换
                fft_res = np.fft.fft(i_total, n=n_fft)
                psd_gate = np.abs(fft_res[:n_freq]) ** 2

                # 累积
                psd_accum[g, :] += psd_gate

        # --- 3. 平均与归一化 ---
        # avg_psd = (psd_accum / n_accum) * psd_norm_factor
        avg_psd = psd_accum

        # 生成频率轴 (MHz)
        freq_axis = np.fft.fftfreq(n_fft, 1 / self.p.sample_rate)[:n_freq] / 1e6

        return {
            'data': avg_psd,  # [Gate, Freq]
            'range_axis': range_axis,
            'freq_axis': freq_axis,
            'debug_waveforms': debug_waveforms  # 包含时域波形数据
        }

    # =========================================================
    # 验证绘图函数 (英文标题/图例)
    # =========================================================

    def verify_1_spectral_comparison(self, res):
        """
        [Verification 1 - Modified]
        Frequency Domain Comparison at Near, Mid, and Far gates.
        Purpose: To verify that the useful signal peak decays with range,
                 while the noise floor remains constant.
        """
        print("\n[Verification 1] Plotting Frequency Domain Spectra...")

        waveforms = res['debug_waveforms']
        num_gates = len(waveforms['sig_components'])
        n_fft = self.p.fft_points
        fs = self.p.sample_rate

        # Select 3 representative gates
        gates_idx = [2, num_gates // 2, num_gates - 5]
        labels = ['Near Range (High SNR)', 'Mid Range (Mid SNR)', 'Far Range (Low SNR)']

        # Calculate Frequency Axis (MHz)
        freq_axis = np.fft.fftfreq(n_fft, 1 / fs)[:n_fft // 2] / 1e6

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        for i, g in enumerate(gates_idx):
            ax = axes[i]

            # 1. Retrieve Time Domain Snapshots (Single Pulse)
            sig_t = waveforms['sig_components'][g]  # Scaled Signal
            noise_t = waveforms['noise_components'][g]  # Noise
            total_t = sig_t + noise_t

            # 2. Perform FFT (to view in Frequency Domain)
            # Normalized to match the physical units A^2/Hz approx
            norm = 2.0 / (fs * n_fft)

            fft_total = np.fft.fft(total_t, n=n_fft)
            psd_total = np.abs(fft_total[:n_fft // 2]) ** 2 * norm

            fft_noise = np.fft.fft(noise_t, n=n_fft)
            psd_noise = np.abs(fft_noise[:n_fft // 2]) ** 2 * norm

            # 3. Plot
            # Plot Noise Floor Reference
            ax.plot(freq_axis, psd_noise, color='gray', alpha=0.4, linewidth=1, label='Noise Floor Only')
            # Plot Total Signal
            ax.plot(freq_axis, psd_total, color='blue', alpha=0.9, linewidth=1.5, label='Total Signal (Sig+Noise)')

            # Annotate Target SNR
            target_snr = waveforms['snr_targets'][g]
            target_snr_db = 10 * np.log10(target_snr)

            ax.set_title(f"{labels[i]} - Gate {g} - Target SNR: {target_snr_db:.1f} dB")
            ax.set_ylabel("PSD ($A^2/Hz$)")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            # Highlight the Signal Peak (around 120MHz or Doppler shifted)
            # Find peak in the signal region (e.g., 50-200MHz) to draw attention
            peak_idx = np.argmax(psd_total)
            ax.plot(freq_axis[peak_idx], psd_total[peak_idx], 'rx')

            if i == 2:
                ax.set_xlabel("Frequency (MHz)")

        plt.suptitle("Spectra at Different Ranges",fontsize=14)
        plt.tight_layout()
        plt.show()

    def verify_2_3d_psd(self, res):
        """
        验证2: 功率谱仿真数据三维展示
        X: Frequency, Y: Range Gate, Z: PSD (A^2/Hz)
        """
        print("\n[Verification 2] Plotting 3D PSD (Linear Scale)...")

        data = res['data']  # [Gate, Freq]
        freqs = res['freq_axis']
        ranges = res['range_axis']

        # 创建网格
        X, Y = np.meshgrid(freqs, ranges)
        Z = data  # 线性单位 A^2/Hz

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制曲面
        surf = ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=False)

        # 英文标签
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Range Gate (m)')
        ax.set_zlabel('PSD ($A^2/Hz$)')
        ax.set_title("3D Power Spectral Density", fontsize=12)

        # 添加色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='PSD ($A^2/Hz$)')

        # 调整视角以便观察
        ax.view_init(elev=30, azim=-60)
        plt.show()

    def verify_3_2d_heatmap(self, res):
        """
        验证3: 功率谱二维展示 (归一化)
        横轴: Frequency, 纵轴: Range Gate, Color: Normalized PSD
        """
        print("\n[Verification 3] Plotting 2D Normalized Spectrogram...")

        data = res['data']
        freqs = res['freq_axis']
        ranges = res['range_axis']

        # 归一化处理: 每个距离门的数据除以该门的最大值
        # 这样可以消除距离衰减的影响，清晰看到远处微弱信号的频移
        max_per_gate = np.max(data, axis=1, keepdims=True)
        # 防止除零
        max_per_gate[max_per_gate == 0] = 1.0
        data_norm = data / max_per_gate

        plt.figure(figsize=(10, 6))

        # 绘制伪彩图
        plt.pcolormesh(freqs, ranges, data_norm, cmap='jet', shading='auto')

        # 英文标签
        plt.colorbar(label='Normalized PSD (0-1)')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Range Gate (m)')
        plt.title("2D Normalized Spectrogram", fontsize=12)

        # 限制显示范围 (聚焦 0-250MHz)
        # plt.xlim(0, 250)
        plt.ylim(0, ranges[-1])
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.show()


if __name__ == "__main__":
    # 实例化仿真器
    sim = LidarSimulator()

    # === 阶段 1: 单径向调试 (Single Radial Debug) ===
    # 设定参数: 常风速, 累积50次用于快速查看
    print("\n=== STEP 1: Running Single Radial Simulation for Verification ===")
    radial_results = sim.simulate_single_radial(azimuth=0.0, wind_mode='constant', n_accum=50)

    # 立即执行三个验证
    sim.verify_1_spectral_comparison(radial_results)  # 验证波形叠加公式是否正确
    sim.verify_2_3d_psd(radial_results)  # 验证能量分布 (线性坐标)
    sim.verify_3_2d_heatmap(radial_results)  # 验证频率提取 (归一化坐标)

    # === 阶段 2: 全方位扫描 (Full Scan) ===
    # 只有当上述图形正确后，您再取消下面的注释进行全扫描
    """
    print("\n=== STEP 2: Running Full 16-Azimuth Scan ===")
    full_scan_data = []
    azimuths = np.linspace(0, 360, 16, endpoint=False)

    for azi in azimuths:
        res = sim.simulate_single_radial(azimuth=azi, wind_mode='constant', n_accum=50)
        full_scan_data.append(res['data'])

    full_scan_data = np.array(full_scan_data) # [16, Gates, Freq]
    print("Full scan complete. Data shape:", full_scan_data.shape)
    """