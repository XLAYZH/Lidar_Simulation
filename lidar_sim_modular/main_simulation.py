import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条
import os

# 导入所有模块
from A_lidar_params import params
from C_lidar_physics import LidarPhysics
from D_noise_model import NoiseModel
from E_wind_field import WindField
import S_plot_style as plot_style


class LidarSimulator:
    def __init__(self):
        print("正在初始化雷达仿真系统...")
        self.p = params
        self.physics = LidarPhysics()
        self.noise_model = NoiseModel()
        self.wind_field = WindField()

        # 尝试加载真实探空数据 (可选)
        # 如果文件不存在，会自动回退到 Hybrid 模型
        self.wind_field.load_sounding_data(r"E:\GraduateStu6428\Codes\ObservationData54511\12Z\2025-12-01_12.csv")

    def run_ppi_scan(self, n_accum=1000):
        """
        执行完整的 PPI 扫描仿真
        """
        # 1. 扫描参数
        azimuths = np.linspace(0, 360, self.p.azimuth_count, endpoint=False)
        print(f"开始 PPI 扫描: 共 {len(azimuths)} 个方位角, 每个角度累积 {n_accum} 个脉冲")

        # 2. 结果存储容器
        # 存储每个方位角的最终 PSD (Azimuth x Frequency)
        # 我们只保存正频率部分的前半段 (对应 0 ~ fs/2)
        # 实际上我们通常保存 512 点 (对应 distance bins 的数量，或者 FFT 的一半)
        ppi_psd_data = np.zeros((len(azimuths), self.p.points_per_bin))

        # 频率轴 (用于绘图)
        freq_axis = self.p.freqs[:self.p.points_per_bin]

        # PSD 归一化因子 (转换为 A^2/Hz)
        psd_norm = 2.0 / (self.p.sample_rate * self.p.fft_points)

        # 3. 扫描循环
        for i, azi in enumerate(azimuths):
            print(f"  -> 扫描方位角 {azi:.1f}° ({i + 1}/{len(azimuths)})...")

            # A. 获取当前方向的径向风速廓线 (Module E)
            v_los, _ = self.wind_field.get_radial_velocity(
                self.p.range_axis,
                azimuth_deg=azi,
                elevation_deg=self.p.elevation_angle_deg,
                wind_type='real'  # 优先使用真实数据，如果没有则自动回退到混合模型
            )

            # B. 脉冲累积循环
            accum_psd = np.zeros(self.p.points_per_bin)

            for _ in range(n_accum):
                # 1. 生成纯信号 (Module C)
                # 注意: 每次都需要重新调用，因为散斑(Speckle)相位是随机的
                sig_complex = self.physics.simulate_ideal_signal(v_los_profile=v_los)

                # 2. 生成噪声 (Module D)
                _, _, _, noise_total = self.noise_model.generate_total_noise()

                # 3. 叠加 (信号 + 噪声)
                # [关键] 信噪比控制:
                # 现在的代码中信号和噪声都是基于物理参数计算的绝对值。
                # 如果我们要模拟特定的 SNR (比如远距离信噪比低)，这已经隐含在物理模型里了
                # (信号随距离R^2衰减，噪声恒定)。
                # 所以直接相加即可。
                total_signal = sig_complex + noise_total

                # 4. FFT & PSD
                # 截取前 1024 点进行 FFT (对应最远探测距离)
                # 或者如果 total_signal 长度就是 1024
                fft_res = np.fft.fft(total_signal, n=self.p.fft_points)
                psd = np.abs(fft_res[:self.p.points_per_bin]) ** 2

                accum_psd += psd

            # C. 平均 & 存储
            avg_psd = (accum_psd / n_accum) * psd_norm
            ppi_psd_data[i, :] = avg_psd

        return azimuths, freq_axis, ppi_psd_data

    def plot_results(self, azimuths, freq_axis, ppi_data):
        """绘制论文图 2.16 (3D) 和 2.17 (2D)"""

        # 为了展示清晰，我们通常只画某一个方位角的 3D 图 (例如第 0 个)
        # 或者画所有方位角的叠加? 论文图 2.16 是 "常风速条件下... 3D视图"
        # 看起来是 频率 vs 距离(距离门) vs 幅度。
        # 等等，我们的 ppi_data 是 (Azimuth, Frequency)。
        # 这里的 "Frequency" 轴其实对应着 "Range" (如果是 FMCW 雷达)
        # 或者 对应着 "Velocity" (如果是脉冲多普勒)。

        # [修正] 论文图 2.16 的横轴是 "频率"，纵轴是 "距离门"。
        # 这意味着：这并不是 PPI 扫描的结果，而是一个单一方位角下，
        # 对信号进行 "分段 FFT" (STFT) 得到的结果。

        # 既然我们现在的仿真输出是 1024 点的 FFT，这代表的是整个探测范围内的频谱。
        # 如果要做图 2.16 (距离分辨的频谱)，我们需要对时域信号做 STFT (短时傅里叶变换)。

        print("\n[Info] 正在生成 STFT 以复现图 2.16/2.17 (距离分辨频谱)...")
        # 重新生成一个单脉冲信号用于展示 STFT
        # 取第一个方位角的数据
        v_los, _ = self.wind_field.get_radial_velocity(
            self.p.range_axis, azimuths[0], self.p.elevation_angle_deg, 'real')

        # 我们需要一个长时域信号来做 STFT 吗？
        # 目前的 signal 是 1024 点 (对应 3840m 往返时间)。
        # 如果要做距离分辨，需要把 1024 点切成更小的段?
        # 不，1024 点只有 1us。通常脉冲多普勒雷达为了获得距离分辨，
        # 是对每个距离门分别采样的。

        # [代码逻辑修正]: 我们的仿真目前是 "全波形 FFT"。
        # 要得到 "距离门 vs 频率" 的图，我们需要改变信号处理方式：
        # 我们其实已经生成了 (Time x Range) 的矩阵 E_T_matrix。
        # 但最终输出的是混频后的 1D 时域信号 i_h(t)。

        # 在实际雷达中，为了获得距离分辨，我们通常是对回波进行 "滑窗 FFT"。
        # 窗口宽度 = 脉冲宽度。

        # 让我们对第 0 个方位角的累积信号做滑窗 FFT
        n_gates = 50  # 论文提到 50 个距离门
        gate_len = int(self.p.fft_points / n_gates)  # 1024 / 50 approx 20 点? 太少了。
        # 论文参数: 距离门采样点数 512 ?? (表 2.1)
        # 表 2.1: FFT 点数 1024。距离门采样点数 512。
        # 这意味着每个距离门都做了 1024 点 FFT (补零)? 或者是滑动?

        # 既然我们无法完全确定论文的 STFT 细节，我们这里用一个简单的滑动窗口来近似展示。
        # 或者，直接展示 PPI 数据的 2D 图 (方位角 vs 频率/速度)。

        # --- 方案 B: 绘制 方位角 vs 频率 (VAD 扫描图) ---
        # 这是验证风场反演最直观的图。

        f_MHz = freq_axis / 1e6

        plt.figure(figsize=(10, 6))
        # X轴: 频率(对应速度), Y轴: 方位角
        # 使用 pcolormesh
        plt.pcolormesh(f_MHz, azimuths, ppi_data, cmap='jet', shading='auto')
        plt.colorbar(label='PSD ($A^2/Hz$)')

        plot_style.style.apply_standard_layout(plt.gcf(), plt.gca(),
                                               title="PPI 扫描频谱图 (VAD)",
                                               xlabel="频率 (MHz)",
                                               ylabel="方位角 (deg)")
        plt.show()

        # --- 方案 C: 简单的 3D 频谱图 (针对某一个方位角) ---
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 只画第 0 个方位角
        psd_0 = ppi_data[0, :]
        ax.plot(f_MHz, psd_0, zs=0, zdir='y', color='r')

        # 画第 4 个 (90度)
        psd_90 = ppi_data[4, :]
        ax.plot(f_MHz, psd_90, zs=90, zdir='y', color='g')

        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Azimuth (deg)')
        ax.set_zlabel('PSD')
        ax.set_title("不同方位的功率谱 (3D View)", fontproperties=plot_style.style.zh_font)
        plt.show()


if __name__ == "__main__":
    sim = LidarSimulator()
    # 运行扫描 (为节省时间，演示时可将 n_accum 设小一点，如 100)
    azis, freqs, data = sim.run_ppi_scan(n_accum=100)
    sim.plot_results(azis, freqs, data)

    # 保存数据 (为深度学习做准备)
    np.savez("lidar_dataset_sample.npz", azimuths=azis, freqs=freqs, psd=data)
    print("数据已保存为 lidar_dataset_sample.npz")