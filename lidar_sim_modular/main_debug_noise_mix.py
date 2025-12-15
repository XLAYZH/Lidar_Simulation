import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# 导入核心模块
from A_lidar_params import params
from C_lidar_physics import LidarPhysics
from D_noise_model import NoiseModel
import S_plot_style as plot_style


def verify_signal_noise_mixing():
    print(">>> 开始 [信号+噪声] 混合验证 <<<")

    # 初始化
    physics = LidarPhysics()
    nm = NoiseModel()

    # 1. 生成物理信号 (设定一个明显的风速)
    # 设定 -15 m/s (迎面风)，预期频移 +19.3 MHz -> 峰值在 139.3 MHz 附近
    print("\n[步骤 1] 生成纯净物理信号...")
    v_test = np.full_like(params.range_axis, -15.0)
    sig_pure = physics.simulate_ideal_signal(v_los_profile=v_test)

    # 2. 生成噪声 (动态长度)
    print(f"[步骤 2] 生成噪声 (长度 {len(sig_pure)})...")
    _, _, _, noise_real = nm.generate_total_noise(n_samples=len(sig_pure))

    # 3. 截取第一个距离门的数据进行分析 (Near Field)
    # 论文要求: 第一个距离门 SNR = 20dB
    gate_len = 512
    # 为了 FFT 分辨率更高，我们这里取前 1024 点进行分析
    # 注意: 这里的切片只是为了计算 SNR 和画图验证
    analysis_len = 1024

    sig_segment = sig_pure[:analysis_len]
    noise_segment = noise_real[:analysis_len]

    # ==========================================
    # 对比 A: 原始物理混合 (Raw Mix)
    # ==========================================
    mix_raw = sig_segment + noise_segment

    # 计算原始 SNR
    p_sig = np.var(sig_segment)
    p_noise = np.var(noise_segment)
    snr_raw_linear = p_sig / p_noise
    snr_raw_db = 10 * np.log10(snr_raw_linear)

    print(f"\n[对比 A] 原始物理状态:")
    print(f"  - 信号功率: {p_sig:.2e}")
    print(f"  - 噪声功率: {p_noise:.2e}")
    print(f"  - 原始 SNR: {snr_raw_db:.2f} dB (非常低，信号被淹没)")

    # ==========================================
    # 对比 B: 强制增强混合 (Scaled Mix)
    # ==========================================
    target_snr_db = 20.0
    target_snr_linear = 10 ** (target_snr_db / 10)  # 100

    # 计算缩放因子 k
    # (k * sig)^2 / noise = 100  =>  k^2 * (p_sig/p_noise) = 100
    # k = sqrt(100 / current_snr_linear)
    scale_factor = np.sqrt(target_snr_linear / snr_raw_linear)

    mix_scaled = (sig_segment * scale_factor) + noise_segment

    print(f"\n[对比 B] 强制增强状态 (目标 20dB):")
    print(f"  - 缩放因子 k: {scale_factor:.2f}")
    print(f"  - 增强后 SNR: {10 * np.log10(np.var(sig_segment * scale_factor) / p_noise):.2f} dB")

    # ==========================================
    # 绘图验证 (频域 PSD 对比)
    # ==========================================
    print("\n[绘图] 生成频谱对比图...")

    # 计算 PSD (简单周期图法)
    def calc_psd(signal):
        fft_res = np.fft.fft(signal, n=1024)  # 补零到 1024
        # 取前一半 (0-500MHz)
        psd = np.abs(fft_res[:512]) ** 2
        return psd

    psd_pure = calc_psd(sig_segment)
    psd_raw = calc_psd(mix_raw)
    psd_scaled = calc_psd(mix_scaled)

    freqs = np.fft.fftfreq(1024, 1 / params.sample_rate)[:512] / 1e6  # MHz

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 图1: 原始混合 (模拟真实物理探测)
    ax1.plot(freqs, 10 * np.log10(psd_raw + 1e-30), color='gray', alpha=0.6, label='Raw Signal (Signal & Noise)')
    ax1.plot(freqs, 10 * np.log10(psd_pure + 1e-30), color='green', linestyle='--', lw=1, label='Pure Signal (Signal Only)')

    plot_style.style.apply_standard_layout(fig, ax1,
                                           title=f"场景A: 原始物理混合 (SNR={snr_raw_db:.1f}dB)",
                                           xlabel="频率 (MHz)", ylabel="PSD (dB)")
    ax1.legend(loc='lower right')
    # ax1.set_ylim(-50, 100)  # 根据实际情况调整
    ax1.set_xlim(0, 200)  # 关注低频区

    # 标记 RIN 噪声墙
    ax1.text(10, 80, "RIN \n(High Intensity at Low Frequency)", color='red', fontsize=9)

    # 图2: 增强混合 (模拟论文算法评估)
    ax2.plot(freqs, 10 * np.log10(psd_scaled + 1e-30), color='blue', alpha=0.8, label='Scaled Signal (Signal Amplified)')
    ax2.plot(freqs, 10 * np.log10(psd_raw + 1e-30), color='gray', alpha=0.3, label='Raw Signal')

    plot_style.style.apply_standard_layout(fig, ax2,
                                           title=f"场景B: 增强混合 (SNR={target_snr_db:.1f}dB)",
                                           xlabel="频率 (MHz)", ylabel="PSD (dB)")

    # 标记信号峰
    peak_freq_theory = 120 + (2 * 15.0 / 1.55e-6 / 1e6)  # ~139.3 MHz
    ax2.axvline(peak_freq_theory, color='r', linestyle=':', label='Doppler Shift')
    ax2.text(peak_freq_theory + 5, np.max(10 * np.log10(psd_scaled)), "Peak", color='red')

    ax2.legend(loc='lower right')
    ax2.set_xlim(0, 200)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    verify_signal_noise_mixing()