import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# 导入基础模块
from A_lidar_params import params
from C_lidar_physics import LidarPhysics
import S_plot_style as plot_style


def verify_system():
    print(">>> 开始分步验证仿真系统 (Debug Mode) <<<")
    physics = LidarPhysics()

    # ==========================================
    # 验证 1: 信号随距离的衰减 (1/R^2)
    # ==========================================
    print("\n[验证 1] 检查距离衰减特性...")

    # 1. 生成纯信号
    v_zero = np.zeros_like(params.range_axis)
    # 注意: 这里返回的已经是实数(Real)信号了
    sig_zero = physics.simulate_ideal_signal(v_los_profile=v_zero)

    # 2. [关键修复] 生成匹配的高分辨率距离轴
    # 信号是时域的，我们要把 时间 -> 距离
    # 距离 = 索引 * (c / 2 / fs)
    n_points = len(sig_zero)
    dist_res_time = params.c / (2 * params.sample_rate)  # ~0.15m
    dist_axis_high_res = np.arange(n_points) * dist_res_time

    print(f"  - 信号长度: {n_points} 点")
    print(f"  - 绘图轴长度: {len(dist_axis_high_res)} 点")

    # 3. 计算包络 (取绝对值 + 平滑)
    envelope = np.abs(sig_zero)
    # 平滑处理，以便看清趋势 (窗口大小 500)
    smooth_env = uniform_filter1d(envelope, size=500)

    # 4. 绘图
    plt.figure(figsize=(10, 4))
    # 绘制原始波形太密集，只画背景灰线
    plt.plot(dist_axis_high_res, envelope, color='gray', alpha=0.3, lw=0.1, label='Instant Waveform')
    # 绘制平滑包络
    plt.plot(dist_axis_high_res, smooth_env, color='red', lw=1.5, label='Intensity Envelope (1/R²)')

    plot_style.style.apply_standard_layout(plt.gcf(), plt.gca(),
                                           title="Intensity of Signal vs Distance",
                                           xlabel="Distance (m)", ylabel="Current (A)")
    plt.xlim(0, 4000)
    # 避开近处盲区的极值，自动调整Y轴
    valid_y = smooth_env[dist_axis_high_res > 200]
    if len(valid_y) > 0:
        plt.ylim(0, np.max(valid_y) * 1.2)
    plt.legend()
    plt.show()

    # ==========================================
    # 验证 2: 120 MHz 载波与多普勒频移
    # ==========================================
    print("\n[验证 2] 检查 AOM 频率和多普勒频移...")

    # 场景 A: 静止
    sig_static = physics.simulate_ideal_signal(v_los_profile=np.zeros_like(params.range_axis))

    # 场景 B: 强风 (20 m/s 远离)
    v_wind = np.full_like(params.range_axis, 20.0)
    sig_moving = physics.simulate_ideal_signal(v_los_profile=v_wind)

    # 选取 500m 处的一个切片进行 FFT
    target_dist = 500  # m
    idx_start = int(target_dist / dist_res_time)
    gate_len = 2048  # 取长一点，频谱更细腻

    seg_static = sig_static[idx_start: idx_start + gate_len]
    seg_moving = sig_moving[idx_start: idx_start + gate_len]

    # FFT
    fft_static = np.abs(np.fft.fft(seg_static))[:gate_len // 2]
    fft_moving = np.abs(np.fft.fft(seg_moving))[:gate_len // 2]
    freqs = np.fft.fftfreq(gate_len, 1 / params.sample_rate)[:gate_len // 2] / 1e6  # MHz

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_static, label='Wind Speed (LOS) 0 m/s', color='blue')
    plt.plot(freqs, fft_moving, label='Wind Speed (LOS) +20 m/s', color='red')

    # 理论线
    plt.axvline(120, color='b', linestyle=':', alpha=0.5, label='Theoretical AOM Shift')
    # 20m/s 对应频移 ~25.8 MHz. 远离雷达 -> 频率减小 -> 120 - 25.8 = 94.2
    doppler_shift = 2 * 20.0 / params.wavelength / 1e6
    plt.axvline(120 - doppler_shift, color='r', linestyle=':', alpha=0.5, label='Theoretical Doppler Shift')

    plot_style.style.apply_standard_layout(plt.gcf(), plt.gca(),
                                           title=f"A Verification of Doppler Shift (@ {target_dist}m)",
                                           xlabel="Frequency (MHz)", ylabel="Intensity")
    plt.xlim(0, 500)  # 聚焦在 AOM 频率附近
    plt.legend()
    plt.show()

    # ==========================================
    # 验证 3: 距离门切片与 PSD 瀑布图
    # ==========================================
    print("\n[验证 3] 生成无噪声的 3D 瀑布图 (验证数据结构)...")

    # 使用常风速 -10 m/s (负号代表迎面风，频率应增加)
    v_const = np.full_like(params.range_axis, -10.0)
    sig_final = physics.simulate_ideal_signal(v_los_profile=v_const)

    # 切片参数
    gate_len_pts = 512  # 对应论文参数
    n_fft = 1024  # 补零到 1024
    num_gates = len(sig_final) // gate_len_pts

    spectrogram = np.zeros((num_gates, n_fft // 2))

    # 计算距离轴 (每个门的中心)
    gate_res_m = dist_res_time * gate_len_pts  # 每个门的物理长度 ~76.8m
    range_axis_gates = np.arange(num_gates) * gate_res_m + gate_res_m / 2

    for i in range(num_gates):
        seg = sig_final[i * gate_len_pts: (i + 1) * gate_len_pts]
        # 加窗减少泄漏
        win_seg = seg * np.hanning(len(seg))
        # FFT
        spec = np.abs(np.fft.fft(win_seg, n=n_fft))[:n_fft // 2]
        spectrogram[i, :] = spec ** 2

    freq_axis = np.fft.fftfreq(n_fft, 1 / params.sample_rate)[:n_fft // 2] / 1e6

    plt.figure(figsize=(9, 6))
    # X: Frequency, Y: Range
    # 使用 Log 标度看衰减
    plt.pcolormesh(freq_axis, range_axis_gates, 10 * np.log10(spectrogram + 1e-30), cmap='jet', shading='auto')
    plt.colorbar(label='Power (dB)')

    plot_style.style.apply_standard_layout(plt.gcf(), plt.gca(),
                                           title="Spectrogram vs Distance, Wind Speed: -10m/s (LOS)",
                                           xlabel="Frequency (MHz)", ylabel="Distance (m)")
    plt.xlim(0, 500)
    plt.ylim(0, 3840)
    plt.show()


if __name__ == "__main__":
    verify_system()