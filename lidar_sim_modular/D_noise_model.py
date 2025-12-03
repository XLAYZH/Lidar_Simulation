import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # 进度条库

# 导入参数模块和绘图风格
from A_lidar_params import params
import S_plot_style as plot_style


class NoiseModel:
    """
    噪声仿真模块 (物理修正版)
    负责生成散粒噪声、热噪声、RIN噪声和探测器NEP噪声。
    """

    def __init__(self):
        self.p = params

        # --- RIN 参数 ---
        self.fr = 660e3  # 弛豫频率 (Hz)
        self.Gamma = 40.8e3  # 阻尼因子
        self.A = 2e12
        self.B = 0.1

        # --- NEP 参数 ---
        try:
            # [物理修正] NEP 单位转换 pW -> W
            raw_nep = np.load('nep_fit_smooth.npy')
            self.nep_profile = raw_nep * 1e-12
        except FileNotFoundError:
            print("Warning: 'nep_fit_smooth.npy' not found. Using zero noise for NEP.")
            self.nep_profile = np.zeros(self.p.points_per_bin)

        self.f_low = 10e3
        self.f_high = 200e6

    def calculate_gaussian_variance(self):
        """计算高斯白噪声方差 (A^2)"""
        shot_power = 2 * self.p.q_electron * self.p.responsivity * \
                     self.p.local_power * self.p.bandwidth
        thermal_power = 4 * self.p.k_boltzmann * self.p.temperature * \
                        self.p.bandwidth / self.p.load_resistance
        return shot_power, thermal_power

    def calculate_rin_psd(self):
        """计算 RIN 电流 PSD (A^2/Hz)"""
        omega = 2 * np.pi * self.p.freqs
        # 避免除零
        den = ((2 * np.pi * self.fr) ** 2 + self.Gamma ** 2 - omega ** 2) ** 2 + (4 * self.Gamma ** 2 * omega ** 2)
        den[den == 0] = 1e-20
        rin_linear = (self.A + self.B * omega ** 2) / den
        RIN_f_db = 10 * np.log10(rin_linear)

        # [物理修正] 电流 PSD 公式
        rin_psd_current = 2 * (self.p.responsivity * self.p.local_power) ** 2 * rin_linear
        return rin_psd_current, RIN_f_db

    def calculate_nep_psd(self):
        """计算 BDN 电流 PSD (A^2/Hz)"""
        H_f = 1 / np.sqrt(1 + (self.f_low / self.p.freqs) ** 16) / \
              np.sqrt(1 + (self.p.freqs / self.f_high) ** 16)

        n_pts = min(len(self.nep_profile), len(self.p.freqs))
        nep_part = self.nep_profile[:n_pts]
        resp_part = (self.p.responsivity * H_f)[:n_pts]

        nep_psd_current = (nep_part * resp_part) ** 2
        return nep_psd_current, resp_part

    def simulate_colored_noise_from_psd(self, psd_onesided):
        """从 PSD 生成有色噪声时域波形"""
        df = self.p.sample_rate / self.p.fft_points
        target_power = np.sum(psd_onesided * df)

        spec_mag = np.sqrt(psd_onesided)
        spec_double = np.concatenate((spec_mag, spec_mag[::-1]))
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(len(spec_double)))

        noise_time = np.fft.ifft(spec_double * random_phase).real

        # 能量校正
        current_var = np.var(noise_time)
        if current_var > 0:
            noise_time *= np.sqrt(target_power / current_var)
        return noise_time

    def generate_total_noise(self):
        """生成组合后的总噪声 (核心接口)"""
        shot_p, thermal_p = self.calculate_gaussian_variance()
        noise_gauss = np.random.normal(0, np.sqrt(shot_p + thermal_p), self.p.fft_points)

        rin_psd, _ = self.calculate_rin_psd()
        noise_rin = self.simulate_colored_noise_from_psd(rin_psd)

        nep_psd, _ = self.calculate_nep_psd()
        noise_nep = self.simulate_colored_noise_from_psd(nep_psd)

        total_noise = noise_gauss + noise_rin + noise_nep
        return noise_rin, noise_gauss, noise_nep, total_noise


# =============================================================================
# 独立绘图验证 (所有图表独立显示)
# =============================================================================
if __name__ == "__main__":
    nm = NoiseModel()

    # 1. 准备坐标轴
    t_axis = np.arange(params.points_per_bin) * params.time_step * 1e9  # ns
    f_axis_std = params.freqs / 1e6  # MHz (标准线性轴)

    # 高分辨率轴 (仅用于 RIN 理论曲线)
    f_plot_hz = np.logspace(5, 8, 2000)
    f_plot_mhz = f_plot_hz / 1e6

    # ==========================================
    # 1. 图 2.7: 散粒噪声与热噪声 (时域)
    # ==========================================
    print("绘图: 散粒与热噪声时域 (Fig 2.7)...")
    shot_p, thermal_p = nm.calculate_gaussian_variance()
    noise_shot = np.random.normal(0, np.sqrt(shot_p), 512) * 1e6  # uA
    noise_therm = np.random.normal(0, np.sqrt(thermal_p), 512) * 1e6  # uA

    # (a) 散粒噪声
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(111)
    ax1.plot(t_axis, noise_shot, color='brown', lw=0.8)
    plot_style.style.apply_standard_layout(fig1, ax1, title="散粒噪声", xlabel="时间 (ns)",
                                           ylabel="电流 ($\u00B5 A$)")
    ax1.set_xlim(0, 500)

    # (b) 热噪声
    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot(111)
    ax2.plot(t_axis, noise_therm, color='orange', lw=0.8)
    plot_style.style.apply_standard_layout(fig2, ax2, title="热噪声", xlabel="时间 (ns)",
                                           ylabel="电流 ($\u00B5 A$)")
    ax2.set_xlim(0, 500)

    plt.show()

    # ==========================================
    # 2. 图 2.9: RIN 噪声特性 (使用高分辨率轴)
    # ==========================================
    print("绘图: RIN 噪声特性 (Fig 2.9)...")

    # [切换轴]
    original_freqs = nm.p.freqs
    nm.p.freqs = f_plot_hz
    rin_psd_high, rin_db_high = nm.calculate_rin_psd()
    nm.p.freqs = original_freqs  # [恢复轴]

    # (a) RIN 频率响应 (dB)
    fig3 = plt.figure(figsize=(10, 10))
    ax3 = fig3.add_subplot(111)
    ax3.semilogx(f_plot_mhz, rin_db_high, color='red')
    plot_style.style.apply_standard_layout(fig3, ax3, title="RIN 频率特性", xlabel="频率 (MHz)",
                                           ylabel="RIN (dB/Hz)")
    ax3.set_xlim(0.1, 100)

    # (b) RIN 功率谱密度 (PSD)
    fig4 = plt.figure(figsize=(10, 10))
    ax4 = fig4.add_subplot(111)
    ax4.semilogx(f_plot_mhz, rin_psd_high, color='red')
    plot_style.style.apply_standard_layout(fig4, ax4, title="RIN 功率谱密度", xlabel="频率 (MHz)",
                                           ylabel="PSD ($A^2/Hz$)")
    ax4.set_xlim(0.1, 100)

    plt.show()

    # ==========================================
    # 3. 图 2.11: 平衡探测器噪声 (使用标准轴)
    # ==========================================
    print("绘图: 平衡探测器噪声特性 (Fig 2.11)...")
    nep_psd, resp = nm.calculate_nep_psd()
    f_axis_nep = f_axis_std[:len(resp)]

    # (b) 探测器响应
    fig5 = plt.figure(figsize=(10, 10))
    ax5 = fig5.add_subplot(111)
    ax5.plot(f_axis_nep, resp, color='blue')
    plot_style.style.apply_standard_layout(fig5, ax5, title="Thorlabs PDB460C平衡探测器带宽响应", xlabel="频率 (MHz)",
                                           ylabel="归一化响应")
    ax5.set_xlim(0, 500)

    # (c) BDN 功率谱密度
    fig6 = plt.figure(figsize=(10, 10))
    ax6 = fig6.add_subplot(111)
    ax6.plot(f_axis_nep, nep_psd, color='red')
    plot_style.style.apply_standard_layout(fig6, ax6, title="BDN 功率谱密度", xlabel="频率 (MHz)",
                                           ylabel="PSD ($A^2/Hz$)")
    ax6.set_xlim(0, 500)

    plt.show()

    # ==========================================
    # 4. 图 2.12: 有色噪声时域波形
    # ==========================================
    print("绘图: 有色噪声时域波形 (Fig 2.12)...")
    # 重新计算标准轴下的 PSD 用于 IFFT
    rin_psd_std, _ = nm.calculate_rin_psd()
    n_rin = nm.simulate_colored_noise_from_psd(rin_psd_std)[:512] * 1e6
    n_nep = nm.simulate_colored_noise_from_psd(nep_psd)[:512] * 1e6

    # (a) RIN 时域
    fig7 = plt.figure(figsize=(10, 10))
    ax7 = fig7.add_subplot(111)
    ax7.plot(t_axis, n_rin, color='teal', lw=0.8)
    plot_style.style.apply_standard_layout(fig7, ax7, title="相对强度噪声", xlabel="时间 (ns)",
                                           ylabel="电流 ($\u00B5 A$)")
    ax7.set_xlim(0)

    # (b) BDN 时域
    fig8 = plt.figure(figsize=(10, 10))
    ax8 = fig8.add_subplot(111)
    ax8.plot(t_axis, n_nep, color='steelblue', lw=0.8)
    plot_style.style.apply_standard_layout(fig8, ax8, title="平衡探测器噪声", xlabel="时间 (ns)",
                                           ylabel="电流 ($\u00B5 A$)")
    ax8.set_xlim(0)

    plt.show()

    # ==========================================
    # 5. 图 2.13 & 2.14: 脉冲累积效果 (耗时)
    # ==========================================
    print("绘图: 脉冲累积对PSD的影响 (Fig 2.13, 2.14)...")

    acc_list = [1, 10, 100, 1000]
    psd_avg_shot = np.zeros((len(acc_list), 512))
    psd_avg_therm = np.zeros((len(acc_list), 512))
    psd_avg_rin = np.zeros((len(acc_list), 512))
    psd_avg_nep = np.zeros((len(acc_list), 512))

    accum_shot = np.zeros(512);
    accum_therm = np.zeros(512)
    accum_rin = np.zeros(512);
    accum_nep = np.zeros(512)

    # 归一化因子 (转换为物理PSD A^2/Hz)
    psd_norm = 2.0 / (params.sample_rate * params.fft_points)

    for i in tqdm(range(1, 1001), desc="Accumulating"):
        # 生成单次噪声
        ns_shot = np.random.normal(0, np.sqrt(shot_p), params.fft_points)
        ns_therm = np.random.normal(0, np.sqrt(thermal_p), params.fft_points)
        ns_rin = nm.simulate_colored_noise_from_psd(rin_psd_std)
        ns_nep = nm.simulate_colored_noise_from_psd(nep_psd)

        # 累积 PSD
        accum_shot += np.abs(np.fft.fft(ns_shot)[:512]) ** 2
        accum_therm += np.abs(np.fft.fft(ns_therm)[:512]) ** 2
        accum_rin += np.abs(np.fft.fft(ns_rin)[:512]) ** 2
        accum_nep += np.abs(np.fft.fft(ns_nep)[:512]) ** 2

        if i in acc_list:
            idx = acc_list.index(i)
            psd_avg_shot[idx] = (accum_shot / i) * psd_norm
            psd_avg_therm[idx] = (accum_therm / i) * psd_norm
            psd_avg_rin[idx] = (accum_rin / i) * psd_norm
            psd_avg_nep[idx] = (accum_nep / i) * psd_norm

    # N=1: 蓝色, N=10: 橙色, N=100: 绿色, N=1000: 红色
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # 为了防止线条太粗遮挡，稍微调整线宽
    # N=1 时线条最细，N=1000 时线条稍粗，以突出重点
    linewidths = [0.5, 0.8, 1.0, 1.5]
    alphas = [0.6, 0.8, 0.9, 1.0]  # 透明度也渐变，让底层线条不那么“实”

    # (a) 散粒噪声累积
    fig9 = plt.figure(figsize=(10, 10))
    ax9 = fig9.add_subplot(111)
    for k, acc in enumerate(acc_list):
        ax9.plot(f_axis_std, psd_avg_shot[k], label=f'N={acc}', color=colors[k], lw=linewidths[k], alpha=alphas[k])
    plot_style.style.apply_standard_layout(fig9, ax9, title="散粒噪声累积功率谱密度", xlabel="频率 (MHz)",
                                           ylabel="PSD ($A^2/Hz$)")
    ax9.set_xlim(0, 500)
    ax9.legend()

    # (b) 热噪声累积
    fig10 = plt.figure(figsize=(10, 10))
    ax10 = fig10.add_subplot(111)
    for k, acc in enumerate(acc_list):
        ax10.plot(f_axis_std, psd_avg_therm[k], label=f'N={acc}', color=colors[k], lw=linewidths[k], alpha=alphas[k])
    plot_style.style.apply_standard_layout(fig10, ax10, title="热噪声累积功率谱密度", xlabel="频率 (MHz)",
                                           ylabel="PSD ($A^2/Hz$)")
    ax10.set_xlim(0, 500)
    ax10.legend()

    # (c) RIN 累积
    fig11 = plt.figure(figsize=(10, 10))
    ax11 = fig11.add_subplot(111)
    for k, acc in enumerate(acc_list):
        ax11.plot(f_axis_std, psd_avg_rin[k], label=f'N={acc}', color=colors[k], lw=linewidths[k], alpha=alphas[k])
    plot_style.style.apply_standard_layout(fig11, ax11, title="RIN 累积功率谱密度", xlabel="频率 (MHz)",
                                           ylabel="PSD ($A^2/Hz$)")
    ax11.set_xlim(0, 500)
    ax11.set_ylim(0, 8e-24)
    ax11.legend()

    # (d) BDN 累积
    fig12 = plt.figure(figsize=(10, 10))
    ax12 = fig12.add_subplot(111)
    for k, acc in enumerate(acc_list):
        ax12.plot(f_axis_std, psd_avg_nep[k], label=f'N={acc}', color=colors[k], lw=linewidths[k], alpha=alphas[k])
    plot_style.style.apply_standard_layout(fig12, ax12, title="BDN 累积功率谱密度", xlabel="频率 (MHz)",
                                           ylabel="PSD ($A^2/Hz$)")
    ax12.set_xlim(0, 500)
    ax12.set_ylim(0)
    ax12.legend()

    plt.show()
    print("所有图表绘制完成。")