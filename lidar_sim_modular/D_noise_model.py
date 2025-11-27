import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

from matplotlib.pyplot import xlabel

import S_plot_style as plot_style

# 导入参数模块
from A_lidar_params import params

class NoiseModel:
    """
    噪声仿真模块。
    负责生成散粒噪声、热噪声、RIN噪声和探测器NEP噪声。
    参考文献: 论文 2.3 节
    """

    def __init__(self):
        self.p = params

        # --- RIN 参数 [cite: 620] ---
        self.fr = 660e3  # 弛豫频率 (Hz)
        self.Gamma = 40.8e3  # 阻尼因子
        self.A = 2e12
        self.B = 0.1

        # --- NEP 参数 ---
        # 加载 nep_fit_smooth.npy
        try:
            raw_nep = np.load('nep_fit_smooth.npy')
            self.nep_profile = raw_nep * 1e-12
        except FileNotFoundError:
            print("Warning: 'nep_fit_smooth.npy' not found. Using zero noise for NEP.")
            self.nep_profile = np.zeros(self.p.points_per_bin)
            # 注意：原代码这里加载的是512点，对应 fs/2 的频率轴

        self.f_low = 10e3  # 响应度低频截止
        self.f_high = 200e6  # 响应度高频截止 (带宽)

    def calculate_gaussian_variance(self):
        """
        计算高斯白噪声的方差 (电流功率 A^2)
        """
        # 1. 散粒噪声功率
        # <i^2> = 2 * e * R * P_LO * B
        shot_power = 2 * self.p.q_electron * self.p.responsivity * \
                     self.p.local_power * self.p.bandwidth

        # 2. 热噪声功率
        # <i^2> = 4 * k * T * B / R_load
        thermal_power = 4 * self.p.k_boltzmann * self.p.temperature * \
                        self.p.bandwidth / self.p.load_resistance

        return shot_power, thermal_power

    def simulate_gaussian_noise(self):
        """
        生成时域高斯白噪声 (散粒+热)
        """
        shot_p, thermal_p = self.calculate_gaussian_variance()
        total_std = np.sqrt(shot_p + thermal_p)

        # 生成随机噪声
        return np.random.normal(0, total_std, self.p.fft_points)

    def calculate_rin_psd(self):
        """
        计算 RIN 噪声的功率谱密度 (A^2/Hz)

        """
        freqs = self.p.freqs
        omega = 2 * np.pi * freqs
        omega_fr = 2 * np.pi * self.fr

        # RIN(f) 相对强度噪声谱 (dB/Hz)
        num = self.A + self.B * omega ** 2
        den = (omega_fr ** 2 + self.Gamma ** 2 - omega ** 2) ** 2 + (4 * self.Gamma ** 2 * omega ** 2)
        # 防止除零 (freqs[0]接近0)
        den[den == 0] = 1e-20

        rin_val_linear = num / den
        RIN_f_db = 10 * np.log10(rin_val_linear)  # 用于绘图验证

        # 转换为电流 PSD (A^2/Hz)
        # PSD = 2 * R^2 * P_LO^2 * 10^(RIN/10)  (系数2来自单边/双边谱定义或论文习惯)
        # 这里我们直接使用物理推导: I_rin^2 = P_LO^2 * 10^(RIN/10) * R^2
        rin_psd_current = 2 * (self.p.responsivity * self.p.local_power) ** 2 * rin_val_linear

        return rin_psd_current, RIN_f_db

    def calculate_nep_psd(self):
        """
        计算探测器(BDN)噪声的功率谱密度 (A^2/Hz)

        """
        # 探测器频率响应 H(f)
        # 使用16次方模拟陡峭的200MHz截止特性
        H_f = 1 / np.sqrt(1 + (self.f_low / self.p.freqs) ** 16) / \
              np.sqrt(1 + (self.p.freqs / self.f_high) ** 16)

        # 有效响应度
        resp_effective = self.p.responsivity * H_f

        # PSD = (NEP * R_eff)^2
        # 注意：self.nep_profile 长度可能只有 512 (对应正频率)，需匹配
        n_pts = min(len(self.nep_profile), len(self.p.freqs))
        nep_part = self.nep_profile[:n_pts]
        resp_part = resp_effective[:n_pts]

        nep_psd_current = (nep_part * resp_part) ** 2

        return nep_psd_current, resp_part

    def simulate_colored_noise_from_psd(self, psd_onesided):
        """
        通用方法：利用频域随机相位法从 PSD 生成时域噪声
        [cite: 492-512]
        """
        N = self.p.fft_points
        df = self.p.sample_rate / N

        # 1. 构造双边幅度谱
        # PSD(单边) -> 功率 P = PSD * df
        # 幅度 A = sqrt(P) * sqrt(N) (为了抵消IFFT的1/N系数，或者遵循能量守恒)
        # 修正逻辑：保持总能量守恒。
        # 时域方差 sigma^2 = sum(PSD * df)
        # IFFT后时域信号 variance 应该是这个值。

        # 简单实现：构造 sqrt(PSD)，然后 IFFT，再缩放
        target_power = np.sum(psd_onesided * df)  # 目标总功率 (A^2)

        # 构造双边谱 (对称)
        # psd_onesided 对应 [0, 1, ..., N/2-1]
        # 镜像翻转用于负频率
        spec_mag = np.sqrt(psd_onesided)
        spec_double = np.concatenate((spec_mag, spec_mag[::-1]))

        # 2. 添加随机相位
        random_phase = np.exp(1j * 2 * np.pi * np.random.rand(len(spec_double)))
        noise_freq = spec_double * random_phase

        # 3. IFFT
        noise_time = np.fft.ifft(noise_freq).real

        # 4. 幅度校正 (强制能量守恒)
        # 当前方差
        current_var = np.var(noise_time)
        if current_var > 0:
            scale = np.sqrt(target_power / current_var)
            noise_time *= scale

        return noise_time

    def generate_total_noise(self):
        """
        生成组合后的总噪声
        """
        # 1. 生成各分量
        shot_p, thermal_p = self.calculate_gaussian_variance()
        noise_gauss = np.random.normal(0, np.sqrt(shot_p + thermal_p), self.p.fft_points)

        rin_psd, _ = self.calculate_rin_psd()
        noise_rin = self.simulate_colored_noise_from_psd(rin_psd)

        nep_psd, _ = self.calculate_nep_psd()
        noise_nep = self.simulate_colored_noise_from_psd(nep_psd)

        # 2. 组合 (直接相加，因为它们是不相关的随机过程)
        # 原论文代码中有一个 "8 * nep" 的权重，这可能是为了模拟 8 个探测器阵列？
        # 或者是一个经验系数。为了物理严谨，我们这里先直接相加。
        # 如果为了复现原代码效果，可以取消下面注释：
        # noise_nep *= 8.0

        total_noise = noise_gauss + noise_rin + noise_nep

        return noise_rin, noise_gauss, noise_nep, total_noise


# =============================================================================
# 验证绘图部分 (复现论文图 2.7 - 2.14)
# =============================================================================
if __name__ == "__main__":
    nm = NoiseModel()
    t_axis = np.arange(params.points_per_bin) * params.time_step * 1e9  # ns
    f_axis = params.freqs / 1e6  # MHz

    # --- 1. 复现图 2.7 (散粒与热噪声时域) ---
    print("绘图 1/6: 散粒与热噪声时域...")
    shot_p, thermal_p = nm.calculate_gaussian_variance()
    noise_shot = np.random.normal(0, np.sqrt(shot_p), 512) * 1e6  # uA
    noise_therm = np.random.normal(0, np.sqrt(thermal_p), 512) * 1e6  # uA

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.plot(t_axis, noise_shot, color='brown', lw=0.8)
    plot_style.style.apply_standard_layout(fig, ax1, title="散粒噪声", xlabel="时间 ($ns$)", ylabel="电流 ($uA$)")
    ax2.plot(t_axis, noise_therm, color='orange', lw=0.8)
    plot_style.style.apply_standard_layout(fig, ax2, title="热噪声", xlabel="时间 ($ns$)", ylabel="电流 ($uA$)")
    plt.show()

    # --- 2. 复现图 2.9 (RIN 特性) ---
    print("绘图 2/6: RIN 噪声特性...")
    rin_psd, rin_db = nm.calculate_rin_psd()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.semilogx(f_axis, rin_db, color='red')
    plot_style.style.apply_standard_layout(fig, ax1, title="RIN频率特性", xlabel="频率 ($MHz$)", ylabel="RIN (dB/Hz)")
    ax1.set_xlim(0.1, 100)

    ax2.semilogx(f_axis, rin_psd, color='red')
    plot_style.style.apply_standard_layout(fig, ax2, title="RIN功率谱密度", xlabel="频率 ($MHz$)", ylabel="PSD ($A^2/Hz$)")
    ax2.set_xlim(0.1, 100)
    plt.show()

    # --- 3. 复现图 2.11 (NEP 特性) ---
    print("绘图 3/6: 平衡探测器噪声特性...")
    nep_psd, resp = nm.calculate_nep_psd()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.plot(f_axis[:len(resp)], resp, color='blue')
    plot_style.style.apply_standard_layout(fig, ax1, title="BDN 工作带宽频率特性", xlabel="频率 ($MHz$)", ylabel="归一化响应")
    ax1.set_xlim(0, 500)

    ax2.plot(f_axis, nep_psd, color='red')
    plot_style.style.apply_standard_layout(fig, ax2, title="BDN功率谱密度", xlabel="频率 ($MHz$)", ylabel="PSD ($A^2/Hz$)")
    ax2.set_xlim(0, 500)
    plt.show()

    # --- 4. 复现图 2.12 (有色噪声时域) ---
    print("绘图 4/6: 有色噪声时域波形...")
    n_rin = nm.simulate_colored_noise_from_psd(rin_psd)[:512] * 1e6  # uA
    n_nep = nm.simulate_colored_noise_from_psd(nep_psd)[:512] * 1e6  # uA

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.plot(t_axis, n_rin, color='teal', lw=0.8)
    plot_style.style.apply_standard_layout(fig, ax1, title="相对强度噪声", xlabel="时间 ($ns$)", ylabel="电流 ($\u00B5 A$)")

    ax2.plot(t_axis, n_nep, color='steelblue', lw=0.8)
    plot_style.style.apply_standard_layout(fig, ax2, title="平衡探测器噪声", xlabel="时间 ($ns$)", ylabel="电流 ($\u00B5 A$)")
    plt.show()

    # --- 5. 复现图 2.13/2.14 (脉冲累积效果 - 耗时!) ---
    print("绘图 5/6: 脉冲累积对PSD的影响 (Fig 2.13, 2.14)...")
    print("  正在进行 1000 次蒙特卡洛模拟，请稍候...")

    # 累积次数列表
    acc_list = [1, 10, 100, 1000]

    # 存储累积后的 PSD
    psd_avg_shot = np.zeros((len(acc_list), 512))
    psd_avg_nep = np.zeros((len(acc_list), 512))

    # 临时累加器
    accum_shot = np.zeros(512)
    accum_nep = np.zeros(512)

    for i in range(1, 1001):
        # 生成噪声
        n_shot = np.random.normal(0, np.sqrt(shot_p), 1024)
        n_nep_t = nm.simulate_colored_noise_from_psd(nep_psd)  # 1024点

        # 计算 PSD (Periodogram: |FFT|^2 / (fs*N))
        # 注意：这里为了显示清晰，可能不需要除以 fs*N，直接看相对值
        # 但为了物理严谨，我们还是算 |FFT|^2
        fft_shot = np.abs(np.fft.fft(n_shot)[:512]) ** 2
        fft_nep = np.abs(np.fft.fft(n_nep_t)[:512]) ** 2

        accum_shot += fft_shot
        accum_nep += fft_nep

        if i in acc_list:
            idx = acc_list.index(i)
            psd_avg_shot[idx] = accum_shot / i
            psd_avg_nep[idx] = accum_nep / i

    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['lightblue', 'skyblue', 'dodgerblue', 'navy']

    for k in range(len(acc_list)):
        ax1.plot(f_axis, 10 * np.log10(psd_avg_shot[k]), label=f'N={acc_list[k]}', color=colors[k], lw=0.8)
    ax1.set_title("散粒噪声 PSD 随累积次数变化 (dB)", fontproperties=plot_style.style.zh_font)
    ax1.legend()

    for k in range(len(acc_list)):
        ax2.plot(f_axis, 10 * np.log10(psd_avg_nep[k]), label=f'N={acc_list[k]}', color=colors[k], lw=0.8)
    ax2.set_title("BDN噪声 PSD 随累积次数变化 (dB)", fontproperties=plot_style.style.zh_font)
    ax2.legend()

    plt.show()
    print("NoiseModel 验证完成。")