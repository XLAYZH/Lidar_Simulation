import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from tqdm import tqdm

from A_lidar_params import params
import S_plot_style as plot_style


class NoiseModel:
    """
    噪声仿真模块（可直接运行版）

    目标：
    1. 用一侧 PSD 严格生成实值高斯随机过程；
    2. 统一在频域定义散粒噪声、热噪声、RIN 与 BDN；
    3. 使输出时域噪声在多次平均后能够回到目标 PSD 包络；
    4. 与主仿真程序接口保持兼容；
    5. 保留原始 D_noise_model.py 中的全部独立验证绘图功能。

    当前默认建模约定：
    - 散粒噪声、热噪声：按“理想白噪声源”处理，默认不乘平衡探测器响应；
    - RIN：保留其解析频谱形状，默认不再额外乘接收机响应；
    - BDN：作为接收机相关噪声，默认保留接收机响应整形；
    - 所有 PSD 均为一侧定义，单位 A^2/Hz。
    """

    def __init__(self):
        self.p = params

        # --- RIN 参数 ---
        self.fr = 660e3      # 弛豫频率 (Hz)
        self.Gamma = 40.8e3  # 阻尼因子 (Hz)
        self.A = 2e12
        self.B = 0.1

        # --- 接收机近似通带 ---
        self.f_low = 10e3
        self.f_high = 200e6

        # --- 建模开关 ---
        # 对于散粒噪声与热噪声，这里默认采用“理想白噪声源”定义：
        # 一侧 PSD 在整个仿真频带内保持常数，不再额外乘接收机响应。
        # 这样其累积平均 PSD 将与频率无关，符合白噪声验证的理论预期。
        # 若后续需要研究接收机输出端的带宽整形效应，可将该开关改为 True。
        self.shape_white_noise = False

        # RIN 是否乘接收机响应。默认关闭，以保留其解析谱型本身。
        self.shape_rin = False

        # BDN 来源于 NEP 曲线，本身代表接收机相关噪声，默认保留响应整形。
        self.shape_bdn = True

        # --- BDN / NEP 数据 ---
        self.nep_profile = None
        self.nep_interp = None
        self._load_nep_profile()

    def print_model_configuration(self):
        """打印当前噪声模型配置，便于核查白噪声是否被接收机响应整形。"""
        print("\n=== Noise model configuration ===")
        print(f"shape_white_noise = {self.shape_white_noise}  (False 表示散粒噪声/热噪声保持平坦 PSD)")
        print(f"shape_rin         = {self.shape_rin}  (False 表示 RIN 保持解析谱型本身)")
        print(f"shape_bdn         = {self.shape_bdn}  (True  表示 BDN 保留接收机响应整形)")
        if self.nep_interp is None:
            print("NEP profile        = not loaded")
        else:
            print("NEP profile        = loaded" if np.any(self.nep_profile > 0) else "NEP profile        = zero fallback")
        print("=================================\n")

    # ------------------------------------------------------------------
    # 基础频率响应与 PSD 定义
    # ------------------------------------------------------------------
    def _load_nep_profile(self):
        """加载平滑 NEP 数据，并构造插值函数。"""
        candidate_paths = [
            Path(__file__).resolve().parent / "nep_fit_smooth.npy",
            Path.cwd() / "nep_fit_smooth.npy",
        ]

        raw_nep = None
        for pth in candidate_paths:
            if pth.exists():
                raw_nep = np.load(pth)
                break

        if raw_nep is None:
            print("Warning: 'nep_fit_smooth.npy' not found. Using zero noise for NEP.")
            self.nep_profile = np.zeros(self.p.fft_points // 2 + 1, dtype=float)
            base_freqs = np.linspace(0.0, self.p.sample_rate / 2.0, len(self.nep_profile))
            self.nep_interp = interp1d(
                base_freqs,
                self.nep_profile,
                bounds_error=False,
                fill_value=(self.nep_profile[0], self.nep_profile[-1]),
            )
            return

        # 假设原始单位为 pW/sqrt(Hz)，转换为 W/sqrt(Hz)
        self.nep_profile = np.asarray(raw_nep, dtype=float) * 1e-12
        base_freqs = np.linspace(0.0, self.p.sample_rate / 2.0, len(self.nep_profile))
        self.nep_interp = interp1d(
            base_freqs,
            self.nep_profile,
            bounds_error=False,
            fill_value=(self.nep_profile[0], self.nep_profile[-1]),
        )

    def receiver_response(self, freqs: np.ndarray) -> np.ndarray:
        """
        接收机幅频响应 H(f)。
        这里采用与原程序相同形式的近似带通：
        - 低频端抑制直流与极低频起伏；
        - 高频端体现有限电带宽滚降。
        """
        freqs = np.asarray(freqs, dtype=float)
        f_safe = np.maximum(freqs, 1e-30)

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            h_low = 1.0 / np.sqrt(1.0 + (self.f_low / f_safe) ** 16)
            h_high = 1.0 / np.sqrt(1.0 + (f_safe / self.f_high) ** 16)
        return h_low * h_high

    def frequency_axis(self, n_samples: int) -> np.ndarray:
        """返回 rFFT 对应的一侧频率轴。"""
        return np.fft.rfftfreq(n_samples, d=1.0 / self.p.sample_rate)

    def calculate_shot_psd(self, freqs: np.ndarray | None = None) -> np.ndarray:
        """
        计算散粒噪声一侧 PSD。

        默认采用理想白噪声源模型：
            S_shot = 2 q I_LO
        其 PSD 与频率无关。

        若 self.shape_white_noise=True，则再乘以 |H(f)|^2，表示接收机输出端噪声。
        """
        if freqs is None:
            freqs = self.frequency_axis(self.p.fft_points)
        freqs = np.asarray(freqs, dtype=float)
        i_lo = self.p.responsivity * self.p.local_power
        s0 = 2.0 * self.p.q_electron * i_lo
        psd = np.full_like(freqs, s0, dtype=float)
        if self.shape_white_noise:
            psd *= self.receiver_response(freqs) ** 2
        return psd

    def calculate_thermal_psd(self, freqs: np.ndarray | None = None) -> np.ndarray:
        """
        计算热噪声一侧 PSD。

        默认采用理想白噪声源模型：
            S_th = 4 k T / R
        其 PSD 与频率无关。

        若 self.shape_white_noise=True，则再乘以 |H(f)|^2，表示接收机输出端噪声。
        """
        if freqs is None:
            freqs = self.frequency_axis(self.p.fft_points)
        freqs = np.asarray(freqs, dtype=float)
        s0 = 4.0 * self.p.k_boltzmann * self.p.temperature / self.p.load_resistance
        psd = np.full_like(freqs, s0, dtype=float)
        if self.shape_white_noise:
            psd *= self.receiver_response(freqs) ** 2
        return psd

    def calculate_gaussian_variance(self):
        """
        返回散粒噪声与热噪声在当前配置下的总方差（A^2）。

        说明：
        - 当 self.shape_white_noise=False 时，对应理想白噪声源在当前仿真带宽内的积分方差；
        - 当 self.shape_white_noise=True 时，对应经接收机响应整形后的积分方差。

        该函数主要用于与旧版接口兼容和快速检查。
        """
        freqs = self.frequency_axis(self.p.fft_points)
        df = freqs[1] - freqs[0]
        shot_var = np.sum(self.calculate_shot_psd(freqs)) * df
        thermal_var = np.sum(self.calculate_thermal_psd(freqs)) * df
        return shot_var, thermal_var

    def calculate_rin_psd(self, freqs: np.ndarray | None = None):
        """
        计算 RIN 电流 PSD。

        仍沿用原代码的 RIN 解析形式：
            RIN(f) = (A + B * omega^2) / den
        再换算为电流 PSD，并乘接收机带宽响应。
        """
        if freqs is None:
            freqs = self.frequency_axis(self.p.fft_points)

        freqs = np.asarray(freqs, dtype=float)
        omega = 2.0 * np.pi * freqs
        den = ((2.0 * np.pi * self.fr) ** 2 + self.Gamma ** 2 - omega ** 2) ** 2 + \
              (4.0 * self.Gamma ** 2 * omega ** 2)
        den = np.maximum(den, 1e-30)

        rin_linear = (self.A + self.B * omega ** 2) / den
        rin_db = 10.0 * np.log10(np.maximum(rin_linear, 1e-300))

        i_lo = self.p.responsivity * self.p.local_power
        rin_psd_current = 2.0 * (i_lo ** 2) * rin_linear
        if self.shape_rin:
            rin_psd_current *= self.receiver_response(freqs) ** 2
        return rin_psd_current, rin_db

    def calculate_nep_psd(self, freqs: np.ndarray | None = None):
        """
        计算 BDN（基于 NEP 曲线）对应的电流 PSD。

        若 nep_fit_smooth.npy 的单位是 W/sqrt(Hz) 的输入等效光功率噪声，
        则输出电流幅度谱密度约为 R * H(f) * NEP(f)，其 PSD 为平方。
        """
        if freqs is None:
            freqs = self.frequency_axis(self.p.fft_points)

        freqs = np.asarray(freqs, dtype=float)
        nep_vals = self.nep_interp(freqs) if self.nep_interp is not None else np.zeros_like(freqs)
        if self.shape_bdn:
            resp_part = self.p.responsivity * self.receiver_response(freqs)
        else:
            resp_part = np.full_like(freqs, self.p.responsivity, dtype=float)
        nep_psd_current = (nep_vals * resp_part) ** 2
        return nep_psd_current, resp_part

    # ------------------------------------------------------------------
    # 频域到时域：严格 PSD 合成
    # ------------------------------------------------------------------
    def simulate_colored_noise_from_psd(
        self,
        psd_onesided: np.ndarray,
        n_samples: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        由一侧 PSD 严格生成实值高斯时域噪声。

        设采样率为 fs，记录长度为 N，频率分辨率 df = fs / N。
        对于内部频点 k=1,...,N/2-1，有
            E[|X[k]|^2] = 0.5 * N * fs * S1(f_k)
        对于 DC / Nyquist 点，则
            E[|X[k]|^2] = N * fs * S1(f_k)
        其中 S1 为一侧 PSD。

        这样在使用普通 periodogram：
            Pxx = |rfft(x)|^2 / (fs*N)
        并对内部点乘 2 后，其多次平均将收敛到目标 S1(f)。
        """
        if rng is None:
            rng = np.random.default_rng()

        psd_onesided = np.asarray(psd_onesided, dtype=float)
        psd_onesided = np.maximum(psd_onesided, 0.0)

        if n_samples is None:
            n_samples = 2 * (len(psd_onesided) - 1)

        n_rfft = n_samples // 2 + 1
        if len(psd_onesided) != n_rfft:
            src_f = np.linspace(0.0, self.p.sample_rate / 2.0, len(psd_onesided))
            dst_f = self.frequency_axis(n_samples)
            psd_onesided = np.interp(dst_f, src_f, psd_onesided)

        fs = self.p.sample_rate
        spec = np.zeros(n_rfft, dtype=np.complex128)

        # DC
        spec[0] = rng.normal(0.0, np.sqrt(fs * n_samples * psd_onesided[0]))

        # 内部频点
        last_interior = n_rfft - 1 if (n_samples % 2 == 1) else n_rfft - 2
        if last_interior >= 1:
            n_interior = last_interior
            sigma = np.sqrt(0.25 * fs * n_samples * psd_onesided[1:last_interior + 1])
            spec[1:last_interior + 1] = sigma * (
                rng.standard_normal(n_interior) + 1j * rng.standard_normal(n_interior)
            )

        # Nyquist（仅偶数长度存在）
        if n_samples % 2 == 0:
            spec[-1] = rng.normal(0.0, np.sqrt(fs * n_samples * psd_onesided[-1]))

        return np.fft.irfft(spec, n=n_samples)

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------
    def component_psds(self, n_samples: int | None = None) -> dict:
        """返回给定长度下各噪声分量的一侧 PSD。"""
        if n_samples is None:
            n_samples = self.p.fft_points
        freqs = self.frequency_axis(n_samples)
        rin_psd, _ = self.calculate_rin_psd(freqs)
        nep_psd, _ = self.calculate_nep_psd(freqs)
        shot_psd = self.calculate_shot_psd(freqs)
        thermal_psd = self.calculate_thermal_psd(freqs)
        return {
            "freqs": freqs,
            "shot": shot_psd,
            "thermal": thermal_psd,
            "rin": rin_psd,
            "bdn": nep_psd,
            "total": shot_psd + thermal_psd + rin_psd + nep_psd,
        }

    def generate_total_noise(
        self,
        n_samples: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        生成总噪声。

        为保持与旧主程序接口兼容，返回：
            noise_rin, noise_gauss, noise_nep, total_noise
        其中 noise_gauss = noise_shot + noise_thermal。
        """
        if rng is None:
            rng = np.random.default_rng()
        if n_samples is None:
            n_samples = self.p.fft_points

        psds = self.component_psds(n_samples)

        noise_shot = self.simulate_colored_noise_from_psd(psds["shot"], n_samples, rng)
        noise_thermal = self.simulate_colored_noise_from_psd(psds["thermal"], n_samples, rng)
        noise_rin = self.simulate_colored_noise_from_psd(psds["rin"], n_samples, rng)
        noise_nep = self.simulate_colored_noise_from_psd(psds["bdn"], n_samples, rng)

        noise_gauss = noise_shot + noise_thermal
        total_noise = noise_gauss + noise_rin + noise_nep
        return noise_rin, noise_gauss, noise_nep, total_noise

    # ------------------------------------------------------------------
    # 验证工具
    # ------------------------------------------------------------------
    def one_sided_periodogram(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        采用矩形窗 periodogram，返回一侧 PSD。
        这里故意不用加窗，是为了与本模块的频域合成公式一一对应。
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        fs = self.p.sample_rate
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spec = np.fft.rfft(x, n=n)
        psd = (np.abs(spec) ** 2) / (fs * n)

        if n % 2 == 0:
            if len(psd) > 2:
                psd[1:-1] *= 2.0
        else:
            if len(psd) > 1:
                psd[1:] *= 2.0
        return freqs, psd

    def average_psd(self, generator, n_samples: int, n_accum: int) -> tuple[np.ndarray, np.ndarray]:
        """对给定噪声生成器做多次平均 periodogram。"""
        psd_acc = None
        freqs = None
        for _ in range(n_accum):
            x = generator(n_samples)
            freqs, pxx = self.one_sided_periodogram(x)
            if psd_acc is None:
                psd_acc = np.zeros_like(pxx)
            psd_acc += pxx
        return freqs, psd_acc / float(n_accum)

    def run_all_validations(self):
        """
        复现原始 D_noise_model.py 中的全部噪声验证内容，并保持与修订版统计定义一致。
        """
        n_time = self.p.points_per_bin
        n_fft = self.p.fft_points
        rng = np.random.default_rng(20260318)

        t_axis_ns = np.arange(n_time) * self.p.time_step * 1e9
        f_axis_std_mhz = self.frequency_axis(n_fft) / 1e6

        # 高频率分辨率轴（仅用于 RIN 理论曲线）
        f_plot_hz = np.logspace(5, 8, 2000)
        f_plot_mhz = f_plot_hz / 1e6

        comp_std = self.component_psds(n_fft)
        shot_psd_std = comp_std["shot"]
        thermal_psd_std = comp_std["thermal"]
        rin_psd_std = comp_std["rin"]
        nep_psd_std = comp_std["bdn"]

        # ==========================================
        # 1. 图 2.7: 散粒噪声与热噪声 (时域)
        # ==========================================
        print("绘图: 散粒与热噪声时域...")
        noise_shot = self.simulate_colored_noise_from_psd(shot_psd_std, n_time, rng) * 1e6
        noise_therm = self.simulate_colored_noise_from_psd(thermal_psd_std, n_time, rng) * 1e6

        fig1 = plt.figure(figsize=(10, 10))
        ax1 = fig1.add_subplot(111)
        ax1.plot(t_axis_ns, noise_shot, color='brown', lw=0.8)
        plot_style.style.apply_standard_layout(fig1, ax1, title="散粒噪声", xlabel="时间 (ns)", ylabel="电流 ($\u00B5 A$)")
        ax1.set_xlim(0, t_axis_ns[-1])

        fig2 = plt.figure(figsize=(10, 10))
        ax2 = fig2.add_subplot(111)
        ax2.plot(t_axis_ns, noise_therm, color='orange', lw=0.8)
        plot_style.style.apply_standard_layout(fig2, ax2, title="热噪声", xlabel="时间 (ns)", ylabel="电流 ($\u00B5 A$)")
        ax2.set_xlim(0, t_axis_ns[-1])

        plt.show()

        # ==========================================
        # 2. 图 2.9: RIN 噪声特性
        # ==========================================
        print("绘图: RIN 噪声特性...")
        rin_psd_high, rin_db_high = self.calculate_rin_psd(f_plot_hz)

        fig3 = plt.figure(figsize=(10, 10))
        ax3 = fig3.add_subplot(111)
        ax3.semilogx(f_plot_mhz, rin_db_high, color='red')
        plot_style.style.apply_standard_layout(fig3, ax3, title="RIN 频率特性", xlabel="频率 (MHz)", ylabel="RIN (dB/Hz)")
        ax3.set_xlim(0.1, 100)

        fig4 = plt.figure(figsize=(10, 10))
        ax4 = fig4.add_subplot(111)
        ax4.semilogx(f_plot_mhz, rin_psd_high, color='red')
        plot_style.style.apply_standard_layout(fig4, ax4, title="RIN 功率谱密度", xlabel="频率 (MHz)", ylabel="PSD ($A^2/Hz$)")
        ax4.set_xlim(0.1, 100)

        plt.show()

        # ==========================================
        # 3. 图 2.11: 平衡探测器噪声
        # ==========================================
        print("绘图: 平衡探测器噪声特性...")
        receiver_resp = self.receiver_response(self.frequency_axis(n_fft))

        fig5 = plt.figure(figsize=(10, 10))
        ax5 = fig5.add_subplot(111)
        ax5.plot(f_axis_std_mhz, receiver_resp, color='blue')
        plot_style.style.apply_standard_layout(fig5, ax5, title="Thorlabs PDB460C平衡探测器带宽响应", xlabel="频率 (MHz)", ylabel="归一化响应")
        ax5.set_xlim(0, 500)

        fig6 = plt.figure(figsize=(10, 10))
        ax6 = fig6.add_subplot(111)
        ax6.plot(f_axis_std_mhz, nep_psd_std, color='red')
        plot_style.style.apply_standard_layout(fig6, ax6, title="BDN 功率谱密度", xlabel="频率 (MHz)", ylabel="PSD ($A^2/Hz$)")
        ax6.set_xlim(0, 500)

        plt.show()

        # ==========================================
        # 4. 图 2.12: 有色噪声时域波形
        # ==========================================
        print("绘图: 有色噪声时域波形...")
        n_rin = self.simulate_colored_noise_from_psd(rin_psd_std, n_time, rng) * 1e6
        n_nep = self.simulate_colored_noise_from_psd(nep_psd_std, n_time, rng) * 1e6

        fig7 = plt.figure(figsize=(10, 10))
        ax7 = fig7.add_subplot(111)
        ax7.plot(t_axis_ns, n_rin, color='teal', lw=0.8)
        plot_style.style.apply_standard_layout(fig7, ax7, title="相对强度噪声", xlabel="时间 (ns)", ylabel="电流 ($\u00B5 A$)")
        ax7.set_xlim(0, t_axis_ns[-1])

        fig8 = plt.figure(figsize=(10, 10))
        ax8 = fig8.add_subplot(111)
        ax8.plot(t_axis_ns, n_nep, color='steelblue', lw=0.8)
        plot_style.style.apply_standard_layout(fig8, ax8, title="平衡探测器噪声", xlabel="时间 (ns)", ylabel="电流 ($\u00B5 A$)")
        ax8.set_xlim(0, t_axis_ns[-1])

        plt.show()

        # ==========================================
        # 5. 图 2.13 & 2.14: 脉冲累积效果
        # ==========================================
        print("绘图: 脉冲累积对PSD的影响...")
        acc_list = [1, 10, 100, 1000]
        n_freq = len(f_axis_std_mhz)

        psd_avg_shot = np.zeros((len(acc_list), n_freq))
        psd_avg_therm = np.zeros((len(acc_list), n_freq))
        psd_avg_rin = np.zeros((len(acc_list), n_freq))
        psd_avg_nep = np.zeros((len(acc_list), n_freq))

        accum_shot = np.zeros(n_freq)
        accum_therm = np.zeros(n_freq)
        accum_rin = np.zeros(n_freq)
        accum_nep = np.zeros(n_freq)

        for i in tqdm(range(1, 1001), desc="Accumulating"):
            ns_shot = self.simulate_colored_noise_from_psd(shot_psd_std, n_fft, rng)
            ns_therm = self.simulate_colored_noise_from_psd(thermal_psd_std, n_fft, rng)
            ns_rin = self.simulate_colored_noise_from_psd(rin_psd_std, n_fft, rng)
            ns_nep = self.simulate_colored_noise_from_psd(nep_psd_std, n_fft, rng)

            _, p_shot = self.one_sided_periodogram(ns_shot)
            _, p_therm = self.one_sided_periodogram(ns_therm)
            _, p_rin = self.one_sided_periodogram(ns_rin)
            _, p_nep = self.one_sided_periodogram(ns_nep)

            accum_shot += p_shot
            accum_therm += p_therm
            accum_rin += p_rin
            accum_nep += p_nep

            if i in acc_list:
                idx = acc_list.index(i)
                psd_avg_shot[idx] = accum_shot / i
                psd_avg_therm[idx] = accum_therm / i
                psd_avg_rin[idx] = accum_rin / i
                psd_avg_nep[idx] = accum_nep / i

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        linewidths = [0.5, 0.8, 1.0, 1.5]
        alphas = [0.6, 0.8, 0.9, 1.0]

        fig9 = plt.figure(figsize=(10, 10))
        ax9 = fig9.add_subplot(111)
        for k, acc in enumerate(acc_list):
            ax9.plot(f_axis_std_mhz, psd_avg_shot[k], label=f'N={acc}', color=colors[k], lw=linewidths[k], alpha=alphas[k])
        ax9.plot(f_axis_std_mhz, shot_psd_std, 'k--', lw=1.2, label='Theoretical PSD')
        plot_style.style.apply_standard_layout(fig9, ax9, title="散粒噪声累积功率谱密度", xlabel="频率 (MHz)", ylabel="PSD ($A^2/Hz$)")
        ax9.set_xlim(0, 500)
        ax9.legend()

        fig10 = plt.figure(figsize=(10, 10))
        ax10 = fig10.add_subplot(111)
        for k, acc in enumerate(acc_list):
            ax10.plot(f_axis_std_mhz, psd_avg_therm[k], label=f'N={acc}', color=colors[k], lw=linewidths[k], alpha=alphas[k])
        ax10.plot(f_axis_std_mhz, thermal_psd_std, 'k--', lw=1.2, label='Theoretical PSD')
        plot_style.style.apply_standard_layout(fig10, ax10, title="热噪声累积功率谱密度", xlabel="频率 (MHz)", ylabel="PSD ($A^2/Hz$)")
        ax10.set_xlim(0, 500)
        ax10.legend()

        fig11 = plt.figure(figsize=(10, 10))
        ax11 = fig11.add_subplot(111)
        for k, acc in enumerate(acc_list):
            ax11.plot(f_axis_std_mhz, psd_avg_rin[k], label=f'N={acc}', color=colors[k], lw=linewidths[k], alpha=alphas[k])
        plot_style.style.apply_standard_layout(fig11, ax11, title="RIN 累积功率谱密度", xlabel="频率 (MHz)", ylabel="PSD ($A^2/Hz$)")
        ax11.set_xlim(0, 500)
        ax11.set_ylim(0, 8e-24)
        ax11.legend()

        fig12 = plt.figure(figsize=(10, 10))
        ax12 = fig12.add_subplot(111)
        for k, acc in enumerate(acc_list):
            ax12.plot(f_axis_std_mhz, psd_avg_nep[k], label=f'N={acc}', color=colors[k], lw=linewidths[k], alpha=alphas[k])
        plot_style.style.apply_standard_layout(fig12, ax12, title="BDN 累积功率谱密度", xlabel="频率 (MHz)", ylabel="PSD ($A^2/Hz$)")
        ax12.set_xlim(0, 500)
        ax12.legend()

        plt.show()
        print("所有图表绘制完成。")


if __name__ == "__main__":
    nm = NoiseModel()
    nm.print_model_configuration()
    nm.run_all_validations()
