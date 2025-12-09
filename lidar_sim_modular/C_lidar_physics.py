import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from A_lidar_params import params
from B_atmosphere_model import AtmosphereModel
import S_plot_style as plot_style


class LidarPhysics:
    """
    雷达物理层仿真模块 (支持风廓线输入版)
    """

    def __init__(self):
        self.p = params
        self.atmo = AtmosphereModel()

        # 预计算脉冲延迟矩阵 (Time x Range)
        # axis 0: time, axis 1: range layer
        delay = 2 * self.p.range_axis / self.p.c
        self.time_diff = self.p.time_axis[:, np.newaxis] - delay[np.newaxis, :]

        # 预计算脉冲形状因子 (高斯包络)
        self.pulse_shape_factor = np.exp(-4 * np.log(2) * (self.time_diff / self.p.pulse_width) ** 2)

    def get_pulse_power_profile(self, t_axis=None):
        if t_axis is None:
            t_axis = np.linspace(-1000e-9, 1000e-9, 1000)
        coeff = self.p.pulse_energy * (2 * np.sqrt(np.log(2))) / (np.sqrt(np.pi) * self.p.pulse_width)
        profile = coeff * np.exp(-4 * np.log(2) * (t_axis / self.p.pulse_width) ** 2)
        return t_axis, profile

    def simulate_ideal_signal(self, v_los_profile=None):
        """
        生成理想外差信号 (未加噪声)

        参数:
            v_los_profile: 径向风速数组 (m/s), 长度需等于 range_axis。
                           如果为 None, 则默认使用 0 m/s。
        """
        # 1. 准备大气参数
        beta_km = np.interp(self.p.height_axis, self.p.height_axis, self.atmo.beta_total)
        beta_m = beta_km / 1000.0
        alpha_km = np.interp(self.p.height_axis, self.p.height_axis, self.atmo.alpha_total)
        alpha_m = alpha_km / 1000.0

        # 双程透过率 T^2
        optical_depth = np.cumsum(alpha_m) * self.p.dist_res
        transmittance_squared = np.exp(-2 * optical_depth)

        # 2. 几何因子 (1/R^2)
        geometric_factor = (self.p.telescope_area / self.p.range_axis ** 2) * \
                           self.p.system_efficiency * self.p.dist_res

        # 3. 散斑 (Speckle)
        # 每个距离门的期望回波强度
        Km_sq_expectation = transmittance_squared * beta_m * geometric_factor
        # 生成瑞利分布幅度 & 均匀分布相位
        rand_phase = np.random.uniform(0, 2 * np.pi, len(Km_sq_expectation))
        rand_amp = np.random.rayleigh(np.sqrt(Km_sq_expectation))
        Km = rand_amp * np.exp(1j * rand_phase)  # 复振幅向量 (N_layers,)

        # 4. 脉冲能量系数
        E_pulse_coeff = self.p.pulse_energy * (2 * np.sqrt(np.log(2))) / (np.sqrt(np.pi) * self.p.pulse_width)
        pulse_power_matrix = E_pulse_coeff * self.pulse_shape_factor
        E_T_matrix = np.sqrt(pulse_power_matrix)  # 电场幅度矩阵 (Time x Range)

        # 5. [核心升级] 多普勒相位矩阵
        # 针对每个距离门计算不同的多普勒频移
        if v_los_profile is None:
            v_los_profile = np.zeros_like(self.p.range_axis)

        # f_d = -2 * v_r / lambda (N_layers,)
        freq_shifts = -2 * v_los_profile / self.p.wavelength

        # 构建相位矩阵: exp(-j * 2 * pi * f_d * t)
        # 利用广播机制: (N_time, 1) * (1, N_layers) -> (N_time, N_layers)
        phase_doppler_matrix = np.exp(-2j * np.pi * self.p.time_axis[:, np.newaxis] * freq_shifts[np.newaxis, :])

        # 6. 信号积分 (相干叠加)
        # Signal = Sum_over_m ( E_T(t, m) * Km(m) * Phase_Doppler(t, m) )
        # 逐元素相乘后，沿距离轴(axis 1)求和
        contribution_matrix = E_T_matrix * Km[np.newaxis, :] * phase_doppler_matrix
        signal_complex = np.sum(contribution_matrix, axis=1)

        # 7. 添加 AOM 频移和本振光混频
        # i_h = 2 * R * sqrt(P_LO) * signal_complex * exp(-j * 2 * pi * f_AOM * t)
        phase_aom = np.exp(-2j * np.pi * self.p.freq_aom * self.p.time_axis)

        i_h = 2 * self.p.responsivity * np.sqrt(self.p.local_power) * \
              phase_aom * signal_complex

        return i_h

# --- 独立验证模块 ---
if __name__ == "__main__":
    physics = LidarPhysics()

    # ==========================================
    # 验证 1: 绘制图 2.5 (脉冲形状)
    # ==========================================
    t_axis, p_profile = physics.get_pulse_power_profile()

    plt.figure(figsize=(8, 5))
    plt.plot(t_axis * 1e9, p_profile, color='tab:red')
    plt.xlabel("时间 ($ns$)", fontproperties=plot_style.style.zh_font)
    plt.ylabel("功率 ($W$)", fontproperties=plot_style.style.zh_font)
    plt.title("高斯脉冲时域分布模型", fontproperties=plot_style.style.zh_font)
    plt.xlim(-1000, 1000)
    plt.show()

    # ==========================================
    # 验证 2: 绘制图 2.6 (时域外差信号) - 优化显示
    # ==========================================
    print("正在生成时域外差信号...")
    test_wind_profile = np.full_like(params.range_axis, 10.0)
    i_h = physics.simulate_ideal_signal(v_los_profile=test_wind_profile)

    dist_axis = params.time_axis * params.c / 2

    # [关键优化]：单位转换为 微安 (uA)
    i_h_uA = np.real(i_h) * 1e6

    # [关键优化]：动态计算 Y 轴范围
    # 获取绝对值的最大值，并稍微留一点余量 (例如 1.1 倍)
    max_amp = np.max(np.abs(i_h_uA))
    ylim_val = max_amp * 1.1

    print(f"信号峰值幅度: {max_amp:.4f} uA")

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制主图
    # linewidth 设细一点，以便看清高频震荡的包络
    ax.plot(dist_axis, i_h_uA, color='blue', linewidth=0.3)

    ax.set_xlabel("距离 ($m$)", fontproperties=plot_style.style.zh_font)
    ax.set_ylabel("外差信号 ($\u00B5 A$)", fontproperties=plot_style.style.zh_font)  # 单位改为 uA
    ax.set_title("时域外差电流信号", fontproperties=plot_style.style.zh_font)

    ax.set_xlim(0, 4000)
    ax.set_ylim(-ylim_val, ylim_val)  # 动态设置范围，确保波形充满画面
    ax.grid(True, alpha=0.3)

    # 绘制插图 (放大 960m - 1140m)
    ax_ins = inset_axes(ax, width="45%", height="30%", loc='upper right', borderpad=2)

    idx_s = np.argmin(np.abs(dist_axis - 960))
    idx_e = np.argmin(np.abs(dist_axis - 1140))

    ax_ins.plot(dist_axis[idx_s:idx_e], i_h_uA[idx_s:idx_e], color='blue', linewidth=0.5)
    # ax_ins.set_title("960m - 1140m 细节", fontproperties=zh_font, fontsize=10)
    ax_ins.grid(True, alpha=0.3)

    # plt.tight_layout()
    plt.show()