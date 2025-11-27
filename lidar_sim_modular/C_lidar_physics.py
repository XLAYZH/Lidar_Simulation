import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import S_plot_style as plot_style

# 导入前序模块
from A_lidar_params import params
from B_atmosphere_model import AtmosphereModel


class LidarPhysics:
    """
    雷达物理层仿真模块。
    负责计算：
    1. 发射激光脉冲的时域波形 P_T(t)
    2. 基于雷达方程和散斑效应的理想外差信号 i_h(t)
    """

    def __init__(self):
        self.p = params
        self.atmo = AtmosphereModel()

        # 预计算脉冲形状函数 (归一化的高斯分布部分)
        # 使用 self.p.range_axis
        delay = 2 * self.p.range_axis / self.p.c
        self.time_diff = self.p.time_axis[:, np.newaxis] - delay[np.newaxis, :]

        # 高斯因子: exp(-4 * ln2 * (t/tau)^2)
        self.pulse_shape_factor = np.exp(-4 * np.log(2) * (self.time_diff / self.p.pulse_width) ** 2)

    def get_pulse_power_profile(self, t_axis=None):
        """
        计算单脉冲的瞬时功率 P_T(t) (瓦特)。

        """
        if t_axis is None:
            t_axis = np.linspace(-1000e-9, 1000e-9, 1000)  # 默认 -1us 到 1us

        # 峰值功率系数 A = E * [2*sqrt(ln2) / (sqrt(pi)*tau)]
        # 仅与单脉冲能量 E 有关，不乘 PRF
        coeff = self.p.pulse_energy * (2 * np.sqrt(np.log(2))) / (np.sqrt(np.pi) * self.p.pulse_width)

        # 时域分布
        profile = coeff * np.exp(-4 * np.log(2) * (t_axis / self.p.pulse_width) ** 2)
        return t_axis, profile

    def simulate_ideal_signal(self, wind_u=5.0):
        """
        模拟单次脉冲的理想外差电流信号 i_h(t)。

        """
        # 1. 获取大气光学参数 (插值到 slant range)
        beta_km = np.interp(self.p.height_axis, self.p.height_axis, self.atmo.beta_total)
        beta_m = beta_km / 1000.0  # km^-1 -> m^-1

        alpha_km = np.interp(self.p.height_axis, self.p.height_axis, self.atmo.alpha_total)
        alpha_m = alpha_km / 1000.0

        # 累积积分计算光学厚度
        optical_depth = np.cumsum(alpha_m) * self.p.dist_res
        transmittance_squared = np.exp(-2 * optical_depth)

        # 2. 计算几何因子
        geometric_factor = (self.p.telescope_area / self.p.range_axis ** 2) * \
                           self.p.system_efficiency * self.p.dist_res

        # 3. 计算散斑效应期望值 K_m^2
        Km_sq_expectation = transmittance_squared * beta_m * geometric_factor

        # 4. 生成随机散斑
        rand_phase = np.random.uniform(0, 2 * np.pi, len(Km_sq_expectation))
        rand_amp = np.random.rayleigh(np.sqrt(Km_sq_expectation))
        Km = rand_amp * np.exp(1j * rand_phase)

        # 5. 计算相位项 (AOM + Doppler)
        phase_aom = np.exp(-2j * np.pi * self.p.freq_aom * self.p.time_axis)

        v_los = wind_u * np.cos(np.radians(self.p.elevation_angle_deg))
        freq_shift_doppler = -2 * v_los / self.p.wavelength
        phase_doppler = np.exp(-2j * np.pi * freq_shift_doppler * self.p.time_axis)

        # 6. 计算脉冲功率矩阵 (单脉冲能量)
        # 修正点：移除 PRF，只用 pulse_energy
        coeff = self.p.pulse_energy * (2 * np.sqrt(np.log(2))) / (np.sqrt(np.pi) * self.p.pulse_width)
        pulse_power_matrix = coeff * self.pulse_shape_factor

        # 7. 积分求和
        E_T_matrix = np.sqrt(pulse_power_matrix)
        signal_complex = np.sum(E_T_matrix * Km[np.newaxis, :], axis=1)

        # 最终外差电流
        i_h = 2 * self.p.responsivity * np.sqrt(self.p.local_power) * \
              phase_aom * phase_doppler * signal_complex

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
    i_h = physics.simulate_ideal_signal(wind_u=10.0)

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