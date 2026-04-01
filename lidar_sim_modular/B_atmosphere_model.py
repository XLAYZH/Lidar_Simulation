import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from A_lidar_params import params  # 导入参数模块
import PlotStyle


class AtmosphereModel:
    """
    大气模型类。
    负责计算消光系数(alpha)和后向散射系数(beta)。
    [cite_start]参考文献: 论文 2.2.1 节 [cite: 403-426]
    """

    def __init__(self):
        self.p = params

        # 1. 计算仿真所需的廓线
        # [修正] 使用 self.p.height_axis (对应 lidar_params.py 中的定义)
        self.alpha_mol, self.alpha_aer, self.alpha_total, \
            self.beta_mol, self.beta_aer, self.beta_total = \
            self.calculate_coefficients(self.p.height_axis)

    def calculate_coefficients(self, height_axis_m):
        """
        核心计算逻辑：根据输入的高度数组(m)，计算对应的消光和散射系数。
        这样设计是为了既能计算仿真用的短距离数据，也能计算绘图用的长距离数据。
        """
        # 垂直高度，单位转换: m -> km
        h_km = height_axis_m / 1000.0

        # 1. 分子消光系数 (Rayleigh)
        # 公式单位: km^-1
        alpha_mol_km = (8 * np.pi / 3) * 1.54e-3 * \
                       (532e-9 / self.p.wavelength) ** 4 * \
                       np.exp(-h_km / 7)

        # 2. 气溶胶消光系数 (Mie)
        # 公式单位: km^-1
        # 注意：这里包含了 20km 处的平流层气溶胶层 (高斯项)
        alpha_aer_km = 50 * (2.47e-3 * np.exp(-h_km / 2) +
                             5.13e-6 * np.exp(-(h_km - 20) ** 2 / 36)) * \
                       (532e-9 / self.p.wavelength)

        # 3. 总消光系数 (km^-1)
        alpha_total_km = alpha_mol_km + alpha_aer_km

        # 4. 后向散射系数 (km^-1 sr^-1)
        beta_mol_km = alpha_mol_km / (8 * np.pi / 3)
        beta_aer_km = alpha_aer_km * 0.02  # 假设气溶胶雷达比为 50sr
        beta_total_km = beta_mol_km + beta_aer_km

        # --- 单位换算 ---
        # 仿真计算通常需要国际单位制 (m^-1)，但绘图通常用 (km^-1)
        # 这里为了通用性，输出对应的物理量，调用者需注意单位
        # 本方法直接返回 km^-1 为单位的数值，方便绘图和逻辑复用

        return alpha_mol_km, alpha_aer_km, alpha_total_km, \
            beta_mol_km, beta_aer_km, beta_total_km

    def get_transmittance_squared(self):
        """
        计算大气双程透过率 T^2(R) (用于雷达仿真)
        """
        # 注意：这里必须使用转化为 m^-1 的系数进行积分，且积分路径是斜径
        # self.alpha_total 是 calculate_coefficients 返回的 km^-1 单位
        # 所以需要除以 1000 转为 m^-1
        alpha_total_m = self.alpha_total / 1000.0

        # 累积积分计算光学厚度 (Optical Depth)
        # 使用累积求和近似积分: sum(alpha * dr)
        # dr 是距离分辨率 self.p.dist_res
        optical_depth = np.cumsum(alpha_total_m) * self.p.dist_res

        # T^2 = exp(-2 * tau)
        t_squared = np.exp(-2 * optical_depth)
        return t_squared


# --- 独立验证模块：绘制图 2.3 ---
if __name__ == "__main__":

    # 实例化模型 (内部会自动计算仿真所需的数据)
    atmo = AtmosphereModel()

    # === 绘图准备 ===
    # 为了复现论文图 2.3，我们需要计算 0-30 km 的数据
    # 而不是使用 atmo 实例中存储的 0-3.8 km 数据

    # 1. 生成绘图专用的高度轴 (0 - 30 km)
    h_plot_m = np.linspace(0, 30000, 1000)  # 0 到 30000 米

    # 2. 调用计算方法
    a_mol, a_aer, a_tot, _, _, _ = atmo.calculate_coefficients(h_plot_m)

    # 3. 绘图
    fig, ax = plt.subplots(figsize=(6, 6))

    # 垂直高度 (km)
    h_axis_km = h_plot_m / 1000.0

    ax.plot(a_mol, h_axis_km, label='Molecules', color='orange', linewidth=1)
    ax.plot(a_aer, h_axis_km, label='Aerosols', color='green', linewidth=1)
    ax.plot(a_tot, h_axis_km, label='Atmosphere (Total)', color='blue', linewidth=1, linestyle='--')

    ax.set_xscale('log')  # X轴对数坐标
    ax.set_xlim(1e-6, 1e-1)  # 设置X轴范围
    ax.set_ylim(0, 30)  # 设置Y轴范围

    ax.set_xlabel("Extinction Coefficient ($km^{-1}$)")
    ax.set_ylabel("Height ($km$)")
    ax.set_title("Extinction Coefficient of Molecules, Aerosols and Total")
    ax.legend(prop={'family': 'Times New Roman'})

    PlotStyle.set_axis(ax,
                       xminor=True,  # 自动次刻度
                       yminor=True,
                       axis_lw=1.2,  # 轴线宽度
                       ticklabel_fontsize=11,  # 刻度字体
                       font_name = 'Times New Roman',  # 设置字体
                       label_fontsize = 12,  # 轴标签字体
                       title_fontsize = 14)

    print("AtmosphereModel 验证完成。")
    plt.show()

    # ==========================================
    # 新增：绘制大气透过率随高度变化曲线 (单程 & 双程)
    # ==========================================
    fig, ax = plt.subplots(figsize=(6, 6))

    # 1. 计算光学厚度 (Optical Depth, tau)
    # 公式: tau(z) = integral(alpha(h) dh)
    # a_tot 单位是 km^-1, 所以高度微分 dh 也必须是 km
    dh_km = (h_plot_m[1] - h_plot_m[0]) / 1000.0

    # 使用累积求和近似积分
    optical_depth = np.cumsum(a_tot) * dh_km

    # 2. 计算透过率 (Transmittance)
    # 单程透过率 T(z) = exp(-tau)
    transmittance_oneway = np.exp(-optical_depth)

    # 双程透过率 T^2(z) = exp(-2 * tau)
    # 这是雷达回波能量实际经历的衰减
    transmittance_twoway = np.exp(-2 * optical_depth)

    # 3. 绘图
    ax.plot(transmittance_oneway, h_axis_km, label='One-way ($T$)', color='purple', linewidth=1.5)
    ax.plot(transmittance_twoway, h_axis_km, label='Two-way ($T^2$)', color='darkred', linewidth=1.5,
             linestyle='--')

    # 4. 设置样式
    ax.set_xlabel("Transmittance (Normalized)")
    ax.set_ylabel("Height ($km$)")
    ax.set_title("Vertical Transmittance Profile")

    ax.set_xlim(0.8, 1.0)  # 透过率在 0~1 之间
    ax.set_ylim(0, 30)  # 高度 0~30 km
    ax.legend(prop={'family': 'Times New Roman'})  # 确保图例中文正常显示

    PlotStyle.set_axis(ax,
                       xminor=0.05,
                       yminor=True,
                       axis_lw=1.2,  # 轴线宽度
                       ticklabel_fontsize=11,  # 刻度字体
                       font_name='Times New Roman',  # 设置字体
                       label_fontsize=12,  # 轴标签字体
                       title_fontsize=14)

    print("大气透过率图表已生成。")
    plt.show()