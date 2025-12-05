import numpy as np
import matplotlib.pyplot as plt
from A_lidar_params import params
import S_plot_style as plot_style


class WindField:
    """
    风场生成模块。
    负责生成大气中的三维风矢量 (u, v, w) 以及计算雷达视线方向的径向风速 (V_r)。
    严格遵循论文扫描参数: 俯仰角 72度, 方位角步进 22.5度 (16个方向)。
    """

    def __init__(self):
        self.p = params

    def get_power_law_profile(self, heights, u_ref=10.0, z_ref=100.0, alpha=0.2):
        """
        生成指数律风廓线 (Power Law Profile)
        公式: u(z) = u_ref * (z / z_ref)^alpha
        用于模拟边界层内随高度增加的风速。

        参数:
            heights: 高度数组 (m)
            u_ref: 参考高度处的水平风速 (m/s)
            z_ref: 参考高度 (m)
            alpha: 风切变指数 (通常 0.1 ~ 0.3)
        返回:
            speed_profile: 对应高度的风速大小数组
        """
        # 避免高度为0或负数导致计算错误
        valid_h = np.maximum(heights, 1e-3)
        speed_profile = u_ref * (valid_h / z_ref) ** alpha
        return speed_profile

    def get_wind_vector_field(self, heights, wind_type='power_law'):
        """
        生成三维风矢量场 (u, v, w)
        假设:
        - 水平风速服从指数律 (或常数)
        - 风向恒定为北风 (0度, 即只有 v 分量, u=0)
        - 垂直风速 w=0

        参数:
            heights: 垂直高度数组 (z)
        返回:
            u, v, w: 三个与 heights 等长的数组
        """
        if wind_type == 'power_law':
            # 生成随高度变化的风速
            hor_speed = self.get_power_law_profile(heights, u_ref=10.0, alpha=0.2)
        else:
            # 常风速 (用于早期验证)
            hor_speed = np.full_like(heights, 10.0)

        # 定义风向: 假设风从北向南吹 (North Wind)
        # 在气象定义中，北风意味着风来自北方。
        # 在数学向量中(x=东, y=北)，风矢量指向南方，即 v < 0。
        # 但论文仿真常假设简单的正向速度，这里我们假设风矢量指向正北 (v > 0) 以便理解
        # u = East, v = North, w = Up

        u = np.zeros_like(heights)  # 东西分量
        v = hor_speed  # 南北分量 (假设全部分量都在南北方向)
        w = np.zeros_like(heights)  # 垂直分量

        return u, v, w

    def get_radial_velocity(self, range_axis, azimuth_deg, elevation_deg, wind_type='power_law'):
        """
        计算视线方向径向风速 (V_los / V_r)

        根据论文公式 (4.1):
        V_r = u * sin(theta) * cos(alpha) + v * cos(theta) * cos(alpha) + w * sin(alpha)

        参数:
            range_axis: 雷达径向距离轴 (m)
            azimuth_deg: 方位角 theta (度)
            elevation_deg: 俯仰角 alpha (度)
            wind_type: 风场类型 ('power_law' or 'constant')
        返回:
            v_los: 沿径向分布的视线风速数组 (m/s)
            heights: 对应的垂直高度数组 (m)
        """
        # 1. 计算各采样点对应的垂直高度
        # z = R * sin(alpha)
        # 注意: elevation_deg 应为 72 度
        el_rad = np.radians(elevation_deg)
        az_rad = np.radians(azimuth_deg)

        heights = range_axis * np.sin(el_rad)

        # 2. 获取该高度处的真实风矢量 (u, v, w)
        u, v, w = self.get_wind_vector_field(heights, wind_type)

        # 3. 投影计算径向风速 V_r
        # 论文 Eq 4.1: V = u sin(θ) cos(α) + v cos(θ) cos(α) + w sin(α)
        # 这里 θ=azimuth, α=elevation
        # 物理含义: 风矢量 V 在视线方向 n 上的投影 (V dot n)

        v_r = u * np.sin(az_rad) * np.cos(el_rad) + \
              v * np.cos(az_rad) * np.cos(el_rad) + \
              w * np.sin(el_rad)

        return v_r, heights


# =============================================================================
# 验证绘图: 模拟雷达扫描一圈 (PPI)
# =============================================================================
if __name__ == "__main__":
    wf = WindField()

    # 1. 检查参数是否符合论文
    print(f"系统参数检查:")
    print(f"  - 俯仰角: {wf.p.elevation_angle_deg}° (论文要求 72°)")
    print(f"  - 方位角数: {wf.p.azimuth_count} (论文要求 16)")

    # 2. 生成扫描序列
    # 0 ~ 360度，步进 22.5度
    azimuths = np.linspace(0, 360, wf.p.azimuth_count, endpoint=False)
    print(f"  - 扫描方位角: {azimuths}")

    # 3. 模拟扫描并绘图
    plt.figure(figsize=(10, 6))

    # 选取几个典型方位角进行绘制
    # 0度(北), 90度(东), 180度(南), 270度(西)
    # 由于我们假设的是正北风 (v>0, u=0):
    # 0度: V_r 投影最大 (cos0 = 1)
    # 90度: V_r 投影为0 (cos90 = 0)
    # 180度: V_r 投影为负 (cos180 = -1)

    target_azimuths = [0, 45, 90, 135, 180]

    for azi in target_azimuths:
        if azi in azimuths:  # 确保只画实际扫描的点
            v_r, h = wf.get_radial_velocity(
                params.range_axis,
                azimuth_deg=azi,
                elevation_deg=params.elevation_angle_deg,
                wind_type='power_law'
            )

            # 绘制 V_r 随高度的变化
            plt.plot(v_r, h, label=f'Azimuth {azi}°', linewidth=1.5)

    plot_style.style.apply_standard_layout(plt.gcf(), plt.gca(),
                                           title="不同方位角的径向风速廓线 (俯仰角72°)",
                                           xlabel="径向风速 ($m/s$)",
                                           ylabel="垂直高度 (m)")
    plt.legend()
    plt.show()
    print("风场模块 (E) 验证完成。")