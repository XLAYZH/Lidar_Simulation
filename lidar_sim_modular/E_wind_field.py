import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import os
import glob
import random

from A_lidar_params import params
import S_plot_style as plot_style


class WindField:
    """
    风场生成模块 (包含三种理论模型 + 真实数据插值)
    """

    def __init__(self):
        self.p = params
        self.real_profile = None


    def load_sounding_data(self, csv_path):
        """加载探空 CSV 数据并构建插值函数"""
        if not os.path.exists(csv_path):
            print(f"[Error] 文件不存在: {csv_path}")
            self.real_profile = None
            return

        try:
            # 读取 CSV (自动处理分隔符)
            df = pd.read_csv(csv_path, sep=None, engine='python', header=0)
            df.columns = df.columns.str.strip()  # 清洗列名空格

            required_cols = ['HGHT', 'SPED', 'DRCT']
            if not all(col in df.columns for col in required_cols):
                print(f"[Error] 缺少列: {required_cols}")
                self.real_profile = None
                return

            # 转数值
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=required_cols, inplace=True)

            z_real = df['HGHT'].values
            speed_real = df['SPED'].values
            dir_real = df['DRCT'].values

            # 风向转 UV
            dir_rad = np.radians(dir_real)
            u_real = -speed_real * np.sin(dir_rad)
            v_real = -speed_real * np.cos(dir_rad)

            # 创建插值
            self.u_interp = interp1d(z_real, u_real, kind='linear', bounds_error=False, fill_value="extrapolate")
            self.v_interp = interp1d(z_real, v_real, kind='linear', bounds_error=False, fill_value="extrapolate")

            self.real_profile = True
            print(f"[Success] 已加载: {os.path.basename(csv_path)}")

        except Exception as e:
            print(f"[Error] 解析失败: {e}")
            self.real_profile = None

    def get_power_law_profile(self, heights, u_ref=10.0, z_ref=100.0, alpha=0.2):
        """指数律模型"""
        valid_h = np.maximum(heights, 1e-3)
        return u_ref * (valid_h / z_ref) ** alpha

    def get_hybrid_profile(self, heights, u_ref=10.0):
        """
        三段式混合模型:
        1. 0~300m: 指数律 (强切变)
        2. 300~1500m: 线性过渡
        3. >1500m: 自由大气 (梯度 + 微扰)
        """
        h_surface = 300.0
        h_free = 1500.0
        v_300 = u_ref * (300.0 / 100.0) ** 0.2
        v_free_base = 15.0

        profile = np.zeros_like(heights)

        # 第一段
        mask1 = heights < h_surface
        h_safe = np.maximum(heights[mask1], 1e-3)
        profile[mask1] = u_ref * (h_safe / 100.0) ** 0.2

        # 第二段
        mask2 = (heights >= h_surface) & (heights < h_free)
        if np.any(mask2):
            slope = (v_free_base - v_300) / (h_free - h_surface)
            profile[mask2] = v_300 + slope * (heights[mask2] - h_surface)

        # 第三段
        mask3 = heights >= h_free
        if np.any(mask3):
            free_shear_slope = 2.0 / 1000.0
            base_wind = v_free_base + free_shear_slope * (heights[mask3] - h_free)
            wave_perturbation = 1.0 * np.sin(2 * np.pi * (heights[mask3] - h_free) / 2000.0)
            profile[mask3] = base_wind + wave_perturbation

        return profile

    def get_wind_vector_field(self, heights, wind_type='hybrid'):
        """生成三维风矢量 (u, v, w)"""
        u = np.zeros_like(heights)
        v = np.zeros_like(heights)
        w = np.zeros_like(heights)

        if wind_type == 'real':
            if self.real_profile:
                u = self.u_interp(heights)
                v = self.v_interp(heights)
            else:
                v = self.get_hybrid_profile(heights)  # 回退

        elif wind_type == 'power_law':
            v = self.get_power_law_profile(heights)
        elif wind_type == 'hybrid':
            v = self.get_hybrid_profile(heights)
        elif wind_type == 'constant':
            v = np.full_like(heights, 10.0)
        else:
            v = self.get_hybrid_profile(heights)

        return u, v, w

    def get_radial_velocity(self, range_axis, azimuth_deg, elevation_deg, wind_type='hybrid'):
        """计算视线方向径向风速"""
        el_rad = np.radians(elevation_deg)
        az_rad = np.radians(azimuth_deg)
        heights = range_axis * np.sin(el_rad)

        u, v, w = self.get_wind_vector_field(heights, wind_type)

        # 投影公式 Eq 4.1
        v_r = u * np.sin(az_rad) * np.cos(el_rad) + \
              v * np.cos(az_rad) * np.cos(el_rad) + \
              w * np.sin(el_rad)

        return v_r, heights


# =============================================================================
# 验证绘图
# =============================================================================
if __name__ == "__main__":
    wf = WindField()
    sim_h = np.linspace(0, 4000, 500)  # 0-4km

    # -----------------------------------------------------------
    # 验证 1: 三种理论模型对比 (您要求新增的图)
    # -----------------------------------------------------------
    print("绘制图表 1: 三种风速模型对比...")
    # 获取三种模型的风速廓线 (这里只看水平风速大小 V)
    _, v_power, _ = wf.get_wind_vector_field(sim_h, 'power_law')
    _, v_hybrid, _ = wf.get_wind_vector_field(sim_h, 'hybrid')
    _, v_const, _ = wf.get_wind_vector_field(sim_h, 'constant')

    plt.figure(figsize=(10, 10))
    plt.plot(v_power, sim_h, label='Power Law', linestyle='--', color='tab:green', alpha=0.7)
    plt.plot(v_const, sim_h, label='Constant', linestyle=':', color='tab:blue', alpha=0.7)
    plt.plot(v_hybrid, sim_h, label='Hybrid', color='tab:red', linewidth=2.5)  # 突出显示混合模型

    # 添加分段参考线
    plt.axhline(300, color='k', linestyle='-.', alpha=0.3)
    plt.text(1, 300, '300m', verticalalignment='bottom', fontsize=9)
    plt.axhline(1500, color='k', linestyle='-.', alpha=0.3)
    plt.text(1, 1500, '1500m', verticalalignment='bottom', fontsize=9)

    plot_style.style.apply_standard_layout(plt.gcf(), plt.gca(),
                                           title="不同风廓线模型仿真水平风速对比",
                                           xlabel="水平风速 (m/s)",
                                           ylabel="高度 (m)")
    plt.legend()
    plt.show()

    # -----------------------------------------------------------
    # 验证 2: 真实数据加载测试 (如果文件存在)
    # -----------------------------------------------------------
    target_file = r"E:\GraduateStu6428\Codes\ObservationData54511\12Z\2025-12-01_12.csv"

    print(f"\n尝试加载探空数据: {os.path.basename(target_file)}")
    wf.load_sounding_data(target_file)

    if wf.real_profile:
        print("绘制图表 2: 真实数据 vs 混合模型...")
        u_real, v_real, _ = wf.get_wind_vector_field(sim_h, 'real')
        speed_real = np.sqrt(u_real ** 2 + v_real ** 2)

        plt.figure(figsize=(20, 10))

        # 左图：廓线对比
        plt.subplot(1, 2, 1)
        plt.plot(speed_real, sim_h, label='Real', color='tab:blue')
        plt.plot(v_hybrid, sim_h, label='Hybrid', color='tab:red', linestyle='--')
        plot_style.style.apply_standard_layout(plt.gcf(), plt.gca(),
                                               title="真实风速 vs 仿真模型",
                                               xlabel="风速 (m/s)", ylabel="高度 (m)")
        plt.legend()

        # 右图：风矢量轨迹
        plt.subplot(1, 2, 2)
        sc = plt.scatter(u_real, v_real, c=sim_h, cmap='viridis', s=15, alpha=0.8)
        plt.colorbar(sc, label='Height (m)')
        plt.axhline(0, color='gray', lw=0.5)
        plt.axvline(0, color='gray', lw=0.5)
        plot_style.style.apply_standard_layout(plt.gcf(), plt.gca(),
                                               title="真实风矢量轨迹 (Hodograph)",
                                               xlabel="U (m/s, East)", ylabel="V (m/s, North)")
        plt.axis('equal')  # 保证 XY 比例一致

        plt.tight_layout()
        plt.show()
    else:
        print("未找到探空文件，跳过真实数据绘图。")