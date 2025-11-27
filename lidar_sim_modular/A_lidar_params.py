"""
这个文件不包含逻辑，只包含数据。这样后续所有模块调用参数时，都以此为准，避免不同文件参数不一致。
"""
import numpy as np


class LidarParams:
    """
    激光雷达系统参数集中管理类。
    所有单位均采用国际单位制 (SI): 米, 秒, 焦耳, 瓦特, 安培。
    """

    def __init__(self):
        # --- 基础常数 ---
        self.c = 299792458.0  # 光速 (m/s)
        self.h_planck = 6.626e-34  # 普朗克常数 (J·s)
        self.k_boltzmann = 1.380649e-23  # 玻尔兹曼常数 (J/K)
        self.q_electron = 1.602e-19  # 电子电荷 (C)

        # --- 激光器参数 (论文表2.1, 2.4) ---
        self.wavelength = 1550e-9  # 波长 1550nm
        self.pulse_energy = 50e-6  # 单脉冲能量 50uJ [cite: 471]
        self.pulse_width = 500e-9  # 脉冲宽度 500ns (FWHM) [cite: 471]
        self.prf = 10e3  # 脉冲重复频率 10kHz [cite: 471]
        self.local_power = 2e-3  # 本振光功率 2mW (论文表2.1) [cite: 471]
        # 注意：论文表2.4中提到的 P偏振1.85mW, S偏振1.64mW，这里取仿真通用值 2mW

        self.freq_aom = 120e6  # AOM移频 120MHz [cite: 471]

        # --- 探测器与采集参数 ---
        self.sample_rate = 1e9  # 采样率 1GS/s [cite: 471]
        self.responsivity = 1.0  # 探测器响应度 1.0 A/W (假设值/表2.3最大值) [cite: 666]
        self.bandwidth = 200e6  # 探测器带宽 200MHz [cite: 666]
        self.temperature = 293.15  # 工作温度 20°C (用于热噪声计算)
        self.load_resistance = 50  # 负载阻抗 50欧姆
        self.fft_points = 1024  # FFT点数 [cite: 471]
        self.points_per_bin = 512  # 距离门采样点数 [cite: 471]

        # --- 频率轴定义 (供噪声模块使用) ---
        # 0 ~ fs/2, 取前 512 点
        self.freqs = np.fft.fftfreq(self.fft_points, 1.0 / self.sample_rate)[:self.points_per_bin]
        self.freqs[0] = 1e-10  # 避免在计算 1/f 噪声时出现除以零错误

        # --- 光学系统参数 ---
        self.telescope_diam = 0.12  # 望远镜口径 120mm [cite: 471]
        self.telescope_area = np.pi * (self.telescope_diam / 2) ** 2
        self.system_efficiency = 0.1  # 系统综合效率 0.1 (论文表2.1) [cite: 471]
        # 注意：之前的幅度过大问题已通过修正脉冲功率公式解决，此处效率保持0.1即可

        # --- 扫描与几何参数 ---
        self.elevation_angle_deg = 72  # 俯仰角 72度 [cite: 471]
        self.azimuth_count = 16  # 方位角数量 [cite: 471]

        # --- 仿真网格设置 ---
        self.max_range = 3840.0  # 最大探测距离 (m)
        self.time_step = 1.0 / self.sample_rate
        self.dist_res = 1.0  # 积分步长 1m

        # 生成距离轴 (用于大气积分)
        self.range_axis = np.arange(self.dist_res, self.max_range + self.dist_res, self.dist_res)
        # 垂直高度 (用于大气模型查找)
        self.height_axis = self.range_axis * np.sin(np.radians(self.elevation_angle_deg))

        # 生成时间轴
        round_trip_time = 2 * self.max_range / self.c
        total_samples = int(np.ceil(round_trip_time / self.time_step))
        self.time_axis = np.arange(total_samples) * self.time_step


# 实例化一个全局参数对象，方便调试
params = LidarParams()

if __name__ == "__main__":
    print("LidarParams 模块测试:")
    print(f"  - 距离分辨率: {params.c * params.time_step / 2 * 1e2:.2f} cm")
    print(f"  - 最大探测时间: {params.time_axis[-1] * 1e6:.2f} us")
    print(f"  - 垂直高度最大值: {params.height_axis[-1]:.2f} m")