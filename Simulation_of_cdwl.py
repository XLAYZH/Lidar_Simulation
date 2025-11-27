class LidarParams:
    """
    相干多普勒激光雷达系统参数类
    包含激光器、探测系统和信号处理相关参数
    """

    def __init__(self):
        # 激光器参数
        self.wavelength = 1550e-9           # 激光波长 (m)
        self.pulse_width = 500e-9           # 激光脉宽 (s)
        self.pulse_energy = 50e-6           # 单脉冲能量 (J)
        self.local_oscillator_power = 2e-3  # 本振光功率 (W)
        self.acousto_optic_frequency_shift = 120e6  # 声光频移量 (Hz)

        # 探测系统参数
        self.telescope_aperture = 0.12     # 望远镜口径 (m)
        self.system_efficiency = 0.1        # 系统效率（无单位）

        # 数据采集与信号处理参数
        self.sampling_rate = 1e9            # 采集卡采样率 (Hz)
        self.range_gate_count = 50          # 距离门个数
        self.samples_per_range_gate = 512   # 每个距离门采样点数
        self.fft_points = 1024              # FFT点数

        # 大气与扫描参数
        self.atmospheric_layer_thickness = 1.0  # 大气分层厚度 (m)
        self.azimuth_step = 22.5            # 方位角步进 (度)
        self.elevation_angle = 72.0         # 俯仰角 (度)



if __name__ == "__main__":
    params = LidarParams()
