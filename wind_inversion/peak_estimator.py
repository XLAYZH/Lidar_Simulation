import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, interp1d
import multiprocessing as mp
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
class PeakFinder:
    @staticmethod
    def find_peak(data):
        """
        峰值法
        """
        return np.argmax(data)

    @staticmethod
    def x_db_bandwidth(data):
        """
        X-dB 带宽方法
        """
        # 计算最大值及其阈值
        R_max = np.max(data)
        R_th = R_max / (10 ** (3 / 10))
        # 找到满足阈值的区域
        threshold_indices = np.where(data >= R_th)[0]
        if len(threshold_indices) == 0:
            return np.nan  # 如果没有满足条件的值，返回 None
        # 确定带宽范围的起始和结束索引
        start_index = threshold_indices[0]
        end_index = threshold_indices[-1]
        # 计算中心位置
        center_index = (start_index + end_index) / 2

        if center_index is None:
            center_index = np.nan
        return center_index

    @staticmethod
    def centroid_method(data, noise):
        """
        质心法
        """
        # 找到峰值位置
        peak_index = np.argmax(data)

        left_index = peak_index
        while left_index > 0 and data[left_index - 1] <= data[left_index]:
            left_index -= 1

        right_index = peak_index
        while right_index < len(data) - 1 and data[right_index + 1] <= data[right_index]:
            right_index += 1

        left_distance = peak_index - left_index
        right_distance = right_index - peak_index
        min_distance = max(left_distance, right_distance)

        # 计算对称索引范围
        balanced_left = max(0, peak_index - min_distance)
        balanced_right = min(len(data) - 1, peak_index + min_distance)

        # 生成均衡索引
        index = np.arange(balanced_left, balanced_right + 1, 1)
        # index = np.arange(left_index, right_index + 1, 1)
        intensity = data[balanced_left:balanced_right + 1]

        centroid = np.sum(index * intensity) / np.sum(intensity)

        # 计算信噪比（SNR）
        signal_power = np.sum(intensity)
        noise_power = np.sum(noise)
        snr = 10 * np.log10(signal_power / noise_power)
        # plt.figure()
        # plt.plot(data)
        # plt.plot(index, intensity)
        # plt.show()
        return centroid, left_index, right_index, snr

    @staticmethod
    def quadratic_fit(x, a2, a1, a0):
        """定义二次多项式函数."""
        return a2 * x ** 2 + a1 * x + a0

    @staticmethod
    def estimate_peak_position(data):
        """
        使用二次多项式拟合估计峰值位置。
        """
        # 找到峰值附近的索引范围
        peak_index = np.argmax(data)
        left_index = max(0, peak_index - 2)  # 取峰值左右5个点作为拟合范围
        right_index = min(len(data) - 1, peak_index + 2)
        index = np.arange(len(data))
        # 提取峰值范围内的数据进行拟合
        fit_index = index[left_index:right_index + 1]
        fit_data = data[left_index:right_index + 1]

        # 二次多项式拟合
        popt, _ = curve_fit(PeakFinder.quadratic_fit, fit_index, fit_data)
        a2, a1, a0 = popt

        # 计算峰值位置
        peak_position = -a1 / (2 * a2)
        return peak_position, popt

    @staticmethod
    def gaussian(x, a, b, c):
        """定义高斯函数"""
        return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

    @staticmethod
    def gaussian_fit(data, resolution, noise):
        """
        高斯拟合
        """
        # 使用索引作为 x 值
        _, x, spec_fft_SM = PeakFinder.spine_fit(data, resolution)

        # 找到插值后数据的峰值位置
        peak_index_high_res = np.argmax(spec_fft_SM)

        # 缩小拟合范围，仅对插值后峰值附近的数据进行拟合
        left_index_high_res = max(0, peak_index_high_res - 10 * resolution)  # 取峰值左右30个点作为拟合范围
        right_index_high_res = min(len(spec_fft_SM) - 1, peak_index_high_res + 10 * resolution)

        # 提取插值后峰值附近的数据进行拟合
        fit_x_data_peak = x[left_index_high_res:right_index_high_res + 1]
        fit_y_data_peak = spec_fft_SM[left_index_high_res:right_index_high_res + 1]

        # 使用插值后峰值附近的数据进行高斯拟合，限制 a 为正值
        initial_guess = [np.max(fit_y_data_peak), fit_x_data_peak[np.argmax(fit_y_data_peak)], 5]  # 初始猜测峰值高度、位置、宽度
        bounds = ([0, -np.inf, 0], [np.inf, np.inf, np.inf])
        try:
            popt, _ = curve_fit(PeakFinder.gaussian, fit_x_data_peak, fit_y_data_peak, p0=initial_guess, bounds=bounds)
            a, b, c = popt
            # 生成拟合曲线（仅在插值后峰值附近范围内）
            fitted_curve_gaussian = PeakFinder.gaussian(fit_x_data_peak, a, b, c)

        except Exception:  # 如果拟合失败
            b = np.nan
            fitted_curve_gaussian = np.nan


        return b, fit_x_data_peak, fitted_curve_gaussian

    @staticmethod
    def quadratic_fit_peak(data, resolution):
        """
        使用二阶多项式拟合寻找峰值。
        """
        _, x, spec_fft_SM = PeakFinder.spine_fit(data, resolution)
        # 找到插值后数据的峰值位置
        peak_index = np.argmax(spec_fft_SM)

        # 取峰值附近的点进行二阶多项式拟合
        left_index = max(0, peak_index - 2*resolution)  # 取峰值左右各 15 个点
        right_index = min(len(spec_fft_SM) - 1, peak_index + 2*resolution)
        fit_x_data = x[left_index:right_index + 1]
        fit_y_data = spec_fft_SM[left_index:right_index + 1]


        # 二阶多项式拟合
        popt = np.polyfit(fit_x_data, fit_y_data, 2)  # 返回 [a2, a1, a0]
        a2, a1, a0 = popt
        # 计算峰值位置（导数为 0 的位置）
        peak_position = -a1 / (2 * a2)
        # 生成二阶多项式拟合曲线，仅在峰值区域有效，其它部分为 0
        fitted_curve_quadratic = np.zeros_like(spec_fft_SM)
        fitted_curve_quadratic[left_index:right_index + 1] = np.polyval(popt, fit_x_data)

        return peak_position, fitted_curve_quadratic

    @staticmethod
    def spine_fit(data, resolution):
        '''
        样条插值拟合
        '''
        # 使用索引作为 x 值
        x_data = np.arange(len(data))
        # 对原始数据进行三次样条插值
        cs = CubicSpline(x_data, data)
        # 生成更高分辨率的 x 值和插值后的 y 值
        num_points = (len(data) - 1) * resolution + 1
        high_res_x = np.linspace(x_data.min(), x_data.max(), num_points)
        high_res_y = cs(high_res_x)
        # 找到插值后数据的峰值位置
        peak_index = high_res_x[np.argmax(high_res_y)]

        return peak_index, high_res_x, high_res_y

    @staticmethod
    def fourier_fit(data, factor=2):
        '''
        傅里叶插值拟合
        '''
        # 使用索引作为 x 值
        x_data = np.arange(len(data))
        # 对原始数据进行傅里叶变换
        fft_data = np.fft.fft(data)
        # 频率分量的数量
        N = len(data)
        # 插值因子：将频谱分辨率提高
        new_N = N * factor
        # 创建一个新的频谱（频域中间插入零）
        new_fft_data = np.zeros(new_N, dtype=complex)
        new_fft_data[:N // 2] = fft_data[:N // 2]
        new_fft_data[-N // 2:] = fft_data[-N // 2:]

        # 对插值后的频谱进行逆傅里叶变换，得到插值后的时域信号
        interpolated_data = np.fft.ifft(new_fft_data)
        # 生成插值后的更高分辨率的 x 值
        high_res_x = np.linspace(x_data.min(), x_data.max(), new_N)
        high_res_y = np.real(interpolated_data)  # 取实部
        # plt.figure()
        # plt.plot(x_data, data)
        # plt.plot(high_res_x, high_res_y)
        # plt.show()
        # 找到插值后数据的峰值位置
        peak_index = high_res_x[np.argmax(high_res_y)]

        return peak_index, high_res_x, high_res_y