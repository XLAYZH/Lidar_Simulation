# -*- coding: utf-8 -*-
"""
vector_wind_solver.py — 矢量风反演算法库

功能
----
从多方向径向风速（LOS velocity）反演三维矢量风 (w, u, v)，并提供水平风速、
垂直风速、风向等衍生量。

支持的算法
----------
+------+----------------+----------------------------------------------+
| 方法 | 函数名         | 原理                                         |
+------+----------------+----------------------------------------------+
| SVD  | SVD()          | 奇异值分解，直接最小二乘求解 V = A⁺·V_los    |
| DSWF | DSWF()         | 直接求和加权拟合（仅一维分支可用）           |
| AIR  | airSWF()       | 自适应迭代重加权，自动抑制异常径向观测       |
| cooks| cooks()        | Cook's 距离剔除，迭代降权有影响力异常点      |
| cooks|cooks_new()     | KNN 预剔除 + Cook's 距离迭代（推荐）         |
| FSWF | FSWF()         | 频率函数加权拟合，优化目标函数求解           |
+------+----------------+----------------------------------------------+

核心入口
--------
SWF(losvelocity, azimuthangle, time_data, num, key, ...)
    统一反演入口，根据 key 参数分发到对应算法。
    支持一维（单窗口单距离门）和二维（滑动窗口多距离门）两种输入模式。

关键辅助函数
------------
is_continuous()             — 判断扫描窗口方位角连续性
clean_data()                — 剔除 NaN 和无效方位角观测
resolution_of_vector()      — 将矢量风投影回径向风速
calculate_R_squared()       — 计算拟合优度 R²
calculate_collinearity_diagnostics() — 共线性诊断

典型调用流程
------------
1. 调用 SWF() 传入径向风速、方位角、时间、算法名称
2. 内部自动完成：输入校验 → 设计矩阵构建 → 连续性判断 → 数据清洗 → 反演
3. 返回 (V_all, hor_speed, ver_speed, wind_dir)

示例
----
.. code-block:: python

    from vector_wind_solver import SWF

    V_all, hor_speed, ver_speed, wind_dir = SWF(
        radial_velocity, azimuth_angle, time_data,
        num=16, key='cooks', elevationangle=72,
    )
    # V_all[:, 0] = w (垂直), V_all[:, 1] = u, V_all[:, 2] = v
"""

import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
def validate_inputs(azimuthangle, elevationangle, losvelocity):
    if len(azimuthangle) != len(losvelocity):
        raise ValueError("方位角、俯仰角和径向风速的数量应一致。")
    if elevationangle <= 0 or elevationangle >= 90:
        raise ValueError("俯仰角应位于0-90°之间。")
    if len(azimuthangle) < 4:
        raise ValueError("至少输入3个不相同的方向。")
def prepare_design_matrix(azimuthangle, elevationangle, num):
    A = np.zeros((num, 3))
    azimuthangle = azimuthangle.flatten()
    for j in range(num):
        Si = np.array([
            np.sin(np.radians(elevationangle)),
            np.cos(np.radians(elevationangle)) * np.sin(np.radians(azimuthangle[j])),
            np.cos(np.radians(elevationangle)) * np.cos(np.radians(azimuthangle[j]))
        ])
        A[j, :] = Si
    return A
def clean_data(A, losvelocity, azimuthangle, bool_mask=None):
    """
    剔除不能参与矢量风反演的径向观测。

    原有算法思路保持不变：这里只剔除已经为 NaN 的径向风速，并叠加扫描连续性/方位角
    判断给出的 bool_mask；不在此处新增 SNR、风速阈值或功率谱形态判据。

    参数
    ----
    A : ndarray, shape=(num, 3)
        由方位角和俯仰角得到的设计矩阵。
    losvelocity : ndarray, shape=(num,)
        当前时间窗口、当前距离门内的径向风速。
    azimuthangle : ndarray, shape=(num,)
        与径向风速逐一对应的实际方位角。
    bool_mask : ndarray, shape=(num,), optional
        来自 is_continuous 的布尔掩膜。若未提供，则默认所有径向在扫描几何上有效；
        这用于修复一维输入分支原先少传 bool_mask 的问题。
    """
    losvelocity = np.asarray(losvelocity)
    azimuthangle = np.asarray(azimuthangle)

    valid_indices = ~np.isnan(losvelocity)
    if bool_mask is None:
        bool_mask = np.ones_like(valid_indices, dtype=bool)
    else:
        bool_mask = np.asarray(bool_mask, dtype=bool)

    if bool_mask.shape[0] != valid_indices.shape[0]:
        raise ValueError("bool_mask 的长度必须与 losvelocity 一致。")

    combined_mask = valid_indices & bool_mask
    A_valid = A[combined_mask, :]
    losvelocity_valid = losvelocity[combined_mask]
    azi_valid = azimuthangle[combined_mask]
    return A_valid, losvelocity_valid, azi_valid
def resolution_of_vector(V_all, azi_data, elevation_angle):
    V_los_all = np.zeros((len(azi_data), 1))
    azi_data = azi_data.flatten()
    for j in range(len(azi_data)):
         Si = np.array([np.sin(np.radians(elevation_angle)),
                        np.cos(np.radians(elevation_angle)) * np.sin(np.radians(azi_data[j])),
                        np.cos(np.radians(elevation_angle)) * np.cos(np.radians(azi_data[j]))])
         V_los = np.dot(V_all, Si)
         V_los_all[j, :] = V_los
    return V_los_all
def SVD(A_valid, losvelocity_valid):
    V = np.linalg.pinv(A_valid).dot(losvelocity_valid)
    # R_squared = calculate_R_squared(A_valid, V, losvelocity_valid)
    # CN_1 = calculate_standard_collinearity_diagnostics(A_valid)
    # CN_2 = calculate_weighted_collinearity_diagnostics(A_valid)
    return V
def calculate_R_squared(A, V, losvelocity):
    '''
    用于计算水平一致性的质量控制参数R^2
    :param A: 俯仰角和方位角计算而得的矩阵
    :param V: 合成矢量风速，w,u,v
    :param losvelocity: 用于合成的径向风速
    :return: R^2
    '''
    V_radial = A.dot(V)
    # 计算 R^2
    V_r = np.sum(losvelocity)  # 求和
    V_r_bar = V_r / len(losvelocity)  # 计算平均值
    SS_res = np.sum((losvelocity - V_radial) ** 2)  # 残差平方和
    SS_tot = np.sum((losvelocity - V_r_bar) ** 2)  # 总平方和
    R_squared = 1 - SS_res / SS_tot
    return R_squared
def calculate_standard_collinearity_diagnostics(A, w=None):
    '''
    :param A:俯仰角和方位角计算而得的矩阵
    :return:
    '''
    if w is None:
        # 根据定义标准化数据矩阵Z
        Z = A.copy()
        for i in range(Z.shape[1]):
            col_norm = np.linalg.norm(A[:, i])
            if col_norm == 0:
                continue
            Z[:, i] = Z[:, i] / col_norm
    else:
        assert np.all(w >= 0) and np.any(w > 0), "权重向量 w 必须是非负的，且不能全部为零。"
        # 创建权重矩阵 W
        W = np.diag(np.sqrt(w))
        B = A @ W
        # 标准化加权设计矩阵
        X = A.copy()
        for i in range(X.shape[1]):
            col_norm = np.sqrt((A[:, i] ** 2).T @ W @ A[:, i])
            if col_norm == 0:
                continue
            X[:, i] = X[:, i] / col_norm

        # 计算加权设计矩阵 X^T W X
        Z = X.T @ W @ X
    # 使用奇异值分解来计算条件数CN(Z)
    try:
        U, s, Vh = np.linalg.svd(Z)
        CN = s.max() / s.min()
    except:
        CN = np.nan
    return CN

def calculate_weighted_collinearity_diagnostics(A, w=None):
    '''
    :param A: 设计矩阵，通常由俯仰角和方位角计算而得
    :param w: 权重向量，对应于 A 中每个观测的权重
    :return:
    '''
    if w is None:
        Z = A.T @ A
    else:
        if np.all(w >= 0) and np.any(w > 0):
            W = np.diag(w)
            Z = A.T @ W @ A
        else:
            print(w)
            print("权重向量 w 必须是非负的，且不能全部为零。")
    # 使用奇异值分解来计算条件数CN(Z)
    try:
        U, s, Vh = np.linalg.svd(Z)
        CN = s.max() / s.min()
    except:
        CN = np.nan
    return CN
def DSWF(A_valid, losvelocity_valid):
    sum_Si_transpose_Si = np.sum([np.outer(A_valid[i, :], A_valid[i, :])
                                  for i in range(A_valid.shape[0])], axis=0)
    sum_Si_transpose_Vr = np.sum([A_valid[i, :] * losvelocity_valid[i]
                                  for i in range(A_valid.shape[0])], axis=0)
    V = np.linalg.pinv(sum_Si_transpose_Si).dot(sum_Si_transpose_Vr)
    return V
def airSWF(A_valid, losvelocity_valid, azimuth_valid, elevationangle,  max_iterations=30):
    w_i = np.ones(len(losvelocity_valid))  # 初始化权重为1
    V = weighted_least_squares(A_valid, losvelocity_valid, w_i)  # 初始参数估计
    V_r = resolution_of_vector(V, azimuth_valid, elevationangle).flatten()  # 初始径向风速估计
    w_i_next, has_converged = air_weights(losvelocity_valid, V_r, w_i, len(losvelocity_valid))  # 权重更新
    t = 0
    while not has_converged and t < max_iterations:
        # print(f"Iteration {t}")
        # 使用更新后的权重重新计算参数估计
        V = weighted_least_squares(A_valid, losvelocity_valid, w_i_next)
        # 重新计算径向风速
        V_r = resolution_of_vector(V, azimuth_valid, elevationangle).flatten()
        # 更新权重并检查收敛条件
        w_i_next, has_converged = air_weights(losvelocity_valid, V_r, w_i_next, len(losvelocity_valid))
        # 更新迭代计数器
        t += 1
    return V, w_i_next
def weighted_least_squares(A, b, w):
    # 构造权重矩阵
    W = np.diag(np.sqrt(w))
    # 应用权重到设计矩阵 A 和观测值 b
    AW = W @ A
    bw = W @ b
    # 使用 SVD 分解加权设计矩阵
    U, sigma, VT = np.linalg.svd(AW, full_matrices=False)
    # 计算 Sigma 的伪逆
    sigma_plus = np.zeros((VT.shape[0], U.shape[1]))
    for i in range(len(sigma)):
        if sigma[i] > 1e-10:  # 避免除以0
            sigma_plus[i, i] = 1.0 / sigma[i]
    # 计算参数的估计值
    V = VT.T @ sigma_plus @ U.T @ bw
    return V
def air_weights(V_ri, V_hat_ri, w_i, p):
    """
    更新权重系数的函数。
    参数:
    V_ri -- 实际测量的径向风速。
    V_hat_ri -- 估计的径向风速。
    w_i -- 当前迭代的权重系数。
    p -- 数据点的数量。
    t -- 当前迭代次数。
    返回:
    w_i_next -- 下一迭代的权重系数。
    has_converged -- 是否满足终止条件。
    """
    # 计算绝对误差
    d_i_t = np.abs(V_ri - V_hat_ri)
    # 计算标准差和平均值
    s_d_t = np.std(d_i_t)
    m_d_t = np.mean(d_i_t)
    # 更新权重系数
    w_i_next = 2 / (1 + np.exp(2 * (d_i_t - (2 * s_d_t - m_d_t)) / s_d_t))
    # 检查终止条件
    has_converged = np.all(np.abs(w_i_next - w_i) / np.abs(w_i) <= 1 / p)
    return w_i_next, has_converged
def calculate_Rw_squared(V, w_i, azimuth_valid, losvelocity_valid, elevation_angle):
    V_radial = resolution_of_vector(V, azimuth_valid, elevation_angle).flatten()
    V_r = np.sum(w_i * losvelocity_valid)
    V_r_bar = V_r / np.sum(w_i)
    SS_res = np.sum(w_i * (losvelocity_valid - V_radial) ** 2)
    SS_tot = np.sum(w_i * (losvelocity_valid - V_r_bar) ** 2)
    R_squared = 1 - SS_res / SS_tot
    return R_squared
def cooks(A_valid, losvelocity_valid, azimuth_valid, elevationangle, min_iterations=2, max_iteration=10):
    w_i = np.ones(len(losvelocity_valid))  # 初始化权重为1
    t = 0
    while True and t < max_iteration:
        V = weighted_least_squares(A_valid, losvelocity_valid, w_i)
        V_r = resolution_of_vector(V, azimuth_valid, elevationangle).flatten()
        cooks_distances = calculate_cooks_distances(V_r, w_i, losvelocity_valid, A_valid, azimuth_valid, elevationangle)
        w_i_next = (1 / np.exp(np.abs(cooks_distances)))
        has_converged = (np.all(np.abs(w_i_next - w_i) / np.abs(w_i) <= 0.05) and t >= min_iterations)
        if has_converged:
            # print(f"Converged after {t} iterations.")
            break
        w_i = w_i_next  # 更新权重以供下次迭代使用
        t += 1
        # print(f"Iteration {t}")
    transformed_weights = (w_i ** 2)
    return V, w_i, transformed_weights
def calculate_cooks_distances(V_r, w_i, losvelocity_valid, A_valid, azimuth_valid, elevation_angle):
    n = len(losvelocity_valid)  # 数据点的数量
    p = 3  # 模型参数的数量（假设是w, u, v）
    cooks_distances = np.zeros(n)

    # 计算原始模型和残差
    residuals = V_r - losvelocity_valid
    original_mse = calculate_mse(residuals)
    # 对每个数据点进行留一分析
    for x in range(len(losvelocity_valid)):
        # 留一法移除一个观测值
        V_temp = np.delete(losvelocity_valid, x)
        A_temp = np.delete(A_valid, x, 0)
        azimuth_temp = np.delete(azimuth_valid, x)
        w_temp = np.delete(w_i, x)
        # 重新计算模型参数
        V_all_temp = weighted_least_squares(A_temp, V_temp, w_temp)
        V_los_all_temp = resolution_of_vector(V_all_temp, azimuth_temp, elevation_angle)
        residuals_temp = V_los_all_temp.flatten() - V_temp

        # 计算留一法后的MSE
        new_mse = calculate_mse(residuals_temp)
        new_rss = np.sum(residuals_temp ** 2)
        original_rss = np.sum(residuals ** 2)
        # 计算Cook's距离
        cooks_distances[x] = (new_rss - original_rss) / original_mse
    return cooks_distances
def calculate_mse(residuals):
    return np.sum(residuals ** 2) / len(residuals)
def count_continuous_azimuth_angles(azimuth_angle, key):
    """
    计算一组中连续的方位角数量
    :param azimuth_angles: 方位角数组
    :return: 连续方位角的数量
    """
    if key == '16':
        azimuth_angle_rounded = np.round(azimuth_angle, 1)
        continuous_count = 1  # 起始点自身也算一个连续点
        for i in range(1, len(azimuth_angle_rounded)):
            diff = azimuth_angle_rounded[i] - azimuth_angle_rounded[i-1]
            if diff == -22.5 or diff == 337.5:  # 考虑循环的情况
                continuous_count += 1
    elif key == '4':
        azimuth_angle_rounded = np.round(azimuth_angle, 1)
        continuous_count = 1  # 起始点自身也算一个连续点
        for i in range(1, len(azimuth_angle_rounded)):
            diff = azimuth_angle_rounded[i] - azimuth_angle_rounded[i-1]
            if diff == 90 or diff == -270:  # 考虑循环的情况
                continuous_count += 1

    return continuous_count
def calculate_cooks_distances_new(V_r, w_i, losvelocity_valid, A_valid, azimuth_valid, elevation_angle):
    n = 8  # 数据点的数量
    p = A_valid.shape[1]  # 模型参数的数量，等于设计矩阵A_valid的列数
    cooks_distances = np.zeros(n)

    # 初次计算模型的残差
    residuals = V_r - losvelocity_valid
    original_rss = np.sum(w_i * (residuals ** 2))

    # 估计残差的方差sigma^2
    sigma_squared = original_rss / (n - p)

    # 对每个数据点计算杠杆值 (h_i)，在有权重的情况下
    W = np.diag(w_i)  # 权重矩阵
    hat_matrix = A_valid @ np.linalg.inv(A_valid.T @ W @ A_valid) @ A_valid.T @ W
    leverages = np.diag(hat_matrix)


    # 对每个数据点计算Cook's距离
    cooks_distances = ((w_i * (residuals ** 2)) / original_rss) * (leverages / ((3 - leverages) ** 2))
    # print('mean', np.mean(cooks_distances))
    # print('std', np.std(cooks_distances))

    has_converged = ((np.max(cooks_distances) - np.mean(cooks_distances)) <= 0.05)
    return cooks_distances, has_converged
# def cooks_new(A_valid, losvelocity_valid, azimuth_valid, elevationangle, min_iterations=2, max_iteration=100, alpha=2, tol=0.05):
#     w_i = np.ones(len(losvelocity_valid))  # 初始化权重为1
#     t = 0
#     while t < max_iteration:
#         V = weighted_least_squares(A_valid, losvelocity_valid, w_i)
#         V_r = resolution_of_vector(V, azimuth_valid, elevationangle).flatten()
#         cooks_distances, has_converged = calculate_cooks_distances_new(V_r, w_i, losvelocity_valid, A_valid, azimuth_valid, elevationangle)
#         if t == 0:
#             print(np.max(cooks_distances))
#         if has_converged:
#             break
#
#         w_i_next = (1 - cooks_distances) * w_i
#         w_i_next = w_i_next * len(losvelocity_valid) / np.sum(w_i_next)
#
#         w_i = w_i_next  # 更新权重以供下次迭代使用
#         t += 1
#     # print(t)
#     return V, w_i
def _estimate_azimuth_grid_phase(azi, step=22.5):
    """
    估计当前扫描窗口相对于 22.5° 网格的整体相位偏移。

    说明：batch_retrieval.py 中会先执行 (azi - 105) % 360，而 105° 不是 22.5° 的整数倍，
    因此校正后的有效方位角不一定落在 0, 22.5, 45, ... 这一绝对网格上。
    这里估计“当前窗口自身的网格相位”，只判断这些方位角是否接近同一个 22.5° 间隔网格，
    不改变后续反演仍使用实际方位角构建设计矩阵的做法。
    """
    finite_azi = np.asarray(azi, dtype=float)
    finite_azi = finite_azi[np.isfinite(finite_azi)]
    if finite_azi.size == 0:
        return 0.0

    # 在 0~step 周期上估计相位。用圆平均避免余数靠近 0/step 边界时的普通均值失效。
    residual = np.mod(finite_azi, step)
    angle = 2 * np.pi * residual / step
    mean_vec = np.nanmean(np.exp(1j * angle))
    phase = (np.angle(mean_vec) % (2 * np.pi)) * step / (2 * np.pi)
    return float(phase)


def is_continuous(azi, time, azi_tolerance=2.0, min_valid_azi=13, max_time_gap_seconds=10,
                  grid_step=22.5, grid_phase=None):
    """
    判断一个扫描窗口是否可作为同一次矢量风合成窗口。

    修正点：不再使用 np.isin(azi, azi)，也不把方位角强行匹配到以 0° 为起点的绝对网格。
    原因是前处理存在系统方位偏置校正，例如 (azi - 105) % 360；105° 不是 22.5° 的整数倍，
    若仍固定匹配 0, 22.5, 45, ...，会把本来有效的扫描窗口全部判为无效。

    当前做法：估计当前窗口相对于 22.5° 网格的整体相位偏移，再判断每个方位角到该相位网格的
    最近距离是否不超过 azi_tolerance。后续构建设计矩阵仍使用实际方位角 azi，不使用网格化角度。
    """
    azi = np.asarray(azi, dtype=float).flatten() % 360

    if len(azi) == 0:
        return False, np.array([], dtype=bool)

    if grid_phase is None:
        grid_phase = _estimate_azimuth_grid_phase(azi, step=grid_step)

    expected_azi = (grid_phase + np.arange(0, 360, grid_step)) % 360
    angle_diff = np.abs(((azi[:, None] - expected_azi[None, :] + 180) % 360) - 180)
    nearest_error = np.nanmin(angle_diff, axis=1)
    matched_angles = np.isfinite(azi) & (nearest_error <= azi_tolerance)

    if np.sum(matched_angles) < min_valid_azi:
        return False, matched_angles

    time_dt = [datetime.strptime(str(t), '%Y-%m-%d %H:%M:%S') for t in time]
    for i in range(1, len(time_dt)):
        # 与 batch_retrieval.py 的外部切分条件保持一致：超过阈值才视为断采。
        if (time_dt[i] - time_dt[i - 1]) > timedelta(seconds=max_time_gap_seconds):
            return False, matched_angles
    return True, matched_angles
# noinspection LanguageDetectionInspection
def detect_anomalies_with_knn(data, weights, threshold_multiplier=1):

    # 仅考虑权重不为0的点
    valid_indices = np.where(weights > 0)[0]
    valid_data = data[valid_indices].reshape(-1, 1)
    k = len(valid_data) - 1
    # # KNN模型
    # nbrs = NearestNeighbors(n_neighbors=k)
    # nbrs.fit(valid_data)
    #
    # # 计算每个点到其K个最近邻的距离
    # distances, indices = nbrs.kneighbors(valid_data)
    #
    # # 计算K个邻居的平均距离
    # avg_distances = distances.mean(axis=1)
    distances = np.abs(valid_data - valid_data.T)
    nearest_distances = np.sort(distances, axis=1)[:, 1:k + 1]
    avg_distances = np.mean(nearest_distances, axis=1)
    # 设定一个阈值，判断异常值
    threshold = np.max((np.mean(avg_distances) + threshold_multiplier * np.std(avg_distances), 6))

    # 识别异常值
    anomalies = np.where(avg_distances > threshold)[0]

    return valid_indices[anomalies]

def cooks_new(A_valid, losvelocity_valid, azimuth_valid, elevationangle, max_iteration=8):
    w_i = np.ones(len(losvelocity_valid))
    t = 0
    a = 0
    prev_w = w_i.copy()
    prev_rss = np.inf
    while t < max_iteration:
        if a == 0:
            anomalies = detect_anomalies_with_knn(losvelocity_valid, w_i)
            if anomalies.size > 0:
                for idx in anomalies:
                    w_i[idx] = 0
                continue
            else:
                a = 1
                continue
        else:
            if np.nansum(w_i) < 3:
                V = np.array([np.nan, np.nan, np.nan])
                break
            try:
                V = weighted_least_squares(A_valid, losvelocity_valid, w_i)
                V_r = resolution_of_vector(V, azimuth_valid, elevationangle).flatten()
                n = len(losvelocity_valid)  # 数据点的数量
                p = A_valid.shape[1]  # 模型参数的数量，等于设计矩阵A_valid的列数
                residuals = V_r - losvelocity_valid
                original_rss = np.sum(w_i * (residuals ** 2))
                # 对每个数据点计算杠杆值 (h_i)，在有权重的情况下
                W = np.diag(w_i)  # 权重矩阵
                hat_matrix = A_valid @ np.linalg.inv(A_valid.T @ W @ A_valid) @ A_valid.T @ W
                leverages = np.diag(hat_matrix)
                # 对每个数据点计算Cook's距离
                cooks_distances = ((w_i * (residuals ** 2)) / (original_rss))
                w_i_new = (1 - cooks_distances) / (1 + leverages)
                w_i[np.where(w_i != 0)] = w_i_new[np.where(w_i != 0)]

                # 计算 RSS 的变化
                rss_change = np.abs(original_rss - prev_rss)
                if rss_change <= 1:
                    break
            except:
                V = np.array([np.nan, np.nan, np.nan])
                break

        # 更新 prev_rss 和 prev_w
        prev_rss = original_rss
        prev_w = w_i.copy()
        t += 1
    return V, prev_w
def FSWF(A_valid, losvelocity_valid):
    x0 = [0, 0, 0]
    bounds = [(-10, 10), (-50, 50), (-50, 50)]
    def objective_function(x):
        return [
            1 / np.sum(np.exp(-(losvelocity_valid - np.dot(A_valid, x)) ** 2 / (2 * 2)),
                       axis=0)]
    initial_value = objective_function(x0)
    if np.isnan(initial_value) or np.isinf(initial_value):
        raise ValueError('Objective function is undefined at the initial point.')
    # 运行优化
    result = minimize(objective_function, x0, bounds=bounds, method='SLSQP',
                      options={'disp': False})

    # 如果发生错误，设置结果为 NaN
    if not result.success:
        result = np.array([np.nan, np.nan, np.nan])
    else:
        result = result.x
    V = result
    return V
def SWF(losvelocity, azimuthangle, time_data, num, key, SNR=None, elevationangle=72,
        azi_tolerance=2.0, min_valid_azi=13, max_time_gap_seconds=10):
    validate_inputs(azimuthangle, elevationangle, losvelocity)
    key_upper = str(key).upper()
    if losvelocity.ndim == 1:
        A = prepare_design_matrix(azimuthangle, elevationangle, num)

        # 一维输入通常对应一个扫描窗口、一个距离门。原代码这里少传 bool_mask，
        # 导致 clean_data 接口不匹配；此处使用同一套连续性/方位角容差判断生成 bool_mask。
        is_window_continuous, bool_mask = is_continuous(
            azimuthangle, time_data,
            azi_tolerance=azi_tolerance,
            min_valid_azi=min_valid_azi,
            max_time_gap_seconds=max_time_gap_seconds
        )
        if not is_window_continuous:
            return np.array([np.nan, np.nan, np.nan])

        A_valid, losvelocity_valid, azimuth_valid = clean_data(A, losvelocity, azimuthangle, bool_mask)
        if len(losvelocity_valid) < 3:
            return np.array([np.nan, np.nan, np.nan])

        if key_upper == 'SVD':
            V = SVD(A_valid, losvelocity_valid)
            R_squared = calculate_R_squared(A_valid, V, losvelocity_valid)
            CN_2 = calculate_weighted_collinearity_diagnostics(A_valid)
            return V, R_squared, CN_2
        elif key_upper == 'DSWF':
            V = DSWF(A_valid, losvelocity_valid)
            return V
        elif key_upper == 'AIR':
            V, w_i = airSWF(A_valid, losvelocity_valid, azimuth_valid, elevationangle)
            R_squared = calculate_Rw_squared(V, w_i, azimuth_valid, losvelocity_valid, elevationangle)
            CN_2 = calculate_weighted_collinearity_diagnostics(A_valid, w_i)

            return V, R_squared, CN_2, w_i
        elif key_upper == 'COOKS':
            V, w_i = cooks_new(A_valid, losvelocity_valid, azimuth_valid, elevationangle)
            R_squared = calculate_Rw_squared(V, w_i, azimuth_valid, losvelocity_valid, elevationangle)
            CN_2 = calculate_weighted_collinearity_diagnostics(A_valid, w_i)
            return V, w_i
        elif key_upper == 'FSWF':
            V1 = FSWF(A_valid, losvelocity_valid)
            return V1
        else:
            raise ValueError(f"未知反演方法 key={key!r}，可选：SVD、DSWF、AIR、cooks、FSWF。")


    else:
        V_all = []
        for j in range(losvelocity.shape[0] - num):
            azi = azimuthangle[j:j+num].copy()
            time = time_data[j:j+num]
            is_window_continuous, bool_mask = is_continuous(
                azi, time,
                azi_tolerance=azi_tolerance,
                min_valid_azi=min_valid_azi,
                max_time_gap_seconds=max_time_gap_seconds
            )
            if is_window_continuous:

                V_all_range = []
                A = prepare_design_matrix(azi, elevationangle, num)
                for k in range(losvelocity.shape[1]):
                    # print(k)
                    A_valid, losvelocity_valid, azimuth_valid = clean_data(A, losvelocity[j:j+num, k], azi, bool_mask)
                    if len(losvelocity_valid) < 13:
                        V_all_range.append([np.nan, np.nan, np.nan])
                        continue
                    # 使用处理后的数据进行计算
                    if key_upper == 'SVD':
                        V = SVD(A_valid, losvelocity_valid)
                        V_all_range.append(V.flatten())
                    elif key_upper == 'AIR':
                        V, w_i = airSWF(A_valid, losvelocity_valid, azimuth_valid, elevationangle)
                        V_all_range.append(V.flatten())
                    elif key_upper == 'COOKS':
                        V, w_i = cooks_new(A_valid, losvelocity_valid, azimuth_valid, elevationangle)
                        V_all_range.append(V.flatten())
                    elif key_upper == 'FSWF':
                        V = FSWF(A_valid, losvelocity_valid)
                        V_all_range.append(V.flatten())
                    else:
                        raise ValueError(f"未知反演方法 key={key!r}，可选：SVD、AIR、cooks、FSWF。")
                V_all.append(np.array(V_all_range))
            else:
                V_all_range = np.full((losvelocity.shape[1], 3), np.nan)
                V_all.append(V_all_range)
    V_all = np.array(V_all)
    hor_speed = np.sqrt(V_all[:, :, 1] ** 2 + V_all[:, :, 2] ** 2)
    ver_speed = np.array(V_all[:, :, 0])
    wind_dir = np.degrees(np.arctan2(V_all[:, :, 1], V_all[:, :, 2])) % 360
    return V_all, hor_speed, ver_speed, wind_dir