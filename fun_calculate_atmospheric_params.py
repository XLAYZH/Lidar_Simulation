import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def molecular_backscatter_coefficient(z, lambda_nm):
    """
    计算大气分子的后向散射系数 β_m(z, λ)

    参数:
    z: 探测距离 (km)
    lambda_nm: 激光波长 (nm)

    返回:
    β_m: 后向散射系数 (km⁻¹·sr⁻¹)
    """
    beta_m_val = 1.54e-3 * (532 / lambda_nm)**4 * np.exp(-z / 7)
    return beta_m_val

def molecular_extinction_coefficient(z, lambda_nm):
    """
    计算大气分子的消光系数 α_m(z, λ)

    参数:
    z: 探测距离 (km)
    lambda_nm: 激光波长 (nm)

    返回:
    α_m: 消光系数 (km⁻¹)
    """
    beta_m_val = molecular_backscatter_coefficient(z, lambda_nm)
    alpha_m_val = (8 * np.pi / 3) * beta_m_val
    return alpha_m_val

def aerosol_backscatter_coefficient(z, lambda_nm):
    """
    计算大气气溶胶的后向散射系数 β_a(z, λ)

    参数:
    z: 探测距离 (km)
    lambda_nm: 激光波长 (nm)

    返回:
    β_a: 后向散射系数 (km⁻¹·sr⁻¹)
    """
    term1 = 2.47e-3 * np.exp(-z / 2)
    term2 = 5.13e-6 * np.exp(-((z - 20) / 6)**2)
    beta_a_val = (term1 + term2) * (532 / lambda_nm)
    return beta_a_val

def aerosol_extinction_coefficient(z, lambda_nm, s_a=50):
    """
    计算大气气溶胶的消光系数 α_a(z, λ)

    参数:
    z: 探测距离 (km)
    lambda_nm: 激光波长 (nm)
    s_a: 气溶胶激光雷达比 (sr), 默认为 50 sr

    返回:
    α_a: 消光系数 (km⁻¹)
    """
    beta_a_val = aerosol_backscatter_coefficient(z, lambda_nm)
    alpha_a_val = s_a * beta_a_val
    return alpha_a_val

def total_extinction_coefficient(z, lambda_nm, s_a=50):
    """
    计算总消光系数 α_total(z, λ) = α_m + α_a

    参数:
    z: 探测距离 (km)
    lambda_nm: 激光波长 (nm)
    s_a: 气溶胶激光雷达比 (sr), 默认为 50 sr

    返回:
    α_total: 总消光系数 (km⁻¹)
    """
    alpha_m_val = molecular_extinction_coefficient(z, lambda_nm)
    alpha_a_val = aerosol_extinction_coefficient(z, lambda_nm, s_a)
    alpha_total_val = alpha_m_val + alpha_a_val
    return alpha_total_val

def atmospheric_transmittance(z, alpha_total):
    """
    计算大气透过率 T(z) = exp(-∫₀ᶻ α(z') dz')

    参数:
    z: 高度数组 (km)
    alpha_total: 总消光系数数组 (km⁻¹)

    返回:
    T: 大气透过率 (无量纲)
    """
    z_m = z
    # 使用累积梯形积分计算 ∫₀ᶻ α(z') dz'
    integral = cumulative_trapezoid(alpha_total, z_m, initial=0)
    # 计算透过率
    transmittance = np.exp(-integral)
    return transmittance

def calculate_atmospheric_parameters(z_range, lambda_nm_val, s_a_val):
    """
    计算大气参数的主要函数

    参数:
    z_range: 高度范围数组 (km)
    lambda_nm_val: 激光波长 (nm)
    s_a_val: 气溶胶激光雷达比 (sr)

    返回:
    dict: 包含所有计算结果的字典
    """
    # 计算各系数
    beta_m = molecular_backscatter_coefficient(z_range, lambda_nm_val)
    alpha_m = molecular_extinction_coefficient(z_range, lambda_nm_val)
    beta_a = aerosol_backscatter_coefficient(z_range, lambda_nm_val)
    alpha_a = aerosol_extinction_coefficient(z_range, lambda_nm_val, s_a_val)
    alpha_total = total_extinction_coefficient(z_range, lambda_nm_val, s_a_val)
    
    # 计算大气透过率
    transmittance = atmospheric_transmittance(z_range, alpha_total)
    
    # 返回结果字典
    results = {
        'z_range': z_range,
        'beta_m': beta_m,
        'alpha_m': alpha_m,
        'beta_a': beta_a,
        'alpha_a': alpha_a,
        'alpha_total': alpha_total,
        'transmittance': transmittance,
        'lambda_nm': lambda_nm_val,
        's_a': s_a_val
    }
    
    return results

def plot_results(results):
    """
    绘制结果图表

    参数:
    results: calculate_atmospheric_parameters函数返回的结果字典
    """
    z_range = results['z_range']
    alpha_m = results['alpha_m']
    alpha_a = results['alpha_a']
    alpha_total = results['alpha_total']
    transmittance = results['transmittance']
    
    # 设置英文字体为Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 绘图 1: 消光系数廓线
    plt.figure(figsize=(8, 8))
    plt.semilogx(alpha_m, z_range, label='Molecular extinction coefficient $\\alpha_m$', color='blue', linewidth=1)
    plt.semilogx(alpha_a, z_range, label='Aerosol extinction coefficient $\\alpha_a$', color='green', linewidth=1)
    plt.semilogx(alpha_total, z_range, label='Total extinction coefficient $\\alpha_{total}$', color='red', linewidth=1)
    
    # 设置坐标轴标签和标题
    plt.ylabel('Altitude z (km)', fontsize=12)
    plt.xlabel('Extinction coefficient (km⁻¹)', fontsize=12)
    plt.title('Atmospheric Molecular, Aerosol and Total Extinction Coefficient Profiles', fontsize=14, pad=20)
    
    # 设置图例
    plt.legend(fontsize=10, loc='upper right', frameon=True, fancybox=True, shadow=True)

    # 设置对数坐标轴的网格线
    plt.grid(True, which='minor', linestyle='-', linewidth=0.05, alpha=0.5, color='gray')  # 细浅色线标注小刻度
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.8, color='gray')  # 粗深色线标注主刻度

    # 设置刻度标签大小
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    
    # 确保网格从y=0开始
    plt.gca().set_ylim(bottom=0, top=30)
    
    plt.tight_layout()
    plt.show()

    # 绘图 2: 大气透过率廓线
    plt.figure(figsize=(8, 8))
    plt.plot(transmittance, z_range, color='blue', linewidth=1)
    
    # 设置坐标轴标签和标题
    plt.ylabel('Altitude z (km)', fontsize=12)
    plt.xlabel('Atmospheric Transmittance $T(z)$', fontsize=12)
    plt.title('Atmospheric Transmittance vs. Altitude', fontsize=14, pad=20)
    
    # 设置网格
    plt.grid(True)
    
    # 设置刻度标签大小
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.gca().set_xlim(right=1)
    plt.gca().set_ylim(bottom=0, top=30)
    
    plt.tight_layout()
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 设置参数
    z_range = np.linspace(0, 30, 1000)  # 高度范围：0 到 30 km
    lambda_nm_val = 1550  # 激光波长：1550 nm
    s_a_val = 50  # 气溶胶激光雷达比

    # 计算大气参数
    results = calculate_atmospheric_parameters(z_range, lambda_nm_val, s_a_val)
    
    # 绘制结果
    plot_results(results)
