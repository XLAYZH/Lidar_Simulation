# -*- coding: utf-8 -*-
"""
benchmark_inversion.py — 矢量风反演算法实时性基准测试

功能
----
随机选取 3 个 NPZ 数据文件，对五种矢量风反演算法进行耗时对比测试，
重复计算 3 次取均值与标准差，结果保存为 xlsx 文件并以学术科研风格绘制图表。

矢量风反演算法：
    SVD   — 奇异值分解
    DSWF  — 直接求和加权拟合
    AIR   — 自适应迭代重加权
    cooks — Cook's 距离剔除
    FSWF  — 频率函数加权拟合

输出文件
--------
- benchmark_results.xlsx    : 统计数据表
- benchmark_results.png/eps : 学术风格柱状对比图

使用方法
--------
直接运行：python benchmark_inversion.py
"""

import os
import random
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import minimize

# ======================== 可配置参数 ========================

# NPZ 文件搜索目录（仅从这三个年份目录中选取）
NPZ_DIRS = [
    r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\year_2024",
    r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\year_2025",
    r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\year_2026",
]

# 随机选取的文件数
N_FILES = 3

# 每个文件-算法组合重复计算次数
N_REPEAT = 10

# 测试用数据规模（若文件不足则取实际值）
N_RADIALS = 16
N_GATES = 60

# 俯仰角
ELEVATION_ANGLE = 72

# 随机种子（确保可复现）
RANDOM_SEED = 42

# 输出文件（保存在脚本所在目录下的 benchmark 子文件夹）
_SCRIPT_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _SCRIPT_DIR / "benchmark"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_XLSX = str(_OUTPUT_DIR / "benchmark_results.xlsx")
OUTPUT_PNG  = str(_OUTPUT_DIR / "benchmark_results.png")
OUTPUT_EPS  = str(_OUTPUT_DIR / "benchmark_results.eps")

# ======================== 学术风格全局设置 ========================

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})
# 中文回退
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ======================== 矢量风反演核心函数 ========================

def prepare_design_matrix(azimuthangle, elevationangle, num):
    A = np.zeros((num, 3))
    azimuthangle = np.asarray(azimuthangle).flatten()
    for j in range(num):
        Si = np.array([
            np.sin(np.radians(elevationangle)),
            np.cos(np.radians(elevationangle)) * np.sin(np.radians(azimuthangle[j])),
            np.cos(np.radians(elevationangle)) * np.cos(np.radians(azimuthangle[j]))
        ])
        A[j, :] = Si
    return A


def vector_svd(A, los):
    return np.linalg.pinv(A).dot(los)


def vector_dswf(A, los):
    sum_StS = np.sum([np.outer(A[i, :], A[i, :]) for i in range(A.shape[0])], axis=0)
    sum_StV = np.sum([A[i, :] * los[i] for i in range(A.shape[0])], axis=0)
    return np.linalg.pinv(sum_StS).dot(sum_StV)


def weighted_ls(A, b, w):
    W = np.diag(np.sqrt(w))
    AW = W @ A
    bw = W @ b
    U, sigma, VT = np.linalg.svd(AW, full_matrices=False)
    sigma_plus = np.zeros((VT.shape[0], U.shape[1]))
    for i in range(len(sigma)):
        if sigma[i] > 1e-10:
            sigma_plus[i, i] = 1.0 / sigma[i]
    return VT.T @ sigma_plus @ U.T @ bw


def resolution_of_vector(V, azi_data, elev_angle):
    V_los_all = np.zeros(len(azi_data))
    for j in range(len(azi_data)):
        Si = np.array([np.sin(np.radians(elev_angle)),
                       np.cos(np.radians(elev_angle)) * np.sin(np.radians(azi_data[j])),
                       np.cos(np.radians(elev_angle)) * np.cos(np.radians(azi_data[j]))])
        V_los_all[j] = np.dot(V, Si)
    return V_los_all


def vector_air(A, los, azi, elev, max_iter=30):
    w = np.ones(len(los))
    V = weighted_ls(A, los, w)
    V_r = resolution_of_vector(V, azi, elev)
    d = np.abs(los - V_r)
    s_d, m_d = np.std(d), np.mean(d)
    w_next = 2 / (1 + np.exp(2 * (d - (2 * s_d - m_d)) / s_d))
    t = 0
    while t < max_iter:
        V = weighted_ls(A, los, w_next)
        V_r = resolution_of_vector(V, azi, elev)
        d = np.abs(los - V_r)
        s_d, m_d = np.std(d), np.mean(d)
        w_new = 2 / (1 + np.exp(2 * (d - (2 * s_d - m_d)) / s_d))
        if np.all(np.abs(w_new - w_next) / np.abs(w_next) <= 1 / len(los)):
            w_next = w_new
            break
        w_next = w_new
        t += 1
    return V


def vector_cooks(A, los, azi, elev, max_iter=8):
    w = np.ones(len(los))
    t = 0
    a = 0
    prev_rss = np.inf
    while t < max_iter:
        if a == 0:
            valid = np.where(w > 0)[0]
            valid_data = los[valid].reshape(-1, 1)
            distances = np.abs(valid_data - valid_data.T)
            k_neighbors = max(1, len(valid_data) - 1)
            nearest = np.sort(distances, axis=1)[:, 1:k_neighbors + 1]
            avg_dist = np.mean(nearest, axis=1)
            threshold = max(np.mean(avg_dist) + np.std(avg_dist), 6)
            anomalies = np.where(avg_dist > threshold)[0]
            if anomalies.size > 0:
                for idx in anomalies:
                    w[valid[idx]] = 0
                continue
            else:
                a = 1
                continue
        else:
            if np.nansum(w) < 3:
                return np.array([np.nan, np.nan, np.nan])
            try:
                V = weighted_ls(A, los, w)
                V_r = resolution_of_vector(V, azi, elev)
                residuals = V_r - los
                rss = np.sum(w * (residuals ** 2))
                W = np.diag(w)
                hat = A @ np.linalg.inv(A.T @ W @ A) @ A.T @ W
                leverages = np.diag(hat)
                cooks_d = (w * (residuals ** 2)) / rss
                w_new = (1 - cooks_d) / (1 + leverages)
                w[np.where(w != 0)] = w_new[np.where(w != 0)]
                if np.abs(rss - prev_rss) <= 1:
                    break
            except Exception:
                return np.array([np.nan, np.nan, np.nan])
        prev_rss = rss
        t += 1
    return V


def vector_fswf(A, los):
    x0 = [0, 0, 0]
    bounds = [(-10, 10), (-50, 50), (-50, 50)]

    def objective(x):
        return [1 / np.sum(np.exp(-(los - np.dot(A, x)) ** 2 / (2 * 2)), axis=0)]

    result = minimize(objective, x0, bounds=bounds, method='SLSQP',
                      options={'disp': False})
    if not result.success:
        return np.array([np.nan, np.nan, np.nan])
    return result.x


# ======================== 算法注册表 ========================

VECTOR_METHODS = {
    'SVD':   vector_svd,
    'DSWF':  vector_dswf,
    'AIR':   vector_air,
    'cooks': vector_cooks,
    'FSWF':  vector_fswf,
}


def _run_one_pass(func, A, radial_v, azi, elev, n_gates):
    """单次完整反演：遍历所有距离门"""
    for k in range(n_gates):
        los = radial_v[:, k]
        if np.any(np.isnan(los)):
            continue
        if func in (vector_air, vector_cooks):
            func(A, los, azi, elev)
        else:
            func(A, los)


# ======================== 数据加载 ========================

def collect_npz_files(npz_dirs):
    """从多个目录收集所有 _radial_wind_*.npz 文件"""
    all_files = []
    for npz_dir in npz_dirs:
        path = Path(npz_dir)
        if not path.exists():
            raise FileNotFoundError(f"目录不存在: {npz_dir}")
        files = sorted(path.glob('*_radial_wind_*.npz'))
        all_files.extend(files)
    if not all_files:
        raise FileNotFoundError(f"未在指定目录中找到 *_radial_wind_*.npz 文件")
    return sorted(all_files)


def load_test_data(npz_path, n_radials, n_gates):
    """加载 NPZ 文件并截取指定规模的数据"""
    data = np.load(npz_path)
    actual_radials = min(data['radial_v_P'].shape[0], n_radials)
    actual_gates = min(data['radial_v_P'].shape[1], n_gates)
    return (
        data['radial_v_P'][:actual_radials, :actual_gates],
        data['azi_data'][:actual_radials],
        data['time'][:actual_radials],
    )


# ======================== 主测试 ========================

def run_benchmark():
    """执行完整基准测试"""
    # 1. 收集并随机选取文件
    all_files = collect_npz_files(NPZ_DIRS)
    random.seed(RANDOM_SEED)
    selected = random.sample(all_files, min(N_FILES, len(all_files)))
    selected = sorted(selected)

    print("=" * 70)
    print("  矢量风反演算法实时性基准测试")
    print("=" * 70)
    print(f"  NPZ 数据源: year_2024 / year_2025 / year_2026")
    print(f"  找到 {len(all_files)} 个文件, 随机选取 {len(selected)} 个 (seed={RANDOM_SEED}):")
    for f in selected:
        print(f"    - {f.name}")
    print(f"  每文件-算法重复: {N_REPEAT} 次（同一批文件对所有算法）")
    print(f"  数据规模上限: {N_RADIALS} 径向 × {N_GATES} 距离门")

    # 2. 逐文件逐算法测试
    all_records = []

    for file_idx, npz_path in enumerate(selected):
        fname = npz_path.stem
        print(f"\n{'─' * 70}")
        print(f"  [{file_idx + 1}/{len(selected)}] 加载: {fname}")

        radial_v, azi, time_data = load_test_data(npz_path, N_RADIALS, N_GATES)
        n_radials = radial_v.shape[0]
        n_gates = radial_v.shape[1]
        print(f"  实际规模: {n_radials} 径向 × {n_gates} 距离门")

        elev = ELEVATION_ANGLE
        A = prepare_design_matrix(azi, elev, n_radials)

        for method_name, method_func in VECTOR_METHODS.items():
            times = []
            for rep in range(N_REPEAT):
                t0 = time.perf_counter()
                _run_one_pass(method_func, A, radial_v, azi, elev, n_gates)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)

            mean_t = np.mean(times)
            std_t = np.std(times, ddof=1) if N_REPEAT > 1 else 0.0

            all_records.append({
                'File': fname,
                'Method': method_name,
                'Radials': n_radials,
                'Gates': n_gates,
                'Rep_1_s': times[0],
                'Rep_2_s': times[1] if len(times) > 1 else np.nan,
                'Rep_3_s': times[2] if len(times) > 2 else np.nan,
                'Mean_s': mean_t,
                'Std_s': std_t,
            })
            print(f"    {method_name:>6s}:  {mean_t:.4f}s ± {std_t:.4f}s  "
                  f"(各次: {', '.join(f'{t:.4f}' for t in times)})")

    # 3. 构建 DataFrame 并保存 xlsx
    df = pd.DataFrame(all_records)
    # 增加相对 SVD 的比值列
    svd_means = df[df['Method'] == 'SVD'].set_index('File')['Mean_s']
    df['Rel_to_SVD'] = df.apply(
        lambda row: row['Mean_s'] / svd_means.get(row['File'], np.nan), axis=1
    )

    print(f"\n{'=' * 70}")
    print("  保存结果至 xlsx ...")
    with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
        # Sheet 1: 原始记录
        df.to_excel(writer, sheet_name='Raw_Records', index=False)

        # Sheet 2: 按算法汇总（跨文件均值）
        summary = df.groupby('Method').agg(
            Mean_Mean_s=('Mean_s', 'mean'),
            Std_Mean_s=('Mean_s', 'std'),
            Mean_Std_s=('Std_s', 'mean'),
            Mean_Rel_to_SVD=('Rel_to_SVD', 'mean'),
        ).reset_index()
        summary = summary.sort_values('Mean_Mean_s')
        summary.to_excel(writer, sheet_name='Summary_by_Method', index=False)
        print("\n  ▶ Summary_by_Method:")
        print(summary.to_string(index=False))

        # Sheet 3: 按文件汇总
        best_idx = df.groupby('File')['Mean_s'].idxmin()
        file_best = df.loc[best_idx, ['File', 'Method']].rename(columns={'Method': 'Best_Method'})
        file_minmax = df.groupby('File').agg(
            Min_Mean_s=('Mean_s', 'min'),
            Max_Mean_s=('Mean_s', 'max'),
        ).reset_index()
        file_summary = file_best.merge(file_minmax, on='File')
        file_summary.to_excel(writer, sheet_name='Summary_by_File', index=False)

    print(f"  结果已保存: {OUTPUT_XLSX}")

    return df, summary


# ======================== 学术风格绘图 ========================

def plot_academic_style(df):
    """绘制学术论文风格的算法对比柱状图"""

    methods_order = ['SVD', 'DSWF', 'AIR', 'cooks', 'FSWF']
    files = sorted(df['File'].unique())

    # 从 Summary_by_Method 构建跨文件统计
    agg = df.groupby('Method').agg(
        mean=('Mean_s', 'mean'),
        std=('Mean_s', 'std'),
    ).reindex(methods_order)

    means = agg['mean'].values
    stds = agg['std'].values

    # ---------- Figure 1: 主柱状图 ----------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    x = np.arange(len(methods_order))
    width = 0.55

    # 学术常用配色（色盲友好）
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5']

    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.6,
                  error_kw={'linewidth': 1.0, 'capthick': 1.0})

    # 在柱上标注数值
    for i, (bar, m, s) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.002,
                f'{m:.4f}s', ha='center', va='bottom', fontsize=8.5,
                fontfamily='serif')

    ax.set_xticks(x)
    ax.set_xticklabels(methods_order, fontsize=11)
    ax.set_ylabel('Computation Time (s)', fontsize=12)
    ax.set_title('Vector Wind Inversion Algorithm Performance Comparison',
                 fontsize=13, fontweight='bold', pad=12)

    # 网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.35, linewidth=0.5)
    ax.set_axisbelow(True)

    # 边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_EPS, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  图表已保存: {OUTPUT_PNG}, {OUTPUT_EPS}")
    plt.show()

    # ---------- Figure 2: 分组柱状图（按文件分面） ----------
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))

    n_files = len(files)
    n_methods = len(methods_order)
    total_groups = n_files * n_methods
    x2 = np.arange(total_groups)
    width2 = 0.6

    file_colors = ['#4472C4', '#ED7D31', '#70AD47']
    bar_colors = []
    bar_means2 = []
    bar_stds2 = []
    labels2 = []

    for fi, fname in enumerate(files):
        for mi, method in enumerate(methods_order):
            subset = df[(df['File'] == fname) & (df['Method'] == method)]
            if not subset.empty:
                bar_means2.append(subset['Mean_s'].values[0])
                bar_stds2.append(subset['Std_s'].values[0])
            else:
                bar_means2.append(0)
                bar_stds2.append(0)
            bar_colors.append(file_colors[fi])
            labels2.append(f"{fname[:20]}\n{method}")

    bars2 = ax2.bar(x2, bar_means2, width2, yerr=bar_stds2, capsize=3,
                    color=bar_colors, edgecolor='black', linewidth=0.5,
                    error_kw={'linewidth': 0.8})

    # 简化的 x 轴标签
    simple_labels = [f"F{fi+1}\n{m}" for fi, fname in enumerate(files) for m in methods_order]
    ax2.set_xticks(x2)
    ax2.set_xticklabels(simple_labels, fontsize=7.5, rotation=0)
    ax2.set_ylabel('Computation Time (s)', fontsize=12)
    ax2.set_title('Per-File Algorithm Timing (3 NPZ Files × 5 Methods)',
                   fontsize=13, fontweight='bold', pad=12)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.35, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=file_colors[i], edgecolor='black',
                             label=f"F{i+1}: {files[i][:30]}") for i in range(n_files)]
    ax2.legend(handles=legend_elements, fontsize=7.5, loc='upper left',
               framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    per_png = str(_OUTPUT_DIR / "benchmark_per_file.png")
    per_eps = str(_OUTPUT_DIR / "benchmark_per_file.eps")
    fig2.savefig(per_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig(per_eps, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  分文件图表已保存: {per_png}, {per_eps}")
    plt.show()


# ======================== 入口 ========================

def main():
    df, summary = run_benchmark()

    print(f"\n{'=' * 70}")
    print("  跨文件汇总排名（快→慢）")
    print("=" * 70)
    for _, row in summary.iterrows():
        print(f"  {row['Method']:>6s}:  {row['Mean_Mean_s']:.4f}s ± {row['Std_Mean_s']:.4f}s  "
              f"(相对SVD: {row['Mean_Rel_to_SVD']:.2f}x)")

    plot_academic_style(df)


if __name__ == '__main__':
    main()
