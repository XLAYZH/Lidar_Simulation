from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# =========================
# 路径参数（请改成你的绝对路径）
# =========================
INPUT_NPZ = Path(r"E:\GraduateStu6428\Codes\MATLAB\Codes_Data.npz")
OUTPUT_NPZ = Path(r"E:\GraduateStu6428\Codes\MATLAB")

OUT_PNG_SINGLE = Path(r"E:\GraduateStu6428\Codes\MATLAB\alpha_profile_single_radial.png")
OUT_PNG_VAD16 = Path(r"E:\GraduateStu6428\Codes\MATLAB\alpha_profile_vad16_mean.png")


# =========================
# 系统与反演参数
# =========================
VISIBILITY_M = 9977.0           # 当天能见度，m
LAMBDA_NM = 1550.0              # 激光波长，nm
SA_AER = 29.978                 # 气溶胶激光雷达比
SM_MOL = 8.0 * np.pi / 3.0      # 分子激光雷达比
K_BETA = 0.2165                 # 地面到最低有效探测门的映射系数

ELEV_DEG = 72.0                 # 仰角
LIDAR_ALT_M = 0.0               # 仪器海拔，若未知先置 0
N_AZIMUTH_PER_VAD = 16          # 一圈 VAD 的方位数

# 绘图时选取的单径向索引
PLOT_SINGLE_RADIAL_INDEX = 0


# =========================
# 能见度 -> 参考边界值
# =========================
def q_from_visibility_km(u_km: float) -> float:
    """
    根据能见度 U(km) 计算经验参数 q
    """
    if u_km <= 0:
        raise ValueError("能见度必须大于 0")

    if u_km > 50.0:
        return 1.6
    elif u_km > 6.0:
        return 1.3
    else:
        return 0.585 * (u_km ** (1.0 / 3.0))


def visibility_to_reference_beta_alpha(
    visibility_m: float,
    lambda_nm: float = LAMBDA_NM,
    sa_aer: float = SA_AER,
    k_beta: float = K_BETA,
) -> tuple[float, float, float, float]:
    """
    由能见度得到：
    1) 近地面气溶胶后向散射系数 beta_near_ground
    2) 最低有效探测门参考后向散射系数 beta_a_ref
    3) 最低有效探测门参考消光系数 alpha_a_ref
    4) q

    全部返回 km 制：
    beta -> km^-1 sr^-1
    alpha -> km^-1
    """
    u_km = visibility_m / 1000.0
    q = q_from_visibility_km(u_km)

    beta_near_ground = 3.91 / (sa_aer * u_km) * ((lambda_nm / 550.0) ** (-q))
    beta_a_ref = k_beta * beta_near_ground
    alpha_a_ref = sa_aer * beta_a_ref
    return beta_near_ground, beta_a_ref, alpha_a_ref, q


# =========================
# 几何关系
# =========================
def slant_range_to_height_km(
    range_m: np.ndarray,
    elev_deg: float = ELEV_DEG,
    lidar_alt_m: float = LIDAR_ALT_M
) -> np.ndarray:
    """
    将斜距转换为高度，返回 km
    """
    range_m = np.asarray(range_m, dtype=np.float64)
    height_m = lidar_alt_m + range_m * np.sin(np.deg2rad(elev_deg))
    return height_m / 1000.0


# =========================
# 分子后向散射系数
# =========================
def molecular_backscatter_km(
    height_km: np.ndarray,
    lambda_nm: float = LAMBDA_NM
) -> np.ndarray:
    """
    按用户给出的公式计算 beta_m(z, lambda)

    这里按下式实现：
        beta_m(z, lambda) = 1.54e-3 * (532 / lambda)^4 * exp(-z / 7)

    约定：
    - z: km
    - lambda: nm
    - beta_m: km^-1 sr^-1

    若你原始文献中的公式还包含额外常数或分母，
    只需要修改此函数即可。
    """
    z_km = np.asarray(height_km, dtype=np.float64)
    beta_m = 1.54e-3 * (532.0 / lambda_nm) ** 4 * np.exp(-z_km / 7.0)
    return beta_m


# =========================
# 构造 Fernald 所需 X(R)
# =========================
def build_total_x(data: np.lib.npyio.NpzFile, eta: float = 1.0) -> np.ndarray:
    """
    用预处理结果构造 Fernald 反演的校正回波功率 X(R)

    建议使用：
        X = p_rcs + eta * s_rcs
    第一版先取 eta = 1.0
    """
    p_rcs = np.asarray(data["p_rcs"], dtype=np.float64)
    s_rcs = np.asarray(data["s_rcs"], dtype=np.float64)

    x_total = p_rcs + eta * s_rcs
    x_total = np.maximum(x_total, 1e-12)
    return x_total


# =========================
# 数值积分
# =========================
def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    返回与 y 同长度的累积梯形积分
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    out = np.zeros_like(y, dtype=np.float64)
    if y.size <= 1:
        return out

    dx = np.diff(x)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)
    return out


# =========================
# Fernald 前向反演（单条径向）
# =========================
def fernald_forward_single(
    x_profile: np.ndarray,
    range_km: np.ndarray,
    beta_m_profile: np.ndarray,
    beta_a_ref: float,
    sa_aer: float = SA_AER,
    sm_mol: float = SM_MOL,
    ref_index: int = 0,
    clip_negative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    单条径向的前向 Fernald 反演

    输入量统一采用 km 制：
    - range_km         : km
    - beta_m_profile   : km^-1 sr^-1
    - beta_a_ref       : km^-1 sr^-1
    - 输出 beta_a      : km^-1 sr^-1
    - 输出 alpha_a     : km^-1

    注：
    这里的积分变量仍为斜距 R，而 beta_m 已按对应高度 z(R) 计算。
    """
    x = np.asarray(x_profile, dtype=np.float64)
    r = np.asarray(range_km, dtype=np.float64)
    beta_m = np.asarray(beta_m_profile, dtype=np.float64)

    if x.ndim != 1 or r.ndim != 1 or beta_m.ndim != 1:
        raise ValueError("x_profile, range_km, beta_m_profile 必须均为一维数组")
    if not (len(x) == len(r) == len(beta_m)):
        raise ValueError("x_profile, range_km, beta_m_profile 长度必须一致")
    if ref_index < 0 or ref_index >= len(r):
        raise IndexError("ref_index 超出范围")

    # 仅处理参考点及其以上部分
    x_seg = np.maximum(x[ref_index:], 1e-12)
    r_seg = r[ref_index:]
    beta_m_seg = beta_m[ref_index:]

    # I(R) = ∫ beta_m(r') dr'
    i_m = cumulative_trapezoid(beta_m_seg, r_seg)

    # G(R) = X(R) * exp[-2(Sa-Sm)∫ beta_m dr]
    g_r = x_seg * np.exp(-2.0 * (sa_aer - sm_mol) * i_m)

    # J(R) = ∫ G(r') dr'
    j_r = cumulative_trapezoid(g_r, r_seg)

    # Fernald 前向公式
    denom = x_seg[0] / (beta_a_ref + beta_m_seg[0]) - 2.0 * sa_aer * j_r

    beta_total_seg = np.full_like(g_r, np.nan, dtype=np.float64)
    valid = denom > 0.0
    beta_total_seg[valid] = g_r[valid] / denom[valid]

    beta_a_seg = beta_total_seg - beta_m_seg
    if clip_negative:
        beta_a_seg = np.maximum(beta_a_seg, 0.0)

    alpha_a_seg = sa_aer * beta_a_seg

    # 补回完整长度
    beta_a = np.full_like(x, np.nan, dtype=np.float64)
    alpha_a = np.full_like(x, np.nan, dtype=np.float64)
    beta_a[ref_index:] = beta_a_seg
    alpha_a[ref_index:] = alpha_a_seg

    return beta_a, alpha_a


# =========================
# 批量单径向反演
# =========================
def invert_all_single_radials(
    x_total: np.ndarray,
    range_km: np.ndarray,
    beta_m_profile: np.ndarray,
    beta_a_ref: float,
    sa_aer: float = SA_AER,
    sm_mol: float = SM_MOL,
    ref_index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    对所有单径向逐条反演
    """
    x_total = np.asarray(x_total, dtype=np.float64)

    n_radial, n_gate = x_total.shape
    beta_a_all = np.full((n_radial, n_gate), np.nan, dtype=np.float64)
    alpha_a_all = np.full((n_radial, n_gate), np.nan, dtype=np.float64)

    for i in range(n_radial):
        beta_a_i, alpha_a_i = fernald_forward_single(
            x_profile=x_total[i],
            range_km=range_km,
            beta_m_profile=beta_m_profile,
            beta_a_ref=beta_a_ref,
            sa_aer=sa_aer,
            sm_mol=sm_mol,
            ref_index=ref_index,
            clip_negative=True
        )
        beta_a_all[i] = beta_a_i
        alpha_a_all[i] = alpha_a_i

    return beta_a_all, alpha_a_all


# =========================
# 16 方位平均
# =========================
def average_x_by_vad_cycle(
    x_total: np.ndarray,
    n_azimuth_per_vad: int = N_AZIMUTH_PER_VAD
) -> tuple[np.ndarray, int]:
    """
    将连续径向按 16 方位组成一个完整 VAD 周期，并先对 X(R) 做平均

    返回：
    - x_vad_mean: shape (N_cycle, N_gate)
    - n_discard:  被舍弃的余数径向个数
    """
    x_total = np.asarray(x_total, dtype=np.float64)
    n_radial, n_gate = x_total.shape

    n_cycle = n_radial // n_azimuth_per_vad
    n_used = n_cycle * n_azimuth_per_vad
    n_discard = n_radial - n_used

    if n_cycle == 0:
        raise ValueError("有效径向数不足一个完整 VAD 周期，无法做 16 方位平均")

    x_trim = x_total[:n_used]
    x_vad_mean = x_trim.reshape(n_cycle, n_azimuth_per_vad, n_gate).mean(axis=1)

    return x_vad_mean, n_discard


def invert_vad_mean_radials(
    x_vad_mean: np.ndarray,
    range_km: np.ndarray,
    beta_m_profile: np.ndarray,
    beta_a_ref: float,
    sa_aer: float = SA_AER,
    sm_mol: float = SM_MOL,
    ref_index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    对 16 方位平均后的各 VAD 周期廓线做 Fernald 反演
    """
    return invert_all_single_radials(
        x_total=x_vad_mean,
        range_km=range_km,
        beta_m_profile=beta_m_profile,
        beta_a_ref=beta_a_ref,
        sa_aer=sa_aer,
        sm_mol=sm_mol,
        ref_index=ref_index
    )


# =========================
# 绘图
# =========================
def plot_single_radial_extinction_profile(
    height_km: np.ndarray,
    alpha_a_single: np.ndarray,
    radial_index: int,
    out_png: Path
) -> None:
    """
    绘制单径向气溶胶消光廓线
    """
    if radial_index < 0 or radial_index >= alpha_a_single.shape[0]:
        raise IndexError("radial_index 超出范围")

    plt.figure(figsize=(6, 8))
    plt.plot(alpha_a_single[radial_index], height_km, linewidth=1.8)
    plt.xlabel(r"Aerosol extinction coefficient $\alpha_a$ (km$^{-1}$)")
    plt.ylabel("Height (km)")
    plt.title(f"Single-radial aerosol extinction profile (radial {radial_index})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_vad16_mean_extinction_profiles(
    height_km: np.ndarray,
    alpha_a_vad16: np.ndarray,
    out_png: Path
) -> None:
    """
    绘制 16 方位平均后的气溶胶消光廓线：
    - 细线：每个完整 VAD 周期的反演结果
    - 粗线：所有完整周期的平均结果
    """
    plt.figure(figsize=(6, 8))

    for i in range(alpha_a_vad16.shape[0]):
        plt.plot(alpha_a_vad16[i], height_km, linewidth=1.0, alpha=0.35)

    alpha_mean = np.nanmean(alpha_a_vad16, axis=0)
    plt.plot(alpha_mean, height_km, linewidth=2.5, label="Mean of all VAD-16 profiles")

    plt.xlabel(r"Aerosol extinction coefficient $\alpha_a$ (km$^{-1}$)")
    plt.ylabel("Height (km)")
    plt.title("Aerosol extinction profiles after 16-azimuth averaging")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# 主程序
# =========================
def main() -> None:
    data = np.load(INPUT_NPZ)

    # 1) 读取预处理结果
    range_m = np.asarray(data["range_m"], dtype=np.float64)
    range_km = range_m / 1000.0
    kept_gate_numbers = np.asarray(data["kept_gate_numbers"], dtype=np.int32)

    # 2) 构造 X(R)
    # 注意：这里直接使用 pre_process.py 产出的 p_rcs / s_rcs，
    # 且不再重复做 R^2 校正
    x_total = build_total_x(data, eta=1.0)

    # 3) 斜距 -> 高度（km）
    height_km = slant_range_to_height_km(
        range_m=range_m,
        elev_deg=ELEV_DEG,
        lidar_alt_m=LIDAR_ALT_M
    )

    # 4) 分子后向散射系数 beta_m(z)
    beta_m_profile = molecular_backscatter_km(
        height_km=height_km,
        lambda_nm=LAMBDA_NM
    )

    # 5) 能见度 -> 最低有效门参考值
    beta_ng, beta_a_ref, alpha_a_ref, q_val = visibility_to_reference_beta_alpha(
        visibility_m=VISIBILITY_M,
        lambda_nm=LAMBDA_NM,
        sa_aer=SA_AER,
        k_beta=K_BETA
    )

    # 6) 单径向 Fernald 反演
    beta_a_single, alpha_a_single = invert_all_single_radials(
        x_total=x_total,
        range_km=range_km,
        beta_m_profile=beta_m_profile,
        beta_a_ref=beta_a_ref,
        sa_aer=SA_AER,
        sm_mol=SM_MOL,
        ref_index=0
    )

    # 7) 16 方位平均后再反演
    x_vad16_mean, n_discard = average_x_by_vad_cycle(
        x_total=x_total,
        n_azimuth_per_vad=N_AZIMUTH_PER_VAD
    )

    beta_a_vad16, alpha_a_vad16 = invert_vad_mean_radials(
        x_vad_mean=x_vad16_mean,
        range_km=range_km,
        beta_m_profile=beta_m_profile,
        beta_a_ref=beta_a_ref,
        sa_aer=SA_AER,
        sm_mol=SM_MOL,
        ref_index=0
    )

    # 8) 多周期平均代表廓线
    beta_a_vad16_mean = np.nanmean(beta_a_vad16, axis=0)
    alpha_a_vad16_mean = np.nanmean(alpha_a_vad16, axis=0)

    # 9) 保存结果
    np.savez_compressed(
        OUTPUT_NPZ,
        range_m=range_m,
        range_km=range_km,
        height_km=height_km,
        kept_gate_numbers=kept_gate_numbers,

        x_total=x_total,
        x_vad16_mean=x_vad16_mean,

        beta_m_profile=beta_m_profile,

        beta_near_ground=beta_ng,
        beta_a_ref=beta_a_ref,
        alpha_a_ref=alpha_a_ref,
        q_visibility=q_val,

        beta_a_single=beta_a_single,
        alpha_a_single=alpha_a_single,

        beta_a_vad16=beta_a_vad16,
        alpha_a_vad16=alpha_a_vad16,

        beta_a_vad16_mean=beta_a_vad16_mean,
        alpha_a_vad16_mean=alpha_a_vad16_mean,

        visibility_m=np.array(VISIBILITY_M, dtype=np.float64),
        sa_aer=np.array(SA_AER, dtype=np.float64),
        sm_mol=np.array(SM_MOL, dtype=np.float64),
        k_beta=np.array(K_BETA, dtype=np.float64),
        elev_deg=np.array(ELEV_DEG, dtype=np.float64),
        n_azimuth_per_vad=np.array(N_AZIMUTH_PER_VAD, dtype=np.int32),
        n_discard_after_vad_mean=np.array(n_discard, dtype=np.int32)
    )

    # 10) 出图
    plot_single_radial_extinction_profile(
        height_km=height_km,
        alpha_a_single=alpha_a_single,
        radial_index=PLOT_SINGLE_RADIAL_INDEX,
        out_png=OUT_PNG_SINGLE
    )

    plot_vad16_mean_extinction_profiles(
        height_km=height_km,
        alpha_a_vad16=alpha_a_vad16,
        out_png=OUT_PNG_VAD16
    )

    # 11) 输出说明
    print("反演完成")
    print(f"输入文件: {INPUT_NPZ}")
    print(f"输出文件: {OUTPUT_NPZ}")
    print(f"单径向消光廓线 shape: {alpha_a_single.shape}")
    print(f"16方位平均后消光廓线 shape: {alpha_a_vad16.shape}")
    print(f"16方位平均后舍弃径向数: {n_discard}")
    print(f"最低有效门参考 beta_a_ref: {beta_a_ref:.6e} km^-1 sr^-1")
    print(f"最低有效门参考 alpha_a_ref: {alpha_a_ref:.6e} km^-1")
    print(f"q = {q_val:.3f}")
    print(f"单径向廓线图: {OUT_PNG_SINGLE}")
    print(f"16方位平均廓线图: {OUT_PNG_VAD16}")


if __name__ == "__main__":
    main()