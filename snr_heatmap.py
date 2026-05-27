import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. 常改参数
# =========================================================

excel_path = r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\radial_wind_data_2026_05_27_260514.xlsx"
# 输出文件夹
output_dir = r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\radial_wind_data_2026_05_27_260514_new"

# Excel中sheet编号从0开始：
# sheet3 -> 2，一般对应 SNR_P
# sheet4 -> 3，一般对应 SNR_S
p_snr_sheet = 2
s_snr_sheet = 3

# 高度计算参数
# 系统脉冲宽度 500 ns，对应距离分辨率约 c*t/2 ≈ 75 m
# 原代码中使用 512 * 0.15 = 76.8 m 作为距离门间隔
range_resolution_m = 512 * 0.15

# 扫描仰角，单位：degree
elevation_angle_deg = 72

# 第一个有效距离门，也就是处理后表格的第一个SNR列，对应原始第4个距离门
# 根据系统说明，第4个距离门约为100 m
first_valid_height_m = 100.0

# 色标范围
# 若希望自动缩放，设为 None
snr_vmin = -30
snr_vmax = 3

# 图片分辨率
dpi = 300


# =========================================================
# 2. 全局绘图字体设置
# =========================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


def build_height_axis(n_gates):
    """
    根据有效距离门数量构造真实高度轴。

    表格中第3列及以后对应原始第4~60距离门。
    这里将第一个有效距离门高度设为 first_valid_height_m，
    后续高度按距离门垂直投影间隔递增。
    """
    vertical_resolution = range_resolution_m * np.sin(np.deg2rad(elevation_angle_deg))
    height = first_valid_height_m + np.arange(n_gates) * vertical_resolution
    return height


def read_snr_sheet(excel_path, sheet_name):
    """
    读取SNR工作表。

    表格结构：
    第1列：Time
    第2列：Azi_Data
    第3列及以后：不同高度/距离门处的SNR
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    time_data = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    azimuth_data = pd.to_numeric(df.iloc[:, 1], errors="coerce")

    snr_df = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")

    # 删除时间为空的行
    valid_time_mask = ~time_data.isna()
    time_data = time_data[valid_time_mask].reset_index(drop=True)
    azimuth_data = azimuth_data[valid_time_mask].reset_index(drop=True)
    snr_df = snr_df[valid_time_mask].reset_index(drop=True)

    return time_data, azimuth_data, snr_df


def plot_height_time_snr(time_data, snr_df, title, save_path):
    """
    绘制高度—时间SNR热力图。

    横轴：采集时间
    纵轴：真实高度，单位 m
    颜色：SNR，单位 dB
    """
    snr_matrix = snr_df.to_numpy(dtype=float).T
    n_gates, n_times = snr_matrix.shape

    height = build_height_axis(n_gates)

    fig, ax = plt.subplots(figsize=(14, 7))

    im = ax.imshow(
        snr_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=snr_vmin,
        vmax=snr_vmax,
        extent=[0, n_times - 1, height[0], height[-1]]
    )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Time", fontsize=15)
    ax.set_ylabel("Height (m)", fontsize=15)

    # x轴时间刻度
    if n_times <= 10:
        x_ticks = np.arange(n_times)
    else:
        x_ticks = np.linspace(0, n_times - 1, 10, dtype=int)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [time_data.iloc[i].strftime("%H:%M:%S") for i in x_ticks],
        rotation=45,
        ha="right",
        fontsize=12
    )

    # y轴高度刻度
    y_min = height[0]
    y_max = height[-1]
    y_ticks = np.linspace(y_min, y_max, 8)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.0f}" for y in y_ticks], fontsize=12)

    ax.tick_params(axis="both", labelsize=12)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("SNR (dB)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def main():
    os.makedirs(output_dir, exist_ok=True)

    # P通道SNR
    p_time, p_azi, p_snr = read_snr_sheet(excel_path, p_snr_sheet)

    # S通道SNR
    s_time, s_azi, s_snr = read_snr_sheet(excel_path, s_snr_sheet)

    plot_height_time_snr(
        time_data=p_time,
        snr_df=p_snr,
        title="P-channel SNR Height-Time Heatmap",
        save_path=os.path.join(output_dir, "P_channel_SNR_height_time_heatmap.png")
    )

    plot_height_time_snr(
        time_data=s_time,
        snr_df=s_snr,
        title="S-channel SNR Height-Time Heatmap",
        save_path=os.path.join(output_dir, "S_channel_SNR_height_time_heatmap.png")
    )


if __name__ == "__main__":
    main()