import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. 常改参数
# =========================================================

excel_path = r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\radial_wind_data_22.5_45_260514.xlsx"

# 输出总文件夹
output_dir = r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\radial_wind_data_22.5_45_260514_new"

# Excel sheet 索引（从0开始）
# sheet3 -> 2，一般为 P 通道 SNR
# sheet4 -> 3，一般为 S 通道 SNR
p_snr_sheet = 2
s_snr_sheet = 3

# 16个标准方位角
nominal_azimuths = np.arange(22.5, 360, 45)

# 方位角匹配容差（与前面讨论一致）
azimuth_tolerance = 2.0  # degree

# -------------------------
# 高度轴参数
# -------------------------
# 原系统脉冲宽度 500 ns，对应单个距离门斜距约 c*t/2 ≈ 75 m
# 原代码中常用：512 * 0.15 = 76.8 m
range_resolution_m = 512 * 0.15

# 仰角
elevation_angle_deg = 72

# 第一个有效距离门（即表格第1个SNR列）对应高度
# 根据你给出的系统说明，第4个距离门约为100 m
first_valid_height_m = 100.0

# -------------------------
# 绘图参数
# -------------------------
snr_vmin = -40
snr_vmax = 0
dpi = 300

# 最少记录数要求，太少则不出图
min_records_per_azimuth = 3


# =========================================================
# 2. 全局绘图风格
# =========================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 3. 工具函数
# =========================================================

def circular_diff_deg(a, b):
    """
    计算角度差的最小圆周差值
    """
    return np.abs((a - b + 180) % 360 - 180)


def build_height_axis(n_gates):
    """
    构造真实高度轴（单位：m）

    表格中第3列以后对应原始第4~60距离门。
    这里采用：
    第一个有效距离门高度 = 100 m
    后续高度按垂直距离分辨率递增
    """
    vertical_resolution = range_resolution_m * np.sin(np.deg2rad(elevation_angle_deg))
    height = first_valid_height_m + np.arange(n_gates) * vertical_resolution
    return height


def read_snr_sheet(excel_path, sheet_name):
    """
    读取SNR工作表

    表格结构：
    第1列：Time
    第2列：Azi_Data
    第3列及以后：不同高度的SNR
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    time_data = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    azimuth_data = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    snr_df = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")

    # 删除无效时间行
    valid_mask = ~time_data.isna() & ~azimuth_data.isna()
    time_data = time_data[valid_mask].reset_index(drop=True)
    azimuth_data = azimuth_data[valid_mask].reset_index(drop=True)
    snr_df = snr_df[valid_mask].reset_index(drop=True)

    # 按时间排序
    sort_idx = np.argsort(time_data.values)
    time_data = time_data.iloc[sort_idx].reset_index(drop=True)
    azimuth_data = azimuth_data.iloc[sort_idx].reset_index(drop=True)
    snr_df = snr_df.iloc[sort_idx].reset_index(drop=True)

    return time_data, azimuth_data, snr_df


def extract_one_azimuth(time_data, azimuth_data, snr_df, target_azimuth, tol=2.0):
    """
    提取某一个方位角的数据
    采用圆周角差 + 容差匹配
    """
    diff = circular_diff_deg(azimuth_data.to_numpy(dtype=float), target_azimuth)
    mask = diff <= tol

    sub_time = time_data[mask].reset_index(drop=True)
    sub_azi = azimuth_data[mask].reset_index(drop=True)
    sub_snr = snr_df[mask].reset_index(drop=True)

    return sub_time, sub_azi, sub_snr


def plot_height_time_heatmap(time_data, snr_df, title, save_path,
                             vmin=-40, vmax=0):
    """
    绘制高度-时间SNR热力图
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
        vmin=vmin,
        vmax=vmax,
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
    y_ticks = np.linspace(height[0], height[-1], 8)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.0f}" for y in y_ticks], fontsize=12)

    ax.tick_params(axis="both", labelsize=12)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("SNR (dB)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def azimuth_to_filename_str(azi):
    """
    将方位角转换为适合文件名的字符串
    如 22.5 -> 22p5
    """
    if float(azi).is_integer():
        return f"{int(azi)}"
    else:
        return str(azi).replace(".", "p")


def process_one_channel(excel_path, sheet_name, channel_name, output_dir):
    """
    对一个通道（P或S）分别输出16个方位角热力图
    """
    time_data, azimuth_data, snr_df = read_snr_sheet(excel_path, sheet_name)

    channel_out_dir = os.path.join(output_dir, f"{channel_name}_channel")
    os.makedirs(channel_out_dir, exist_ok=True)

    summary = []

    for azi in nominal_azimuths:
        sub_time, sub_azi, sub_snr = extract_one_azimuth(
            time_data=time_data,
            azimuth_data=azimuth_data,
            snr_df=snr_df,
            target_azimuth=azi,
            tol=azimuth_tolerance
        )

        n_records = len(sub_time)

        if n_records < min_records_per_azimuth:
            summary.append({
                "Azimuth": azi,
                "Count": n_records,
                "Status": "Skipped"
            })
            print(f"[{channel_name}] Azimuth {azi:>5.1f}° : only {n_records} records, skipped.")
            continue

        azi_str = azimuth_to_filename_str(azi)

        title = f"{channel_name}-channel SNR Heatmap (Azimuth = {azi:.1f}°)"
        save_path = os.path.join(
            channel_out_dir,
            f"{channel_name}_channel_SNR_azimuth_{azi_str}deg.png"
        )

        plot_height_time_heatmap(
            time_data=sub_time,
            snr_df=sub_snr,
            title=title,
            save_path=save_path,
            vmin=snr_vmin,
            vmax=snr_vmax
        )

        summary.append({
            "Azimuth": azi,
            "Count": n_records,
            "Status": "Saved"
        })

        print(f"[{channel_name}] Azimuth {azi:>5.1f}° : {n_records} records, figure saved.")

    # 保存统计表
    summary_df = pd.DataFrame(summary)
    summary_df.to_excel(
        os.path.join(channel_out_dir, f"{channel_name}_channel_azimuth_summary.xlsx"),
        index=False
    )


# =========================================================
# 4. 主程序
# =========================================================

def main():
    os.makedirs(output_dir, exist_ok=True)

    # P通道
    process_one_channel(
        excel_path=excel_path,
        sheet_name=p_snr_sheet,
        channel_name="P",
        output_dir=output_dir
    )

    # S通道
    process_one_channel(
        excel_path=excel_path,
        sheet_name=s_snr_sheet,
        channel_name="S",
        output_dir=output_dir
    )

    print("All figures have been generated.")


if __name__ == "__main__":
    main()