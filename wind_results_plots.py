import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. 常改参数
# =========================================================

# 可以是单个 Excel 文件，也可以是包含多个 wind_results_*.xlsx 的文件夹
input_path = r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\wind_vector_results\wind_results_22deg5_0_P-cooks_S-cooks.xlsx"
# input_path = r"D:\your_path\wind_results_folder"

output_dir = r"F:\3220240787\Lidar_Simulation\wind_inversion\los_velocity_and_snr\wind_vector_results\22deg5\Figures\-23"

# 选择要画的通道
# 如果你的新版 batch_retrieval 使用的是 "S Wind Speed"，把下面的 "S1" 改成 "S"
channels = ["P", "S"]

# sheet 名称模板
# 原始 vector1.py / VectorWindCompositing.py 通常使用这些名称
sheet_map = {
    "horizontal_speed": "{channel} Wind Speed",
    "vertical_speed": "{channel} Vertical Speed",
    "wind_direction": "{channel} Wind Direction",
}

# 色标范围
plot_config = {
    "horizontal_speed": {
        "title_name": "Horizontal Wind Speed",
        "cbar_label": "Horizontal Wind Speed (m/s)",
        "vmin": 0,
        "vmax": 25,
        "cmap": "jet",
        "filename_key": "horizontal_speed",
    },
    "vertical_speed": {
        "title_name": "Vertical Wind Speed",
        "cbar_label": "Vertical Wind Speed (m/s)",
        "vmin": -3,
        "vmax": 3,
        "cmap": "jet",
        "filename_key": "vertical_speed",
    },
    "wind_direction": {
        "title_name": "Wind Direction",
        "cbar_label": "Wind Direction (degree)",
        "vmin": 0,
        "vmax": 360,
        "cmap": "hsv",
        "filename_key": "wind_direction",
    },
}

# 若 Excel 的高度列名已经是真实高度，则设为 True
# batch_retrieval 生成的 Excel 通常已经把 height 作为列名，因此建议 True
use_height_from_columns = True

# 如果 Excel 列名不是高度，而是 Col_1, Col_2...，则使用下面参数重建高度
range_resolution_m = 512 * 0.15
elevation_angle_deg = 72
first_valid_height_m = None
# None 表示沿用原 batch_retrieval 的高度写法：
# height = np.arange(0.5, n_gates + 0.5, 1) * range_resolution_m * sin(elevation)


# 时间轴显示数量
n_time_ticks = 10

# 图片分辨率
dpi = 300


# =========================================================
# 2. 字体设置
# =========================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 3. 工具函数
# =========================================================

def collect_excel_files(input_path):
    """
    支持单个 Excel 文件或文件夹输入。
    """
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "wind_results_*.xlsx")))
        if len(files) == 0:
            files = sorted(glob.glob(os.path.join(input_path, "*.xlsx")))
        return files

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def build_height_axis(n_gates):
    """
    当 Excel 列名不是高度时，重建高度轴。
    """
    vertical_resolution = range_resolution_m * np.sin(np.deg2rad(elevation_angle_deg))

    if first_valid_height_m is None:
        # 与原 batch_retrieval/vector1.py 保持一致
        return np.arange(0.5, n_gates + 0.5, 1) * vertical_resolution
    else:
        return first_valid_height_m + np.arange(n_gates) * vertical_resolution


def try_convert_columns_to_height(columns):
    """
    尝试把 Excel 的列名转换成高度。
    """
    heights = []
    for col in columns:
        try:
            heights.append(float(col))
        except Exception:
            return None
    return np.asarray(heights, dtype=float)


def read_wind_sheet(excel_file, sheet_name):
    """
    读取一个风场结果 sheet。

    sheet 结构：
    第1列：时间
    第2列以后：不同高度的结果值
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    time_data = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    value_df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    valid_time_mask = ~time_data.isna()
    time_data = time_data[valid_time_mask].reset_index(drop=True)
    value_df = value_df[valid_time_mask].reset_index(drop=True)

    if use_height_from_columns:
        height = try_convert_columns_to_height(value_df.columns)
        if height is None:
            height = build_height_axis(value_df.shape[1])
    else:
        height = build_height_axis(value_df.shape[1])

    return time_data, height, value_df


def concatenate_same_sheet(excel_files, sheet_name):
    """
    把多个 wind_results_*.xlsx 中同名 sheet 按时间拼接。
    """
    time_list = []
    value_list = []
    height_ref = None

    for excel_file in excel_files:
        try:
            time_data, height, value_df = read_wind_sheet(excel_file, sheet_name)
        except ValueError:
            print(f"Sheet not found: {sheet_name} in {os.path.basename(excel_file)}")
            continue

        if len(time_data) == 0:
            continue

        if height_ref is None:
            height_ref = height
        else:
            if len(height) != len(height_ref):
                raise ValueError(
                    f"Height dimension mismatch in {excel_file}, sheet {sheet_name}"
                )

        time_list.append(time_data)
        value_list.append(value_df)

    if len(time_list) == 0:
        return None, None, None

    all_time = pd.concat(time_list, ignore_index=True)
    all_values = pd.concat(value_list, ignore_index=True)

    # 按时间排序
    sort_idx = np.argsort(all_time.values)
    all_time = all_time.iloc[sort_idx].reset_index(drop=True)
    all_values = all_values.iloc[sort_idx].reset_index(drop=True)

    return all_time, height_ref, all_values


def plot_height_time_heatmap(time_data, height, value_df, title, cbar_label,
                             save_path, cmap, vmin=None, vmax=None):
    """
    绘制高度—时间热力图。
    """
    data_matrix = value_df.to_numpy(dtype=float).T
    n_gates, n_times = data_matrix.shape

    fig, ax = plt.subplots(figsize=(14, 7))

    im = ax.imshow(
        data_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[0, n_times - 1, height[0], height[-1]]
    )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Time", fontsize=15)
    ax.set_ylabel("Height (m)", fontsize=15)

    # x轴时间刻度
    if n_times <= n_time_ticks:
        x_ticks = np.arange(n_times)
    else:
        x_ticks = np.linspace(0, n_times - 1, n_time_ticks, dtype=int)

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
    cbar.set_label(cbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def safe_name(text):
    """
    将字符串转换为适合文件名的形式。
    """
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


# =========================================================
# 4. 主流程
# =========================================================

def main():
    excel_files = collect_excel_files(input_path)

    if len(excel_files) == 0:
        raise FileNotFoundError("No Excel files found.")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(excel_files)} Excel file(s).")

    for channel in channels:
        for variable_key, sheet_template in sheet_map.items():
            sheet_name = sheet_template.format(channel=channel)
            cfg = plot_config[variable_key]

            time_data, height, value_df = concatenate_same_sheet(
                excel_files=excel_files,
                sheet_name=sheet_name
            )

            if time_data is None:
                print(f"Skipped: {sheet_name}")
                continue

            title = f"{channel}-channel {cfg['title_name']} Height-Time Heatmap"

            save_path = os.path.join(
                output_dir,
                f"{safe_name(channel)}_{cfg['filename_key']}_height_time_heatmap.png"
            )

            plot_height_time_heatmap(
                time_data=time_data,
                height=height,
                value_df=value_df,
                title=title,
                cbar_label=cfg["cbar_label"],
                save_path=save_path,
                cmap=cfg["cmap"],
                vmin=cfg["vmin"],
                vmax=cfg["vmax"]
            )

            print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()