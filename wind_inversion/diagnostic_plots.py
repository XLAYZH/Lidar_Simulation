import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from multiprocessing import Pool
from joblib import Parallel, delayed
from vector_wind_solver import *
import time
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# def plot_radial_velocity_snr(radial_v_P, SNR_P, radial_v_S, SNR_S, time, azi_data, group_by_time=True):
#     # 对方位角进行处理
#     azi_data = np.round(azi_data, 1)
#     azi_data[azi_data == 360] = 0
#
#     # 创建 DataFrame
#     df = pd.DataFrame({
#         'time': pd.to_datetime(time),  # 将时间转换为 datetime 对象
#         'azi_data': azi_data,
#     })
#
#     # 添加小时列
#     df['hour'] = df['time'].dt.floor('H')
#
#     # 根据是否按时间分组，选择分组方式
#     if group_by_time:
#         grouped = df.groupby(['hour', 'azi_data'])
#     else:
#         grouped = df.groupby('azi_data')
#
#     # 遍历每个分组并绘制图形
#     for group_key, indices in grouped:
#         # 提取对应的 radial_v_P 和 SNR 数据
#         radial_P_values = radial_v_P[indices.index, :].flatten()
#         SNR_P_values = SNR_P[indices.index, :].flatten()
#         radial_S_values = radial_v_S[indices.index, :].flatten()
#         SNR_S_values = SNR_S[indices.index, :].flatten()
#
#         # 创建子图，设置共享 y 轴
#         fig, axs = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
#
#         # 绘制第一个子图（P 通道）
#         axs[0].scatter(radial_P_values, SNR_P_values, s=1)
#         axs[0].axhline(y=-40, color='r', linestyle='--')
#         # 设置标题
#         if group_by_time:
#             axs[0].set_title(f'方位角：{group_key[1]}\n时间：{group_key[0]}')
#         else:
#             axs[0].set_title(f'方位角：{group_key}')
#         axs[0].set_xlabel('P通道径向风速(m/s)')
#         axs[0].set_ylabel('信噪比(dB)')
#         axs[0].set_xlim(-30, 30)  # 设置 x 轴范围
#
#         # 绘制第二个子图（S 通道）
#         axs[1].scatter(radial_S_values, SNR_S_values, s=1)
#         axs[1].axhline(y=-40, color='r', linestyle='--')
#         if group_by_time:
#             axs[1].set_title(f'方位角：{group_key[1]}\n时间：{group_key[0]}')
#         else:
#             axs[1].set_title(f'方位角：{group_key}')
#         axs[1].set_xlabel('S通道径向风速(m/s)')
#         axs[1].set_ylabel('信噪比(dB)')
#         axs[1].set_xlim(-30, 30)  # 设置 x 轴范围
#
#         plt.show()

def plot_radial_velocity_snr(radial_v_P, SNR_P, radial_v_S, SNR_S, time, azi_data, group_by_time=True):
    # 对方位角进行处理
    azi_data = np.round(azi_data, 1)
    azi_data[azi_data == 360] = 0

    # 创建 DataFrame
    df = pd.DataFrame({
        'time': pd.to_datetime(time),  # 将时间转换为 datetime 对象
        'azi_data': azi_data,
    })

    # 添加小时列
    df['hour'] = df['time'].dt.floor('H')

    # 根据是否按时间分组，选择分组方式
    if group_by_time:
        grouped = df.groupby(['hour', 'azi_data'])
    else:
        grouped = df.groupby('azi_data')

    # 遍历每个分组并绘制图形
    for group_key, indices in grouped:
        # 提取对应的 radial_v_P 和 SNR 数据
        radial_P_values = radial_v_P[indices.index, :].flatten()
        SNR_P_values = SNR_P[indices.index, :].flatten()
        radial_S_values = radial_v_S[indices.index, :].flatten()
        SNR_S_values = SNR_S[indices.index, :].flatten()

        # 绘制 P 通道图
        plt.figure(figsize=(10, 8))
        plt.scatter(radial_P_values, SNR_P_values, s=5, color='blue', alpha=0.7)  # 增加透明度和点的大小
        plt.axhline(y=-40, color='red', linestyle='--', linewidth=4, label='SNR Threshold')  # 加粗虚线
        # 设置标题
        # title = f'方位角：{group_key[1]}\n时间：{group_key[0]}' if group_by_time else f'方位角：{group_key}'
        # plt.title(title, fontsize=20)
        # 设置坐标轴
        plt.xlabel('Velocity (Light of Sight) (m/s)', fontsize=22)
        plt.ylabel('Signal-to-Noise Ratio (dB)', fontsize=22)
        plt.xlim(-30, 30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线
        plt.tight_layout()
        plt.show()

        # 绘制 S 通道图
        plt.figure(figsize=(10, 8))
        plt.scatter(radial_S_values, SNR_S_values, s=10, color='blue', alpha=0.7)  # 增加透明度和点的大小
        plt.axhline(y=-40, color='red', linestyle='--', linewidth=2)  # 加粗虚线
        # 设置标题
        title = f'Azimuth：{group_key[1]}\nTime：{group_key[0]}' if group_by_time else f'Azimuth：{group_key}'
        plt.title(title, fontsize=20)
        # 设置坐标轴
        plt.xlabel('Velocity (Light of Sight) (m/s)', fontsize=18)
        plt.ylabel('Signal-to-Noise Ratio (dB)', fontsize=18)
        plt.xlim(-30, 30)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线
        plt.tight_layout()
        plt.show()

def process_folder(date_str):
    print(date_str)
    data = np.load(rf'F:\3220240787\Lidar_Simulation\wind_inversion\inversion_results\{date_str}_radial_wind_0506.npz')
    # 提取数据
    radial_v_P = data['radial_v_P']
    radial_v_S = data['radial_v_S']
    azi_data = data['azi_data']
    time_data = data['time']
    SNR_P = data['SNR_P']
    SNR_S = data['SNR_S']

    # 校正方位角
    azi_data = np.round(azi_data, 1)
    azi_data = np.round(azi_data * 2) / 2
    azi_data[azi_data == 360] = 0
    azi_data = (azi_data - 105) % 360

    # plot_radial_velocity_snr(radial_v_P, SNR_P, radial_v_S, SNR_S, time_data, azi_data, group_by_time=False)
    # unique_azi_data = np.unique(azi_data)
    # print(unique_azi_data)
    # 根据信噪比对径向风速进行滤除，阈值为-40dB
    # radial_v_P[SNR_P <= -40] = np.nan
    # radial_v_S[SNR_S <= -40] = np.nan
    # 根据时间划分数据矩阵
    time_diff = np.diff(pd.to_datetime(time_data))
    time_diff_seconds = time_diff / np.timedelta64(1, 's')
    # 找出超过 2 秒的时间间隔点索引
    exceed_2s_indices = np.where(time_diff_seconds > 10)[0]
    split_indices = exceed_2s_indices + 1
    radial_v_P_splits = np.split(radial_v_P, split_indices, axis=0)
    radial_v_S_splits = np.split(radial_v_S, split_indices, axis=0)
    azi_data_splits = np.split(azi_data, split_indices, axis=0)
    time_data_splits = np.split(time_data, split_indices, axis=0)
    SNR_P_splits = np.split(SNR_P, split_indices, axis=0)
    SNR_S_splits = np.split(SNR_S, split_indices, axis=0)

    # 定义 P 和 S 数据的反演结果列表
    hor_speed_results_P, ver_speed_results_P, wind_dir_results_P, u_results_P, v_results_P = [], [], [], [], []
    hor_speed_results_S, ver_speed_results_S, wind_dir_results_S, u_results_S, v_results_S = [], [], [], [], []
    range_bin = 512 * 0.15 * np.sin(np.radians(72))
    height = np.arange(0.5, 57.5, 1) * range_bin
    # plt.figure()
    # plt.plot(azi_data_splits[0][14345:14345+16], radial_v_P_splits[0][14345:14345+16, 7], marker='o')
    # plt.show()
    for i in range(len(split_indices) + 1):  # +1 是因为最后一个区段
        # 获取当前的子矩阵
        azi_data_sub = azi_data_splits[i]
        radial_v_P_sub = radial_v_P_splits[i]
        radial_v_S_sub = radial_v_S_splits[i]
        time_data_sub = time_data_splits[i]
        computation_times = {
            "SWF_P": [],
            "SWF_S1": [],
            "SWF_S2": [],
            "SWF_S3": [],
            "SWF_S4": []
        }

        # 记录SWF函数P部分的计算时间
        start_time = time.time()  # 记录开始时间
        V_all_P, hor_speed_P, ver_speed_P, wind_dir_P = SWF(radial_v_P_sub, azi_data_sub, time_data_sub, 16, 'FSWF')
        end_time = time.time()  # 记录结束时间
        computation_times["SWF_P"].append(end_time - start_time)

        # 计算 S 数据的 SWF并记录时间
        start_time = time.time()  # 记录开始时间
        V_all_S1, hor_speed_S1, ver_speed_S1, wind_dir_S1 = SWF(radial_v_P_sub, azi_data_sub, time_data_sub, 16, 'SVD')
        end_time = time.time()  # 记录结束时间
        computation_times["SWF_S1"].append(end_time - start_time)

        plt.figure()
        plt.pcolor(time_data_sub[:100], height, hor_speed_S1[:100, :].T, cmap='jet', vmax=20)
        plt.colorbar()
        ax =plt.gca()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.show()
        a = 0
#这里曾经被注释
        # start_time = time.time()  # 记录开始时间
        # V_all_S2, hor_speed_S2, ver_speed_S2, wind_dir_S2 = SWF(radial_v_S_sub, azi_data_sub, time_data_sub, 16, 'AIR')
        # end_time = time.time()  # 记录结束时间
        # computation_times["SWF_S2"].append(end_time - start_time)
        #
        # start_time = time.time()  # 记录开始时间
        # V_all_S3, hor_speed_S3, ver_speed_S3, wind_dir_S3 = SWF(radial_v_S_sub, azi_data_sub, time_data_sub, 16,
        #                                                         'cooks')
        # end_time = time.time()  # 记录结束时间
        # computation_times["SWF_S3"].append(end_time - start_time)
        #
        # start_time = time.time()  # 记录开始时间
        # V_all_S4, hor_speed_S4, ver_speed_S4, wind_dir_S4 = SWF(radial_v_S_sub, azi_data_sub, time_data_sub, 16, 'FSWF')
        # end_time = time.time()  # 记录结束时间
        # computation_times["SWF_S4"].append(end_time - start_time)


        # df_times = pd.DataFrame(computation_times)
        # df_times.to_excel(f"computation_times_{date_str}_{i}.xlsx", index=False)

        # hor_speed_results_P.append(hor_speed_P)
        # ver_speed_results_P.append(ver_speed_P)
        # wind_dir_results_P.append(wind_dir_P)
        # u_results_P.append((V_all_P[:, :, 1]))
        # v_results_P.append((V_all_P[:, :, 2]))


        # hor_speed_results_S.append(hor_speed_S)
        # ver_speed_results_S.append(ver_speed_S)
        # wind_dir_results_S.append(wind_dir_S)
        # u_results_S.append((V_all_S[:, :, 1]))
        # v_results_S.append((V_all_S[:, :, 2]))

        with pd.ExcelWriter(f'wind_results_{date_str}_{i}.xlsx') as writer:
            pd.DataFrame(hor_speed_P, index=time_data_sub[:-16], columns=height).to_excel(
                writer, sheet_name='P Wind Speed')
            pd.DataFrame(ver_speed_P, index=time_data_sub[:-16], columns=height).to_excel(
                writer, sheet_name='P Vertical Speed')
            pd.DataFrame(wind_dir_P, index=time_data_sub[:-16], columns=height).to_excel(
                writer, sheet_name='P Wind Direction')
            pd.DataFrame(V_all_P[:, :, 1], index=time_data_sub[:-16], columns=height).to_excel(
                writer, sheet_name='P U Component')
            pd.DataFrame(V_all_P[:, :, 2], index=time_data_sub[:-16], columns=height).to_excel(
                writer, sheet_name='P V Component')

            pd.DataFrame(hor_speed_S1, index=time_data_sub[:-16], columns=height).to_excel(
                writer, sheet_name='S1 Wind Speed')
            pd.DataFrame(ver_speed_S1, index=time_data_sub[:-16], columns=height).to_excel(
                writer, sheet_name='S1 Vertical Speed')
            pd.DataFrame(wind_dir_S1, index=time_data_sub[:-16], columns=height).to_excel(
                writer, sheet_name='S1 Wind Direction')
            pd.DataFrame(V_all_S1[:, :, 1], index=time_data_sub[:-16], columns=height).to_excel(
                writer, sheet_name='S1 U Component')
            pd.DataFrame(V_all_S1[:, :, 2], index=time_data_sub[:-16], columns=height).to_excel(
                writer, sheet_name='S1 V Component')

            # pd.DataFrame(hor_speed_S2, index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S2 Horizontal Speed')
            # pd.DataFrame(ver_speed_S2, index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S2 Vertical Speed')
            # pd.DataFrame(wind_dir_S2, index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S2 Wind Direction')
            # pd.DataFrame(V_all_S2[:, :, 1], index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S2 U Component')
            # pd.DataFrame(V_all_S2[:, :, 2], index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S2 V Component')
            #
            # pd.DataFrame(hor_speed_S3, index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S3 Horizontal Speed')
            # pd.DataFrame(ver_speed_S3, index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S3 Vertical Speed')
            # pd.DataFrame(wind_dir_S3, index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S3 Wind Direction')
            # pd.DataFrame(V_all_S3[:, :, 1], index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S3 U Component')
            # pd.DataFrame(V_all_S3[:, :, 2], index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S3 V Component')
            #
            # pd.DataFrame(hor_speed_S4, index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S4 Horizontal Speed')
            # pd.DataFrame(ver_speed_S4, index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S4 Vertical Speed')
            # pd.DataFrame(wind_dir_S4, index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S4 Wind Direction')
            # pd.DataFrame(V_all_S4[:, :, 1], index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S4 U Component')
            # pd.DataFrame(V_all_S4[:, :, 2], index=time_data_sub[:-16], columns=height).to_excel(
            #     writer, sheet_name='S4 V Component')

        print(f'save wind_results_{date_str}_{i}.xlsx')
#这里也是

if __name__ == '__main__':
    main_folder = r"F:\3220240787\Lidar_Simulation\wind_inversion\inversion_results"
    # 获取所有子文件夹名
    subfolders = [name for name in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, name))]

    # Parallel(n_jobs=16)(
    #     delayed(process_folder)(date_str) for date_str in subfolders)


    process_folder('data_00_00_01')



        #     if i < len(split_indices):
        #         # 计算时间间隔（以秒为单位）并确定需要的 NaN 行数
        #         time_interval = time_diff[split_indices[i] - 1] / np.timedelta64(1, 's')
        #         nan_rows_needed = int(time_interval / 2)  # 时间间隔 / 2 秒，取整
        #
        #         # 定义 NaN 行矩阵，行数为 nan_rows_needed，列数为 num_columns
        #         nan_rows_P = np.full((nan_rows_needed, radial_v_P_sub.shape[1]), np.nan)
        #         nan_rows_S = np.full((nan_rows_needed, radial_v_S_sub.shape[1]), np.nan)
        #
        #         # 在 P 和 S 结果列表中添加 NaN 行
        #         hor_speed_results_P.append(nan_rows_P)
        #         ver_speed_results_P.append(nan_rows_P)
        #         wind_dir_results_P.append(nan_rows_P)
        #
        #         hor_speed_results_S.append(nan_rows_S)
        #         ver_speed_results_S.append(nan_rows_S)
        #         wind_dir_results_S.append(nan_rows_S)
        #
        # # 合并 P 和 S 结果
        # hor_speed_P_final = np.vstack(hor_speed_results_P)
        # ver_speed_P_final = np.vstack(ver_speed_results_P)
        # wind_dir_P_final = np.vstack(wind_dir_results_P)
        #
        # hor_speed_S_final = np.vstack(hor_speed_results_S)
        # ver_speed_S_final = np.vstack(ver_speed_results_S)
        # wind_dir_S_final = np.vstack(wind_dir_results_S)
        #
        # # 设置高度轴
        #
        #
        # # 找到 hor_speed_P_final 每行最后一个非 NaN 值的索引
        # last_non_nan_indices = np.array([np.max(np.where(~np.isnan(row))[0]) if np.any(~np.isnan(row)) else np.nan
        #                                  for row in hor_speed_P_final])
        # # 获取最大非 NaN 列索引，以此来确定最大有效高度
        # max_non_nan_index = int(np.nanmax(last_non_nan_indices)) + 1  # 取最大有效索引
        #
        # height = height[:max_non_nan_index]
        # # 设置时间轴
        # time_data1 = pd.to_datetime(time_data)
        # time_data_num = mdates.date2num(time_data1)
        # # 创建3x2的子图
        # fig, axes = plt.subplots(3, 2, figsize=(18, 16), sharex=True)
        #
        # # 第一行：水平风速
        # c1 = axes[0, 0].imshow(hor_speed_P_final[:, :max_non_nan_index].T, aspect='auto',
        #                        extent=[time_data_num[0], time_data_num[-16], height[-1], height[0]],
        #                        cmap='jet', vmin=0, vmax=25)
        # fig.colorbar(c1, ax=axes[0, 0], label='P Horizontal Wind Speed (m/s)')
        # axes[0, 0].set_ylabel('Height (m)')
        # axes[0, 0].invert_yaxis()
        # axes[0, 0].set_title('P Horizontal Wind Speed')
        #
        # c2 = axes[0, 1].imshow(hor_speed_S_final[:, :max_non_nan_index].T, aspect='auto',
        #                        extent=[time_data_num[0], time_data_num[-16], height[-1], height[0]],
        #                        cmap='jet', vmin=0, vmax=25)
        # fig.colorbar(c2, ax=axes[0, 1], label='S Horizontal Wind Speed (m/s)')
        # axes[0, 1].invert_yaxis()
        # axes[0, 1].set_title('S Horizontal Wind Speed')
        #
        # # 第二行：水平风向
        # c3 = axes[1, 0].imshow(wind_dir_P_final[:, :max_non_nan_index].T, aspect='auto',
        #                        extent=[time_data_num[0], time_data_num[-16], height[-1], height[0]],
        #                        cmap='hsv', vmin=0, vmax=360)
        # fig.colorbar(c3, ax=axes[1, 0], label='P Horizontal Wind Direction (°)')
        # axes[1, 0].set_ylabel('Height (m)')
        # axes[1, 0].invert_yaxis()
        # axes[1, 0].set_title('P Horizontal Wind Direction')
        #
        # c4 = axes[1, 1].imshow(wind_dir_S_final[:, :max_non_nan_index].T, aspect='auto',
        #                        extent=[time_data_num[0], time_data_num[-16], height[-1], height[0]],
        #                        cmap='hsv', vmin=0, vmax=360)
        # fig.colorbar(c4, ax=axes[1, 1], label='S Horizontal Wind Direction (°)')
        # axes[1, 1].invert_yaxis()
        # axes[1, 1].set_title('S Horizontal Wind Direction')
        #
        # # 第三行：垂直风速
        # c5 = axes[2, 0].imshow(ver_speed_P_final[:, :max_non_nan_index].T, aspect='auto',
        #                        extent=[time_data_num[0], time_data_num[-16], height[-1], height[0]],
        #                        cmap='jet', vmin=-2, vmax=2)
        # fig.colorbar(c5, ax=axes[2, 0], label='P Vertical Wind Speed (m/s)')
        # axes[2, 0].set_xlabel('Time')
        # axes[2, 0].set_ylabel('Height (m)')
        # axes[2, 0].invert_yaxis()
        # axes[2, 0].set_title('P Vertical Wind Speed')
        #
        # c6 = axes[2, 1].imshow(ver_speed_S_final[:, :max_non_nan_index].T, aspect='auto',
        #                        extent=[time_data_num[0], time_data_num[-16], height[-1], height[0]],
        #                        cmap='jet', vmin=-2, vmax=2)
        # fig.colorbar(c6, ax=axes[2, 1], label='S Vertical Wind Speed (m/s)')
        # axes[2, 1].set_xlabel('Time')
        # axes[2, 1].invert_yaxis()
        # axes[2, 1].set_title('S Vertical Wind Speed')
        #
        # # 设置时间刻度格式
        # for ax in axes[2, :]:  # 只需设置第三行的x轴时间格式
        #     ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        #     ax.set_xlim(time_data_num[0], time_data_num[-16])
        #
        # fig.autofmt_xdate()  # 自动旋转日期标签
        #
        # fig.suptitle(f"{date_str}", fontsize=16)
        #
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #
        # # save_path = f"plot_data\\{date_str}.png"
        # # plt.savefig(save_path)
        #
        # # plt.figure()
        # # plt.plot(time_data_num[:-16], wind_dir_P[:, 7] + 90, label=f'{height[7]}')
        # # plt.legend()
        # # plt.ylim(0, 360)
        #
        # # np.savez(rf'{date_str}_wind_result.npz',
        # #         hor_speed_P_final = hor_speed_P_final,
        # #         ver_speed_P_final = ver_speed_P_final,
        # #         wind_dir_P_final = wind_dir_P_final,
        # #
        # #         hor_speed_S_final = hor_speed_S_final,
        # #         ver_speed_S_final = ver_speed_S_final,
        # #         wind_dir_S_final = wind_dir_S_final,
        # #         time = time_data)
        # plt.figure()
        # plt.plot(time_data1[:-16], wind_dir_P_final[:, 7])
        # plt.xlabel('time')
        # plt.ylabel('wind direction(°)')
        # plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # plt.show()

