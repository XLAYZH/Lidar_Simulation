import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- Matplotlib 设置 ---
# (确保您的系统支持这些字体，或者注释掉)
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = (10, 7)  # 设置图像大小

# --- 绘图脚本 ---
try:
    # 1. 加载数据
    nep_data = np.load('nep_fit_smooth.npy')
    print(f"成功加载 'nep_fit_smooth.npy'，数据形状: {nep_data.shape}")

    # 2. 创建 X 轴 (频率)
    # 根据 new_simulation_fryj.py 和 lidar_sim.py 中的参数
    sample_rate = 1e9  # 1 GHz
    FFT_points = 1024  # FFT 点数

    # 计算频率轴（取前512个点）
    freqs_hz = np.fft.fftfreq(FFT_points, 1 / sample_rate)[:512]

    # 转换为 MHz
    freqs_mhz = freqs_hz / 1e6

    # 3. 绘图
    plt.figure()
    plt.plot(freqs_mhz, nep_data, color='tab:red', linewidth=2)

    # 4. 设置标签和标题 (参考 论文 图2.10)
    plt.title("NEP (Noise Equivalent Power) Spectrum (nep_fit_smooth.npy)", fontsize=16)
    plt.xlabel("Frequency (MHz)", fontsize=14)
    # Y轴标签使用LaTeX格式，参考论文
    plt.ylabel(r"NEP (pW/Hz$^{1/2}$)", fontsize=14)

    # 设置坐标轴范围 (参考 论文 图2.10)
    plt.xlim(0, 500)
    plt.ylim(bottom=4)  # Y轴从4开始

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 5. 保存和显示图像
    plt.savefig("nep_spectrum_plot.png")
    plt.show()
    print("图像已保存为 'nep_spectrum_plot.png'")

except FileNotFoundError:
    print("错误: 'nep_fit_smooth.npy' 文件未找到。")
except Exception as e:
    print(f"绘图时发生错误: {e}")