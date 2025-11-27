import numpy as np
from A_lidar_params import params
from D_noise_model import NoiseModel


def verify_noise_power():
    print("=== 噪声功率验证 (Power = Variance <i^2>) ===\n")

    # 1. 初始化模型
    nm = NoiseModel()

    # 2. 获取代码计算的理论值
    # 这里调用的是 D_noise_model.py 中的 calculate_gaussian_variance 方法
    code_shot, code_thermal = nm.calculate_gaussian_variance()

    # 3. 手动计算理论值 (基于 A_lidar_params.py 的参数)
    # 散粒噪声: 2 * e * R * P_LO * B
    manual_shot = 2 * params.q_electron * params.responsivity * \
                  params.local_power * params.bandwidth

    # 热噪声: 4 * k * T * B / R
    manual_thermal = 4 * params.k_boltzmann * params.temperature * \
                     params.bandwidth / params.load_resistance

    # 4. 统计验证 (蒙特卡洛模拟)
    # 生成 1,000,000 个点以确保统计收敛 (比默认的 1024 点更准)
    N_samples = 1024000

    # 模拟散粒噪声
    sim_shot_noise = np.random.normal(0, np.sqrt(code_shot), N_samples)
    stat_shot_var = np.var(sim_shot_noise)  # 计算实际方差

    # 模拟热噪声
    sim_thermal_noise = np.random.normal(0, np.sqrt(code_thermal), N_samples)
    stat_thermal_var = np.var(sim_thermal_noise)  # 计算实际方差

    # --- 输出对比结果 ---

    print(f"【参数检查】")
    print(f"  本振功率 P_LO: {params.local_power * 1000:.1f} mW")
    print(f"  温度 T: {params.temperature:.2f} K")
    print(f"  带宽 B: {params.bandwidth / 1e6:.0f} MHz")
    print("-" * 60)

    print(f"【散粒噪声验证】")
    print(f"  1. 手算公式值 : {manual_shot:.4e} A^2")
    print(f"  2. 代码计算值 : {code_shot:.4e} A^2")
    print(f"  3. 仿真统计值 : {stat_shot_var:.4e} A^2 (N={N_samples})")
    print(f"  -> 代码逻辑偏差: {(code_shot - manual_shot) / manual_shot * 100:.8f}%")
    print(f"  -> 随机统计偏差: {(stat_shot_var - manual_shot) / manual_shot * 100:.4f}%")
    print("-" * 60)

    print(f"【热噪声验证】")
    print(f"  1. 手算公式值 : {manual_thermal:.4e} A^2")
    print(f"  2. 代码计算值 : {code_thermal:.4e} A^2")
    print(f"  3. 仿真统计值 : {stat_thermal_var:.4e} A^2 (N={N_samples})")
    print(f"  -> 代码逻辑偏差: {(code_thermal - manual_thermal) / manual_thermal * 100:.8f}%")
    print(f"  -> 随机统计偏差: {(stat_thermal_var - manual_thermal) / manual_thermal * 100:.4f}%")
    print("-" * 60)

    print("【结论】")
    if abs(code_shot - manual_shot) < 1e-20 and abs(code_thermal - manual_thermal) < 1e-20:
        print("✅ 代码计算公式与理论公式完全一致。")
    else:
        print("❌ 代码计算公式存在误差，请检查参数或公式。")


if __name__ == "__main__":
    verify_noise_power()