import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from A_lidar_params import params
from C_lidar_physics import LidarPhysics
from D_noise_model_fixed import NoiseModel
from E_wind_field import WindField
import S_plot_style as plot_style


class LidarSimulator:
    """
    主仿真程序（可直接运行版 v5）

    设计原则：
    1. 保留物理信号自身随距离门衰减的幅度，不再人为指定每个距离门的 SNR。
    2. 噪声作为平稳随机过程叠加到整段时域记录中，使噪声基底在各距离门间总体保持一致。
    3. 使用带窗的一侧 PSD 估计，并做正确的 A^2/Hz 归一化。
    4. 增加“去低频热图”和“超噪声图”，避免低频脊掩盖有色噪声与信号细节。
    """

    def __init__(self):
        self.p = params
        self.physics = LidarPhysics()
        self.noise_model = NoiseModel()
        self.wind_field = WindField()

        # 可选：尝试加载真实探空，失败时自动退回理论风场
        self.sounding_path = r"E:\GraduateStu6428\Codes\ObservationData54511\12Z\2025-12-01_12.csv"
        try:
            self.wind_field.load_sounding_data(self.sounding_path)
        except Exception:
            print("注意: 未找到探空数据，将使用理论风场。")

        # 打印噪声模型配置，便于确认白噪声是否被接收机响应整形
        if hasattr(self.noise_model, "print_model_configuration"):
            self.noise_model.print_model_configuration()

    @staticmethod
    def _one_sided_psd(x: np.ndarray, fs: float, n_fft: int, remove_dc: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        基于 Hann 窗的一侧功率谱密度估计。

        参数
        ----
        x : 时域信号
        fs : 采样率
        n_fft : FFT 点数，可大于 len(x)
        remove_dc : 是否先去均值

        返回
        ----
        freqs_hz : 一侧频率轴 (Hz)
        psd      : 一侧 PSD (A^2/Hz)
        """
        x = np.asarray(x, dtype=float)
        if remove_dc:
            x = x - np.mean(x)

        win = np.hanning(len(x))
        u = np.sum(win ** 2)
        xw = x * win

        spec = np.fft.rfft(xw, n=n_fft)
        psd = (np.abs(spec) ** 2) / (fs * u)

        if n_fft % 2 == 0:
            if len(psd) > 2:
                psd[1:-1] *= 2.0
        else:
            if len(psd) > 1:
                psd[1:] *= 2.0

        freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        return freqs_hz, psd

    @staticmethod
    def _db(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), floor))

    @staticmethod
    def _select_freq_band(freqs: np.ndarray, data: np.ndarray, fmin_mhz: float | None = None, fmax_mhz: float | None = None):
        mask = np.ones_like(freqs, dtype=bool)
        if fmin_mhz is not None:
            mask &= freqs >= fmin_mhz
        if fmax_mhz is not None:
            mask &= freqs <= fmax_mhz
        return freqs[mask], data[..., mask], mask

    def simulate_single_radial(
        self,
        azimuth: float = 0.0,
        wind_mode: str = "constant",
        n_accum: int = 50,
        signal_gain: float = 1.0,
        remove_dc: bool = True,
    ) -> dict:
        """
        仿真单个方位角的功率谱数据。

        参数
        ----
        azimuth     : 方位角 (deg)
        wind_mode   : 风场模式，传递给 WindField
        n_accum     : 累积脉冲数
        signal_gain : 对整段物理信号统一乘的比例因子，仅用于整体可视化调节，
                      不改变不同距离门之间的相对衰减关系
        remove_dc   : FFT 前是否去均值
        """
        gate_len = self.p.points_per_bin
        n_fft = self.p.fft_points
        fs = self.p.sample_rate

        total_points = len(self.p.time_axis)
        num_gates = total_points // gate_len
        valid_points = num_gates * gate_len
        n_freq = n_fft // 2 + 1

        gate_range_res = self.p.c * gate_len / (2.0 * fs)
        range_axis = (np.arange(num_gates) + 0.5) * gate_range_res

        v_los, _ = self.wind_field.get_radial_velocity(
            self.p.range_axis,
            azimuth,
            self.p.elevation_angle_deg,
            wind_type=wind_mode,
        )

        psd_accum = np.zeros((num_gates, n_freq), dtype=float)
        noise_psd_accum = np.zeros((num_gates, n_freq), dtype=float)

        first_pulse = {
            "signal_time": [],
            "noise_time": [],
            "total_time": [],
            "signal_psd": [],
            "noise_psd": [],
            "total_psd": [],
        }

        print(
            f">>> 单径向仿真开始 | azimuth={azimuth:.1f} deg | wind_mode={wind_mode} | "
            f"n_accum={n_accum} | gates={num_gates}"
        )

        for p_idx in tqdm(range(n_accum), desc="Accumulating pulses"):
            sig_full = signal_gain * self.physics.simulate_ideal_signal(v_los_profile=v_los)
            sig_full = sig_full[:valid_points]

            _, _, _, noise_full = self.noise_model.generate_total_noise(n_samples=valid_points)
            total_full = sig_full + noise_full

            for g in range(num_gates):
                s = g * gate_len
                e = s + gate_len

                sig_gate = sig_full[s:e]
                noise_gate = noise_full[s:e]
                total_gate = total_full[s:e]

                freq_hz, psd_total = self._one_sided_psd(total_gate, fs, n_fft, remove_dc=remove_dc)
                _, psd_noise = self._one_sided_psd(noise_gate, fs, n_fft, remove_dc=remove_dc)

                psd_accum[g] += psd_total
                noise_psd_accum[g] += psd_noise

                if p_idx == 0:
                    _, psd_sig = self._one_sided_psd(sig_gate, fs, n_fft, remove_dc=remove_dc)
                    first_pulse["signal_time"].append(sig_gate.copy())
                    first_pulse["noise_time"].append(noise_gate.copy())
                    first_pulse["total_time"].append(total_gate.copy())
                    first_pulse["signal_psd"].append(psd_sig.copy())
                    first_pulse["noise_psd"].append(psd_noise.copy())
                    first_pulse["total_psd"].append(psd_total.copy())

        avg_psd = psd_accum / float(n_accum)
        avg_noise_psd = noise_psd_accum / float(n_accum)
        avg_signal_excess_db = self._db(avg_psd / np.maximum(avg_noise_psd, 1e-30))
        freq_axis_mhz = freq_hz / 1e6

        return {
            "data": avg_psd,
            "noise_data": avg_noise_psd,
            "excess_db": avg_signal_excess_db,
            "range_axis": range_axis,
            "freq_axis": freq_axis_mhz,
            "first_pulse": first_pulse,
            "meta": {
                "gate_len": gate_len,
                "n_fft": n_fft,
                "fs": fs,
                "n_accum": n_accum,
                "signal_gain": signal_gain,
                "azimuth": azimuth,
                "wind_mode": wind_mode,
            },
        }

    def plot_spectral_slices(self, res: dict, gates: list[int] | None = None, xlim=(0, 500), fmin_mhz: float | None = None):
        """绘制若干距离门的谱线切片，显示信号峰衰减与噪声底稳定性。"""
        data = res["data"]
        noise_data = res["noise_data"]
        freqs = res["freq_axis"]
        ranges = res["range_axis"]

        if gates is None:
            n_gate = len(ranges)
            gates = [1, n_gate // 2, n_gate - 2]

        freqs_sel, _, mask = self._select_freq_band(freqs, data, fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])

        fig, axes = plt.subplots(len(gates), 1, figsize=(10, 9), sharex=True)
        if len(gates) == 1:
            axes = [axes]

        for ax, g in zip(axes, gates):
            psd_db = self._db(data[g, mask])
            noise_db = self._db(noise_data[g, mask])

            ax.plot(freqs_sel, noise_db, color="gray", lw=1.0, alpha=0.9, label="Noise floor")
            ax.plot(freqs_sel, psd_db, color="tab:blue", lw=1.2, label="Signal + noise")
            ax.set_ylabel("PSD (dB, A$^2$/Hz)")
            ax.set_title(f"Gate {g} | Center range = {ranges[g]:.1f} m")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Frequency (MHz)")
        axes[-1].set_xlim(max(xlim[0], fmin_mhz or xlim[0]), xlim[1])
        plt.suptitle("Spectral slices at selected range gates", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_2d_heatmap(
            self,
            res: dict,
            xlim=(0, 500),
            dyn_range_db=35.0,
            fmin_mhz: float | None = None,
            mode: str = "db",  # "db" 或 "normalized"
            norm_mode: str = "per_gate",  # "per_gate" 或 "global"
            eps: float = 1e-30,
    ):
        """
        二维功率谱热图。

        mode="db":
            画绝对 PSD 的 dB 图，单位 dB(A^2/Hz)

        mode="normalized":
            画归一化功率谱密度图，范围 0~1

        norm_mode="per_gate":
            每个距离门分别按本门最大值归一化，更接近论文中的“距离门归一化”效果

        norm_mode="global":
            全局按一个最大值归一化，保留不同距离门之间的相对强弱
        """
        data = res["data"]
        freqs = res["freq_axis"]
        ranges = res["range_axis"]

        freqs_sel, data_sel, _ = self._select_freq_band(
            freqs, data, fmin_mhz=fmin_mhz, fmax_mhz=xlim[1]
        )

        plt.figure(figsize=(11, 6))

        if mode == "db":
            data_db = self._db(data_sel)
            vmax = np.nanmax(data_db)
            vmin = vmax - dyn_range_db

            plt.pcolormesh(
                freqs_sel, ranges, data_db,
                shading="auto", cmap="jet",
                vmin=vmin, vmax=vmax
            )
            cbar = plt.colorbar(label="PSD (dB, A$^2$/Hz)")
            title = "Range-Frequency power spectrum"

        elif mode == "normalized":
            if norm_mode == "per_gate":
                denom = np.max(data_sel, axis=1, keepdims=True)
            elif norm_mode == "global":
                denom = np.max(data_sel)
            else:
                raise ValueError("norm_mode must be 'per_gate' or 'global'")

            data_norm = data_sel / np.maximum(denom, eps)
            data_norm = np.clip(data_norm, 0.0, 1.0)

            plt.pcolormesh(
                freqs_sel, ranges, data_norm,
                shading="auto", cmap="jet",
                vmin=0.0, vmax=1.0
            )
            cbar = plt.colorbar(label="Normalized PSD")
            title = "Normalized power spectral density"

        else:
            raise ValueError("mode must be 'db' or 'normalized'")

        cbar.ax.minorticks_off()
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Range gate center (m)")

        if fmin_mhz is None:
            plt.title(title)
        else:
            plt.title(f"{title} (>{fmin_mhz:.1f} MHz)")

        plt.xlim(freqs_sel[0], freqs_sel[-1])
        plt.ylim(ranges[0], ranges[-1])
        plt.grid(True, linestyle="--", alpha=0.2)
        plt.tight_layout()
        plt.show()

    def plot_noise_heatmap(self, res: dict, xlim=(0, 500), dyn_range_db=20.0, fmin_mhz: float | None = None):
        """仅绘制噪声基底热图，便于检查有色噪声是否随频率变化。"""
        noise_data = res["noise_data"]
        freqs = res["freq_axis"]
        ranges = res["range_axis"]

        freqs_sel, noise_sel, _ = self._select_freq_band(freqs, noise_data, fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])
        noise_db = self._db(noise_sel)
        vmax = np.nanmax(noise_db)
        vmin = vmax - dyn_range_db

        plt.figure(figsize=(11, 6))
        plt.pcolormesh(freqs_sel, ranges, noise_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(label="Noise PSD (dB, A$^2$/Hz)")
        cbar.ax.minorticks_off()
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Range gate center (m)")
        if fmin_mhz is None:
            plt.title("Range-Frequency noise floor")
        else:
            plt.title(f"Range-Frequency noise floor (>{fmin_mhz:.1f} MHz)")
        plt.xlim(freqs_sel[0], freqs_sel[-1])
        plt.ylim(ranges[0], ranges[-1])
        plt.grid(True, linestyle="--", alpha=0.2)
        plt.tight_layout()
        plt.show()

    def plot_mean_noise_spectrum(self, res: dict, xlim=(0, 500), fmin_mhz: float | None = None, show_theory: bool = True):
        """
        绘制跨距离门平均后的噪声谱。
        该图比二维热图更容易看出随频率变化的确定性包络。
        """
        noise_data = res["noise_data"]
        freqs = res["freq_axis"]

        freqs_sel, noise_sel, _ = self._select_freq_band(freqs, noise_data, fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])
        mean_noise = np.mean(noise_sel, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(freqs_sel, self._db(mean_noise), lw=1.8, color='tab:blue', label='Empirical mean noise PSD')

        if show_theory:
            comp = self.noise_model.component_psds(n_samples=res['meta']['n_fft'])
            comp_freqs = comp['freqs'] / 1e6
            comp_freqs_sel, shot_sel, mask = self._select_freq_band(comp_freqs, comp['shot'][np.newaxis, :], fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])
            thermal_sel = comp['thermal'][mask]
            rin_sel = comp['rin'][mask]
            bdn_sel = comp['bdn'][mask]
            total_sel = comp['total'][mask]

            plt.plot(comp_freqs_sel, self._db(total_sel), '--', lw=1.6, color='k', label='Theory total noise PSD')
            plt.plot(comp_freqs_sel, self._db(shot_sel[0]), ':', lw=1.2, color='tab:orange', label='Shot')
            plt.plot(comp_freqs_sel, self._db(thermal_sel), ':', lw=1.2, color='tab:green', label='Thermal')
            plt.plot(comp_freqs_sel, self._db(rin_sel), ':', lw=1.2, color='tab:red', label='RIN')
            if np.any(bdn_sel > 0):
                plt.plot(comp_freqs_sel, self._db(bdn_sel), ':', lw=1.2, color='tab:purple', label='BDN')
            else:
                print('注意: 当前运行环境未加载到 nep_fit_smooth.npy，BDN 分量为 0。')

        plt.xlim(max(xlim[0], fmin_mhz or xlim[0]), xlim[1])
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('PSD (dB, A$^2$/Hz)')
        plt.title('Range-averaged noise spectrum')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_excess_over_noise(self, res: dict, gates: list[int] | None = None, xlim=(0, 500), fmin_mhz: float | None = None):
        """绘制总谱相对于噪声底的超噪声比，突出有用信号峰。"""
        excess_db = res["excess_db"]
        freqs = res["freq_axis"]
        ranges = res["range_axis"]

        if gates is None:
            n_gate = len(ranges)
            gates = [1, n_gate // 2, n_gate - 2]

        freqs_sel, _, mask = self._select_freq_band(freqs, excess_db, fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])

        plt.figure(figsize=(10, 7))
        for g in gates:
            plt.plot(freqs_sel, excess_db[g, mask], lw=1.2, label=f"Gate {g} ({ranges[g]:.0f} m)")
        plt.xlim(max(xlim[0], fmin_mhz or xlim[0]), xlim[1])
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Excess over noise (dB)")
        plt.title("Signal excess relative to noise floor")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_excess_heatmap(self, res: dict, xlim=(0, 500), fmin_mhz: float | None = None, vmax_db: float = 20.0, vmin_db: float = -2.0):
        """绘制总谱相对于噪声底的二维热图，更直观地显示信号峰随距离衰减。"""
        excess_db = res["excess_db"]
        freqs = res["freq_axis"]
        ranges = res["range_axis"]

        freqs_sel, excess_sel, _ = self._select_freq_band(freqs, excess_db, fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])

        plt.figure(figsize=(11, 6))
        plt.pcolormesh(freqs_sel, ranges, excess_sel, shading="auto", cmap="jet", vmin=vmin_db, vmax=vmax_db)
        cbar = plt.colorbar(label="Excess over noise (dB)")
        cbar.ax.minorticks_off()
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Range gate center (m)")
        if fmin_mhz is None:
            plt.title("Signal excess over noise floor")
        else:
            plt.title(f"Signal excess over noise floor (>{fmin_mhz:.1f} MHz)")
        plt.xlim(freqs_sel[0], freqs_sel[-1])
        plt.ylim(ranges[0], ranges[-1])
        plt.grid(True, linestyle="--", alpha=0.2)
        plt.tight_layout()
        plt.show()

    def plot_3d_psd(self, res: dict, xlim=(0, 500), fmin_mhz: float | None = None):
        """三维 PSD 图，便于观察峰值与噪声底的空间分布。"""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        data = res["data"]
        freqs = res["freq_axis"]
        ranges = res["range_axis"]

        freqs_sel, data_sel, _ = self._select_freq_band(freqs, data, fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])
        X, Y = np.meshgrid(freqs_sel, ranges)
        Z = self._db(data_sel)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="jet", linewidth=0, antialiased=False)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Range gate center (m)")
        ax.set_zlabel("PSD (dB, A$^2$/Hz)")
        if fmin_mhz is None:
            ax.set_title("3D power spectral density")
        else:
            ax.set_title(f"3D power spectral density (>{fmin_mhz:.1f} MHz)")
        fig.colorbar(surf, ax=ax, shrink=0.60, aspect=15, label="PSD (dB, A$^2$/Hz)")
        ax.view_init(elev=28, azim=-62)
        plt.tight_layout()
        plt.show()

    def plot_first_pulse_time_series(self, res: dict, gate_idx: int = 1):
        """查看某一距离门第一脉冲的时域波形。"""
        fp = res["first_pulse"]
        gate_len = res["meta"]["gate_len"]
        time_ns = np.arange(gate_len) / self.p.sample_rate * 1e9

        sig = fp["signal_time"][gate_idx] * 1e6
        noise = fp["noise_time"][gate_idx] * 1e6
        total = fp["total_time"][gate_idx] * 1e6

        plt.figure(figsize=(10, 5))
        plt.plot(time_ns, noise, label="Noise", alpha=0.8)
        plt.plot(time_ns, sig, label="Signal", alpha=0.8)
        plt.plot(time_ns, total, label="Signal + Noise", lw=1.2)
        plt.xlabel("Time within gate (ns)")
        plt.ylabel("Current (µA)")
        plt.title(f"First pulse time series at gate {gate_idx}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sim = LidarSimulator()

    # 说明：
    # 1. fmin_mhz 用于去除低频脊，便于观察 50~200 MHz 内的有色噪声与信号细节；
    # 2. 若仅想查看总谱，可调用 plot_2d_heatmap / plot_3d_psd；
    # 3. 若想突出噪声形状或信号相对噪声的超额，建议同时查看 plot_noise_heatmap 与 plot_excess_heatmap。
    res = sim.simulate_single_radial(
        azimuth=0.0,
        wind_mode="constant",
        n_accum=50,
        signal_gain=1.0,
        remove_dc=True,
    )

    # 全频段图（0~500 MHz）
    sim.plot_spectral_slices(res, xlim=(0, 500), fmin_mhz=None)
    sim.plot_2d_heatmap(
        res,
        xlim=(0, 500),
        mode="normalized",
        norm_mode="per_gate"
    )
    sim.plot_3d_psd(res, xlim=(0, 500), fmin_mhz=None)

    # 去低频图（>10 MHz, 仍显示至 500 MHz）
    sim.plot_spectral_slices(res, xlim=(10, 500), fmin_mhz=10.0)
    sim.plot_noise_heatmap(res, xlim=(10, 500), dyn_range_db=50.0, fmin_mhz=10.0)
    sim.plot_mean_noise_spectrum(res, xlim=(10, 500), fmin_mhz=10.0, show_theory=True)
    sim.plot_2d_heatmap(
        res,
        xlim=(10, 500),
        mode="normalized",
        norm_mode="global"
    )#归一化功率谱密度。如果保留不同距离门之间的相对强弱，norm_mode选择global；如果每个门都强制为1，选择per_gate
    sim.plot_2d_heatmap(
        res,
        xlim=(0, 500),
        mode="db",
        dyn_range_db=50.0
    )#绝对PSD
    sim.plot_3d_psd(res, xlim=(10, 500), fmin_mhz=10.0)
    sim.plot_excess_over_noise(res, xlim=(10, 500), fmin_mhz=10.0)
    sim.plot_excess_heatmap(res, xlim=(10, 500), fmin_mhz=10.0, vmax_db=20.0, vmin_db=-2.0)

    # 时域检查
    sim.plot_first_pulse_time_series(res, gate_idx=1)
