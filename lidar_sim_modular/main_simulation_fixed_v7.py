from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from A_lidar_params import params
from C_lidar_physics import LidarPhysics

from D_noise_model_fixed import NoiseModel


from E_wind_field import WindField
from PlotStyle import set_axis


class LidarSimulator:
    """
    主仿真程序（v6）

    相较于 v5 的主要改动：
    1. 适配新的 E_wind_field.py，仅保留 constant / profile 两种主模式；
    2. 支持直接传入 profile_path，使用标准化 npz 风廓线驱动仿真；
    3. 去除旧版 load_sounding_data() 依赖；
    4. 保留 v5 的 PSD 估计、噪声叠加和绘图接口，方便连续使用。
    """

    def __init__(self, profile_path: str | Path | None = None):
        self.p = params
        self.physics = LidarPhysics()
        self.noise_model = NoiseModel()
        self.wind_field = WindField()

        self.profile_path: Optional[Path] = None
        if profile_path is not None:
            self.set_profile(profile_path)

        if hasattr(self.noise_model, "print_model_configuration"):
            self.noise_model.print_model_configuration()

    def set_profile(self, profile_path: str | Path) -> None:
        """加载并缓存标准化风廓线 npz。"""
        path = Path(profile_path)
        ok = self.wind_field.load_profile_npz(path)
        if not ok:
            raise FileNotFoundError(f"无法加载风廓线文件: {path}")
        self.profile_path = path

    @staticmethod
    def _one_sided_psd(
        x: np.ndarray,
        fs: float,
        n_fft: int,
        remove_dc: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        基于 Hann 窗的一侧功率谱密度估计。

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
    def _select_freq_band(
        freqs: np.ndarray,
        data: np.ndarray,
        fmin_mhz: float | None = None,
        fmax_mhz: float | None = None,
    ):
        mask = np.ones_like(freqs, dtype=bool)
        if fmin_mhz is not None:
            mask &= freqs >= fmin_mhz
        if fmax_mhz is not None:
            mask &= freqs <= fmax_mhz
        return freqs[mask], data[..., mask], mask

    def _resolve_profile_path(self, wind_mode: str, profile_path: str | Path | None) -> Path | None:
        if wind_mode != "profile":
            return None

        if profile_path is not None:
            return Path(profile_path)

        if self.profile_path is not None:
            return self.profile_path

        raise ValueError(
            "当 wind_mode='profile' 时，必须提供 profile_path，"
            "或在初始化 LidarSimulator(profile_path=...) 时预先加载。"
        )

    def _apply_plot_style(
        self,
        ax,
        xminor=False,
        yminor=False,
        xmajor=None,
        ymajor=None,
    ) -> None:
        """统一调用 PlotStyle.py 设置 2D 图风格。"""
        kwargs = dict(
            axis_lw=1.5,
            major_width=1.5,
            minor_width=1.2,
            major_length=6,
            minor_length=3,
            ticklabel_fontsize=12,
            label_fontsize=14,
            title_fontsize=15,
            font_name="Times New Roman",
        )
        if xmajor is not None:
            kwargs["xmajor"] = xmajor
        if ymajor is not None:
            kwargs["ymajor"] = ymajor
        set_axis(ax, xminor=xminor, yminor=yminor, **kwargs)

    @staticmethod
    def _style_colorbar(cbar) -> None:
        for label in cbar.ax.get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(12)
        cbar.ax.yaxis.label.set_fontname("Times New Roman")
        cbar.ax.yaxis.label.set_fontsize(14)

    def simulate_single_radial(
        self,
        azimuth: float = 0.0,
        wind_mode: str = "constant",
        profile_path: str | Path | None = None,
        u_const: float = 0.0,
        v_const: float = 10.0,
        w_const: float = 0.0,
        n_accum: int = 50,
        signal_gain: float = 1.0,
        remove_dc: bool = True,
    ) -> dict:
        """
        仿真单个方位角的功率谱数据。

        参数
        ----
        azimuth     : 方位角 (deg)
        wind_mode   : 风场模式，仅支持 'constant' 或 'profile'
        profile_path: 当 wind_mode='profile' 时，指定标准化 npz 风廓线路径
        u_const,
        v_const,
        w_const     : constant 模式下的常风速分量
        n_accum     : 累积脉冲数
        signal_gain : 对整段物理信号统一乘的比例因子，仅用于整体可视化调节
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

        active_profile = self._resolve_profile_path(wind_mode, profile_path)
        v_los, heights = self.wind_field.get_radial_velocity(
            self.p.range_axis,
            azimuth,
            self.p.elevation_angle_deg,
            wind_type=wind_mode,
            profile_path=active_profile,
            u_const=u_const,
            v_const=v_const,
            w_const=w_const,
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
        if wind_mode == "profile":
            print(f"    使用风廓线: {active_profile}")
        else:
            print(f"    常风速: u={u_const:.3f}, v={v_const:.3f}, w={w_const:.3f} m/s")

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
            "height_axis": np.asarray(heights, dtype=float),
            "vlos_true": np.asarray(v_los, dtype=float),
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
                "profile_path": str(active_profile) if active_profile is not None else None,
                "u_const": float(u_const),
                "v_const": float(v_const),
                "w_const": float(w_const),
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
            self._apply_plot_style(ax, xminor=True, yminor=True)

        axes[-1].set_xlabel("Frequency (MHz)")
        axes[-1].set_xlim(max(xlim[0], fmin_mhz or xlim[0]), xlim[1])
        fig.suptitle("Spectral slices at selected range gates", fontsize=14, fontname="Times New Roman")
        plt.tight_layout()
        plt.show()

    def plot_2d_heatmap(self, res: dict, xlim=(0, 500), dyn_range_db=35.0, fmin_mhz: float | None = None):
        data = res["data"]
        freqs = res["freq_axis"]
        ranges = res["range_axis"]

        freqs_sel, data_sel, _ = self._select_freq_band(freqs, data, fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])
        data_db = self._db(data_sel)
        vmax = np.nanmax(data_db)
        vmin = vmax - dyn_range_db

        fig, ax = plt.subplots(figsize=(11, 6))
        pcm = ax.pcolormesh(freqs_sel, ranges, data_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(pcm, ax=ax, label="PSD (dB, A$^2$/Hz)")
        cbar.ax.minorticks_off()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Range gate center (m)")
        if fmin_mhz is None:
            ax.set_title("Range-Frequency power spectrum")
        else:
            ax.set_title(f"Range-Frequency power spectrum (>{fmin_mhz:.1f} MHz)")
        ax.set_xlim(freqs_sel[0], freqs_sel[-1])
        ax.set_ylim(ranges[0], ranges[-1])
        ax.grid(True, linestyle="--", alpha=0.2)
        self._apply_plot_style(ax, xminor=True, yminor=True)
        self._style_colorbar(cbar)
        plt.tight_layout()
        plt.show()

    def plot_noise_heatmap(self, res: dict, xlim=(0, 500), dyn_range_db=20.0, fmin_mhz: float | None = None):
        noise_data = res["noise_data"]
        freqs = res["freq_axis"]
        ranges = res["range_axis"]

        freqs_sel, noise_sel, _ = self._select_freq_band(freqs, noise_data, fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])
        noise_db = self._db(noise_sel)
        vmax = np.nanmax(noise_db)
        vmin = vmax - dyn_range_db

        fig, ax = plt.subplots(figsize=(11, 6))
        pcm = ax.pcolormesh(freqs_sel, ranges, noise_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(pcm, ax=ax, label="Noise PSD (dB, A$^2$/Hz)")
        cbar.ax.minorticks_off()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Range gate center (m)")
        if fmin_mhz is None:
            ax.set_title("Range-Frequency noise floor")
        else:
            ax.set_title(f"Range-Frequency noise floor (>{fmin_mhz:.1f} MHz)")
        ax.set_xlim(freqs_sel[0], freqs_sel[-1])
        ax.set_ylim(ranges[0], ranges[-1])
        ax.grid(True, linestyle="--", alpha=0.2)
        self._apply_plot_style(ax, xminor=True, yminor=True)
        self._style_colorbar(cbar)
        plt.tight_layout()
        plt.show()

    def plot_mean_noise_spectrum(self, res: dict, xlim=(0, 500), fmin_mhz: float | None = None, show_theory: bool = True):
        noise_data = res["noise_data"]
        freqs = res["freq_axis"]

        freqs_sel, noise_sel, _ = self._select_freq_band(freqs, noise_data, fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])
        mean_noise = np.mean(noise_sel, axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(freqs_sel, self._db(mean_noise), lw=1.8, color='tab:blue', label='Empirical mean noise PSD')

        if show_theory and hasattr(self.noise_model, 'component_psds'):
            comp = self.noise_model.component_psds(n_samples=res['meta']['n_fft'])
            comp_freqs = comp['freqs'] / 1e6
            comp_freqs_sel, shot_sel, mask = self._select_freq_band(comp_freqs, comp['shot'][np.newaxis, :], fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])
            thermal_sel = comp['thermal'][mask]
            rin_sel = comp['rin'][mask]
            bdn_sel = comp['bdn'][mask]
            total_sel = comp['total'][mask]

            ax.plot(comp_freqs_sel, self._db(total_sel), '--', lw=1.6, color='k', label='Theory total noise PSD')
            ax.plot(comp_freqs_sel, self._db(shot_sel[0]), ':', lw=1.2, color='tab:orange', label='Shot')
            ax.plot(comp_freqs_sel, self._db(thermal_sel), ':', lw=1.2, color='tab:green', label='Thermal')
            ax.plot(comp_freqs_sel, self._db(rin_sel), ':', lw=1.2, color='tab:red', label='RIN')
            if np.any(bdn_sel > 0):
                ax.plot(comp_freqs_sel, self._db(bdn_sel), ':', lw=1.2, color='tab:purple', label='BDN')

        ax.set_xlim(max(xlim[0], fmin_mhz or xlim[0]), xlim[1])
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('PSD (dB, A$^2$/Hz)')
        ax.set_title('Range-averaged noise spectrum')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        self._apply_plot_style(ax, xminor=5, yminor=True, xmajor=20)
        plt.tight_layout()
        plt.show()

    def plot_excess_over_noise(self, res: dict, gates: list[int] | None = None, xlim=(0, 500), fmin_mhz: float | None = None):
        excess_db = res["excess_db"]
        freqs = res["freq_axis"]
        ranges = res["range_axis"]

        if gates is None:
            n_gate = len(ranges)
            gates = [1, n_gate // 2, n_gate - 2]

        freqs_sel, _, mask = self._select_freq_band(freqs, excess_db, fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])

        fig, ax = plt.subplots(figsize=(10, 7))
        for g in gates:
            ax.plot(freqs_sel, excess_db[g, mask], lw=1.2, label=f"Gate {g} ({ranges[g]:.0f} m)")
        ax.set_xlim(max(xlim[0], fmin_mhz or xlim[0]), xlim[1])
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Excess over noise (dB)")
        ax.set_title("Signal excess relative to noise floor")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        self._apply_plot_style(ax, xminor=5, yminor=True, xmajor=20)
        plt.tight_layout()
        plt.show()

    def plot_excess_heatmap(self, res: dict, xlim=(0, 500), fmin_mhz: float | None = None, vmax_db: float = 20.0, vmin_db: float = -2.0):
        excess_db = res["excess_db"]
        freqs = res["freq_axis"]
        ranges = res["range_axis"]

        freqs_sel, excess_sel, _ = self._select_freq_band(freqs, excess_db, fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])

        fig, ax = plt.subplots(figsize=(11, 6))
        pcm = ax.pcolormesh(freqs_sel, ranges, excess_sel, shading="auto", cmap="jet", vmin=vmin_db, vmax=vmax_db)
        cbar = fig.colorbar(pcm, ax=ax, label="Excess over noise (dB)")
        cbar.ax.minorticks_off()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Range gate center (m)")
        if fmin_mhz is None:
            ax.set_title("Signal excess over noise floor")
        else:
            ax.set_title(f"Signal excess over noise floor (>{fmin_mhz:.1f} MHz)")
        ax.set_xlim(freqs_sel[0], freqs_sel[-1])
        ax.set_ylim(ranges[0], ranges[-1])
        ax.grid(True, linestyle="--", alpha=0.2)
        self._apply_plot_style(ax, xminor=True, yminor=True)
        self._style_colorbar(cbar)
        plt.tight_layout()
        plt.show()

    def plot_3d_psd(self, res: dict, xlim=(0, 500), fmin_mhz: float | None = None):
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
        ax.set_xlabel("Frequency (MHz)", fontname="Times New Roman", fontsize=13)
        ax.set_ylabel("Range gate center (m)", fontname="Times New Roman", fontsize=13)
        ax.set_zlabel("PSD (dB, A$^2$/Hz)", fontname="Times New Roman", fontsize=13)
        if fmin_mhz is None:
            ax.set_title("3D power spectral density", fontname="Times New Roman", fontsize=15)
        else:
            ax.set_title(f"3D power spectral density (>{fmin_mhz:.1f} MHz)", fontname="Times New Roman", fontsize=15)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.60, aspect=15, label="PSD (dB, A$^2$/Hz)")
        self._style_colorbar(cbar)
        for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(11)
        ax.view_init(elev=28, azim=-62)
        plt.tight_layout()
        plt.show()

    def plot_first_pulse_time_series(self, res: dict, gate_idx: int = 1):
        fp = res["first_pulse"]
        gate_len = res["meta"]["gate_len"]
        time_ns = np.arange(gate_len) / self.p.sample_rate * 1e9

        sig = fp["signal_time"][gate_idx] * 1e6
        noise = fp["noise_time"][gate_idx] * 1e6
        total = fp["total_time"][gate_idx] * 1e6

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_ns, noise, label="Noise", alpha=0.8)
        ax.plot(time_ns, sig, label="Signal", alpha=0.8)
        ax.plot(time_ns, total, label="Signal + Noise", lw=1.2)
        ax.set_xlabel("Time within gate (ns)")
        ax.set_ylabel("Current (µA)")
        ax.set_title(f"First pulse time series at gate {gate_idx}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        self._apply_plot_style(ax, xminor=True, yminor=True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # --------------------------
    # 示例 1：常风速
    # --------------------------
    # sim = LidarSimulator()
    # res = sim.simulate_single_radial(
    #     azimuth=0.0,
    #     wind_mode="constant",
    #     u_const=0.0,
    #     v_const=10.0,
    #     w_const=0.0,
    #     n_accum=20,
    #     signal_gain=1.0,
    #     remove_dc=True,
    # )
    #
    # sim.plot_2d_heatmap(res, xlim=(0, 500), dyn_range_db=35.0, fmin_mhz=None)
    # sim.plot_excess_heatmap(res, xlim=(10, 500), fmin_mhz=10.0, vmax_db=20.0, vmin_db=-2.0)

    # --------------------------
    # 示例 2：探空风廓线（取消注释后使用）
    # --------------------------
    profile = r"E:\GraduateStu6428\Codes\Python\sonde_profiles_npz\2025-12-01_12.npz"
    sim_profile = LidarSimulator(profile_path=profile)
    res_profile = sim_profile.simulate_single_radial(
        azimuth=0.0,
        wind_mode="profile",
        n_accum=20,
        signal_gain=1.0,
        remove_dc=True,
    )
    sim_profile.plot_2d_heatmap(res_profile, xlim=(0, 500), dyn_range_db=35.0, fmin_mhz=None)
