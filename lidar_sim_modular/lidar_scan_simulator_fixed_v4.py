from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from main_simulation_fixed_v7 import LidarSimulator


def _scan_worker(task: dict) -> dict:
    """顶层 worker，供 Windows 下多进程安全调用。"""
    sim = LidarSimulator(profile_path=task.get("profile_path"))
    res = sim.simulate_single_radial(
        azimuth=float(task["azimuth"]),
        wind_mode=str(task["wind_mode"]),
        profile_path=task.get("profile_path"),
        u_const=float(task.get("u_const", 0.0)),
        v_const=float(task.get("v_const", 10.0)),
        w_const=float(task.get("w_const", 0.0)),
        n_accum=int(task["n_accum"]),
        signal_gain=float(task.get("signal_gain", 1.0)),
        remove_dc=bool(task.get("remove_dc", True)),
    )
    return {
        "index": int(task["index"]),
        "azimuth": float(task["azimuth"]),
        "result": res,
    }


class LidarScanSimulator(LidarSimulator):
    """在单径向仿真内核基础上，增加整圈圆周扫描、断点续跑、数据保存与快速质检功能。"""

    def _get_aom_frequency_hz(self) -> float:
        for name in ("aom_frequency", "freq_aom"):
            if hasattr(self.p, name):
                return float(getattr(self.p, name))
        raise AttributeError("LidarParams 中未找到 AOM 频率参数；请检查 aom_frequency 或 freq_aom。")

    def _get_elevation_deg(self) -> float:
        for name in ("elevation_angle_deg", "elevation_deg"):
            if hasattr(self.p, name):
                return float(getattr(self.p, name))
        raise AttributeError("LidarParams 中未找到仰角参数；请检查 elevation_angle_deg 或 elevation_deg。")

    def _preflight_check(self) -> None:
        required = ("sample_rate", "fft_points", "points_per_bin", "wavelength")
        missing = [k for k in required if not hasattr(self.p, k)]
        if missing:
            raise AttributeError(f"LidarParams 缺少必要参数: {missing}")
        _ = self._get_aom_frequency_hz()
        _ = self._get_elevation_deg()
        if not hasattr(self, "simulate_single_radial"):
            raise AttributeError("当前仿真器缺少 simulate_single_radial()。")
        if not hasattr(self, "wind_field"):
            raise AttributeError("当前仿真器缺少 wind_field。")

    @staticmethod
    def _build_freq_mask(freq_axis_mhz: np.ndarray, store_band_mhz: tuple[float, float] | None):
        if store_band_mhz is None:
            return slice(None), freq_axis_mhz
        fmin, fmax = map(float, store_band_mhz)
        if not (fmin < fmax):
            raise ValueError("store_band_mhz 必须满足 fmin < fmax")
        mask = (freq_axis_mhz >= fmin) & (freq_axis_mhz <= fmax)
        if not np.any(mask):
            raise ValueError(f"store_band_mhz={store_band_mhz} 没有选中任何频点")
        return mask, freq_axis_mhz[mask]

    @staticmethod
    def _checkpoint_save(scan_res_partial: dict, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            data=np.asarray(scan_res_partial["data"], dtype=np.float32),
            noise_data=np.asarray(scan_res_partial["noise_data"], dtype=np.float32),
            excess_db=np.asarray(scan_res_partial["excess_db"], dtype=np.float32),
            vlos_true=np.asarray(scan_res_partial["vlos_true"], dtype=np.float32),
            u_true=np.asarray(scan_res_partial["u_true"], dtype=np.float32),
            v_true=np.asarray(scan_res_partial["v_true"], dtype=np.float32),
            w_true=np.asarray(scan_res_partial["w_true"], dtype=np.float32),
            azimuths_deg=np.asarray(scan_res_partial["azimuths_deg"], dtype=np.float32),
            range_axis=np.asarray(scan_res_partial["range_axis"], dtype=np.float32),
            height_axis=np.asarray(scan_res_partial["height_axis"], dtype=np.float32),
            freq_axis_mhz=np.asarray(scan_res_partial["freq_axis_mhz"], dtype=np.float32),
            freq_axis_hz=np.asarray(scan_res_partial["freq_axis_hz"], dtype=np.float64),
            completed_mask=np.asarray(scan_res_partial["completed_mask"], dtype=bool),
            meta=np.asarray(json.dumps(scan_res_partial.get("meta", {}), ensure_ascii=False)),
        )

    @staticmethod
    def _checkpoint_load(checkpoint_path: str | Path) -> dict:
        with np.load(checkpoint_path, allow_pickle=True) as ckpt:
            meta = {}
            if "meta" in ckpt:
                raw = ckpt["meta"]
                if isinstance(raw, np.ndarray):
                    raw = raw.item()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                if isinstance(raw, str) and raw:
                    meta = json.loads(raw)
            return {
                "data": ckpt["data"],
                "noise_data": ckpt["noise_data"],
                "excess_db": ckpt["excess_db"],
                "vlos_true": ckpt["vlos_true"],
                "u_true": ckpt["u_true"],
                "v_true": ckpt["v_true"],
                "w_true": ckpt["w_true"],
                "azimuths_deg": ckpt["azimuths_deg"],
                "range_axis": ckpt["range_axis"],
                "height_axis": ckpt["height_axis"],
                "freq_axis_mhz": ckpt["freq_axis_mhz"],
                "freq_axis_hz": ckpt["freq_axis_hz"],
                "completed_mask": ckpt["completed_mask"],
                "meta": meta,
            }

    @staticmethod
    def _choose_max_workers(requested: int | None, n_tasks: int) -> int:
        if n_tasks <= 0:
            return 1
        if requested is not None:
            return max(1, min(int(requested), n_tasks))
        cpu = os.cpu_count() or 1
        # Windows + FFT/噪声模型 + 大数组下，保守一些更稳
        if cpu <= 2:
            safe = 1
        elif cpu <= 4:
            safe = cpu - 1
        elif cpu <= 8:
            safe = cpu // 2
        else:
            safe = min(cpu // 2, 8)
        return max(1, min(safe, n_tasks))

    @staticmethod
    def _is_checkpoint_compatible(
        ckpt: dict,
        azimuths_deg: np.ndarray,
        wind_mode: str,
        active_profile: Path | None,
        n_accum: int,
        store_band_mhz: tuple[float, float] | None,
    ) -> tuple[bool, str]:
        try:
            if ckpt["data"].shape[0] != azimuths_deg.size:
                return False, "方位角数量不一致"
            if not np.allclose(np.asarray(ckpt["azimuths_deg"], dtype=float), azimuths_deg, atol=1e-6):
                return False, "方位角数组不一致"
            meta = ckpt.get("meta", {})
            if int(meta.get("n_accum", n_accum)) != int(n_accum):
                return False, "n_accum 不一致"
            if str(meta.get("wind_mode", wind_mode)) != str(wind_mode):
                return False, "wind_mode 不一致"
            ckpt_profile = meta.get("profile_path", None)
            now_profile = None if active_profile is None else str(active_profile)
            if ckpt_profile != now_profile:
                return False, "profile_path 不一致"
            ckpt_band = meta.get("store_band_mhz", None)
            now_band = None if store_band_mhz is None else [float(store_band_mhz[0]), float(store_band_mhz[1])]
            if ckpt_band != now_band:
                return False, "store_band_mhz 不一致"
        except Exception as exc:
            return False, f"校验异常: {exc}"
        return True, "OK"

    def extract_radial_result(self, scan_res: dict, azimuth_index: int) -> dict:
        azimuths = np.asarray(scan_res["azimuths_deg"])
        if azimuth_index < 0 or azimuth_index >= len(azimuths):
            raise IndexError("azimuth_index 超出范围")

        out = {
            "data": np.asarray(scan_res["data"][azimuth_index]),
            "noise_data": np.asarray(scan_res["noise_data"][azimuth_index]),
            "excess_db": np.asarray(scan_res["excess_db"][azimuth_index]),
            "range_axis": np.asarray(scan_res["range_axis"]),
            "height_axis": np.asarray(scan_res["height_axis"]),
            "freq_axis": np.asarray(scan_res["freq_axis_mhz"]),
            "freq_axis_hz": np.asarray(scan_res["freq_axis_hz"]),
            "vlos_true": np.asarray(scan_res["vlos_true"][azimuth_index]),
            "u_true": np.asarray(scan_res["u_true"]),
            "v_true": np.asarray(scan_res["v_true"]),
            "w_true": np.asarray(scan_res["w_true"]),
            "first_pulse": scan_res.get("first_radial", {}).get("first_pulse")
            if scan_res.get("first_radial") and azimuth_index == 0
            else None,
            "meta": dict(scan_res.get("meta", {})),
        }
        out["meta"]["azimuth"] = float(azimuths[azimuth_index])
        out["meta"]["azimuth_index"] = int(azimuth_index)
        return out

    def _compute_gate_truth(
        self,
        range_axis: np.ndarray,
        azimuth_deg: float,
        wind_mode: str,
        profile_path: str | Path | None,
        u_const: float,
        v_const: float,
        w_const: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        vlos_gate, height_gate = self.wind_field.get_radial_velocity(
            range_axis,
            float(azimuth_deg),
            self._get_elevation_deg(),
            wind_type=wind_mode,
            profile_path=profile_path,
            u_const=u_const,
            v_const=v_const,
            w_const=w_const,
        )
        return np.asarray(vlos_gate, dtype=float), np.asarray(height_gate, dtype=float)

    def simulate_full_scan(
        self,
        azimuths_deg: np.ndarray | list[float] | None = None,
        n_azimuth: int = 16,
        wind_mode: str = "constant",
        profile_path: str | Path | None = None,
        u_const: float = 0.0,
        v_const: float = 10.0,
        w_const: float = 0.0,
        n_accum: int = 50,
        signal_gain: float = 1.0,
        remove_dc: bool = True,
        keep_first_radial: bool = True,
        store_band_mhz: tuple[float, float] | None = None,
        checkpoint_every: int | None = None,
        checkpoint_path: str | None = None,
        data_dtype: Any = np.float32,
        parallel: bool = True,
        max_workers: int | None = None,
        resume_from_checkpoint: bool = True,
    ) -> dict:
        self._preflight_check()

        if azimuths_deg is None:
            azimuths_deg = np.linspace(0.0, 360.0, n_azimuth, endpoint=False)
        azimuths_deg = np.asarray(azimuths_deg, dtype=float)
        if azimuths_deg.ndim != 1 or azimuths_deg.size == 0:
            raise ValueError("azimuths_deg 必须为一维非空数组")
        if checkpoint_every is not None and checkpoint_every <= 0:
            raise ValueError("checkpoint_every 必须为正整数或 None")

        active_profile = self._resolve_profile_path(wind_mode, profile_path)
        aom_frequency_hz = self._get_aom_frequency_hz()
        elevation_deg = self._get_elevation_deg()
        n_azi = azimuths_deg.size

        if checkpoint_every is not None and checkpoint_path is None:
            checkpoint_path = "scan_checkpoint_partial.npz"

        # 尝试从 checkpoint 断点续跑
        checkpoint = None
        if resume_from_checkpoint and checkpoint_path is not None and Path(checkpoint_path).exists():
            try:
                checkpoint = self._checkpoint_load(checkpoint_path)
                ok, msg = self._is_checkpoint_compatible(
                    checkpoint, azimuths_deg, wind_mode, active_profile, n_accum, store_band_mhz
                )
                if ok:
                    print(f">>> 发现兼容 checkpoint，将断点续跑: {checkpoint_path}")
                else:
                    print(f">>> checkpoint 不兼容，将忽略并重算: {msg}")
                    checkpoint = None
            except Exception as exc:
                print(f">>> 读取 checkpoint 失败，将忽略并重算: {exc}")
                checkpoint = None

        # 初始化数组：若有 checkpoint 优先使用其维度与已完成结果
        first_radial = None
        if checkpoint is not None:
            spec = np.asarray(checkpoint["data"], dtype=data_dtype)
            noise = np.asarray(checkpoint["noise_data"], dtype=data_dtype)
            excess = np.asarray(checkpoint["excess_db"], dtype=data_dtype)
            vlos_true = np.asarray(checkpoint["vlos_true"], dtype=data_dtype)
            u_true = np.asarray(checkpoint["u_true"], dtype=data_dtype)
            v_true = np.asarray(checkpoint["v_true"], dtype=data_dtype)
            w_true = np.asarray(checkpoint["w_true"], dtype=data_dtype)
            range_axis = np.asarray(checkpoint["range_axis"], dtype=float)
            height_axis = np.asarray(checkpoint["height_axis"], dtype=float)
            freq_axis_mhz = np.asarray(checkpoint["freq_axis_mhz"], dtype=float)
            freq_axis_hz = np.asarray(checkpoint["freq_axis_hz"], dtype=float)
            completed_mask = np.asarray(checkpoint["completed_mask"], dtype=bool)
            n_gate, n_freq = spec.shape[1], spec.shape[2]
        else:
            # 选取第一个待计算方位角进行初始化
            init_index = 0
            first = self.simulate_single_radial(
                azimuth=float(azimuths_deg[init_index]),
                wind_mode=wind_mode,
                profile_path=active_profile,
                u_const=u_const,
                v_const=v_const,
                w_const=w_const,
                n_accum=n_accum,
                signal_gain=signal_gain,
                remove_dc=remove_dc,
            )
            if keep_first_radial and init_index == 0:
                first_radial = first

            range_axis = np.asarray(first["range_axis"], dtype=float)
            freq_axis_full_mhz = np.asarray(first["freq_axis"], dtype=float)
            freq_mask, freq_axis_mhz = self._build_freq_mask(freq_axis_full_mhz, store_band_mhz)
            freq_axis_hz = freq_axis_mhz * 1e6

            first_data = np.asarray(first["data"], dtype=data_dtype)[:, freq_mask]
            first_noise = np.asarray(first["noise_data"], dtype=data_dtype)[:, freq_mask]
            first_excess = np.asarray(first["excess_db"], dtype=data_dtype)[:, freq_mask]
            n_gate, n_freq = first_data.shape

            first_vlos_gate, height_axis = self._compute_gate_truth(
                range_axis, azimuths_deg[init_index], wind_mode, active_profile, u_const, v_const, w_const
            )
            if first_vlos_gate.shape[0] != n_gate:
                raise ValueError(
                    f"门级真值长度与功率谱距离门数不一致: vlos={first_vlos_gate.shape[0]}, n_gate={n_gate}"
                )

            spec = np.zeros((n_azi, n_gate, n_freq), dtype=data_dtype)
            noise = np.zeros_like(spec)
            excess = np.zeros_like(spec)
            vlos_true = np.zeros((n_azi, n_gate), dtype=data_dtype)
            completed_mask = np.zeros(n_azi, dtype=bool)

            spec[init_index] = first_data
            noise[init_index] = first_noise
            excess[init_index] = first_excess
            vlos_true[init_index] = np.asarray(first_vlos_gate, dtype=data_dtype)
            completed_mask[init_index] = True

            u_true, v_true, w_true = self.wind_field.get_wind_vector_field(
                height_axis,
                wind_type=wind_mode,
                profile_path=active_profile,
                u_const=u_const,
                v_const=v_const,
                w_const=w_const,
            )

        # 若使用 checkpoint，则需由 checkpoint 元数据恢复频段掩码；后续只需将结果切片到当前保存频段
        # 这里根据 saved freq_axis_mhz 反推是否全带保存已无必要，直接对新结果使用范围截取即可
        n_remaining = int(np.count_nonzero(~completed_mask))
        max_workers_eff = self._choose_max_workers(max_workers, n_remaining)
        parallel_eff = bool(parallel and max_workers_eff > 1 and n_remaining > 1)

        print(
            f">>> 开始整圈扫描仿真 | n_azimuth={n_azi} | 已完成={np.count_nonzero(completed_mask)} | "
            f"剩余={n_remaining} | wind_mode={wind_mode} | n_accum={n_accum} | "
            f"store_band_mhz={store_band_mhz} | parallel={parallel_eff} | max_workers={max_workers_eff}"
        )
        if wind_mode == "profile":
            print(f"    使用风廓线: {active_profile}")
        else:
            print(f"    常风速: u={u_const:.3f}, v={v_const:.3f}, w={w_const:.3f} m/s")

        def save_partial() -> None:
            if checkpoint_every is None or checkpoint_path is None:
                return
            partial = {
                "data": spec,
                "noise_data": noise,
                "excess_db": excess,
                "vlos_true": vlos_true,
                "u_true": np.asarray(u_true, dtype=np.float32),
                "v_true": np.asarray(v_true, dtype=np.float32),
                "w_true": np.asarray(w_true, dtype=np.float32),
                "azimuths_deg": azimuths_deg,
                "range_axis": range_axis,
                "height_axis": height_axis,
                "freq_axis_mhz": freq_axis_mhz,
                "freq_axis_hz": freq_axis_hz,
                "completed_mask": completed_mask,
                "meta": {
                    "completed_azimuths": int(np.count_nonzero(completed_mask)),
                    "n_azimuth_total": int(n_azi),
                    "n_accum": int(n_accum),
                    "wind_mode": wind_mode,
                    "profile_path": str(active_profile) if active_profile is not None else None,
                    "u_const": float(u_const),
                    "v_const": float(v_const),
                    "w_const": float(w_const),
                    "store_band_mhz": None if store_band_mhz is None else [float(store_band_mhz[0]), float(store_band_mhz[1])],
                    "aom_frequency_hz": float(aom_frequency_hz),
                    "elevation_deg": float(elevation_deg),
                    "parallel": bool(parallel_eff),
                    "max_workers": int(max_workers_eff),
                },
            }
            self._checkpoint_save(partial, checkpoint_path)
            print(f"[Checkpoint] 已保存: {checkpoint_path} | 已完成 {np.count_nonzero(completed_mask)}/{n_azi} 个方位角")

        # 构建待处理任务
        tasks = []
        for i in range(n_azi):
            if completed_mask[i]:
                continue
            tasks.append(
                {
                    "index": int(i),
                    "azimuth": float(azimuths_deg[i]),
                    "wind_mode": wind_mode,
                    "profile_path": None if active_profile is None else str(active_profile),
                    "u_const": float(u_const),
                    "v_const": float(v_const),
                    "w_const": float(w_const),
                    "n_accum": int(n_accum),
                    "signal_gain": float(signal_gain),
                    "remove_dc": bool(remove_dc),
                }
            )

        # 用当前保存频率轴确定新结果的频率切片
        def _slice_to_saved_band(res: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            full_freq = np.asarray(res["freq_axis"], dtype=float)
            mask = (full_freq >= float(freq_axis_mhz[0]) - 1e-9) & (full_freq <= float(freq_axis_mhz[-1]) + 1e-9)
            if np.count_nonzero(mask) != len(freq_axis_mhz):
                # 更严格匹配，避免端点比较问题
                inds = [int(np.argmin(np.abs(full_freq - f))) for f in freq_axis_mhz]
                data = np.asarray(res["data"], dtype=data_dtype)[:, inds]
                noise_i = np.asarray(res["noise_data"], dtype=data_dtype)[:, inds]
                excess_i = np.asarray(res["excess_db"], dtype=data_dtype)[:, inds]
            else:
                data = np.asarray(res["data"], dtype=data_dtype)[:, mask]
                noise_i = np.asarray(res["noise_data"], dtype=data_dtype)[:, mask]
                excess_i = np.asarray(res["excess_db"], dtype=data_dtype)[:, mask]
            return data, noise_i, excess_i

        completed_since_ckpt = 0
        if parallel_eff and tasks:
            try:
                with ProcessPoolExecutor(max_workers=max_workers_eff) as ex:
                    futures = {ex.submit(_scan_worker, task): task for task in tasks}
                    for fut in as_completed(futures):
                        worker_out = fut.result()
                        i = int(worker_out["index"])
                        res = worker_out["result"]
                        data_i, noise_i, excess_i = _slice_to_saved_band(res)
                        spec[i] = data_i
                        noise[i] = noise_i
                        excess[i] = excess_i
                        vlos_gate_i, _ = self._compute_gate_truth(
                            range_axis, azimuths_deg[i], wind_mode, active_profile, u_const, v_const, w_const
                        )
                        vlos_true[i] = np.asarray(vlos_gate_i, dtype=data_dtype)
                        completed_mask[i] = True
                        completed_since_ckpt += 1
                        print(f"[Done] azimuth index={i:02d}, azimuth={azimuths_deg[i]:.1f}°")
                        if checkpoint_every is not None and completed_since_ckpt >= checkpoint_every:
                            save_partial()
                            completed_since_ckpt = 0
            except KeyboardInterrupt:
                print("\n>>> 检测到中断，正在保存 checkpoint ...")
                save_partial()
                raise
        else:
            for task in tasks:
                worker_out = _scan_worker(task)
                i = int(worker_out["index"])
                res = worker_out["result"]
                data_i, noise_i, excess_i = _slice_to_saved_band(res)
                spec[i] = data_i
                noise[i] = noise_i
                excess[i] = excess_i
                vlos_gate_i, _ = self._compute_gate_truth(
                    range_axis, azimuths_deg[i], wind_mode, active_profile, u_const, v_const, w_const
                )
                vlos_true[i] = np.asarray(vlos_gate_i, dtype=data_dtype)
                completed_mask[i] = True
                completed_since_ckpt += 1
                print(f"[Done] azimuth index={i:02d}, azimuth={azimuths_deg[i]:.1f}°")
                if checkpoint_every is not None and completed_since_ckpt >= checkpoint_every:
                    save_partial()
                    completed_since_ckpt = 0

        if checkpoint_every is not None and completed_since_ckpt > 0:
            save_partial()

        return {
            "data": spec,
            "noise_data": noise,
            "excess_db": excess,
            "vlos_true": vlos_true,
            "u_true": np.asarray(u_true, dtype=data_dtype),
            "v_true": np.asarray(v_true, dtype=data_dtype),
            "w_true": np.asarray(w_true, dtype=data_dtype),
            "azimuths_deg": azimuths_deg,
            "range_axis": range_axis,
            "height_axis": height_axis,
            "freq_axis_mhz": freq_axis_mhz,
            "freq_axis_hz": freq_axis_hz,
            "completed_mask": completed_mask,
            "first_radial": first_radial if keep_first_radial else None,
            "meta": {
                "n_azimuth": int(n_azi),
                "n_accum": int(n_accum),
                "signal_gain": float(signal_gain),
                "remove_dc": bool(remove_dc),
                "wind_mode": wind_mode,
                "profile_path": str(active_profile) if active_profile is not None else None,
                "u_const": float(u_const),
                "v_const": float(v_const),
                "w_const": float(w_const),
                "elevation_deg": float(elevation_deg),
                "sample_rate_hz": float(self.p.sample_rate),
                "fft_points": int(self.p.fft_points),
                "points_per_bin": int(self.p.points_per_bin),
                "wavelength_m": float(self.p.wavelength),
                "aom_frequency_hz": float(aom_frequency_hz),
                "store_band_mhz": None if store_band_mhz is None else [float(store_band_mhz[0]), float(store_band_mhz[1])],
                "n_freq_saved": int(spec.shape[2]),
                "data_dtype": str(np.dtype(data_dtype)),
                "parallel": bool(parallel_eff),
                "max_workers": int(max_workers_eff),
                "resume_from_checkpoint": bool(resume_from_checkpoint),
            },
        }

    @staticmethod
    def save_scan_hdf5(scan_res: dict, filepath: str, overwrite: bool = True) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("保存 HDF5 需要安装 h5py") from exc

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"文件已存在: {path}")

        with h5py.File(path, "w") as f:
            f.attrs["description"] = "Simulated coherent Doppler wind lidar power spectra for full azimuth scan"
            f.attrs["meta_json"] = json.dumps(scan_res.get("meta", {}), ensure_ascii=False)

            g_axes = f.create_group("axes")
            g_axes.create_dataset("azimuth_deg", data=np.asarray(scan_res["azimuths_deg"], dtype=np.float32))
            g_axes.create_dataset("range_m", data=np.asarray(scan_res["range_axis"], dtype=np.float32))
            g_axes.create_dataset("height_m", data=np.asarray(scan_res["height_axis"], dtype=np.float32))
            g_axes.create_dataset("frequency_mhz", data=np.asarray(scan_res["freq_axis_mhz"], dtype=np.float32))
            g_axes.create_dataset("frequency_hz", data=np.asarray(scan_res["freq_axis_hz"], dtype=np.float64))

            g_spectra = f.create_group("spectra")
            g_spectra.create_dataset("specData", data=np.asarray(scan_res["data"], dtype=np.float32), compression="gzip", compression_opts=4)
            g_spectra.create_dataset("noiseData", data=np.asarray(scan_res["noise_data"], dtype=np.float32), compression="gzip", compression_opts=4)
            g_spectra.create_dataset("excessOverNoise_dB", data=np.asarray(scan_res["excess_db"], dtype=np.float32), compression="gzip", compression_opts=4)

            g_truth = f.create_group("truth")
            g_truth.create_dataset("vlos_mps", data=np.asarray(scan_res["vlos_true"], dtype=np.float32), compression="gzip", compression_opts=4)
            g_truth.create_dataset("u_mps", data=np.asarray(scan_res["u_true"], dtype=np.float32))
            g_truth.create_dataset("v_mps", data=np.asarray(scan_res["v_true"], dtype=np.float32))
            g_truth.create_dataset("w_mps", data=np.asarray(scan_res["w_true"], dtype=np.float32))

            g_state = f.create_group("state")
            g_state.create_dataset("completed_mask", data=np.asarray(scan_res.get("completed_mask", np.ones(len(scan_res["azimuths_deg"]), dtype=bool)), dtype=bool))

            if scan_res.get("first_radial") is not None:
                g_dbg = f.create_group("debug_first_radial")
                g_dbg.attrs["azimuth_deg"] = float(scan_res["azimuths_deg"][0])
                fr = scan_res["first_radial"]
                if fr.get("first_pulse") is not None:
                    for key, val in fr["first_pulse"].items():
                        g_dbg.create_dataset(key, data=np.asarray(val, dtype=np.float32), compression="gzip", compression_opts=4)

    @staticmethod
    def save_scan_npz(scan_res: dict, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            data=np.asarray(scan_res["data"], dtype=np.float32),
            noise_data=np.asarray(scan_res["noise_data"], dtype=np.float32),
            excess_db=np.asarray(scan_res["excess_db"], dtype=np.float32),
            vlos_true=np.asarray(scan_res["vlos_true"], dtype=np.float32),
            u_true=np.asarray(scan_res["u_true"], dtype=np.float32),
            v_true=np.asarray(scan_res["v_true"], dtype=np.float32),
            w_true=np.asarray(scan_res["w_true"], dtype=np.float32),
            azimuths_deg=np.asarray(scan_res["azimuths_deg"], dtype=np.float32),
            range_axis=np.asarray(scan_res["range_axis"], dtype=np.float32),
            height_axis=np.asarray(scan_res["height_axis"], dtype=np.float32),
            freq_axis_mhz=np.asarray(scan_res["freq_axis_mhz"], dtype=np.float32),
            freq_axis_hz=np.asarray(scan_res["freq_axis_hz"], dtype=np.float64),
            completed_mask=np.asarray(scan_res.get("completed_mask", np.ones(len(scan_res["azimuths_deg"]), dtype=bool)), dtype=bool),
            meta=np.asarray(json.dumps(scan_res.get("meta", {}), ensure_ascii=False)),
        )

    def plot_scan_quicklook(
        self,
        scan_res: dict,
        azimuth_indices: list[int] | None = None,
        mode: str = "normalized",
        norm_mode: str = "per_gate",
        xlim=(0, 500),
        fmin_mhz: float | None = None,
        dyn_range_db: float = 35.0,
    ) -> None:
        if azimuth_indices is None:
            n_azi = len(scan_res["azimuths_deg"])
            azimuth_indices = sorted(set([0, n_azi // 4, n_azi // 2, 3 * n_azi // 4]))

        n_show = len(azimuth_indices)
        fig, axes = plt.subplots(n_show, 1, figsize=(11, 2.9 * n_show), sharex=True)
        if n_show == 1:
            axes = [axes]

        mesh = None
        for ax, idx in zip(axes, azimuth_indices):
            radial = self.extract_radial_result(scan_res, idx)
            data = radial["data"]
            freqs = scan_res["freq_axis_mhz"]
            ranges = scan_res["range_axis"]

            freqs_sel, data_sel, _ = self._select_freq_band(freqs, data, fmin_mhz=fmin_mhz, fmax_mhz=xlim[1])
            if mode == "normalized":
                if norm_mode == "per_gate":
                    denom = np.max(data_sel, axis=1, keepdims=True)
                elif norm_mode == "global":
                    denom = np.max(data_sel)
                else:
                    raise ValueError("norm_mode 必须为 'per_gate' 或 'global'")
                img = np.clip(data_sel / np.maximum(denom, 1e-30), 0.0, 1.0)
                mesh = ax.pcolormesh(freqs_sel, ranges, img, shading="auto", cmap="jet", vmin=0.0, vmax=1.0)
            elif mode == "db":
                img = self._db(data_sel)
                vmax = np.nanmax(img)
                vmin = vmax - dyn_range_db
                mesh = ax.pcolormesh(freqs_sel, ranges, img, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
            else:
                raise ValueError("mode 必须为 'normalized' 或 'db'")

            ax.set_ylabel("Range (m)")
            ax.set_title(f"Azimuth = {scan_res['azimuths_deg'][idx]:.1f}°")
            ax.grid(True, linestyle="--", alpha=0.15)
            self._apply_plot_style(ax, xminor=True, yminor=True)

        axes[-1].set_xlabel("Frequency (MHz)")
        axes[-1].set_xlim(max(xlim[0], fmin_mhz or xlim[0]), min(xlim[1], float(np.max(scan_res["freq_axis_mhz"]))))

        cbar_label = "Normalized PSD" if mode == "normalized" else "PSD (dB, A$^2$/Hz)"
        fig.subplots_adjust(left=0.10, right=0.90, bottom=0.07, top=0.93, hspace=0.30)
        cax = fig.add_axes([0.92, 0.18, 0.022, 0.64])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(cbar_label)
        self._style_colorbar(cbar)

        fig.suptitle("Quick-look of full azimuth scan", fontsize=14, fontname="Times New Roman")
        plt.show()
