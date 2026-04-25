"""
Point-wise verification for the CKM1D (Coupled Keller--Miksis 1D Translation) coupled-bubble GPU solver.

Runs the same initial value problem across many identical environments and checks
that the outputs match (within tolerance). Optionally plots and saves dense output.
"""

from __future__ import annotations
import dev_bootstap
import tyro
import time
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal, List

from coupledbubble_control.models import CKM1D
from coupledbubble_control.models import MaterialProperties
from coupledbubble_control.backends import BackendName, KernelVariant, get_current_device_name


@dataclass(frozen=True)
class Scene:
    R0: List[float]     # List of bubble sizes      [um]
    PA0: float          # Pressure Amplitude        [bar]
    FREQ: float         # Excitation Frequency      [Hz]
    T: float            # Time domain end           [-]
    D0: float           # Initial bubble distance   [um]   


def create_scene(R0, PA0, FREQ, T, D0):
    return Scene(
        R0 = [float(R) for R in R0],
        PA0 = -PA0 * 1.013,
        FREQ = float(FREQ),
        T = float(T),
        D0 = float(D0)
    )


SCENE_DICT = {
    "2-a" : create_scene([7.0, 5.0],  1.2, 20.0, 100.0, 200.0),
    "2-b" : create_scene([6.0, 4.0], 1.21, 20.0, 100.0, 200.0),
    "2-c" : create_scene([3.0, 5.0], 1.22, 20.0, 100.0, 200.0),
    "2-d" : create_scene([5.0, 4.0],  1.2, 20.0, 100.0, 200.0),
    "2-e" : create_scene([4.0, 3.0], 1.21, 20.0, 100.0, 200.0),
    "2-f" : create_scene([3.0, 2.0],  1.3, 20.0, 100.0, 150.0),
    "3-a" : create_scene([7.0, 5.0],  1.2, 20.0,  10.0, 200.0),
    "3-b" : create_scene([5.0, 4.0],  1.2, 20.0,  10.0, 200.0),
    "4-a" : create_scene([6.0, 5.0],  1.2, 20.0,  7.50, 300.0),
    "4-b" : create_scene([6.0, 4.0], 1.21, 20.0,  10.0, 200.0)
}


@dataclass
class Config:
    scene_id: Literal["2-a", "2-b", "2-c", "2-d", "2-e", "2-f", "3-a", "3-b", "4-a", "4-b"] = "4-a"
    num_systems: int = 4096
    systems_per_block: int = 64
    backend: Literal["numba", "cupy"] = "numba"
    variant: Literal["shared", "warp"] = "warp"
    num_dense_output: int = 4096
    main_system: int = 0
    plot_results: bool = True
    save_results: bool = True

    def __post_init__(self):
        if self.main_system >= self.num_systems:
            self.main_system = 0


if __name__ == "__main__":
    config = tyro.cli(Config)
    scene_parameters = SCENE_DICT[config.scene_id]
    BACKEND = BackendName(config.backend)
    VARIANT = KernelVariant(config.variant)
    NS      = config.num_systems
    SPB     = config.systems_per_block
    UPS     = len(scene_parameters.R0)  # (2)
    NK      = 1
    AC      = "CONST"
    NDO     = config.num_dense_output

    # Shape: (NS, UPS)
    R0      = np.tile(
                np.array(scene_parameters.R0, dtype=np.float64),
                NS).reshape(NS, UPS)                                                
    
    # Shape (NK, NS)
    PA      = np.full((NK, NS), scene_parameters.PA0, dtype=np.float64)
    FR      = np.full((NK, NS), scene_parameters.FREQ, dtype=np.float64)
    PS      = np.zeros((NK, NS), dtype=np.float64)  # Dummy
    CL      = MaterialProperties.CL
    LR      = CL / (scene_parameters.FREQ * 1000)           # Reference length (wave-length) [m]
    REF_FREQ= scene_parameters.FREQ

    device = get_current_device_name()

    # Initialize model
    model = CKM1D(
        num_systems = NS,
        systems_per_block = SPB,
        units_per_system = UPS,
        num_components = NK,
        acoustic_field = AC,
        num_stored_points = NDO,
        # Physical parameters
        R0 = R0,
        PA = PA,
        FR = FR,
        PS = PS,
        REF_FREQ = REF_FREQ,
        backend = BACKEND,
        variant = VARIANT,
        # CUDA-OPTS
        cuda_mode = "release",
        compiler = "nvrtc",
        max_registers = 192,
        fastmath = True
    )

    # Update host buffers
    model.runtime.host_buffers.state.time_end[:] = scene_parameters.T
    model.runtime.host_buffers.state.actual_state[1, :, 1] = scene_parameters.D0 * 1e-6 / LR
    #print(R0)
    #print(PA)
    #print(FR)

    # RUN SIMULATIONS
    start_time = time.time()
    model.solve(sync_to_host=True)
    end_time = time.time()
    print(f"The simulation time was {end_time - start_time:0.2f} s")

    # CHECK CORRECTNESS
    TILE = model._execution_spec.tile
    main = config.main_system
    db = model.runtime.host_buffers.dense     # type: ignore
    # 1) dense_index valdity range
    di = db.dense_index.astype(np.int64)
    assert np.all(di >= 0), f"dense_index contains negative values: min={di.min()}"
    assert np.all(di <= NDO), f"dense_index exceeds NDO={NDO}: max={di.max()}"

    # 2) Do all systems have the same number of stored dense points?
    unique_counts, unique_freq = np.unique(di, return_counts=True)
    if unique_counts.size == 1:
        npts = int(unique_counts[0])
        print(f"[CHECK] All systems have identical dense_index = {npts}.")
    else:
        # Report distribution and choose safe common slice length
        print("[CHECK] dense_index differs across systems.")
        print("        counts (npts -> how many systems):")
        for c, f in zip(unique_counts.tolist(), unique_freq.tolist()):
            print(f"        {c:6d} -> {f:6d}")

        npts = int(di.min())
        print(f"[CHECK] Using common slice length npts_common = min(dense_index) = {npts}.")
    
    assert npts > 0, f"No dense output points stored (npts={npts})."
    
    # 3) Point-wise equality check vs main
    #    This assumes time grids are identical (or at least identical for first npts).
    #    This assumtion should be valid for identical systems

    t_ref = db.dense_time[:npts, main]
    t_all = db.dense_time[:npts,    :]

    # Time grid consistensy check
    # If it fails, point-wise compaarison of states is not meaningful!
    max_time_dev = np.max(np.abs(t_all - t_ref[:, np.newaxis]))
    print(f"[CHECK] Max |t_env - t_main| over first {npts} points: {max_time_dev:.3e}")
    time_tol = 1.0e-16  # set to e.g. 1e-16 if you expect tiny FP jitter
    if max_time_dev > time_tol:
        print("[WARN] Time grids differ across systems; state point-wise comparison may be invalid.")
    else:
        abs_tol = 1e-12
        rel_tol = 1e-12
        worst_abs = 0.0
        worst_rel = 0.0
        worst_sys = None

        state_ref = db.dense_state[:npts, :, main, :UPS]
        for s in range(NS):
            if s == main:
                continue
            state_cur = db.dense_state[:npts, :, s, :UPS]
            diff = np.abs(state_cur - state_ref)
            denom = np.maximum(np.abs(state_ref), abs_tol)
            rel = diff / denom

            a = float(np.max(diff))
            r = float(np.max(rel))

            if a > worst_abs or r > worst_rel:
                worst_abs = max(worst_abs, a)
                worst_rel = max(worst_rel, r)
                worst_sys = s

        print(f"[CHECK] worst_abs={worst_abs:.3e}, worst_rel={worst_rel:.3e} (vs main={main}, worst_sys={worst_sys})")

        if worst_abs > abs_tol and worst_rel > rel_tol:
            print("[WARN] Some systems differ from main beyond tolerances.")

    # PLOT AND SAVE RESULTS
    if config.plot_results:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

        t = db.dense_time[:npts, main]
        for u in range(UPS):
            R_um = db.dense_state[:npts, 0, main, u] * scene_parameters.R0[u]
            x_um = db.dense_state[:npts, 1, main, u] * LR * 1e6

            axes[0].plot(t, R_um, label = f"Bubble {u}")
            axes[1].plot(t, x_um, label = f"Bubble {u}")

            if u < UPS-1:
                R_um_next = db.dense_state[:npts, 0, main, u+1] * scene_parameters.R0[u+1]
                x_um_next = db.dense_state[:npts, 1, main, u+1] * LR * 1e6

                axes[2].plot(t, R_um_next + R_um, label=f"R{u+1} + R{u}")
                axes[2].plot(t, x_um_next - x_um, label=f"x{u+1} - x{u}")

        axes[0].set_ylabel("Radius [µm]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].set_ylabel("Position [µm]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        axes[2].set_ylabel("Rj+Ri & Dji [µm]")
        axes[2].set_xlabel("Acoustic cycles")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")

        fig.suptitle(
            f"Coupled bubbles | main_system={main}"
        )

        fig.tight_layout()
        plt.show()


    if config.save_results:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        out_dir = Path(__file__).resolve().parent / "CKM1D_Pointwise"
        out_dir.mkdir(parents=True, exist_ok=True)

        tag = (
            f"{config.scene_id}"
            f"_NS_{config.num_systems}"
            f"_SPB_{config.systems_per_block}"
            f"_NDO_{config.num_dense_output}"
            f"_{config.backend}_{config.variant}"
            f"_{ts}"
        )

        npz_path  = out_dir / f"{tag}.npz"
        meta_path = out_dir / f"{tag}.json"
        csv_path  = out_dir / f"{tag}.csv"

        np.savez_compressed(
            npz_path,
            dense_index = db.dense_index,
            dense_time  = db.dense_time,
            dense_state = db.dense_state,
            simulation_time_s = end_time - start_time,
            device = device
        )

        print(f"Saved: {npz_path}")

        meta = {
            "config" : {
                "scene_id"          : config.scene_id,
                "num_systems"       : config.num_systems,
                "systems_per_block" : config.systems_per_block,
                "backed"            : config.backend,
                "variant"           : config.variant,
                "device"            : device
            },
            "simulation_time_s"     : float(end_time - start_time),
            "scene_parameters"      : asdict(scene_parameters)
        }

        meta_path.write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

        print(f"Saved: {meta_path}")

        t = db.dense_time[:npts, main]
        R_um0 = db.dense_state[:npts, 0, main, 0] * scene_parameters.R0[0]
        R_um1 = db.dense_state[:npts, 0, main, 1] * scene_parameters.R0[1]
        x_um0 = db.dense_state[:npts, 1, main, 0] * LR * 1e6
        x_um1 = db.dense_state[:npts, 1, main, 1] * LR * 1e6

        main_arr = np.column_stack(
            [t, R_um0, R_um1, x_um0, x_um1]
        )
        header = "t_s,R0_um,R1_um,x0_um,x1_um"
        np.savetxt(csv_path, main_arr, delimiter=",", fmt="%16.12e", header=header, comments="")

        print(f"Saved: {csv_path}")