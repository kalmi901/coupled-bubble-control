"""
Parameter-study verification for the CKM1D (Coupled Keller--Miksis 1D Translation) coupled-bubble GPU solver.

Runs a sweep over selected physical/solver parameters and verifies that results are
consistent across runs/systems (e.g. no stuck divergence). Optionally saves outputs.
"""
from __future__ import annotations
import dev_bootstap
import tyro
import time
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal, List, Union, Optional

from coupledbubble_control.models import CKM1D
from coupledbubble_control.models import MaterialProperties
from coupledbubble_control.backends import BackendName, KernelVariant, get_current_device_name


@dataclass(frozen=True)
class Sweep:
    """ Meghatároz egy paraméter: lehet fix érték vagy tartomány """
    values: List[float]
    unit_multiplier: float = 1.0

    @property
    def scaled(self) -> List[float]:
        return [v * self.unit_multiplier for v in self.values]
    
@dataclass(frozen=True)
class Scene:
    R0: List[Sweep]         # Bubble Size               [um]
    PA: List[Sweep]         # Pressure Amplitude        [bar]
    FR: List[Sweep]         # Excitation Frequency      [kHz]
    X0: List[Sweep]         # Initial bubble positions  [-]
    T: float                # Time domain end           [-]
    sweep_param: Literal["PA", "PA", "R0", "X0"]
    sweep_axis : int = 0


def create_scene(R0, PA, FR, X0, T, sweep_param, sweep_axis,
                m_R0=1e-6, m_PA=1e5, m_FREQ=1e3, m_X0=1.0):
    
    return Scene(
        R0 = [Sweep(r, m_R0) for r in R0],
        PA = [Sweep(p, m_PA) for p in PA],
        FR = [Sweep(f, m_FREQ) for f in FR],
        X0 = [Sweep(x, m_X0) for x in X0],
        T  = float(T),
        sweep_param = sweep_param,
        sweep_axis = sweep_axis
    )

# Pre-Defined Scenes
SCENE_DICT = {
    # 2 buborékos PA0 paramétersöprés
    "2B_PA1_CASE1" : create_scene(
        R0  = [[60.0], [60.0]],
        PA  = [[0.0, 1.0], [0.0]],
        FR  = [[25.0], [50.0]],
        X0  = [[-0.1], [0.1]],
        T   = 500,
        sweep_param="PA",
        sweep_axis=0
    ),
    # DualFrequency
    "2B_PA1_CASE2" : create_scene(
        R0  = [[60.0], [60.0]],
        PA  = [[0.0, 1.0], [0.2]],
        FR  = [[25.0], [50.0]],
        X0  = [[-0.1], [0.1+1e-6]],
        T   = 500,
        sweep_param="PA",
        sweep_axis=0
    ),
    "2B_PA1_CASE3" : create_scene(
        R0  = [[60.0], [60.0]],
        PA  = [[0.0, 1.0], [0.5]],
        FR  = [[25.0], [50.0]],
        X0  = [[-0.1], [0.1+1e-6]],
        T   = 500,
        sweep_param="PA",
        sweep_axis=0
    ),
    # Single Frequency (PA2=0)
    "3B_PA1_CASE1" : create_scene(
        R0  = [[60.0], [60.0], [60.0]],
        PA  = [[0.0, 1.0], [0.0]],
        FR  = [[25.0], [50.0]],
        X0  = [[-0.1], [1e-12], [0.1]],
        T   = 500,
        sweep_param="PA",
        sweep_axis=0
    ),
    # Dual Frequency 
    "3B_PA1_CASE2" : create_scene(
        R0  = [[60.0], [60.0], [60.0]],
        PA  = [[0.0, 1.0], [0.2]],
        FR  = [[25.0], [50.0]],
        X0  = [[-0.12], [1e-6], [0.12]],
        T   = 500,
        sweep_param="PA",
        sweep_axis=0
    ),
    "3B_PA1_CASE3" : create_scene(
        R0  = [[60.0], [60.0], [60.0]],
        PA  = [[0.0, 1.0], [0.5]],
        FR  = [[25.0], [50.0]],
        X0  = [[-0.12], [1e-6], [0.12]],
        T   = 500,
        sweep_param="PA",
        sweep_axis=0
    ),
    # Single Frequency (PA2=0)
    "4B_PA1_CASE1" : create_scene(
        R0 = [[60.0], [60.0], [60.0], [60.0]],
        PA = [[0.0, 1.0], [0.0]],
        FR = [[25.0], [50.0]],
        X0 = [[-0.2], [-0.05], [0.05], [0.2]],
        T  = 500,
        sweep_param="PA",
        sweep_axis=0
    ),
    # Dual Frequency 
    "4B_PA1_CASE2" : create_scene(
        R0 = [[60.0], [60.0], [60.0], [60.0]],
        PA = [[0.0, 1.0], [0.2]],
        FR = [[25.0], [50.0]],
        X0 = [[-0.4], [-0.1], [0.1+1e-6], [0.4]],
        T  = 500,
        sweep_param="PA",
        sweep_axis=0
    ),
    "4B_PA1_CASE3" : create_scene(
        R0 = [[60.0], [60.0], [60.0], [60.0]],
        PA = [[0.0, 1.0], [0.5]],
        FR = [[25.0], [50.0]],
        X0 = [[-0.4], [-0.1], [0.1+1e-6], [0.4]],
        T  = 500,
        sweep_param="PA",
        sweep_axis=0
    )
}


def build_parameter_ranges(config, scene_parameters):

    NS = config.num_systems
    UPS = len(scene_parameters.R0)
    NK  = len(scene_parameters.PA)

    def get_sweep_points(sweep_obj: Sweep, num_systems: int):
        vals = sweep_obj.values
        if len(vals) == 1:
            # Fix érték --> repeat x num_systems 
            return np.full((num_systems, ), vals[0], dtype=np.float64)
        elif len(vals) == 2:
            # Ha két érték van, generálunk egy rácsot
            return np.linspace(vals[0], vals[1], num_systems, endpoint=True, dtype=np.float64)
        else:
            raise ValueError(
                f"Expected 1 or 2 values in Sweep, got {len(vals)}: {sweep_obj.values}"
            )


    # R0, Shape (NS, UPS)
    R0 = np.empty((NS, UPS), dtype=np.float64)
    X0 = np.empty((NS, UPS), dtype=np.float64)
    for u in range(UPS):
        R0[:, u] = get_sweep_points(
                    scene_parameters.R0[u],
                    NS)
        
        X0[:, u] = get_sweep_points(
                    scene_parameters.X0[u],
                    NS)
        

    # PA, Shape (NK, NS)
    PA = np.empty((NK, NS), dtype=np.float64)
    FR = np.empty((NK, NS), dtype=np.float64)
    PS = np.zeros((NK, NS), dtype=np.float64)
    for k in range(NK):
        PA[k, :] = get_sweep_points(
                    scene_parameters.PA[k],
                    NS)
        FR[k, :] = get_sweep_points(
                    scene_parameters.FR[k],
                    NS)

    return R0, X0, PA, FR, PS


@dataclass
class Config:
    scene_id: str = "4B_PA1_CASE1"
    num_systems: int = 4096
    systems_per_block: int = 32
    backend: Literal["numba", "cupy"] = "numba"
    variant: Literal["shared", "warp"] = "warp"
    num_dense_output: int = 4096
    main_system: int = -1
    plot_results: bool = True
    save_results: bool = True
    plot_resolution: int = 16

    def __post_init__(self):
        if self.main_system >= self.num_systems:
            self.main_system = 0


if __name__ == "__main__":
    config = tyro.cli(Config)
    if config.scene_id not in SCENE_DICT.keys():
        raise ValueError(
            f"Unknown scene_id '{config.scene_id}'. "
            f"Available: {list(SCENE_DICT.keys())}"
        )
    scene_parameters = SCENE_DICT[config.scene_id]
    BACKEND = BackendName(config.backend)
    VARIANT = KernelVariant(config.variant)
    NS      = config.num_systems
    SPB     = config.systems_per_block
    UPS     = len(scene_parameters.R0)
    NK      = len(scene_parameters.PA)
    AC      = "SW_A"
    NDO     = config.num_dense_output


    R0, X0, \
    PA, FR, PS \
        = build_parameter_ranges(config, scene_parameters)

    CL = MaterialProperties.CL
    LR = CL / (scene_parameters.FR[0].scaled[0])     # Reference length (wave-length) [m]
    REF_FREQ = scene_parameters.FR[0].values[0]

    #print(R0)
    #print(PA)
    device = get_current_device_name()

    # INITIALIZE THE MODEL
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

    # UPDATE HOST BUFFERS
    model.runtime.host_buffers.state.time_end[:] = scene_parameters.T
    model.runtime.host_buffers.state.actual_state[1, :, :UPS] = X0

    # RUN SIMULATIONS
    start_time = time.time()
    kernel_times = model.solve(
                        max_steps=int(1e6),
                        kernel_steps=2048,
                        sync_to_host=True,
                        benchmark=True,
                        debug=True)
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

    if config.plot_results:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

        npts = db.dense_index[main]
        t = db.dense_time[:npts, main]
        for u in range(UPS):
            R_um = db.dense_state[:npts, 0, main, u] * R0[main, u]
            x_um = db.dense_state[:npts, 1, main, u] * LR * 1e6

            axes[0].plot(t, R_um, label = f"Bubble {u}")
            axes[1].plot(t, x_um, label = f"Bubble {u}")

            if u < UPS-1:
                R_um_next = db.dense_state[:npts, 0, main, u+1] * R0[main, u]
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

        # -- Plot 3D Figs --
        plot_n = min(config.plot_resolution, config.num_systems)
        sel    = np.linspace(0, NS-1, plot_n, dtype=np.int64)
        cmap_name = "tab10" if UPS <= 10 else "tab20"
        cmap = plt.get_cmap(cmap_name)
        unit_colors = [cmap(u % cmap.N) for u in range(UPS)]
         
        x_label = "Acoustic cycles"
        view_elev, view_azim = 25, -60

        fig3D = plt.figure(figsize=(12, 6))
        axR = fig3D.add_subplot(1, 3, 1, projection="3d")
        axX = fig3D.add_subplot(1, 3, 2, projection="3d")
        axD = fig3D.add_subplot(1, 3, 3, projection="3d")

        if scene_parameters.sweep_param == "PA":
            idx = scene_parameters.sweep_axis
            y_vals = PA[idx, :]
            y_label = f"PA{idx} [bar]"  
        elif scene_parameters.sweep_param == "R0":
            pass
        else:
            # TODO raise runtime error
            pass

        for sys_id in sel:
            npts = db.dense_index[sys_id]
            t    = db.dense_time[:npts, sys_id]
            y    = np.full_like(t, y_vals[sys_id], dtype=np.float32)

            for u in range(UPS):
                color = unit_colors[u]
                R_um = db.dense_state[:npts, 0, sys_id, u] * R0[sys_id, u]
                x_um = db.dense_state[:npts, 1, sys_id, u] * LR * 1e6

                axR.plot(t, y, R_um, label = f"Bubble {u}")
                axX.plot(t, y, x_um, label = f"Bubble {u}")

                if u < UPS-1:
                    R_um_next = db.dense_state[:npts, 0, sys_id, u+1] * R0[sys_id, u]
                    x_um_next = db.dense_state[:npts, 1, sys_id, u+1] * LR * 1e6

                    axD.plot(t, y, R_um_next + R_um, label=f"R{u+1} + R{u}")
                    axD.plot(t, y, x_um_next - x_um, label=f"x{u+1} - x{u}")

        plt.show()

    if config.save_results:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        out_dir = Path(__file__).resolve().parent / "CKM1D_ParameterStudy"
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



