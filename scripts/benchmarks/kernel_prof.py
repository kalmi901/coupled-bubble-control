from __future__ import annotations
import tyro
import time
import benchmark_bootstrap
import numpy as np
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, List, Dict, Optional
from coupledbubble_control.models import CKM1D
from coupledbubble_control.backends import BackendName, KernelVariant, get_current_device_name
from coupledbubble_control.models import MaterialProperties

RandKey = Literal["R0", "PA", "FR", "X0"]

@dataclass(frozen=True)
class Sweep:
    """ Meghatároz egy paraméter: lehet fix érték vagy tartomány """
    values: List[float]
    unit_multiplier: float = 1.0
    randomize: bool = False

    @property
    def scaled(self) -> List[float]:
        return [v * self.unit_multiplier for v in self.values]


@dataclass(frozen=True)
class Scene:
    R0: List[Sweep]         # Bubble Size               [um]
    PA: List[Sweep]         # Pressure Amplitude        [bar]
    FR: List[Sweep]         # Excitation Frequency      [kHz]
    X0: List[Sweep]         # Intiail bubble positions  [-]
    T: float                # Time domain end           [-]

def create_scene(
        R0, PA, FR, X0, T,
        RAND_VALS: Optional[Dict[RandKey, List[int]]] = None,
        m_R0=1e-6, m_PA=1e5, m_FREQ=1e3, m_X0=1.0):
    
    RAND_VALS = RAND_VALS or {}

    def build_sweeps(raw_values, multiplier, key: RandKey):
        randomized_indices = set(RAND_VALS.get(key, []))
        sweeps = []
        for i, vals in enumerate(raw_values):
            sweeps.append(
                Sweep(
                    values = list(vals),
                    unit_multiplier = multiplier,
                    randomize=(i in randomized_indices)
                )
            )
        return sweeps

    return Scene(
        R0 = build_sweeps(R0, 1e-6, "R0"),
        PA = build_sweeps(PA, 1e5,  "PA"),
        FR = build_sweeps(FR, 1e3,  "FR"),
        X0 = build_sweeps(X0, 1.0,  "X0"),
        T = float(T)
    )


SCENE_DICT = {
    "2B_TYPICAL" : create_scene(
        R0 = [[60.0], [60.0]],
        PA = [[0.0, 1.0], [0.0, 1.0]],
        FR = [[25.0], [50.0]],
        X0 = [[-0.45, -0.05], [0.05, 0.45]],
        T = 5.0,
        RAND_VALS={"PA": [0, 1],
                  "X0": [0, 1]}
    ),
    "3B_TYPICAL" : create_scene(
        R0 = [[60.0], [60.0], [60.0]],
        PA = [[0.0, 1.0], [0.0, 1.0]],
        FR = [[25.0], [50.0]],
        X0 = [[-0.45, -0.2], [-0.15, 0.15], [0.2, 0.45]],
        T = 5.0,
        RAND_VALS={"PA": [0, 1],
                   "X0": [0, 1, 2]}
    ),
    "4B_TYPICAL" : create_scene(
        R0 = [[60.0], [60.0], [60.0], [60.0]],
        PA = [[0.0, 1.0], [0.0, 1.0]],
        FR = [[25.0], [50.0]],
        X0 = [[-0.45, -0.3], [-0.25, -0.025], 
              [0.025, 0.25], [0.3, 0.45]],
        T = 5.0,
        RAND_VALS={"PA": [0, 1],
                   "X0": [0, 1, 2, 3]}
    )
}


def build_parameter_ranges(config, scene_parameters):
    NS = config.num_systems
    UPS = len(scene_parameters.R0)
    NK  = len(scene_parameters.PA)
    rng = np.random.default_rng(config.seed)


    def get_values(sweep_obj: Sweep, num_systems: int):
        vals = sweep_obj.values
        randomize = sweep_obj.randomize
        if len(vals) == 1:
            # Fix érték --> repeat x num_systems
            return np.full((num_systems,), vals[0], dtype=np.float64)
        elif len(vals) == 2:
            if not randomize:
                # Két érték van, nem random, lineáris rácsot generálunk
                return np.linspace(vals[0], vals[1], num_systems, endpoint=True, dtype=np.float64)
            return rng.uniform(vals[0], vals[1], size=(num_systems,)).astype(np.float64)
        else:
            raise ValueError(
                f"Expected 1 or 2 values in Sweep, got {len(vals)}: {sweep_obj.values}"
            )

    # R0, Shape (NS, UPS)
    R0 = np.empty((NS, UPS), dtype=np.float64)
    X0 = np.empty((NS, UPS), dtype=np.float64)
    for u in range(UPS):
        R0[:, u] = get_values(
                    scene_parameters.R0[u],
                    NS)
        
        X0[:, u] = get_values(
                    scene_parameters.X0[u],
                    NS)

    # PA, Shape (NK, NS)
    PA = np.empty((NK, NS), dtype=np.float64)
    FR = np.empty((NK, NS), dtype=np.float64)
    PS = np.zeros((NK, NS), dtype=np.float64)
    for k in range(NK):
        PA[k, :] = get_values(
                    scene_parameters.PA[k],
                    NS)
        FR[k, :] = get_values(
                    scene_parameters.FR[k],
                    NS)

    return R0, X0, PA, FR, PS


@dataclass
class ProfilerConfig:
    scene_id: str = "4B_TYPICAL"
    backend: Literal["numba", "cupy"] = "numba"
    variant: Literal["shared", "warp"] = "warp"
    num_systems: int = 4096
    systems_per_block: int = 64
    kernel_steps: int = 32
    max_kernel_steps: int = 100000
    seed: int = 11
    warm_up_steps: int = 1
    measured_steps: int = 10
    debug: bool = False
    # CUDA-OPTS
    cuda_mode: Literal["debug", "profile", "release"] = "profile"
    compiler: Literal["nvrtc", "nvcc"] = "nvrtc"
    max_registers: int = 128
    fastmath: bool = True
    save_file_name: Optional[str] = None

@dataclass
class StepStats:
    step_idx: int
    launch_times_s: List

    def __post_init__(self):
        self.num_launches = len(self.launch_times_s)
        self.step_time_s  = float(np.sum(self.launch_times_s))
        self.mean_launch_s = float(np.mean(self.launch_times_s))
        self.std_launch_s  = float(np.std(self.launch_times_s))
        self.min_launch_s  = float(np.min(self.launch_times_s))
        self.max_launch_s  = float(np.max(self.launch_times_s))

@dataclass
class Summary:
    steps: List[StepStats]
    total_time: float
    def __post_init__(self):
        step_times = np.array([s.step_time_s for s in self.steps])
        launches = np.array([s.num_launches for s in self.steps])
        all_launch_times = np.concatenate([np.array(s.launch_times_s) for s in self.steps]) if self.steps else np.array([])

        self.total_steps = len(self.steps)
        self.total_wall_time_s = self.total_time
        self.mean_step_time_s = step_times.mean()
        self.std_step_time_s = step_times.std()
        self.min_step_time_s = step_times.min()
        self.max_step_time_s = step_times.max()
        self.mean_launches_per_step = launches.mean()
        self.mean_launch_time_s = all_launch_times.mean() if all_launch_times.size else 0.0
        self.std_launch_time_s = all_launch_times.std() if all_launch_times.size else 0.0


def print_header(config, scene_parameters, device):
    print("=" * 60)
    print("CKM1D Solver Benchmark")
    print("-" * 60)
    print(f"Scene              : {config.scene_id}")
    print(f"Backend / Variant  : {config.backend} / {config.variant}")
    print(f"Num systems        : {config.num_systems}")
    print(f"Systems per block  : {config.systems_per_block}")
    print(f"Kernel steps       : {config.kernel_steps}")
    print(f"Max kernel steps   : {config.max_kernel_steps}")
    print(f"Time horizon (T)   : {scene_parameters.T}")
    print(f"Device             : {device}")
    print("=" * 60)
    print("")

def print_step(step_idx, total, stats):
    print(
        f"[{step_idx:02d}/{total:02d}] "
        f"launches={stats.num_launches:2d} | "
        f"step={stats.step_time_s:8.5f} s | "
        f"mean={stats.mean_launch_s:8.5f} s | "
        f"std={stats.std_launch_s:8.5f} s"
    )

def print_summary(summary, sim_T):
    print("")
    print("=" * 60)
    print("Summary")
    print("-" * 60)

    print(f"Total measured steps     : {summary.total_steps}")
    print(f"Total wall time [s]      : {summary.total_wall_time_s:.5f}")

    print("")
    print("Step statistics:")
    print(f"  mean [s]               : {summary.mean_step_time_s:.5f}")
    print(f"  std  [s]               : {summary.std_step_time_s:.5f}")
    print(f"  min  [s]               : {summary.min_step_time_s:.5f}")
    print(f"  max  [s]               : {summary.max_step_time_s:.5f}")

    print("")
    print("Kernel launch statistics:")
    print(f"  mean launches / step   : {summary.mean_launches_per_step:.2f}")
    print(f"  mean launch time [s]   : {summary.mean_launch_time_s:.6f}")
    print(f"  std  launch time [s]   : {summary.std_launch_time_s:.6f}")

    # Simulation speed
    if summary.mean_step_time_s > 0:
        sim_speed = sim_T / summary.mean_step_time_s
        print("")
        print("Simulation speed:")
        print(f"  sim-time / wall-second : {sim_speed:.3f}")
        print(f"  wall-sec / sim-second  : {1.0 / sim_speed:.3f}")

    print("=" * 60)


def append_summary(config, summary, device):
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "simu_times" / device
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / config.save_file_name
    exists = file_path.exists()

    row = {
        "scene_id": config.scene_id,
        "backend": config.backend,
        "variant": config.variant,
        "num_systems": config.num_systems,
        "systems_per_block": config.systems_per_block,
        "kernel_steps": config.kernel_steps,
        "max_registers": config.max_registers,
        "seed": config.seed,
        "mean_step_time_s": summary.mean_step_time_s,
        "std_step_time_s": summary.std_step_time_s,
        "mean_launch_time_s": summary.mean_launch_time_s,
        "mean_launches_per_step": summary.mean_launches_per_step,
    }

    with file_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    config = tyro.cli(ProfilerConfig)
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
    NDO     = 0

    R0, X0, \
    PA, FR, PS \
        = build_parameter_ranges(config, scene_parameters)
    
    REF_FREQ = scene_parameters.FR[0].values[0]

    device = get_current_device_name()

    print_header(config, scene_parameters, device)
    #print("R0 \n", R0[:100])
    #print("X0 \n", X0[:100])

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
        cuda_mode = config.cuda_mode,
        compiler = config.compiler,
        max_registers = config.max_registers,
        fastmath = config.fastmath
    )

    # UPDATE HOST BUFFERS
    model.runtime.host_buffers.state.time_end[:] = scene_parameters.T
    model.runtime.host_buffers.state.actual_state[1, :, :UPS] = X0

    # RUN SIMULATIONS
    start_time = time.time()
    print("Warm-up started")
    for step in range(1,config.warm_up_steps+1):
        print(f"Step: {step:d}/{config.warm_up_steps}")
        run_times = model.solve(
            max_steps = config.max_kernel_steps,
            kernel_steps = config.kernel_steps,
            benchmark = True,
            debug = config.debug,
            sync_to_host=False,
            sync_from_host=True
            )
    print("Warm-up finished\n")

    print("Measurement started")
    step_stats = []
    start_time = time.time()
    for step in range(1, config.measured_steps+1):
        print(f"Step: {step:d}/{config.measured_steps}")
        run_times = model.solve(
            max_steps = config.max_kernel_steps,
            kernel_steps = config.kernel_steps,
            benchmark = True,
            debug = config.debug,
            sync_to_host=False,
            sync_from_host=True
            )
        stats = StepStats(step, run_times)
        step_stats.append(stats)
        print_step(step, config.measured_steps, stats)

    end_time = time.time()

    summary = Summary(step_stats, end_time - start_time)
    print_summary(summary, scene_parameters.T)

    if config.save_file_name is not None:
        append_summary(config, summary, device)
