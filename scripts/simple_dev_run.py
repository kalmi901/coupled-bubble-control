from __future__ import annotations
import dev_bootstap
from coupledbubble_control.models import CKM1D
import numpy as np
import matplotlib.pyplot as plt

from coupledbubble_control.models import MaterialProperties
from coupledbubble_control.backends import BackendName, KernelVariant
P0 = MaterialProperties.P0

if __name__ == "__main__":
    
    NS = 3
    SPB = 2
    UPS = 2
    NK = 1
    AC = "CONST"
    NDO = 4096
    BACKEND = BackendName.CUPY
    VARIANT = KernelVariant.SHARED

    R0 = np.repeat(
        np.linspace(10, 100, NS, endpoint=True, dtype=np.float64),
        UPS).reshape(NS, UPS)

    #R0[:, 0] = 7.0
    #R0[:, 1] = 5.0
    R0[:, 0] = 6.0
    R0[:, 1] = 5.0
    #R0[:, 2] = 6.0
    print(R0)
    input()

    # Shapes (K, NUM_ENVS)
    PA = np.stack([
        np.full((NS,), -1.20 * P0*1e-5, dtype=np.float64),
        #np.full((NS,), 0.0, dtype=np.float64)
    ], axis=0)

    FR = np.stack([
        np.full((NS,), 20.0, dtype=np.float64),
        #np.full((NS,), 40.0, dtype=np.float64)
    ], axis=0)

    PS = np.zeros((NK, NS), dtype=np.float64)
    LR   = 1500 / (20 * 1000)      # Reference Lenght (m)

    print("PA", PA)
    print("FREQ", FR)

    model = CKM1D(
        num_systems = NS,
        systems_per_block = SPB,
        units_per_system = UPS,
        num_components = NK,
        acoustic_field = AC,
        num_stored_points = NDO,
        R0 = R0,
        PA = PA,
        FR = FR,
        PS = PS,
        REF_FREQ = 20.0,
        backend = BACKEND,
        variant = VARIANT,
        # CUDA-OPTS
        cuda_mode = "debug",
        compiler = "nvrtc",
        max_registers = 128,
        fastmath = True
    )

    model.runtime.host_buffers.state.time_end[:] = 8.0
    #model.runtime.host_buffers.state.actual_state[0, :, :] = [7, 8]
    #model.runtime.host_buffers.state.actual_state[1, :, 1] = 200 * 1e-6 / LR
    model.runtime.host_buffers.state.actual_state[1, :, 1] = 300 * 1e-6 / LR
    #model.runtime.host_buffers.state.actual_state[1, :, 2] = 600 * 1e-6 / LR
    #model.runtime.host_buffers.state.actual_state[2, :, :] = [3, 6]
    #model.runtime.host_buffers.state.actual_state[3, :, :] = [2, 4]

    #print(model.runtime.host_buffers.params.unit_params)
    #print(model.runtime.host_buffers.state.actual_state[0])
    #print(model.system_params)
    #print(model.coupling_matrices)
    #print(model.control_params)

    kernel_times = model.solve(benchmark=True, debug=True)
    #print(model.runtime.device_buffers.params.unit_params[0, 0, 0]) #type: ignore
    #print(model.runtime.host_buffers.params.unit_params[:, 0, 0])
    #print(model.runtime.host_buffers.params.unit_params.shape)
    print(kernel_times)
    #print(model.state_shape)