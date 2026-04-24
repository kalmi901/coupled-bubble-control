from __future__ import annotations
import numpy as np
import numba as nb
from numba import cuda
from typing import Dict, Optional

import matplotlib.pyplot as plt

KERNEL_KWARGS = {
    "opt" : True,
    "fastmath": False
}

# Cash–Karp konstansok (tuple literal – biztonságos CUDA-n)
A  = (1.0/5.0, 3.0/10.0, 3.0/5.0, 1.0, 7.0/8.0)
B21 = 1.0/5.0
B31 = (3.0/40.0, 9.0/40.0)
B41 = (3.0/10.0, -9.0/10.0, 6.0/5.0)
B51 = (-11.0/54.0, 5.0/2.0, -70.0/27.0, 35.0/27.0)
B61 = (1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0)

C  = (37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0)
CP = (2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 1.0/4.0)
CE = (C[0]-CP[0], C[1]-CP[1], C[2]-CP[2], C[3]-CP[3], C[4]-CP[4], C[5]-CP[5])

#  ODE Status
ACTIVE              = 0             # The solver is currently stepping, end time not yet reached.
SUCCESS             = 1             # The integration reached the requested end time successfully.
EVENT_TERMINAL      = 2             # A user-defined event triggered a stop before reaching the end time.
MAX_STEPS_REACHED   = -1            # The solver reached the maximum number of allowed steps.
STEP_SIZE_LIMIT     = -2            # Step size became too small to maintain precision (e.g., Stiffness issues)
CONVERGENCE_FAILURE = -3            # Step size became too small due to linalg iteration failures
SINGULARITY_OR_NAN  = -4            # The actual state or timestep returned NaN or Inf (Numerical instability)


def solver(
    kernel,     # numba.cuda.jit() # kernel
    d_buffers,
    model_spec,
    execution_spec,
    solver_spec,
    max_steps: int,
    kernel_steps: int = 2048   # 2048
):
    #print("ich bin numba solver")
    #print(execution_spec)

    threads = execution_spec.block_size
    blocks = execution_spec.grid_size

    #print("device dense_state")
    #print(d_buffers.dense.dense_state.copy_to_host())

    print("threads", threads)
    print("blocks", blocks)

    max_launches = (max_steps + kernel_steps - 1) // kernel_steps
    for launch in range(max_launches):
        print("iter-nb", launch+1)

        kernel[blocks, threads](
            kernel_steps,
            # state 
            d_buffers.state.actual_time,
            d_buffers.state.time_end,
            d_buffers.state.time_begin,
            d_buffers.state.time_step,
            d_buffers.state.actual_state,
            # parameters
            d_buffers.params.unit_params,
            d_buffers.params.system_params,
            d_buffers.params.control_params,
            d_buffers.params.coupling_matrices,
            # dense output
            d_buffers.dense.dense_index,
            d_buffers.dense.dense_time,
            d_buffers.dense.dense_state,
            # status buffers
            d_buffers.status.actual_event,
            d_buffers.status.status_flags,
        )
        cuda.synchronize()
        print(d_buffers.state.actual_time.copy_to_host())
        print(d_buffers.state.time_step.copy_to_host())
        dense_index = d_buffers.dense.dense_index.copy_to_host()
        actual_event = d_buffers.status.actual_event.copy_to_host()
        print(dense_index)
        print(actual_event)
        if np.all(d_buffers.status.status_flags.copy_to_host() != 0):
            print("all-done")
            break
        #input()

    #print("device dense_time")
    #print(d_buffers.dense.dense_time.copy_to_host())
    #print("device dense_state")
    #print(d_buffers.dense.dense_state.copy_to_host())

    # Debug print delete later --> 
    dense_time = d_buffers.dense.dense_time.copy_to_host()
    dense_index = d_buffers.dense.dense_index.copy_to_host()
    dense_state = d_buffers.dense.dense_state.copy_to_host()

    print(dense_time)
    print(dense_index)
    #print(dense_state)

    SYS = 0
    UNIT = 0

    plt.figure("NUMBA1")
    plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 0, SYS, UNIT], 'g-')
    plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 0, SYS, UNIT+1], 'b-')
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 0, SYS, UNIT+2], 'r-')

    SYS = 0
    UNIT = 0

    plt.figure("NUMBA2")
    plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 1, SYS, UNIT], 'g-')
    plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 1, SYS, UNIT+1], 'b-')
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 1, SYS, UNIT+2], 'r-')
    plt.show()



def make_kernel(
        ode_fun,   #cuda.jit(device=True, ...)
        event_fun, #cuda.jit(device=True, ...)
        model_spec, execution_spec, solver_spec):
    
    # --- Model Configuration ---
    SD = model_spec.sd              # (unit) System dimension
    NUP = model_spec.nup            # Number of unit parameters
    NSP = model_spec.nsp            # System parameters
    NCP = model_spec.ncp            # Number of control parameters
    NK  = model_spec.nk             # Number of components (k - index in paper)
    NE  = model_spec.ne             # NUmber of Events
    # TODO:
    NCF = 4     # Number of coupling factors
    NCT = 8     # Number of coupling terms

    # --- System Configuration ---
    NS = execution_spec.ns          # Number of Systems
    SPB = execution_spec.spb        # Systems per block
    TILE = execution_spec.tile      # Treads per system (with padding)
    UPS = execution_spec.ups        # Units per system (real)
    NDO = execution_spec.ndo        # Number of stored dense output

    # --- Solver Configuration ---
    MIN_STEP = solver_spec.min_step
    MAX_STEP = solver_spec.max_step
    ATOL = solver_spec.atol
    RTOL = solver_spec.rtol

    ETOL = 1e-6

    # ----- Helper Functions ------
    @cuda.jit(device=True, **KERNEL_KWARGS, inline=True)
    def store_dense_output(
        valid_lane, l_active,
        gsid,
        lsid, luid,
        l_actual_time,
        l_time_end,
        l_actual_state,
        s_dense_output_store,   # Flag / indices
        s_dense_output_time_index,
        s_dense_output_time,
        s_dense_output_min_time_step,
        g_dense_time,
        g_dense_state,
    ):
        # per-system decision and timestep store
        if valid_lane and l_active and luid == 0:
            #print("store-dense")
            # System leader
            do_store   = -1
            #store_idx  = -1
            buffer_idx = s_dense_output_time_index[lsid]
            #print("buffer_idx", buffer_idx)
            min_time   = s_dense_output_time[lsid]
            min_timestep = s_dense_output_min_time_step[lsid]
            #print("min_time", min_time, "l_actual_time", l_actual_time)
            #print("min_timestep", min_timestep, "l_actual_time", l_actual_time)
            # Check storage capacity and time-interval
            if buffer_idx < NDO and min_time <= l_actual_time:
                do_store = buffer_idx
                # -- Store time instance --
                g_dense_time[buffer_idx, gsid] = l_actual_time
                s_dense_output_time_index[lsid] = buffer_idx + 1    # New index
                s_dense_output_store[lsid] = do_store
                # -- Propose next time --
                s_dense_output_time[lsid] = min(l_actual_time + min_timestep, l_time_end)
        cuda.syncthreads()  # type: ignore
        # per-lane store
        store_idx = s_dense_output_store[lsid]
        if valid_lane and l_active and store_idx != -1:
            for j in range(SD):
                g_dense_state[store_idx, j, gsid, luid] = l_actual_state[j]


    # --- Main Kernel function ---
    @cuda.jit(**KERNEL_KWARGS)
    def rkck45_coupled_solver(
        kernel_steps,
        # state
        g_actual_time,
        g_time_end,
        g_time_begin,
        g_time_step,
        g_actual_state,
        # parameters
        g_unit_params,
        g_system_params,
        g_control_params,
        g_coupling_matrices,
        # dense buffers
        g_dense_index,
        g_dense_time,
        g_dense_state,
        # status buffers
        g_actual_event,
        g_status_flags
                ):
        # ----  Thread Management ----
        gtid = cuda.grid(1)         # type: ignore Global Thread ID 
        tid  = cuda.threadIdx.x     # Local Thread ID
        bid  = cuda.blockIdx.x      # Block ID

        lsid = tid // TILE          # Local system ID within the block
        luid = tid % TILE           # Local unit ID within the system
        gsid = bid * SPB + lsid     # Global system ID

        valid_system = (gsid < NS)
        valid_unit   = (luid < UPS)
        valid_lane   = valid_system and valid_unit

        # ---- System Management ----
        s_terminated_system = cuda.shared.array((SPB,), dtype=nb.int32)      # type: ignore
        s_all_terminated    = cuda.shared.array((1,), dtype=nb.int32)        # type: ignore  (including invalid systems)
        if tid == 0:
            s_all_terminated[0] = 0     # type: ignore
        cuda.syncthreads()              # type: ignore

        #print("gtid", gtid, "tid", tid, "bid", bid, "lsid", lsid, "luid", luid, "gsid", gsid, "valid_system", int(valid_system), "valid_unit", int(valid_unit), "valid_lane", int(valid_lane))

        # -- GLOBAL LOAD --
        # ---- Load Env Specific Data ----
        s_actual_time = cuda.shared.array((SPB,), dtype=nb.float64)     # type: ignore
        s_time_end = cuda.shared.array((SPB,), dtype=nb.float64)        # type: ignore
        s_time_step = cuda.shared.array((SPB,), dtype=nb.float64)       # type: ignore
        s_system_params = cuda.shared.array((NSP, SPB), dtype=nb.float64)       # type: ignore
        s_control_params = cuda.shared.array((NCP, SPB),dtype=nb.float64)       # type: ignore
        if NDO > 0:
            s_dense_output_time_index = cuda.shared.array((SPB, ), dtype=nb.int32)      # type: ignore
            s_dense_output_store      = cuda.shared.array((SPB, ), dtype=nb.int32)      # type: ignore
            s_dense_output_time       = cuda.shared.array((SPB, ), dtype=nb.float64)    # type: ignore
            s_dense_output_min_time_step = cuda.shared.array((SPB,), dtype=nb.float64)  # type: ignore

        if NE > 0:
            s_actual_event_value    = cuda.shared.array((SPB,), dtype=nb.float64)    # type: ignore
            s_next_event_value      = cuda.shared.array((SPB,), dtype=nb.float64)    # type: ignore
            s_event_detected        = cuda.shared.array((SPB,), dtype=nb.int32)      # type: ignore
            s_event_ratio           = cuda.shared.array((SPB,), dtype=nb.float64)    # type: ignore

        if tid < SPB:
            gid = bid * SPB + tid
            if gid < NS:
                actual_time = g_actual_time[gid]
                time_end    = g_time_end[gid]
                time_begin  = g_time_begin[gid]
                terminated  = (actual_time >= time_end)

                s_actual_time[tid] = actual_time   # type: ignore
                s_time_end[tid]    = time_end      # type: ignore
                s_time_step[tid]   = g_time_step[gid]     # type: ignore

                s_terminated_system[tid] = int(terminated)      # type: ignore
                g_status_flags[gtid]     = int(terminated)      # type: ignore

                if NDO > 0:
                    s_dense_output_time_index[tid] = g_dense_index[gid]      # type: ignore
                    s_dense_output_time[tid]       = actual_time             # type: ignore
                    s_dense_output_store[tid]      = 0                       # type: ignore
                    if NDO == 1:
                        s_dense_output_min_time_step[tid] = (time_end - time_begin)                # type: ignore
                    else:
                        s_dense_output_min_time_step[tid] = (time_end - time_begin) / (NDO - 1)    # type: ignore

                if NE > 0:
                    # Load Event Buffer and terminate system if event has already reached
                    actual_event = g_actual_event[gid]
                    terminated   = terminated or (abs(actual_event) < ETOL)
                    s_actual_event_value[tid] = actual_event        # type: ignore
                    s_terminated_system[tid]  = int(terminated)     # type: ignore
                    g_status_flags[gtid]      = int(terminated) * EVENT_TERMINAL    # type: ignore
            else:
                s_terminated_system[tid] = 1                         # type: ignore
        
        cuda.syncthreads()  # type: ignore
        
        if tid < SPB:
            cuda.atomic.add(s_all_terminated, 0, s_terminated_system[tid])    # type: ignore
        cuda.syncthreads()                                      # type: ignore

        all_terminated = (s_all_terminated[0] >= SPB)           # type: ignore
        #if luid == 0:
        #    print("gsid", gsid, "all_terminated", int(all_terminated), "s_all_terminated", s_all_terminated[0]) # type: ignore
        
        #if all_terminated:
        #    print(gtid, "early-return")
        #    return


        # --- System Parameters ---
        for idx in range(tid, SPB * NSP, cuda.blockDim.x):      # type: ignore
            s = idx // NSP
            j = idx % NSP
            g = bid * SPB + s
            if g < NS:
                s_system_params[j, s] = g_system_params[j, g]   # type: ignore

        # --- Glocal Parameters ---
        for idx in range(tid, SPB * NCP, cuda.blockDim.x):      # type: ignore
            s = idx // NCP
            j = idx % NCP
            g = bid * SPB + s
            if g < NS:
                s_control_params[j, s]  = g_control_params[j, g]  # type: ignore  

        cuda.syncthreads()  # type: ignore
        
        # --- Load Unit Specific Data ---
        l_actual_state  = cuda.local.array((SD,), dtype=nb.float64)  # type: ignore
        l_unit_params   = cuda.local.array((NUP,), dtype=nb.float64)  # type: ignore
        l_system_params = cuda.local.array((NSP,), dtype=nb.float64)  # type: ignore
        l_control_params= cuda.local.array((NCP, ), dtype=nb.float64)  # type: ignore
        
        l_actual_time   = 0.0
        l_time_end      = 0.0
        l_time_step     = 0.0
        l_active        = False

        if valid_lane:
            l_actual_time = s_actual_time[lsid]     # type: ignore
            l_time_end    = s_time_end[lsid]        # type: ignore 
            l_time_step   = s_time_step[lsid]       # type: ignore
            #l_active      = l_actual_time < l_time_end
            l_active = valid_lane and (s_terminated_system[lsid] == 0)  # type: ignore TODO erre

            for j in range(NSP):
                l_system_params[j] = s_system_params[j, lsid]       # type: ignore

            for j in range(NCP):
                l_control_params[j] = s_control_params[j, lsid]     # type: ignore
            
            for j in range(SD):
                l_actual_state[j] = g_actual_state[j, gsid, luid]     # type: ignore

            for j in range(NUP):
                l_unit_params[j] = g_unit_params[j, gsid, luid]       # type: ignore

        cuda.syncthreads()  # type: ignore
            
        #if gtid == -1:
        #    print("l_actual_time", l_actual_time)
        #    print("l_time_end", l_time_end)
        #    print("l_time_step", l_time_step)
        #    print("l_active", int(l_active) )

        #    for j in range(SD):
        #        print("l_actual_state[", j, "] =", l_actual_state[j])   # type: ignore

        #    for j in range(NSP):
        #        print("l_system_params[", j, "] =", l_system_params[j])   # type: ignore

        #    for j in range(NCP):
        #        print("l_control_params[", j, "] =", l_control_params[j])   # type: ignore

        #    for j in range(NUP):
        #        print("l_unit_params[", j, "] =", l_unit_params[j])   # type: ignore

        # -- INITIALIZE WORK ARRAYS --
        k1 = cuda.local.array((SD,), dtype=nb.float64)      # type: ignore
        k2 = cuda.local.array((SD,), dtype=nb.float64)      # type: ignore
        k3 = cuda.local.array((SD,), dtype=nb.float64)      # type: ignore
        k4 = cuda.local.array((SD,), dtype=nb.float64)      # type: ignore
        k5 = cuda.local.array((SD,), dtype=nb.float64)      # type: ignore
        k6 = cuda.local.array((SD,), dtype=nb.float64)      # type: ignore

        l_temp_state = cuda.local.array((SD,), dtype=nb.float64)  # type: ignore
        l_new_state = cuda.local.array((SD,), dtype=nb.float64)   # type: ignore

        sh_ratio = cuda.shared.array((SPB, TILE), dtype=nb.float64)         # type: ignore
        sh_ok    = cuda.shared.array((SPB, TILE), dtype=nb.int32)           # type: ignore

        # -- Coupling factors and Coupling terms --
        l_coupling_factors = cuda.local.array((NCF,), dtype=nb.float64)               # type: ignore
        s_coupling_terms   = cuda.shared.array((NCT, SPB, TILE), dtype=nb.float64)    # type: ignore

        # -- Linalg work arrays --
        s_lin_converged   = cuda.shared.array((SPB,), dtype=nb.int32)                # type: ignore
        s_all_converged   = cuda.shared.array(( 1, ), dtype=nb.int32)                # type: ignore
        s_norm_temp       = cuda.shared.array((SPB,), dtype=nb.float64)              # type: ignore
        s_vector_temp     = cuda.shared.array((2, SPB, TILE), dtype=nb.float64)      # type: ignore
        
        # -- CASK-KARP COEFFICIENTS
        a2, a3, a4, a5, a6 = A
        c1, c2, c3, c4, c5, c6 = C
        ce1, ce2, ce3, ce4, ce5, ce6 = CE

        b21 = B21
        b31, b32 = B31
        b41, b42, b43 = B41
        b51, b52, b53, b54 = B51
        b61, b62, b63, b64, b65 = B61

        # Store Initial Conditions as the first dense output
        if NDO > 0:
            store_dense_output(
                valid_lane, l_active,
                gsid,
                lsid, luid,
                l_actual_time, l_time_end, l_actual_state,
                s_dense_output_store,
                s_dense_output_time_index,
                s_dense_output_time,
                s_dense_output_min_time_step,
                g_dense_time,
                g_dense_state)
            
        # Initial Event value
        if NE > 0:
            event_fun(
                valid_lane, l_active,
                gsid,
                lsid, luid,
                s_actual_event_value,
                l_actual_time,
                l_actual_state,
                l_unit_params, l_system_params, l_control_params)

            cuda.syncthreads()  # type: ignore
        # ------------------------------------------------------------
        # --------------------- MAIN KERNEL LOOP ---------------------
        # ------------------------------------------------------------
        for step in range(kernel_steps):
            #if gtid == 0:
            #    print("step", step)
            # -- Clamping time-step --
            if valid_lane and l_active:
                if l_time_step < MIN_STEP:
                    l_time_step = MIN_STEP
                if l_time_step > MAX_STEP:
                    l_time_step = MAX_STEP
                l_remain = l_time_end - l_actual_time
                if l_time_step > l_remain:
                    l_time_step = l_remain

            # ---------------------------------------------------
            # ------------------- RKCK-Stages -------------------
            # ---------------------------------------------------
            # ------ k1 -------
            ode_fun(valid_lane, l_active,
                    gsid,
                    lsid, luid,
                    l_actual_time, 
                    l_actual_state,
                    k1,
                    l_unit_params, l_system_params, l_control_params,
                    l_coupling_factors, s_coupling_terms,
                    g_coupling_matrices,
                    # Linalg Solver
                    s_lin_converged, s_all_converged,
                    s_norm_temp, s_vector_temp)

            # ------ k2 -------
            l_temp_time = l_actual_time + a2 * l_time_step
            for j in range(SD):
                l_temp_state[j] = l_actual_state[j] + l_time_step * (b21 * k1[j])       # type: ignore
            
            ode_fun(valid_lane, l_active,
                    gsid,
                    lsid, luid,
                    l_temp_time,
                    l_temp_state,
                    k2,
                    l_unit_params, l_system_params, l_control_params,
                    l_coupling_factors, s_coupling_terms,
                    g_coupling_matrices,
                    # Linalg Solver
                    s_lin_converged, s_all_converged,
                    s_norm_temp, s_vector_temp)

            # ------ k3 -------
            l_temp_time = l_actual_time + a3 * l_time_step
            for j in range(SD):
                l_temp_state[j] = l_actual_state[j] + l_time_step * (b31 * k1[j] + b32 * k2[j])     # type: ignore

            ode_fun(valid_lane, l_active,
                    gsid,
                    lsid, luid,
                    l_temp_time,
                    l_temp_state,
                    k3,
                    l_unit_params, l_system_params, l_control_params,
                    l_coupling_factors, s_coupling_terms,
                    g_coupling_matrices,
                    # Linalg Solver
                    s_lin_converged, s_all_converged,
                    s_norm_temp, s_vector_temp)

            # ------ k4 -------
            l_temp_time = l_actual_time + a4 * l_time_step
            for j in range(SD):
                l_temp_state[j] = l_actual_state[j] + l_time_step * (b41 * k1[j] + b42 * k2[j] + b43 * k3[j])   # type: ignore

            ode_fun(valid_lane, l_active,
                    gsid,
                    lsid, luid,
                    l_temp_time,
                    l_temp_state,
                    k4,
                    l_unit_params, l_system_params, l_control_params,
                    l_coupling_factors, s_coupling_terms,
                    g_coupling_matrices,
                    # Linalg Solver
                    s_lin_converged, s_all_converged,
                    s_norm_temp, s_vector_temp)

            # ------ k5 -------
            l_temp_time = l_actual_time + a5 * l_time_step
            for j in range(SD):
                l_temp_state[j] = l_actual_state[j] + l_time_step * (b51 * k1[j] + b52 * k2[j] + b53 * k3[j] + b54 * k4[j])     # type: ignore

            ode_fun(valid_lane, l_active,
                    gsid,
                    lsid, luid,
                    l_temp_time,
                    l_temp_state,
                    k5,
                    l_unit_params, l_system_params, l_control_params,
                    l_coupling_factors, s_coupling_terms,
                    g_coupling_matrices,
                    # Linalg Solver
                    s_lin_converged, s_all_converged,
                    s_norm_temp, s_vector_temp)
            
            # ------ k6 -------
            l_temp_time = l_actual_time + a6 * l_time_step
            for j in range(SD):
                l_temp_state[j] = l_actual_state[j] + l_time_step * (b61 * k1[j] + b62 * k2[j] + b63 * k3[j] + b64 * k4[j] + b65 * k5[j])   # type: ignore

            ode_fun(valid_lane, l_active,
                    gsid,
                    lsid, luid,
                    l_temp_time,
                    l_temp_state,
                    k6,
                    l_unit_params, l_system_params, l_control_params,
                    l_coupling_factors, s_coupling_terms,
                    g_coupling_matrices,
                    # Linalg Solver
                    s_lin_converged, s_all_converged,
                    s_norm_temp, s_vector_temp)

            # --- NEW CANDIDATE STATE ---
            for j in range(SD):
                l_new_state[j] = l_actual_state[j] + l_time_step * (c1 * k1[j] + c2 * k2[j] + c3 * k3[j] + c4 * k4[j] + c5 * k5[j] + c6 * k6[j])   # type: ignore

            # --- NEW EVENT VALUE ---
            if NE > 0:
                event_fun(
                    valid_lane, l_active,
                    gsid,
                    lsid, luid,
                    s_next_event_value,
                    l_actual_time + l_time_step,
                    l_new_state,
                    l_unit_params, l_system_params, l_control_params)

            # ---------------------------------------------------
            # ---------------- Event Handling -------------------
            # ---------------------------------------------------
            if NE > 0 and tid < SPB:
                g_old = s_actual_event_value[tid]       # type: ignore
                g_new = s_next_event_value[tid]         # type: ignore
                #terminated = s_terminated_system[tid]   # type: ignore

                # Előjelváltás detektálása (0 átlépés)
                has_crossing = (g_old * g_new < 0.0)
                # Linear interpolation
                # h_new = h_old * ( |g_old| / (|g_old| + |g_new|) )
                # Biztonsági okokból 0.99-al szorozva, hogy ne lépjünk túl rajta a kerekítés miatt
                abs_old = abs(g_old)
                denom   = abs_old + abs(g_new)
                ratio_cal = (abs_old / (denom + 1e-15)) * 0.99
                s_event_detected[tid] = 1 if has_crossing else 0           # type: ignore
                s_event_ratio[tid]    = ratio_cal if has_crossing else 1   # type: ignore
            cuda.syncthreads()  # type: ignore

            # ---------------------------------------------------
            # ----------------- Error Handling ------------------
            # ---------------------------------------------------
            # -- Check Unit Tolerances --
            unit_ok = 1
            unit_ratio = 1.0e30
            sys_event_detected = (s_event_detected[lsid] == 1) if NE > 0 else False     # type: ignore
     
            if valid_lane and l_active and not sys_event_detected:
                for j in range(SD):
                    e = abs(l_time_step * (ce1 * k1[j] + ce2 * k2[j] + ce3 * k3[j] + ce4 * k4[j] + ce5 * k5[j] + ce6 * k6[j])) + 1.0e-18    # type: ignore
                    tol = max(RTOL * max(abs(l_new_state[j]), abs(l_actual_state[j])), ATOL)   # type: ignore
                    r = tol / e
                    if r < unit_ratio:
                        unit_ratio = r
                    if r < 1.0:
                        unit_ok = 0

                sh_ratio[lsid, luid] = unit_ratio   # type: ignore
                sh_ok[lsid, luid]    = unit_ok      # type: ignore
            cuda.syncthreads()                      # type: ignore
            
            #if luid == 0:
            #    print("gsid", gsid, "sh_ratio[0] =", sh_ratio[lsid][0], "sh_ratio[1] =", sh_ratio[lsid][1]) # type: ignore
            #    print("gsid", gsid, "sh_ok[0] =", sh_ok[lsid][0], "sh_ok[1] =", sh_ok[lsid][1]) # type: ignore

            # System Leader --> reduction
            if valid_lane and l_active and luid == 0:
                if sys_event_detected:
                    # Event Correction: reject last step and trunc the timestep
                    #sys_ok = 0
                    sys_ratio = s_event_ratio[lsid]                 # type: ignore Interpolated h value := h_new = h_old * ( |g_old| / (|g_old| + |g_new|) )
                    s_time_step[lsid]   = l_time_step * sys_ratio   # type: ignore
                    s_actual_time[lsid] = l_actual_time             # type: ignore
                    sh_ok[lsid, 0] = 0                              # type: ignore
                else:
                    # Normal Error handling
                    # Initial guess from local observations
                    sys_ok = unit_ok
                    sys_ratio = unit_ratio
                    # Check for neighbors
                    for u in range(1, UPS):
                        ok = sh_ok[lsid, u]     # type: ignore
                        rr = sh_ratio[lsid, u]  # type: ignore
                        if ok == 0:
                            sys_ok = 0
                        if rr < sys_ratio:
                            sys_ratio = rr

                    sh_ok[lsid, 0] = sys_ok          # type: ignore --> system-wide information!
                    
                    # New time step candidate
                    #fac_ok = 0.9 * sys_ratio**0.2
                    #fac_bad = 0.9 * sys_ratio**0.25

                    #if fac_ok < 0.2:
                    #    fac_ok = 0.2
                    #if fac_ok > 5.0:
                    #    fac_ok = 5.0
                    #if fac_bad < 0.1:
                    #    fac_bad = 0.1
                    #if fac_bad > 0.5:
                    #    fac_bad = 0.5

                    fac_ok = min(5.0, max(0.2, 0.9 * sys_ratio**0.2))
                    fac_bad = min(0.5, max(0.1, 0.9 * sys_ratio**0.25))

                    s_time_step[lsid]   = l_time_step * (fac_ok if sys_ok else fac_bad)     # type: ignore
                    s_actual_time[lsid] = l_actual_time + (l_time_step if sys_ok else 0.0)  # type: ignore
                    if NE > 0 and sys_ok:
                        s_actual_event_value[lsid] = s_next_event_value[lsid]               # type: ignore
            cuda.syncthreads()  # type: ignore

            # Accept new state, update
            if valid_lane and l_active:
                sys_ok = sh_ok[lsid, 0]     # type: ignore
                if sys_ok == 1:
                    #l_actual_time = l_actual_time + l_time_step
                    for j in range(SD):
                        l_actual_state[j] = l_new_state[j]  # type: ignore
                # Read new time-step candidate
                l_time_step   = s_time_step[lsid]       # type: ignore
                l_actual_time = s_actual_time[lsid]     # type: ignore
            cuda.syncthreads()  # type: ignore

            if NDO > 0:
                store_dense_output(
                    valid_lane, l_active,
                    gsid,
                    lsid, luid,
                    l_actual_time, l_time_end, l_actual_state,
                    s_dense_output_store,
                    s_dense_output_time_index,
                    s_dense_output_time,
                    s_dense_output_min_time_step,
                    g_dense_time,
                    g_dense_state)
                
            # Am I finished?
            #l_active = valid_lane and (l_actual_time < l_time_end)
            #print("gsid", gsid, "luid", int(l_active))

            if valid_system and l_active and luid == 0:
                #s_terminated_system[lsid] = int(s_actual_time[lsid] >= s_time_end[lsid])  # type: ignore
                if s_actual_time[lsid] >= s_time_end[lsid]:     # type: ignore
                    s_terminated_system[lsid] = 1               # type: ignore
                    g_status_flags[gsid]      = SUCCESS
                    cuda.atomic.add(s_all_terminated, 0, 1)     # type: ignore
                if NE > 0 and (s_terminated_system[lsid] == 0): # type: ignore
                    if(s_actual_event_value[lsid] < ETOL):      # type: ignore
                        #print("gsid", gsid, "event-terminated")
                        g_status_flags[gsid]      = EVENT_TERMINAL
                        s_terminated_system[lsid] = 1           # type: ignore
                        cuda.atomic.add(s_all_terminated, 0, 1) # type: ignore
            cuda.syncthreads()  # type: ignore

            #if tid == 0:
            #    nterm = 0
            #    for s in range(SPB):
            #        nterm += s_terminated_system[s]  # type: ignore
            #    s_all_terminated[0] = nterm          # type: ignore
            #cuda.syncthreads()  # type: ignore

            all_terminated = (s_all_terminated[0] >= SPB)  # type: ignore
            if all_terminated:
            #    print(gtid, "iter-break")
                break

            l_active = valid_lane and (s_terminated_system[lsid] == 0)  # type: ignore
            
            #print("gsid", gsid, "luid", int(l_active), "s_terminated_system[lsid]", s_terminated_system[lsid], "s_all_terminated[0]", s_all_terminated[0])
            


        # -- GLOBAL STORE --
        if valid_lane:
            if luid == 0:
                s_actual_time[lsid] = l_actual_time     # type: ignore
                s_time_step[lsid]   = l_time_step       # type: ignore

            for j in range(SD):
                g_actual_state[j, gsid, luid] = l_actual_state[j]      # type: ignore
            
        cuda.syncthreads()      # type: ignore

        if tid < SPB:
            gid = bid * SPB + tid
            g_actual_time[gid] = s_actual_time[tid]   # type: ignore
            g_time_step[gid]   = s_time_step[tid]     # type: ignore
            
            if NDO > 0:
                g_dense_index[gid] = s_dense_output_time_index[tid] # type: ignore

            if NE > 0:
                g_actual_event[gid] = s_actual_event_value[tid]     # type: ignore

        cuda.syncthreads()  # type: ignore


    return rkck45_coupled_solver