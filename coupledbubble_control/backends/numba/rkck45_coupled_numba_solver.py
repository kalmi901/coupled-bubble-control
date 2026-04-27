from __future__ import annotations
import numpy as np
from numba import cuda
import time

import matplotlib.pyplot as plt

def solver(
    kernel,     #numba.cuda.jit()
    d_buffers,
    model_spec,
    execution_spec,
    solver_spec,
    max_steps: int,
    kernel_steps: int = 2048,
    benchmark:bool = False,
    debug: bool = False
):

    threads = execution_spec.block_size
    blocks = execution_spec.grid_size

    if debug:
        print("NUMBA.CUDA kernel")
        print("threads", threads)
        print("blocks", blocks)
    kernel_times = [] if benchmark else None
    cuda.synchronize()
    max_launches = (max_steps + kernel_steps - 1) // kernel_steps
    for launch in range(max_launches):
        #if debug:
        print("iter-nb", launch+1)

        t0 = time.time()
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
        t1 = time.time()
        if benchmark:
            kernel_times.append(t1 - t0)    # type: ignore

        if debug:
            actual_time =  d_buffers.state.actual_time.copy_to_host()
            time_step = d_buffers.state.time_step.copy_to_host()
            print("actual_time", actual_time)
            print("time_step", time_step)
            dense_index = d_buffers.dense.dense_index.copy_to_host()
            actual_event = d_buffers.status.actual_event.copy_to_host()
            print("dense_index", dense_index)
            print("actual_event", actual_event)
            flags = d_buffers.status.status_flags.copy_to_host()
            print(flags)
            print(flags != 0)
            print(np.sum(flags == 0))
            zero_idx = np.where(flags == 0)[0]
            print(zero_idx)
            print("actual_time_failure", np.sum(actual_time <= 1e-16))
            print("time_step_failure", np.sum(time_step <=1e-12))
            failed_idx = np.where(actual_time <=1e-12)
            print(failed_idx)
            print(actual_time[failed_idx])
            print(time_step[failed_idx])
            print("actual_time[zero_idx]", actual_time[zero_idx])
            print("time_step[zero_idx]", time_step[zero_idx])
            print(len(zero_idx))
            #print(actual_event[zero_idx])
            input()

        if np.all(d_buffers.status.status_flags.copy_to_host() != 0):
            if debug:
                print("all-done")
            break
        #input()

    #print("device dense_time")
    #print(d_buffers.dense.dense_time.copy_to_host())
    #print("device dense_state")
    #print(d_buffers.dense.dense_state.copy_to_host())

    # Debug plot delete later --> 
    #dense_time = d_buffers.dense.dense_time.copy_to_host()
    #dense_index = d_buffers.dense.dense_index.copy_to_host()
    #dense_state = d_buffers.dense.dense_state.copy_to_host()

    #print(dense_time)
    #print(dense_index)
    #print(dense_state)

    #SYS = 0
    #UNIT = 0

    #plt.figure("NUMBA1")
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 0, SYS, UNIT], 'g-')
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 0, SYS, UNIT+1], 'b-')
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 0, SYS, UNIT+2], 'r-')

    #SYS = 0
    #UNIT = 0

    #plt.figure("NUMBA2")
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 1, SYS, UNIT], 'g-')
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 1, SYS, UNIT+1], 'b-')
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 1, SYS, UNIT+2], 'r-')
    #plt.show()

    return kernel_times

