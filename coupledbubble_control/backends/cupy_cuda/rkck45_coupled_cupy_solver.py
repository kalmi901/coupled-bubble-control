from __future__ import annotations
import cupy as cp
from typing import Dict, Optional
import time

import matplotlib.pyplot as plt

def solver(
    kernel,     # cupy.RawKernel
    d_buffers,
    model_spec,
    execution_spec,
    solver_spec,
    max_steps: int,
    kernel_steps: int = 2048,   # 2048
    benchmark: bool = False,
    debug: bool = False
):

    threads = execution_spec.block_size
    blocks = execution_spec.grid_size

    if debug:
        print("CUPY-CUDA-C Rawkernel")
        print("threads", threads)
        print("blocks", blocks)
    kernel_times = [] if benchmark else None

    max_launches = (max_steps + kernel_steps - 1) // kernel_steps
    cp.cuda.runtime.deviceSynchronize()
    for launch in range(max_launches):
        if debug:
            print("iter-cp", launch + 1)

        t0 = time.time()
        kernel(
            (blocks, ), (threads, ),
            (
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
                # dense buffer
                d_buffers.dense.dense_index,
                d_buffers.dense.dense_time,
                d_buffers.dense.dense_state,
                # status buffer
                d_buffers.status.actual_event,
                d_buffers.status.status_flags
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        t1 = time.time()
        if benchmark:
            kernel_times.append(t1 - t0)    # type: ignore

        if debug:
            print("actual_time", d_buffers.state.actual_time.get()[28])
            print("time-step", d_buffers.state.time_step.get()[28])
            #print("dense_index", d_buffers.dense.dense_index.get())
            #print("actual_event", d_buffers.status.actual_event.get())
            #print("status_flags", d_buffers.status.status_flags.get())
            print(d_buffers.status.status_flags != 0)
            print(d_buffers.status.status_flags[28] != 0)
            #print(cp.sum(d_buffers.status.status_flags == 0))
            input()

        if cp.alltrue(d_buffers.status.status_flags != 0):
            if debug:
                print("all-done")
            break
        #input()

    # Debug print delete later --> 
    #dense_time = d_buffers.dense.dense_time.get()
    #dense_index = d_buffers.dense.dense_index.get()
    #dense_state = d_buffers.dense.dense_state.get()

    #print(dense_time)
    #print(dense_index)

    #SYS = 95
    #UNIT = 0

    #plt.figure("CUPY1")
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 0, SYS, UNIT], 'g-')
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 0, SYS, UNIT+1], 'b-')
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 0, SYS, UNIT+2], 'r-')

    #SYS = 95
    #UNIT = 0

    #plt.figure("CUPY2")
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 1, SYS, UNIT], 'g-')
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 1, SYS, UNIT+1], 'b-')
    #plt.plot(dense_time[:dense_index[SYS], SYS], dense_state[:dense_index[SYS], 1, SYS, UNIT+2], 'r-')
    plt.show()

    return kernel_times