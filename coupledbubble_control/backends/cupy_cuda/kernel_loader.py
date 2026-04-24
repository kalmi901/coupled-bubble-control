from __future__ import annotations
from pathlib import Path
import cupy as cp
from typing import Union, Optional, List, Literal
import time

from ..cuda_opts import CUDAOpts

def load_coupled_kernel(
    solver_file: str,
    system_definition: str,      
    stepper_fun_name: str,
    model_spec,
    execution_spec,
    solver_spec,
    # CUDA-OPTS
    cuda_opts: CUDAOpts,
):
    cuda_dir = Path(__file__).resolve().parent / "cuda_src"
    src = (cuda_dir / solver_file).read_text(encoding="utf-8")
    
    system_def = f'-DRHS_HEADER="{system_definition}"'
    print("system_def", system_def)

    # Acoustic field selection
    if model_spec.ac == "CONST":
        ac_def = "-DCONST=1"
    elif model_spec.ac == "SW_N":
        ac_def = "-DSW_N=1"
    elif model_spec.ac == "SW_A":
        ac_def = "-DSW_A=1"
    else:
        raise ValueError(f"Unknown AC model: {model_spec.ac}")

    options = [
        "-std=c++17",
        f"-I{str(cuda_dir)}",
        # Model spec
        f"-DSD={model_spec.sd}",
        f"-DNUP={model_spec.nup}",
        f"-DNSP={model_spec.nsp}",
        f"-DNCM={model_spec.ncm}",
        f"-DNK={model_spec.nk}",
        f"-DNE={model_spec.ne}",
        f"-DNCP={model_spec.ncp}",
        f"-DNCF={model_spec.ncf}",
        f"-DNCT={model_spec.nct}",
        # Execution spec
        f"-DUPS={execution_spec.ups}",
        f"-DNS={execution_spec.ns}",
        f"-DSPB={execution_spec.spb}",
        f"-DNDO={execution_spec.ndo}",
        f"-DTILE={execution_spec.tile}",
        # Solver spec
        f"-DMIN_STEP={solver_spec.min_step:.1e}",
        f"-DMAX_STEP={solver_spec.max_step:.1e}",
        f"-DATOL={solver_spec.atol:.1e}",
        f"-DRTOL={solver_spec.rtol:.1e}",
        f"-DLIN_ATOL={solver_spec.lin_atol:.1e}",
        f"-DLIN_RTOL={solver_spec.lin_rtol:.1e}",
        f"-DLIN_MAX_ITER={solver_spec.lin_max_iter}",
        # Compile
        f"-DREBUILD_TS={int(time.time())}",
        system_def,
        ac_def
    ]

    #cuda_opts = []
    #cuda_opts += ["--maxrregcount=64"]
    #cu_opts=[
    #    "-maxrregcount=192",
    #    "-Xptxas=-v",
    #    "-Xptxas=-warn-spills",
    #    "-lineinfo"
    #]

    #print(cu_opts)

    if cuda_opts is not None:
        cuda_compiler = cuda_opts.compiler
        cu_opts = cuda_opts.to_cupy_options()
        options.extend(cu_opts)


    print("Build Options", options)
    #input()
    mod = cp.RawModule(
            code = src,
            options = tuple(options),
            backend = cuda_compiler
        )

    show_compile_log = True
    if show_compile_log:
        import sys
        mod.compile(log_stream=sys.stdout)   # itt jön a compiler output

    kernel = mod.get_function(stepper_fun_name)
    return kernel

