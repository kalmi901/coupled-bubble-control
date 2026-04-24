"""
TODO
"""
from __future__ import annotations
import numpy as np
import numba as nb
from numba import cuda
import math
from ...cuda_opts import CUDAOpts

TWO_PI = 2.0 * np.pi

ODE_KWARGS = {
    "opt": True,
    "inline": True,
    "lineinfo": False,
    "fastmath": False
}

def make_model(model_spec, execution_spec, solver_spec,
               cuda_opts: CUDAOpts):
    # GLOBAL CONSTANTS
    K = model_spec.nk
    AC = model_spec.ac
    TILE = execution_spec.tile
    UPS = execution_spec.ups
    SPB = execution_spec.spb
    
    LIN_ATOL = solver_spec.lin_atol
    LIN_RTOL = solver_spec.lin_rtol
    LIN_MAX_ITER = solver_spec.lin_max_iter
    LIN_RELAXATION = solver_spec.lin_relaxation

    LIN_ATOL2 = np.float64(LIN_ATOL * LIN_ATOL)
    LIN_RTOL2 = np.float64(LIN_RTOL * LIN_RTOL)

    if cuda_opts is not None:
        ODE_KWARGS.update(cuda_opts.to_numba_kwargs())

    if AC == "SW_N":
        @cuda.jit(device=True, **ODE_KWARGS)
        def acoustic_field_1d_n(t, x, sp, dp):
            """
            t:  (N, )       # dimensionless time (t = t_s / T)
            x:  (N, )       # dimensionless bubble position (x / λ_ref)
            dp: (4K, N)     # acoustic field parameters
              dp[ 0:K, :]   # pressure amplitude (Pa)
              dp[K:2K, :]   # angular frequency (rad/s)
              dp[2K:3K,:]   # phase shift (Rad)
              dp[3K:4K,:]   # Wave number (rad/m)       
            """
            p = 0.0        # Pressures
            pt = 0.0       # Time derivative of pressure
            px = 0.0       # Pressure gradient
            ux = 0.0       # Particle velocity
            w0 = TWO_PI * sp[1]         # 2π/ω_ref SP[1] = 1 / ω_ref
            l0 = TWO_PI * sp[2]         # λ_ref/2π
            u0 = sp[4] 
            for i in range(K):
                amps  = dp[      i]      # Pressure amplitude
                omega = dp[  K + i]      # Angular frequency
                phase = dp[2*K + i]      # Phase shift
                wave  = dp[3*K + i]      # Wave number
                th = w0 * omega * t + phase
                xh = l0 * wave  * x + phase

                st, ct = np.sin(th), np.cos(th)
                sx, cx = np.sin(xh), np.cos(xh)

                p += amps * sx * st
                pt+= amps * omega * sx * ct
                px+= amps * wave  * cx * st
                ux+= amps * u0 * cx * ct

            return p, pt, px, ux
        
        acoustic_field_1d = acoustic_field_1d_n

    elif AC == "SW_A":
        @cuda.jit(device=True, **ODE_KWARGS)
        def acoustic_field_1d_a(t, x, sp, dp):
            """
            t:  (N, )       # dimensionless time (t = t_s / T)
            x:  (N, )       # dimensionless bubble position (x / λ_ref)
            dp: (4K, N)     # acoustic field parameters
              dp[ 0:K, :]   # pressure amplitude (Pa)
              dp[K:2K, :]   # angular frequency (rad/s)
              dp[2K:3K,:]   # phase shift (Rad)
              dp[3K:4K,:]   # Wave number (rad/m)       
            """
            p = 0.0        # Pressures
            pt = 0.0       # Time derivative of pressure
            px = 0.0       # Pressure gradient
            ux = 0.0       # Particle velocity
            w0 = TWO_PI * sp[1]         # 2π/ω_ref SP[1] = 1 / ω_ref
            l0 = TWO_PI * sp[2]
            u0 = sp[4]                  # 1/ rho/c
            for i in range(K):
                amps  = dp[      i]      # Pressure amplitude
                omega = dp[  K + i]      # Angular frequency
                phase = dp[2*K + i]      # Phase shift
                wave  = dp[3*K + i]      # Wave number
                th = w0 * omega * t + phase
                xh = l0 * wave  * x + phase

                st, ct = np.sin(th), np.cos(th)
                sx, cx = np.sin(xh), np.cos(xh)

                p += amps * cx * st
                pt+= amps * omega * cx * ct
                px-= amps * wave  * sx * st
                ux-= amps * u0 * sx * ct

            return p, pt, px, ux
        
        acoustic_field_1d = acoustic_field_1d_a

    elif AC == "CONST":
        @cuda.jit(device=True, **ODE_KWARGS)
        def acoustic_field_1d_c(t, x, sp, dp):
            """
            t:  (N, )       # dimensionless time (t = t_s / T)
            x:  (N, )       # dimensionless bubble position (x / λ_ref)
            dp: (4K, N)     # acoustic field parameters
              dp[ 0:K, :]   # pressure amplitude (Pa)
              dp[K:2K, :]   # angular frequency (rad/s)
              dp[2K:3K,:]   # phase shift (Rad)
              dp[3K:4K,:]   # Wave number (rad/m)       
            """
            p = 0.0        # Pressures
            pt = 0.0       # Time derivative of pressure
            px = 0.0       # Pressure gradient
            ux = 0.0       # Particle velocity
            w0 = TWO_PI * sp[1]         # 2π/ω_ref SP[1] = 1 / ω_ref
            #l0 = TWO_PI * sp[2]
            #u0 = sp[4]                  # 1/ rho/c
            for i in range(K):
                amps  = dp[      i]      # Pressure amplitude
                omega = dp[  K + i]      # Angular frequency
                phase = dp[2*K + i]      # Phase shift
                #wave  = dp[3*K + i]      # Wave number
                th = w0 * omega * t + phase
                #xh = l0 * wave  * x + phase

                st, ct = np.sin(th), np.cos(th)
                #sx, cx = np.sin(xh), np.cos(xh)

                p += amps * st
                pt+= amps * omega * ct
                #px-= amps * wave  * sx * st
                #ux-= amps * u0 * sx * ct

            return p, pt, px, ux

        acoustic_field_1d = acoustic_field_1d_c
    else:
        raise NotImplementedError
    
    @cuda.jit(device=True, **ODE_KWARGS)
    def sign(x):
        return (x > 0) - (x < 0)

    @cuda.jit(device=True, **ODE_KWARGS)
    def explicit_coupling(
        valid_lane, l_active,
        gsid,
        lsid, luid,
        t,
        x, dx,
        cpf,
        gcm
    ):
        if valid_lane and l_active:
            x0 = x[0]       # Dimensionless Radius
            x1 = x[1]       # Dimensionless Position
            x2 = x[2]       # Dimensionless Wall Velocity
            x3 = x[3]       # Dimensionless Translational Velocity
            
            x02 = x0 * x0
            x03 = x02 * x0
            x22 = x2 * x2
            x32 = x3 * x3
            
            h0i = x0 * x22
            h1i = x02 * x2
            h2i = x02 * x2 * x3
            h3i = x03 * x3
            h4i = x03 * x32

            # 2. Aktív szálak maszkolása (azok a rendszerek, amik még nem konvergáltak)
            current_mask = cuda.activemask()

            # Temp buffers
            sum_rad_g0 = 0.0
            sum_rad_g1 = 0.0
            sum_trn_ng = 0.0
            sum_trn_g2 = 0.0
            sum_trn_g3 = 0.0

            # Matrix - Vector products
            tile_start_lane = (cuda.laneid // TILE) * TILE
            for j in range(UPS):
                # Kiszámoljuk, melyik lane a célpont a warp-on belül
                # (laneid // 4) * 4 -> a rendszer első szála, + j -> a j-edik buborék
                target_lane = tile_start_lane + j
                # -- Calculate the distance and the direction --
                xj = cuda.shfl_sync(current_mask, x1, target_lane)
                delta = (x1 - xj)
                s = nb.float64(sign(delta))
                delta = abs(delta)
                #s = nb.float64((luid>j) - (j>luid))
                m = 1.0 - nb.float64(luid == j)     # Diagonal Mask 1 ha nem diagonal, 0 ha diagonal
                r_delta   = m / max(delta, 1e-30)
                r_delta2  = r_delta * r_delta
                sr_delta2 = r_delta2 * s
                r_delta3  = r_delta2 * r_delta

                # -- Copuling Terms --
                h0 = cuda.shfl_sync(current_mask, h0i, target_lane)
                h1 = cuda.shfl_sync(current_mask, h1i, target_lane)
                h2 = cuda.shfl_sync(current_mask, h2i, target_lane)
                h3 = cuda.shfl_sync(current_mask, h3i, target_lane)
                h4 = cuda.shfl_sync(current_mask, h4i, target_lane)

                # -- Coupling Matrices --
                m0 = gcm[0, j, gsid, luid]
                m1 = gcm[1, j, gsid, luid]
                m2 = gcm[2, j, gsid, luid]

                sum_rad_g0 += m0 * (
                    -2.0 * r_delta   * h0
                    -2.5 * sr_delta2 * h2
                    -1.0 * r_delta3  * h4
                )

                sum_rad_g1 += m0 * (
                    -0.5 * sr_delta2 * h1
                    -0.5 * r_delta3  * h3
                )

                sum_trn_ng += m1 * (
                    2.0 * sr_delta2 * h0
                    +5.0 * r_delta3  * h2
                )

                sum_trn_g2 += m1 * (
                        sr_delta2 * h1
                    +   r_delta3  * h3
                )

                sum_trn_g3 += m2 * (
                        sr_delta2 * h1
                    +   r_delta3  * h3
                )

            # Correct RHS
            dx[2] += sum_rad_g0 * cpf[0] + sum_rad_g1 * cpf[1]
            dx[3] += sum_trn_ng + sum_trn_g2 * cpf[2] + sum_trn_g3 * cpf[3]

    
    @cuda.jit(device=True, **ODE_KWARGS)
    def mat_vec_product(
        gsid,
        lsid, luid,
        x,
        cpf,
        gcm,
        v,
        axv
    ):
        x0 = x[0]       # Dimensionless Radius
        x1 = x[1]       # Dimensionless Position

        vri = v[0]
        vti = v[1]
        #x02 = x0 * x0
        #x03 = x02 * x0
        #h5i = x02
        #h6i = x03
        h5i = x0 * x0
        h6i = h5i * x0

        av_rad = 0.0
        av_trn = 0.0
        current_mask = cuda.activemask()
        tile_start_lane = (cuda.laneid // TILE) * TILE
        for j in range(UPS):
            target_lane = tile_start_lane + j
            # -- Calculate the distance and the direction --
            xj = cuda.shfl_sync(current_mask, x1, target_lane)   
            delta = (x1 - xj)
            s = nb.float64(sign(delta))
            delta = abs(delta)
            #s = nb.float64((luid>j) - (j>luid))
            m = 1.0 - nb.float64(luid == j)     # Diagonal Mask 1 ha nem diagonal, 0 ha diagonal
            r_delta   = m / max(delta, 1e-30)
            r_delta2  = r_delta * r_delta
            sr_delta2 = r_delta2 * s
            r_delta3  = r_delta2 * r_delta

            # -- Coupling Terms --
            h5 = cuda.shfl_sync(current_mask, h5i, target_lane)
            h6 = cuda.shfl_sync(current_mask, h6i, target_lane)

            # -- Coupled vector --
            vr = cuda.shfl_sync(current_mask, vri, target_lane)
            vt = cuda.shfl_sync(current_mask, vti, target_lane)
            
            # -- Coupling Matrices --
            m0 = gcm[0, j, gsid, luid]
            m1 = gcm[1, j, gsid, luid]

            av_rad += m0 * (
                    r_delta   * h5 * vr
                +0.5 * sr_delta2 * h6 * vt
            )

            av_trn += -m1 * (
                    sr_delta2 * h5 * vr
                +   r_delta3  * h6 * vt
            )

        axv[0] = av_rad * cpf[0] + vri
        axv[1] = av_trn + vti


    @cuda.jit(device=True, **ODE_KWARGS)
    def implicit_coupling(
        valid_lane, l_active,
        gsid,
        lsid, luid,
        t,
        x, dx,
        cpf,
        gcm,
        # Linalg Solver
        s_lin_converged, s_all_converged,
        s_norm_temp
    ):
        v   = cuda.local.array((2,), dtype=nb.float64)       # type: ignore
        axv = cuda.local.array((2,), dtype=nb.float64)       # type: ignore
        tid = cuda.threadIdx.x  # type: ignore
        if tid == 0:
            s_all_converged[0] = 0

        if luid == 0:
            s_norm_temp[lsid]   = 0.0                           # type: ignore  --> bnorm^2
            #s_lin_converged[lsid] = 0 if valid_lane else 1      # type: ignore / terminated (inactive etc)
            s_lin_converged[lsid] = 0 if (valid_lane and l_active) else 1
        cuda.syncthreads()                                      # type: ignore
    
        if tid < SPB:
            lin_converged = s_lin_converged[tid]
            cuda.atomic.add(s_all_converged, 0, lin_converged)  # type: ignore
        cuda.syncthreads()  # type: ignore

        all_converged = (s_all_converged[0] >= SPB)
        if all_converged:
            return

        if valid_lane and l_active:
            # v := x = b
            dx2 = dx[2]
            dx3 = dx[3]
            acc = dx2 * dx2 + dx3 * dx3
            v[0] = dx2      # type: ignore
            v[1] = dx3      # type: ignore
            cuda.atomic.add(s_norm_temp, lsid, acc)  # type: ignore
        cuda.syncthreads()  # type: ignore

        tol2 = LIN_ATOL2
        if valid_lane:
            tol2 = max(LIN_ATOL2, LIN_RTOL2 * s_norm_temp[lsid])  # type: ignore
        l_converged = (s_lin_converged[lsid] != 0)
        
        for k in range(LIN_MAX_ITER):
            if valid_lane and l_active and not l_converged and luid == 0:
                s_norm_temp[lsid] = 0.0     # type: ignore ---> rnorm^2
            cuda.syncthreads()              # type: ignore

            # Matrix vector product
            if valid_lane and l_active and not l_converged:
                mat_vec_product(
                    gsid,
                    lsid, luid,
                    x,
                    cpf,
                    gcm, 
                    v, axv)

            #if gsid == 0:
            #    print("luid", luid, "axv[0]", axv[0], "axv[1]", axv[1])     # type: ignore

            # Calculate residual --> reuse axv-temp array as residual vector 
            if valid_lane and l_active and not l_converged:
                # r = b - axv --> v = b
                r0 = dx[2] - axv[0]   # type: ignore
                r1 = dx[3] - axv[1]    # type: ignore
                acc = r0 * r0 + r1 * r1
                axv[0] = r0 # type: ignore
                axv[1] = r1 # type: ignore
                cuda.atomic.add(s_norm_temp, lsid, acc)  # type: ignore
            cuda.syncthreads()                      # type: ignore

            #if gsid == 0:
            #    print("luid", luid, "axv[0]:=r[0]", axv[0], "axv[1]:=r[1]", axv[1])     # type: ignore
            #    print("lsid", lsid, "rnorm2", s_norm_temp[lsid])                             # type: ignore

            # Convergence Check and update
            if valid_lane and l_active and not l_converged and luid == 0:
                if s_norm_temp[lsid] < tol2:         # type: ignore
                    s_lin_converged[lsid] = 1        # type: ignore
                    cuda.atomic.add(s_all_converged, 0, 1)  # type: ignore
            cuda.syncthreads()                  # type: ignore

            l_converged = (s_lin_converged[lsid] != 0)   # type: ignore
            if valid_lane and l_active and not l_converged:
                v[0] += axv[0]  # type: ignore
                v[1] += axv[1]  # type: ignore
            cuda.syncthreads()                  # type: ignore

            all_converged = (s_all_converged[0] >= SPB)
            if all_converged:
                break

        # Update dx
        if valid_lane and l_active:
            dx[2] = v[0]    # type: ignore
            dx[3] = v[1]    # type: ignore


    @cuda.jit(device=True, **ODE_KWARGS)
    def ode_fun(
        valid_lane, l_active,
        gsid,
        lsid, luid, 
        t, x, dx, up, sp, cp,
        cpf,
        gcm,
        # Linalg Solver
        s_lin_converged, s_all_converged,
        s_norm_temp):
        if valid_lane and l_active:
            x0 = x[0]       # Dimensionless Radius
            x1 = x[1]       # Dimensionless Position
            x2 = x[2]       # Dimensionless Wall Velocity
            x3 = x[3]       # Dimensionless Translational Velocity

            rx0 = 1.0 / x0
            p = rx0 ** sp[0]

            pa, pat, pax, uax = acoustic_field_1d(t, x1, sp, cp)

            N = (up[0] + up[1]*x2) * p \
                - up[2] * (1 + up[7]*x2) \
                - up[3] * rx0 \
                - up[4]*x2*rx0 \
                - 1.5 * (1.0 - up[7]*x2 * (1.0/3.0))*x2*x2 \
                - (1 + up[7]*x2) * up[5] * pa \
                - up[6] * pat * x0 \
                + up[8] * x3*x3                         # Feedback term

            D = x0 - up[7]*x0*x2 + up[4]*up[7]
            rD = 1.0 / D

            Fb1 = - up[10]*x0*x0*x0 * pax               # Primary Bjerknes Force
            Fd  = - up[11]*x0 * (x3*sp[3] - uax )       # Drag Force

            dx[0] = x2
            dx[1] = x3
            dx[2] = N * rD
            dx[3] = 3*(Fb1+Fd)*up[9]*rx0*rx0*rx0 - 3.0*x2*rx0*x3

            # --> Coupling factors
            cpf[0] = rD
            cpf[1] = x3 * rD
            cpf[2] = x2 * rx0
            cpf[3] = rx0 * rx0

        explicit_coupling(
            valid_lane, l_active,
            gsid,
            lsid, luid,
            t,
            x, dx,
            cpf,
            gcm
        )

        implicit_coupling(
            valid_lane, l_active,
            gsid,
            lsid, luid,
            t,
            x, dx,
            cpf,
            gcm,
            # Linalg Solver
            s_lin_converged, s_all_converged,
            s_norm_temp)

        cuda.syncthreads()  # type: ignore

    @cuda.jit(device=True, **ODE_KWARGS)
    def event_fun(
        valid_lane, l_active,
        gsid,
        lsid, luid,
        s_g,                      # system_event function
        t, x,
        up, sp, cp
    ):
        #if luid == 0:
            # 0. Shared memory reset - Csak egy szál csinálja blokkonként
        #    s_g[lsid] = 1e30
        #cuda.syncthreads()  # type: ignore

        if valid_lane and l_active:
            # 1. Saját adatok előkészítése
            ri = x[0] * up[12]
            xi = x[1]
            # Aktuális maszk az aktív szálaknak
            mask = cuda.activemask()
            # A target_lane a warp-on belüli szomszéd lesz
            target_lane = min(cuda.laneid + 1, 31)   # type: ignore --> Elvileg lehet 32, de így szebb

            # --- JAVÍTÁS: A shuffle-t KI KELL HOZNI az if luid < UPS-1 alól ---
            # Mindenki részt vesz az adatcserében --> "Mindenki kiabál"
            rj = cuda.shfl_sync(mask, ri, target_lane)
            xj = cuda.shfl_sync(mask, xi, target_lane)

            gap = 1e30
            # Csak akkor kérünk adatot, ha van szomszéd a rendszeren belül
            # "De csak valid UNIT figyel"
            if luid < UPS - 1:
                #print("gsid", gsid, "luid", luid, "ri", ri, "rj", rj)
                #print("gsid", gsid, "luid", luid, "xi", xi, "xj", xj)
                # Távolság számítása (collision threshold-dal)
                # delta = |xi - xj| - (ri + rj) * (1 + threshold)
                min_delta = (ri + rj) * (1.0 + 0.1) # __COL_THRESHOLD = 0.1
                gap = math.fabs(xi - xj) - min_delta

            ## 3. Redukció: Megkeressük a legkisebb gap-et a rendszeren belül
            # Kis TILE (pl. 2, 4, 8) esetén a legegyszerűbb, ha minden szál 
            # megnézi a többiek gap-jét egy belső loop-ban shuffle-lel
            
            res_gap = gap
            tile_start_lane = (cuda.laneid // TILE) * TILE
            for j in range(UPS):
                # Mindenki elküldi a saját gap-jét a rendszer többi szálának
                source_lane = tile_start_lane + j
                remote_gap = cuda.shfl_sync(mask, res_gap, source_lane)
                if remote_gap < res_gap:
                    res_gap = remote_gap

            # 4. Az eredményt csak a 0. szál írja ki (vagy mindenki, ha kell)
            if luid == 0:
                s_g[lsid] = res_gap

            #cuda.atomic.min(s_g, lsid, gap)    # type: ignore
        
        cuda.syncthreads()  # type: ignore
        #if luid == 0:
            #print("gsid", gsid, "lsid", lsid, "res_gap", s_g[lsid])

    return ode_fun, event_fun