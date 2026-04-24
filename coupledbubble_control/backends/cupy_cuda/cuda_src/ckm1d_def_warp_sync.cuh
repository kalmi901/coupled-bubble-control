#pragma once

static constexpr double TWO_PI = 6.28318530717958647692;
static constexpr double LIN_ATOL2 = LIN_ATOL * LIN_ATOL;
static constexpr double LIN_RTOL2 = LIN_RTOL * LIN_RTOL;

__forceinline__ __device__
void acoustic_field_1d(
    const double t,
    const double x,
    const double* __restrict__ sp,
    const double* __restrict__ cp,
    double &p, double &pt, double &px, double &ux
)
{
    p = 0.0;                                // Pressure
    pt = 0.0;                               // Time derivative of pressures
    px = 0.0;                               // Pressure gradient
    ux = 0.0;                               // Particle velocity

    const double w0 = TWO_PI * sp[1];       // 2π/ω_ref SP[1] = 1 / ω_ref
    const double l0 = TWO_PI * sp[2];       // λ_ref/2π
    const double u0 = TWO_PI * sp[4];       // 1/ rho/c

    for (int i = 0; i < NK; i++){
        double amps     = cp[       i];     // Pressure Amplitude
        double omega    = cp[NK   + i];     // Angular Frequency
        double phase    = cp[2*NK + i];     // Phase shift
        double wave     = cp[3*NK + i];     // Wave number
        double th = w0 * omega * t + phase;
        double xh = l0 * wave  * x + phase;
        double st, ct, sx, cx;
        sincos(th, &st, &ct);
        sincos(xh, &sx, &cx);
        #if SW_N
        p += amps * sx * st;
        pt+= amps * omega * sx * ct;
        px+= amps * wave  * cx * st;
        ux+= amps * u0 * cx * ct;
        #elif SW_A
        p += amps * cx * st;
        pt+= amps * omega * cx * ct;
        px-= amps * wave  * sx * st;
        ux-= amps * u0 * sx * ct;
        #elif CONST
        p += amps * st;
        pt+= amps * omega * ct;
        #else
        #error "No acoustic field selection
        #endif
    }
}

// Helpers
#define CPT(j, l, u) cpt[(j) * (SPB * TILE) + (l) * TILE + (u)]
#define GCM(i, j, gsid, luid) \
    gcm[(i) * (UPS * NS * TILE) + (j) * (NS * TILE) + (gsid) * TILE + (luid)]


__forceinline__ __device__
double sign(double x) {
    return (x > 0) - (x < 0);
}


__forceinline__ __device__
void explicit_coupling(
    const bool valid_lane,
    const bool l_active,
    const int gsid,
    const int lsid,
    const int luid,
    const double t,
    const double* __restrict__ x,
    double* __restrict__ dx,
    const double* __restrict__ cpf,
    const double* __restrict__ gcm
){
    if (valid_lane and l_active){
        const double x0 = x[0];
        const double x1 = x[1];
        const double x2 = x[2];
        const double x3 = x[3];

        const double x02 = x0 * x0;
        const double x03 = x02 * x0;
        const double x22 = x2 * x2;
        const double x32 = x3 * x3;

        const double h0i = x0 * x22;
        const double h1i = x02 * x2;
        const double h2i = x02 * x2 * x3;
        const double h3i = x03 * x3;
        const double h4i = x03 * x32;

        unsigned int current_mask = __activemask();

        // Temp buffers
        double sum_rad_g0 = 0.0;
        double sum_rad_g1 = 0.0;
        double sum_trn_ng = 0.0;
        double sum_trn_g2 = 0.0;
        double sum_trn_g3 = 0.0;

        double xj;
        double s;
        double m;
        double delta;
        double r_delta;
        double r_delta2;
        double sr_delta2;
        double r_delta3;

        double h0, h1, h2, h3, h4;
        double m0, m1, m2;
        
        int laneid = threadIdx.x & 0x1f;
        int tile_start_lane = (laneid / TILE) * TILE;
        int target_lane;
        // Matrix - Vector product
        for (int j = 0; j < UPS; j++){
            target_lane = tile_start_lane + j;
            // -- Calculate the distance and the direction --
            xj = __shfl_sync(current_mask, x1, target_lane);
            delta = x1 - xj;
            s = sign(delta);
            delta = abs(delta);
            m = 1.0 - (double)(luid == j);
            r_delta = m / max(delta, 1e-30);
            r_delta2  = r_delta * r_delta;
            sr_delta2 = r_delta2 * s;
            r_delta3  = r_delta2 * r_delta;

            // -- Coupling Terms --
            h0 = __shfl_sync(current_mask, h0i, target_lane);
            h1 = __shfl_sync(current_mask, h1i, target_lane);
            h2 = __shfl_sync(current_mask, h2i, target_lane);
            h3 = __shfl_sync(current_mask, h3i, target_lane);
            h4 = __shfl_sync(current_mask, h4i, target_lane);

            // -- Coupling matrices --
            m0 = GCM(0, j, gsid, luid);
            m1 = GCM(1, j, gsid, luid);
            m2 = GCM(2, j, gsid, luid);

            sum_rad_g0 += m0 * (
                -2.0 * r_delta   * h0
                -2.5 * sr_delta2 * h2
                -1.0 * r_delta3  * h4
            );

            sum_rad_g1 += m0 * (
                -0.5 * sr_delta2 * h1
                -0.5 * r_delta3  * h3
            );

            sum_trn_ng += m1 * (
                2.0 * sr_delta2 * h0
                +5.0 * r_delta3  * h2
            );

            sum_trn_g2 += m1 * (
                    sr_delta2 * h1
                +   r_delta3  * h3
            );

            sum_trn_g3 += m2 * (
                    sr_delta2 * h1
                +   r_delta3  * h3
            );
        }

        // Correct RHS
        dx[2] += sum_rad_g0 * cpf[0] + sum_rad_g1 * cpf[1];
        dx[3] += sum_trn_ng + sum_trn_g2 * cpf[2] + sum_trn_g3 * cpf[3];
    }
   
}

__forceinline__ __device__
void mat_vec_product(
    const int gsid,
    const int lsid,
    const int luid,
    const double* __restrict__ x,
    const double* __restrict__ cpf,
    const double* __restrict__ gcm,
    const double* __restrict__ v,
    double* __restrict__ axv
){
    //if (lsid == 0 && luid == 0){
    //    printf("matrix-vector \n");
    //}
    const double x0 = x[0];
    const double x1 = x[1];       // Local Bubble Position
    const double vri = v[0];
    const double vti = v[1];
    const double h5i = x0 * x0;
    const double h6i = h5i * x0;

    double av_rad = 0.0;
    double av_trn = 0.0;

    double xj;
    double s;
    double m;
    double delta;
    double r_delta;
    double r_delta2;
    double sr_delta2;
    double r_delta3;

    double h5, h6;
    double vr, vt;
    double m0, m1;
    unsigned int current_mask = __activemask();
    int laneid = threadIdx.x & 0x1f;
    int tile_start_lane = (laneid / TILE) * TILE;
    int target_lane;
    for (int j = 0; j < UPS; j++){
        target_lane = tile_start_lane + j;
        // -- Calculate the distance and the direction --
        //xj = CPT(7, lsid, j);
        xj = __shfl_sync(current_mask, x1, target_lane);
        delta = x1 - xj;
        s = sign(delta);
        delta = abs(delta);
        m = 1.0 - (double)(luid == j);
        r_delta = m / max(delta, 1e-30);
        r_delta2  = r_delta * r_delta;
        sr_delta2 = r_delta2 * s;
        r_delta3  = r_delta2 * r_delta;

        // -- Coupling Temrs --
        h5 = __shfl_sync(current_mask, h5i, target_lane);
        h6 = __shfl_sync(current_mask, h6i, target_lane);

        // -- Coupled vector --
        vr = __shfl_sync(current_mask, vri, target_lane);
        vt = __shfl_sync(current_mask, vti, target_lane);

        // -- Coupling Matrices --
        m0 = GCM(0, j, gsid, luid);
        m1 = GCM(1, j, gsid, luid);

        av_rad += m0 * (
                r_delta   * h5 * vr
            +0.5 * sr_delta2 * h6 * vt
        );

        av_trn += -m1 * (
                sr_delta2 * h5 * vr
            +   r_delta3  * h6 * vt
        );
    }
    axv[0] = av_rad * cpf[0] + vri;
    axv[1] = av_trn + vti;
}


__forceinline__ __device__
void implicit_coupling(
    const bool valid_lane,
    const bool l_active,
    const int gsid,
    const int lsid,
    const int luid,
    const double t,
    const double* __restrict__ x,
    double* __restrict__ dx,
    const double* __restrict__ cpf,
    const double* __restrict__ gcm,
    // Linalg Solver
    int* s_lin_converged,
    int* s_all_converged,
    double* s_norm_temp
){
    //if (lsid == 0 && luid == 0){
    //    printf("implicit coupling\n");
    //}
    double v[2];
    double axv[2];
    int tid = threadIdx.x;
    if (tid == 0){
        s_all_converged[0] = 0;
    }

    if (luid == 0){
        s_norm_temp[lsid] = 0.0;                        // ---> bnorm^2
        s_lin_converged[lsid] = (int)(!valid_lane || !l_active);
    }
    __syncthreads();
    if (tid < SPB){
        int lin_converged = s_lin_converged[tid];
        atomicAdd(&s_all_converged[0], lin_converged);
    }
    __syncthreads();
    
    bool all_converged = (s_all_converged[0] >= SPB);
    if (all_converged) return;

    if (valid_lane && l_active){
        double dx2 = dx[2];
        double dx3 = dx[3];
        double acc = dx2 * dx2 + dx3 * dx3;
        v[0] = dx2;
        v[1] = dx3;
        atomicAdd(&s_norm_temp[lsid], acc);
    }
    __syncthreads();

    double tol2 = LIN_ATOL2;
    if (valid_lane){
        tol2 = max(LIN_ATOL2, LIN_RTOL2 * s_norm_temp[lsid]);
    }
    bool l_converged = (s_lin_converged[lsid] != 0);

    for (int k = 0; k < LIN_MAX_ITER; k++){
        if (valid_lane && l_active && luid == 0 && !l_converged){
            s_norm_temp[lsid] = 0.0;            // ----> rnorm^2
        }
        __syncthreads();

        // Matrix-Vector product
        if (valid_lane && l_active && !l_converged){
            mat_vec_product(
                gsid,
                lsid, luid,
                x,
                cpf,
                gcm, 
                v, axv);
        }
        __syncthreads();
        //if (gsid == 0){
        //    printf("luid = %d, axv[0] = %.12f, axv[1] = %.12f \n", luid, axv[0], axv[1]);
        //}

        // Calculate residual --> reuse axv-temp array as residual vector
        if (valid_lane && l_active && !l_converged){
            // r = b - axv --> v = v
            double r0 = dx[2] - axv[0];
            double r1 = dx[3] - axv[1];
            double acc = r0 * r0 + r1 * r1;
            axv[0] = r0;
            axv[1] = r1;
            atomicAdd(&s_norm_temp[lsid], acc);
        }
        __syncthreads();
        //if (gsid == 0){
        //    printf("k = %d, luid = %d, axv[0]:=r[0] = %.12f, axv[1]:=r[1] = %.12f \n", k, luid, axv[0], axv[1]);
        //    printf("k = %d, lsid = %d, rnorm2 = %.12f \n", k, lsid, s_norm_temp[lsid]);
        //}
        // Convergence Check and update
        if (valid_lane && l_active && luid == 0 && !l_converged){
            if (s_norm_temp[lsid] < tol2){
                s_lin_converged[lsid] = 1;
                atomicAdd(&s_all_converged[0], 1);
            }
        }
        __syncthreads();
        l_converged = (s_lin_converged[lsid] != 0);

        if (valid_lane && l_active && !l_converged){
            v[0] += axv[0];
            v[1] += axv[1];
        }
        
        all_converged = (s_all_converged[0] >= SPB);
        //printf("tid %d, all_converged %d, k %d\n", tid, (int)all_converged, k);
        //__syncthreads();
        if (all_converged){
            //printf("break, tid %d, all_converged %d, k %d\n", tid, (int)all_converged, k);
            break;
        } 
    }

    // Update dx
    if (valid_lane && l_active){
        dx[2] = v[0];
        dx[3] = v[1];
    }
}


__forceinline__ __device__
void ode_fun(
    const bool valid_lane,
    const bool l_active,
    const int gsid,
    const int lsid,
    const int luid,
    const double t,
    const double* __restrict__ x,
    double* __restrict__ dx,
    const double* __restrict__ up,
    const double* __restrict__ sp,
    const double* __restrict__ cp,
    double* __restrict__ cpf,
    const double* __restrict__ gcm,
    // Linalg Solver
    int* s_lin_converged, 
    int* s_all_converged,
    double* s_norm_temp
){
    if (valid_lane & l_active){
        const double x0 = x[0];          // Dimensionless Radius
        const double x1 = x[1];          // Dimensionless Position
        const double x2 = x[2];          // Dimensionless Wall Velocity
        const double x3 = x[3];          // Dimensionless Translation Velocity

        const double rx0 = 1.0 / x0;
        const double p   = pow(rx0, sp[0]);

        double pa, pat, pax, uax;
        acoustic_field_1d(t, x1, sp, cp, pa, pat, pax, uax);

        const double N = (up[0] + up[1]*x2) * p
            - up[2] * (1 + up[7]*x2)
            - up[3] * rx0
            - up[4] * x2*rx0
            - 1.5 * (1.0 - up[7]*x2 * (1.0/3.0))*x2*x2
            - (1 + up[7]*x2) * up[5] * pa
            - up[6] * pat * x0
            + up[8] * x3*x3;                         // Feedback term

        const double D = x0 - up[7]*x0*x2 + up[4]*up[7];
        const double rD = 1.0 / D;

        const double Fb1 = - up[10]*x0*x0*x0 * pax;
        const double Fd  = - up[11]*x0 * (x3*sp[3] - uax );

        dx[0] = x2;
        dx[1] = x3;
        dx[2] = N * rD;
        dx[3] = 3*(Fb1+Fd)*up[9]*rx0*rx0*rx0 - 3.0*x2*rx0*x3;

        // --> Coupling factor
        cpf[0] = rD;
        cpf[1] = x3 * rD;
        cpf[2] = x2 * rx0;
        cpf[3] = rx0 * rx0;

    }

    explicit_coupling(
        valid_lane, l_active,
        gsid,
        lsid, luid,
        t,
        x, dx,
        cpf,
        gcm
    );

    implicit_coupling(
        valid_lane, l_active,
        gsid,
        lsid, luid,
        t,
        x, dx,
        cpf,
        gcm,
        // Linalg Solver
        s_lin_converged, s_all_converged,
        s_norm_temp
    );

    __syncthreads();
}


__forceinline__ __device__
void event_fun(
    const bool valid_lane,
    const bool l_active,
    const int gsid,
    const int lsid,
    const int luid,
    double* __restrict__ s_g,
    const double t,
    const double* __restrict__ x,
    const double* __restrict__ up,
    const double* __restrict__ sp,
    const double* __restrict__ cp
){
   
    if (valid_lane && l_active){
        // 1. Saját adatok előkészítése
        double ri = x[0] * up[12];
        double xi = x[1];
        // Aktuáls maszk és aktív szálak
        unsigned int mask = __activemask();
        int laneid = threadIdx.x & 0x1f;
        int target_lane = (laneid + 1 < 31) ? laneid + 1 : 31;

        // 2. Adatcsere (shuffle)
        // Mindenki részt vesz
        double rj = __shfl_sync(mask, ri, target_lane);
        double xj = __shfl_sync(mask, xi, target_lane);

        double gap = 1e30;
        if (luid < UPS - 1){
            double min_delta = (ri + rj) * (1.0 + 0.1); // __COL_THRESHOLD = 0.1
            gap = abs(xi - xj) - min_delta;
        }

        // 3. Redukci: Megkeressük a legkisebb gap-et a rendszeren belül
        double res_gap = gap;
        int tile_start_lane = (laneid / TILE) * TILE;
        for (int j = 0; j < UPS; j++){
            int source_lane = tile_start_lane + j;
            double remote_gap = __shfl_sync(mask, res_gap, source_lane);
            //if (remote_gap < res_gap){
            //    res_gap = remote_gap;
            //}
            res_gap = min(res_gap, remote_gap);
        }
        // 4. Az eredményt csak a 0. szál írja ki
        if (luid == 0){
            s_g[lsid] = res_gap;
        }
        __syncthreads();
    }
}