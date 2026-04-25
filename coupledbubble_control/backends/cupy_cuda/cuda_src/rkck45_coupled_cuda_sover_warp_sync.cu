// ---- MODEL SPEC DEFINITIONS ----
// System Dimension
#ifndef SD
#define SD 4
#endif

// Number of unit parameters
#ifndef NUP
#define NUP 12
#endif

// Number of shared parameters
#ifndef NSP
#define NSP 5
#endif

// Number of coupling matrices
#ifndef NCM
#define NCM 3
#endif

// Number of harmonic components
#ifndef NK
#define NK 2
#endif

// Number of events
#ifndef NE
#define NE 1
#endif

// Number of control parameters
#ifndef NCP
#define NCP (4 * NK)
#endif

//Coupling
#ifndef NCF
#define NCF 4
#endif

#ifndef NCT
#define NCT 8
#endif

// ---- ACOUSTIC FIELD TYPE -----
#ifndef SW_A
#define SW_A 0
#endif

#ifndef SW_N
#define SW_N 0
#endif

#ifndef CONST
#define CONST 0
#endif

// ---- EXUTION SPEC ----
// Units per systems
#ifndef UPS
#define UPS 2
#endif

// Number of systems
#ifndef NS
#define NS 1
#endif

// Systems per block
#ifndef SPB
#define SPB 1
#endif

// Number of dense ouputs
#ifndef NDO
#define NDO 0
#endif

// Subwarpt tile size
#ifndef TILE
#define TILE 2
#endif

// ---- SOLVER SPEC ----
#ifndef MIN_STEP 
#define MIN_STEP 1.0e-16
#endif

#ifndef MAX_STEP
#define MAX_STEP 1.0e-2
#endif

#ifndef ATOL
#define ATOL 1.0e-9
#endif

#ifndef RTOL
#define RTOL 1.0e-9
#endif

#ifndef ETOL
#define ETOL 1.0e-6
#endif

// TODO linear solver parameters
#ifndef LIN_ATOL
#define LIN_ATOL 1.0e-9
#endif

#ifndef LIN_RTOL
#define LIN_RTOL 1.0e-9
#endif

#ifndef LIN_MAX_ITER
#define LIN_MAX_ITER 100
#endif

#ifndef REBUILD_TS
#define REBUILD_TS 0
#endif

// System definitions
#ifndef RHS_HEADER
#define RHS_HEADER "ckm1d_def_warp_sync.cuh"
#endif

#if defined(RHS_HEADER)
    // ok
#else
#error "RHS_HEADER is not defined"
#endif

#include "cuda_runtime.h"
#include RHS_HEADER

namespace rkck{
    static constexpr double a2=1.0/5.0, a3=3.0/10.0, a4=3.0/5.0, a5=1.0, a6=7.0/8.0;
    static constexpr double b21=1.0/5.0;
    static constexpr double b31=3.0/40.0, b32=9.0/40.0;
    static constexpr double b41=3.0/10.0, b42=-9.0/10.0, b43=6.0/5.0;
    static constexpr double b51=-11.0/54.0, b52=5.0/2.0, b53=-70.0/27.0, b54=35.0/27.0;
    static constexpr double b61=1631.0/55296.0, b62=175.0/512.0, b63=575.0/13824.0, b64=44275.0/110592.0, b65=253.0/4096.0;
    static constexpr double c1=37.0/378.0, c2=0.0, c3=250.0/621.0, c4=125.0/594.0, c5=0.0, c6=512.0/1771.0;
    static constexpr double c1p=2825.0/27648.0, c2p=0.0, c3p=18575.0/48384.0, c4p=13525.0/55296.0, c5p=277.0/14336.0, c6p=1.0/4.0;
    static constexpr double ce1=(c1-c1p), ce2=(c2-c2p), ce3=(c3-c3p), ce4=(c4-c4p), ce5=(c5-c5p), ce6=(c6-c6p);
}

// ODE Status
#define ACTIVE 0
#define SUCCESS 1
#define EVENT_TERMINAL 2

// Helpers:
#define SYS(j, s)  s_system_params[(j) * SPB + (s)]
#define CTRL(j, s) s_control_params[(j) * SPB + (s)]
#define DENSE_TIME(t, s) g_dense_time[(t) * NS * TILE + (s)]

__forceinline__ __device__
void store_dense_output(
    const bool valid_lane,
    const bool l_active,
    const int gsid,
    const int gtid,
    const int lsid,
    const int luid,
    const double l_actual_time,
    const double l_time_end,
    const double* __restrict__ l_actual_state,
    int* __restrict__ s_dense_output_store,
    int* __restrict__ s_dense_output_time_index,
    double* __restrict__ s_dense_output_time,
    double* __restrict__ s_dense_output_min_time_step,
    double* __restrict__ g_dense_time,
    double* __restrict__ g_dense_state
){
    // per-system decision and timestep store
    if (valid_lane && l_active && (luid == 0)){
        int    do_store     = -1;
        int    buffer_idx   = s_dense_output_time_index[lsid];
        double min_time     = s_dense_output_time[lsid];
        double min_timestep = s_dense_output_min_time_step[lsid];
        // Check storage capacity and time-interval
        if ((buffer_idx < NDO) && (min_time <= l_actual_time)){
            do_store = buffer_idx;
            // -- Store time instance --
            g_dense_time[buffer_idx * NS + gsid] = l_actual_time;
            s_dense_output_time_index[lsid]      = buffer_idx + 1;
            s_dense_output_store[lsid]           = do_store;
            // -- Propose next time --
            s_dense_output_time[lsid] = min(l_actual_time + min_timestep, l_time_end);
        }
    }
    __syncthreads();
    // per-lane store
    int store_idx = s_dense_output_store[lsid];
    if (valid_lane && l_active && store_idx != -1){
        int offset = store_idx * SD * NS * TILE;
        for (int j = 0; j < SD; j++){
            g_dense_state[j * NS * TILE + offset + gtid] = l_actual_state[j];
        }
    }
}


extern "C" __global__
void rkck45_coupled_solver(
    const int kernel_steps,
    // state
    double* __restrict__ g_actual_time,
    double* __restrict__ g_time_end,
    double* __restrict__ g_time_begin,
    double* __restrict__ g_time_step,
    double* __restrict__ g_actual_state,
    // parameters
    const double* __restrict__ g_unit_params,
    const double* __restrict__ g_system_params,
    const double* __restrict__ g_control_params,
    const double* __restrict__ g_coupling_matrices,
    int* __restrict__ g_dense_index,
    double* __restrict__ g_dense_time,
    double* __restrict__ g_dense_state,
    // status buffers
    double* __restrict__ g_actual_event,
    int* __restrict__ g_status_flags
)
{
    // ---- Thread Management ----
    int tid = threadIdx.x;          // Local Thread ID
    int bid = blockIdx.x;           // Block ID
    int gtid = bid * blockDim.x + tid;  // Global Thread ID

    int lsid = tid / TILE;          // Local system ID within the block
    int luid = tid % TILE;          // Local unit ID within the systems
    int gsid = bid * SPB + lsid;    // Gloval system ID

    bool valid_system = (gsid < NS);
    bool valid_unit   = (luid < UPS);
    bool valid_lane   = valid_system && valid_unit;

    // ---- System Management ----
    __shared__ int s_terminated_system[SPB];
    __shared__ int s_all_terminated[1];
    if (tid == 0){
        s_all_terminated[0] = 0;
    }
    __syncthreads();

    //printf("gtid=%d tid=%d bid=%d lsid=%d luid=%d gsid=%d valid_system=%d valid_unit=%d valid_lane=%d\n",
    //   gtid, tid, bid, lsid, luid, gsid,
    //   (int)valid_system, (int)valid_unit, (int)valid_lane);

    // -- GLOBAL LOAD --
    // ---- Load Env Specific Data ----
    __shared__ double s_actual_time[SPB];
    __shared__ double s_time_end[SPB];
    __shared__ double s_time_step[SPB];
    __shared__ double s_system_params[NSP * SPB];
    __shared__ double s_control_params[NCP * SPB];
    #if NDO > 0
    __shared__ int s_dense_output_time_index[SPB];
    __shared__ int s_dense_output_store[SPB];
    __shared__ double s_dense_output_time[SPB];
    __shared__ double s_dense_output_min_time_step[SPB];
    #endif

    #if NE > 0
    __shared__ double s_actual_event_value[SPB];
    __shared__ double s_next_event_value[SPB];
    __shared__ int s_event_detected[SPB];
    __shared__ double s_event_ratio[SPB];
    #endif

    if (tid < SPB){
        int gid = bid * SPB + tid;
        if (gid < NS){
            double actual_time  = g_actual_time[gid];
            double time_end     = g_time_end[gid];
            double time_begin   = g_time_begin[gid];
            bool terminated     = (actual_time >= time_end);

            s_actual_time[tid]  = actual_time;
            s_time_end[tid]     = time_end;
            s_time_step[tid]    = g_time_step[gid];

            s_terminated_system[tid] = (int)terminated;
            g_status_flags[gid]      = (int)terminated;

            #if NDO > 0
                s_dense_output_time_index[tid]  = g_dense_index[gid];
                s_dense_output_time[tid]        = actual_time;
                s_dense_output_store[tid]       = 0;
                if (NDO == 1) {
                    s_dense_output_min_time_step[tid] = (time_end - time_begin);
                } else {
                    s_dense_output_min_time_step[tid] = (time_end - time_begin) / (NDO - 1);
                }
            #endif

            #if NE > 0
                // Load Event Buffer and terminate system if event has already reached
                double actual_event = g_actual_event[gid];
                bool event_occured  = (abs(actual_event) < ETOL);
                terminated          = terminated || event_occured;
                s_actual_event_value[tid] = actual_event;
                s_terminated_system[tid]  = (int)terminated;
                g_status_flags[gid]      += (int)event_occured;
            #endif

        } else {
            s_terminated_system[tid] = 1;
        }
    }
    __syncthreads();
    
    if (tid < SPB){
        atomicAdd(&s_all_terminated[0], s_terminated_system[tid]);
    }
    __syncthreads();
    bool all_terminated = (s_all_terminated[0] >= SPB);
    
    //if (luid == 0){
    //    printf("gsid = %d, all_terminated = %d, s_all_terminated = %d, s_terminated_system[lsid] = %d \n", gsid, (int)all_terminated, s_all_terminated[0], s_terminated_system[lsid]);
    //}

    if (all_terminated){
        //printf("gtid = %d, early-return \n", gtid);
        return;
    }

    // --- System Parameters ---
    for (int idx = tid; idx < SPB * NSP; idx += blockDim.x) {
        int s = idx / NSP;
        int j = idx % NSP;
        int g = bid * SPB + s;
        if (g < NS) {
            SYS(j, s) = g_system_params[j * NS + g];
        }
    }

    // --- Global Parameters ---
    for (int idx = tid; idx < SPB * NCP; idx += blockDim.x) {
        int s = idx / NCP;
        int j = idx % NCP;
        int g = bid * SPB + s;
        if (g < NS) {
            CTRL(j, s) = g_control_params[j * NS + g];
        }
    }

    __syncthreads();

    // --- Load Unit Specific Data ---
    double l_actual_state[SD];
    double l_unit_params[NUP];
    double l_system_params[NSP];
    double l_control_params[NCP];

    double l_actual_time = 0.0;
    double l_time_end    = 0.0;
    double l_time_step   = 0.0;
    bool   l_active      = false;

    if (valid_lane) {
        l_actual_time = s_actual_time[lsid];
        l_time_end    = s_time_end[lsid];
        l_time_step   = s_time_step[lsid];
        //l_active      = (l_actual_time < l_time_end);
        l_active      = valid_lane && (s_terminated_system[lsid] == 0);

        for (int j = 0; j < NSP; j++){
            l_system_params[j] = SYS(j, lsid);
        }

        for (int j = 0; j < NCP; j++){
            l_control_params[j] = CTRL(j, lsid);
        }

        for (int j = 0; j < SD; j++){
            l_actual_state[j] = g_actual_state[(j * NS * TILE) + gtid];
        }

        for (int j = 0; j < NUP; j++){
            l_unit_params[j] = g_unit_params[(j * NS * TILE) + gtid];
        }
    }

    __syncthreads();

    //if (gtid == 0) {
    //    printf("g_sid         = %d\n", gsid);
    //    printf("l_actual_time = %.10e\n", l_actual_time);
    //    printf("l_time_end    = %.10e\n", l_time_end);
    //    printf("l_time_step   = %.10e\n", l_time_step);
    //    printf("l_active      = %d\n", (int)l_active);

        // actual state
    //    #pragma unroll
    //    for (int j = 0; j < SD; j++) {
    //        printf("l_actual_state[%d] = %.12e\n", j, l_actual_state[j]);
    //    }

        // system params
    //    #pragma unroll
    //    for (int j = 0; j < NSP; j++) {
    //        printf("l_system_params[%d] = %.12e\n", j, l_system_params[j]);
    //    }

        // control params
    //    #pragma unroll
    //    for (int j = 0; j < NCP; j++) {
    //        printf("l_control_params[%d] = %.12e\n", j, l_control_params[j]);
    //    }

        // unit params
    //    #pragma unroll
    //    for (int j = 0; j < NUP; j++) {
    //        printf("l_unit_params[%d] = %.12e\n", j, l_unit_params[j]);
    //    }
    //}

    // -- INITIALIZE WORK ARRAYS --
    double k1[SD];
    double k2[SD];
    double k3[SD];
    double k4[SD];
    double k5[SD];
    double k6[SD];

    double l_temp_time;
    double l_temp_state[SD];
    double l_new_state[SD];

    __shared__ double sh_ratio[SPB][TILE];
    __shared__ int sh_ok[SPB][TILE];

    // -- Coupling factors and coupling terms --
    double l_coupling_factors[NCF];

    // -- Linalg work arrays --
    __shared__ int s_lin_converged[SPB];
    __shared__ int s_all_converged[1];
    __shared__ double s_norm_temp[SPB];

    #if NDO > 0
    store_dense_output(
        valid_lane, l_active,
        gsid, gtid,
        lsid, luid,
        l_actual_time,
        l_time_end,
        l_actual_state,
        s_dense_output_store,
        s_dense_output_time_index,
        s_dense_output_time,
        s_dense_output_min_time_step,
        g_dense_time,
        g_dense_state);
    #endif

    #if NE > 0
    event_fun(
        valid_lane, l_active,
        gsid,
        lsid, luid,
        s_actual_event_value,
        l_actual_time,
        l_actual_state,
        l_unit_params, l_system_params, l_control_params);
    #endif

    // ------------------------------------------------------------
    // --------------------- MAIN KERNEL LOOP ---------------------
    // ------------------------------------------------------------
    for (int step = 0; step < kernel_steps; step++){
        //if (gtid == 0){
        //    printf("step %d", step);
        //}
        // -- Calmping time-step --
        if (valid_lane && l_active){
            if (l_time_step < MIN_STEP) l_time_step = MIN_STEP;
            if (l_time_step > MAX_STEP) l_time_step = MAX_STEP;
            double l_remain = l_time_end - l_actual_time;
            if (l_time_step > l_remain) l_time_step = l_remain;
        }

        // -------------------------------------------------------
        // --------------------- RKCK-Stages ---------------------
        // -------------------------------------------------------
        // ------ k1 -------
        ode_fun(valid_lane, l_active,
            gsid,
            lsid, luid,
            l_actual_time,
            l_actual_state,
            k1,
            l_unit_params, l_system_params, l_control_params,
            l_coupling_factors,
            g_coupling_matrices,
            // Linalg Solver
            s_lin_converged, s_all_converged,
            s_norm_temp
            );
        
        // ------ k2 -------
        l_temp_time = l_actual_time + rkck::a2 * l_time_step;
        #pragma unroll
        for (int j = 0; j < SD; j++){
            l_temp_state[j] = l_actual_state[j] + l_time_step * (rkck::b21 * k1[j]);
        }
        ode_fun(valid_lane, l_active,
            gsid,
            lsid, luid,
            l_temp_time,
            l_temp_state,
            k2,
            l_unit_params, l_system_params, l_control_params,
            l_coupling_factors,
            g_coupling_matrices,
            // Linalg Solver
            s_lin_converged, s_all_converged,
            s_norm_temp);
        
        // ------ k3 -------
        l_temp_time = l_actual_time + rkck::a3 * l_time_step;
        #pragma unroll
        for (int j = 0; j < SD; j++){
            l_temp_state[j] = l_actual_state[j] + l_time_step * (rkck::b31 * k1[j] + rkck::b32 * k2[j]);
        }
        ode_fun(valid_lane, l_active,
            gsid,
            lsid, luid,
            l_temp_time,
            l_temp_state,
            k3,
            l_unit_params, l_system_params, l_control_params,
            l_coupling_factors,
            g_coupling_matrices,
            // Linalg Solver
            s_lin_converged, s_all_converged,
            s_norm_temp);
        
        // ------ k4 -------
        l_temp_time = l_actual_time + rkck::a4 * l_time_step;
        #pragma unroll
        for (int j = 0; j < SD; j++){
            l_temp_state[j] = l_actual_state[j] + l_time_step * (rkck::b41 * k1[j] + rkck::b42 * k2[j] + rkck::b43 * k3[j]);
        }
        ode_fun(valid_lane, l_active,
            gsid,
            lsid, luid,
            l_temp_time,
            l_temp_state,
            k4,
            l_unit_params, l_system_params, l_control_params,
            l_coupling_factors,
            g_coupling_matrices,
            // Linalg Solver
            s_lin_converged, s_all_converged,
            s_norm_temp);

        // ------ k5 -------
        l_temp_time = l_actual_time + rkck::a5 * l_time_step;
        #pragma unroll
        for (int j = 0; j < SD; j++){
            l_temp_state[j] = l_actual_state[j] + l_time_step * (rkck::b51 * k1[j] + rkck::b52 * k2[j] + rkck::b53 * k3[j] + rkck::b54 * k4[j]);
        }
        ode_fun(valid_lane, l_active,
            gsid,
            lsid, luid,
            l_temp_time,
            l_temp_state,
            k5,
            l_unit_params, l_system_params, l_control_params,
            l_coupling_factors,
            g_coupling_matrices,
            // Linalg Solver
            s_lin_converged, s_all_converged,
            s_norm_temp);

        // ------ k6 -------
        l_temp_time = l_actual_time + rkck::a6 * l_time_step;
        #pragma unroll
        for (int j = 0; j < SD; j++){
            l_temp_state[j] = l_actual_state[j] + l_time_step * (rkck::b61 * k1[j] + rkck::b62 * k2[j] + rkck::b63 * k3[j] + rkck::b64 * k4[j] + rkck::b65 * k5[j]);
 
        }
        ode_fun(valid_lane, l_active,
            gsid,
            lsid, luid,
            l_temp_time,
            l_temp_state,
            k6,
            l_unit_params, l_system_params, l_control_params,
            l_coupling_factors,
            g_coupling_matrices,
            // Linalg Solver
            s_lin_converged, s_all_converged,
            s_norm_temp);
        
        // --- NEW CANDIDATE STATE ---
        #pragma unroll
        for (int j = 0; j < SD; j++){
            l_new_state[j] = l_actual_state[j] + l_time_step * (rkck::c1 * k1[j] + rkck::c2 * k2[j] + rkck::c3 * k3[j] + rkck::c4 * k4[j] + rkck::c5 * k5[j] + rkck::c6 * k6[j]);
        }

        // --- NEW EVENT VALUE ---
        #if NE > 0
        event_fun(
            valid_lane, l_active,
            gsid,
            lsid, luid,
            s_next_event_value,
            l_actual_time + l_time_step,
            l_new_state,
            l_unit_params, l_system_params, l_control_params);
        #endif

        // ---------------------------------------------------
        // ---------------- Event Handling -------------------
        // ---------------------------------------------------        
        #if NE > 0
        if (tid < SPB && s_terminated_system[tid] == 0){
            double g_old = s_actual_event_value[tid];
            double g_new = s_next_event_value[tid];
            //int terminated = s_terminated_system[tid];

            // Előjelváltás detektálása
            bool has_crossing = (g_old * g_new < 0.0);
            // Linear interpolation
            // h_new = h_old * ( |g_old| / (|g_old| + |g_new|) )
            // Biztonsági okokból 0.99-al szorozva, hogy ne lépjünk túl rajta a kerekítés miatt
            double abs_old = abs(g_old);
            double denom   = abs_old + abs(g_new);
            double ratio_cal = (abs_old / (denom + 1.0e-15)) * 0.99;
            s_event_detected[tid] = (int)has_crossing;
            s_event_ratio[tid]    = has_crossing ? ratio_cal : 1;
        }
        __syncthreads();
        #endif

        // ---------------------------------------------------
        // ----------------- Error Handling ------------------
        // ---------------------------------------------------
        // -- Check Unit Tolerances --
        int unit_ok = 1;
        double unit_ratio = 1.0e300;
        #if NE > 0
        bool sys_event_detected = (s_event_detected[lsid] == 1);
        #else
        bool sys_event_detected = false;
        #endif
        if (valid_lane && l_active && !sys_event_detected){
            double e, r, tol;
            for (int j = 0; j < SD; j++){
                e = abs(l_time_step * (rkck::ce1 * k1[j] + rkck::ce2 * k2[j] + rkck::ce3 * k3[j] + rkck::ce4 * k4[j] + rkck::ce5 * k5[j] + rkck::ce6 * k6[j])) + 1.0e-18;
                tol = max(RTOL * max(abs(l_new_state[j]), abs(l_actual_state[j])), ATOL);
                r = tol / e;
                if (r < unit_ratio) unit_ratio = r;
                if (r < 1.0) unit_ok = 0;
            }
            sh_ratio[lsid][luid] = unit_ratio;
            sh_ok[lsid][luid]    = unit_ok;
        }
        __syncthreads();

        //if (luid == 0 && gsid == 28){
        //    printf("gsid %d, sh_ratio[0] = %.12f, sh_ratio[1] = %.12f \n", gsid, sh_ratio[lsid][0], sh_ratio[lsid][1]);
        //    printf("gsid %d, sh_ok[0] = %d, sh_ok[1] = %d \n", gsid, sh_ok[lsid][0], sh_ok[lsid][1]);
        //}

        //if (tid < SPB && s_terminated_system[tid] == 0){
        //    bool event_detected = (s_event_detected[tid] == 1);
        //    double event_ratio  = s_event_ratio[tid];
        //    double actual_time  = s_actual_time[tid];
        //    double time_end     = s_time_end[tid];
        //    double time_step    = s_time_step[tid];
        //    double rem_time     = time_end - (actual_time + time_step);
        //    int sys_ok          = 1;
        //    double sys_ratio    = 1.0e30;
        //    int ok;
        //    double rr;
        //    for (int u = 0; u < UPS; u++){
        //        ok = sh_ok[tid][u];
        //        rr = sh_ratio[tid][u];
        //        sys_ok = min(sys_ok, ok);
        //        sys_ratio = fmin(sys_ratio, rr);
        //    }
        //    sh_ok[tid][0] = sys_ok;         // system-wide information!
        //    double fac_ok = fmin(5.0, fmax(0.2, 0.9 * pow(sys_ratio, 0.2)));
        //    double fac_bad = fmin(0.5, fmax(0.1, 0.9 * pow(sys_ratio, 0.25)));
        //    double multiplier = (!event_detected) ? ((sys_ok == 1) ? fac_ok : fac_bad) : event_ratio;
        //    double new_time_step = fmin(fmin(fmax(time_step * multiplier, MIN_STEP), MAX_STEP), rem_time);
        //    s_time_step[tid] = new_time_step;
        //    s_actual_time[tid] = actual_time + ((sys_ok == 1 && !event_detected) ? time_step : 0.0);
        //    #if NE > 0
        //    s_actual_event_value[tid] = s_next_event_value[tid];
        //    #endif
        //}
        //__syncthreads();

        // System Leader --> Reduction
        if (valid_lane && l_active && (luid==0)){
            if (sys_event_detected){
                // Event Correction: reject last step and trunc the timestep
                //int sys_ok = 0;
                double sys_ratio    = s_event_ratio[lsid];
                s_time_step[lsid]   = l_time_step * sys_ratio;
                s_actual_time[lsid] = l_actual_time;
                sh_ok[lsid][0] = 0;
            } else {
                // Normal Error handgling
                // Initial guess from local observations
                int sys_ok = unit_ok;
                double sys_ratio = unit_ratio;

                //printf("before, gsid %d, sys_ok %d\n", gsid, sys_ok);
                //printf("before, gsid %d, sys_ratio %.12f\n", gsid, sys_ratio);

                int ok;
                double rr;
                for (int u = 1; u < UPS; u++){
                    ok = sh_ok[lsid][u];
                    rr = sh_ratio[lsid][u];
                    if (ok == 0) sys_ok = 0;
                    if (rr < sys_ratio) sys_ratio = rr;
                }
                sh_ok[lsid][0] = sys_ok;        // system-wide information!

                //printf("after, gsid %d, sys_ok %d\n", gsid, sys_ok);
                //printf("after, gsid %d, sys_ratio %.12f\n", gsid, sys_ratio);

                // New timestep candidate
                //double fac_ok = 0.9 * pow(sys_ratio, 0.2);
                //double fac_bad = 0.9 * pow(sys_ratio, 0.25);

                //if (fac_ok < 0.2) fac_ok = 0.2;
                //if (fac_ok > 5.0) fac_ok = 5.0;
                //if (fac_bad < 0.1) fac_bad = 0.1;
                //if (fac_bad > 0.5) fac_bad = 0.5;

                double fac_ok = min(5.0, max(0.2, 0.9 * pow(sys_ratio, 0.2)));
                double fac_bad = min(0.5, max(0.1, 0.9 * pow(sys_ratio, 0.25)));

                s_time_step[lsid] = (sys_ok == 1) ? l_time_step * fac_ok : l_time_step * fac_bad;
                s_actual_time[lsid] = l_actual_time + ((sys_ok == 1) ? l_time_step : 0.0);
                #if NE > 0
                if (sys_ok == 1){
                    s_actual_event_value[lsid] = s_next_event_value[lsid];
                }
                #endif
            }
        }
        __syncthreads();

        // Accept new state and update
        if (valid_lane && l_active){
            int sys_ok = sh_ok[lsid][0];
            if (sys_ok == 1){
                //l_actual_time = l_actual_time + l_time_step;
                #pragma unroll
                for (int j = 0; j < SD; j++){
                    l_actual_state[j] = l_new_state[j];
                }
            }
            // Read new time-step candidate
            l_time_step = s_time_step[lsid];
            l_actual_time = s_actual_time[lsid];
        }
        __syncthreads();

        #if NDO > 0
        store_dense_output(
            valid_lane, l_active,
            gsid, gtid,
            lsid, luid,
            l_actual_time,
            l_time_end,
            l_actual_state,
            s_dense_output_store,
            s_dense_output_time_index,
            s_dense_output_time,
            s_dense_output_min_time_step,
            g_dense_time,
            g_dense_state);
        #endif

        // Am I finished?
        //if (valid_lane){
        //    l_active = (l_actual_time < l_time_end);
        //}

        if (valid_lane && l_active && (luid == 0)){
            if (s_actual_time[lsid] >= s_time_end[lsid]){
                s_terminated_system[lsid] = 1;
                g_status_flags[gsid] = SUCCESS;
                atomicAdd(&s_all_terminated[0], 1);
            }
            #if NE > 0
            if (s_terminated_system[lsid] == 0 && abs(s_actual_event_value[lsid]) < ETOL){
                //printf("gsid = %d, event-terminated \n", gsid);
                s_terminated_system[lsid] = 1;
                g_status_flags[gsid] = EVENT_TERMINAL;
                atomicAdd(&s_all_terminated[0], 1);
            }
            #endif
        }
        __syncthreads();
        all_terminated = (s_all_terminated[0] >= SPB);
        if (all_terminated){
            //printf("gtid = %d, iter-break\n", gtid);
            break;
        }

        l_active = valid_lane && (s_terminated_system[lsid] == 0);
        //printf("step %d", step);
    }

    //  --- GLOBAL STORE ---
    if (valid_lane){
        if (luid == 0){
            s_actual_time[lsid] = l_actual_time;
            s_time_step[lsid]   = l_time_step;      // Can be removed?
        }
        #pragma unroll
        for (int j = 0; j < SD; j++){
            g_actual_state[(j * NS * TILE) + gtid] = l_actual_state[j];
        }
    }
    __syncthreads();
    if (tid < SPB) {
        int gid = bid * SPB + tid;
        g_actual_time[gid] = s_actual_time[tid];
        g_time_step[gid]   = s_time_step[tid];

        #if NDO > 0
        g_dense_index[gid] = s_dense_output_time_index[tid];
        #endif
        #if NE > 0
        g_actual_event[gid] = s_actual_event_value[tid];
        #endif
    }
    __syncthreads();

}