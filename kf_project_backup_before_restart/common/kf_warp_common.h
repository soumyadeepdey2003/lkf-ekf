/* kf_warp_common.h - shared pieces of the "warp-per-track" mapping (paper Sec. 3.1 and 3.6.2).
 *
 * IDEA: in the thread-per-track mapping one GPU thread computes a whole filter update alone. The scalar equations of the
 * upgraded filters are INDEPENDENT inside a stage (e.g. the 6 unique entries of A-, the 9 entries of C-, the 6 gain columns...),
 * so here the 32 threads ("lanes") of ONE warp cooperate on ONE track: each lane computes one or a few scalar results and the
 * warp synchronises at the boundary between dependent stages ("phases").
 *
 * Rules every phase must obey (checked by tests/test_warp.cpp, which runs the lanes of a phase in random order):
 *   1. a phase only READS values that were finished in EARLIER phases (or are inputs),
 *   2. two lanes never WRITE the same memory cell in the same phase.
 * Ctx = the warp's private scratch space; on the GPU it lives in fast __shared__ memory. */
#ifndef KF_WARP_COMMON_H
#define KF_WARP_COMMON_H
#include <string.h>
#include "kf_types.h"

#define KF_WARP 32               /* threads (lanes) per warp */
#define KF_WARPS_PER_BLOCK 4     /* 4 warps = 128 threads per block: 4 tracks per block */

/* the 6 unique entries of a symmetric 3x3: u = 0..5 -> (a,b) = (0,0) (0,1) (0,2) (1,1) (1,2) (2,2) */
KF_HD static inline void kf_uidx(int u, int *a, int *b) {
    *a = (u < 3) ? 0 : ((u < 5) ? 1 : 2);
    *b = (u < 5) ? ((u < 3) ? u : u - 2) : 2;
}

/* phase 0: the lanes copy the track from global memory into the warp's scratch blocks A, C, D and the state x */
template <class Ctx> KF_HD static inline void kf_warp_load(Ctx &c, const Track *t, int lane) {
    for (int idx = lane; idx < 36; idx += KF_WARP) {
        const int r = idx / 6, cc = idx % 6; const float v = t->P[idx];
        if (r < 3 && cc < 3) c.A[r*3+cc] = v;
        else if (r < 3)      c.C[r*3+cc-3] = v;
        else if (cc >= 3)    c.D[(r-3)*3+cc-3] = v;              /* lower-left block = C^T: not needed */
    }
    if (lane < 6) c.x[lane] = t->x[lane];
}
/* phase 1: state + covariance prediction. 24 INDEPENDENT scalar results -> lanes 0..23 */
template <class Ctx> KF_HD static inline void kf_warp_predict(Ctx &c, const KFParams *p, int lane) {
    const float dt = p->dt, dt2 = dt * dt;
    const float qpp = p->sa2 * dt2 * dt2 * 0.25f, qpv = p->sa2 * dt2 * dt * 0.5f, qvv = p->sa2 * dt2;
    if (lane < 3) {                                              /* lanes 0-2: position += dt * velocity */
        c.x[lane] = c.x[lane] + dt * c.x[lane + 3];
    } else if (lane < 9) {                                       /* lanes 3-8: the 6 unique entries of A- */
        int a, b; kf_uidx(lane - 3, &a, &b); float v;
        if (a == b) { const float t = dt * c.C[a*3+a]; v = c.A[a*3+a] + t + t + dt2 * c.D[a*3+a] + qpp; }
        else        { v = c.A[a*3+b] + dt * c.C[a*3+b] + dt * c.C[b*3+a] + dt2 * c.D[a*3+b]; }
        c.Ap[a*3+b] = v; c.Ap[b*3+a] = v;
    } else if (lane < 18) {                                      /* lanes 9-17: all 9 entries of C- */
        const int i = lane - 9, a = i / 3, b = i % 3;
        float v = c.C[i] + dt * c.D[a*3+b]; if (a == b) v += qpv;
        c.Cp[i] = v;
    } else if (lane < 24) {                                      /* lanes 18-23: the 6 unique entries of D- */
        int a, b; kf_uidx(lane - 18, &a, &b);
        float d = c.D[a*3+b]; if (a == b) d += qvv;
        c.Dp[a*3+b] = d; c.Dp[b*3+a] = d;
    }
}
/* when there is no update (no measurement / unusable S) the predicted blocks are the result: 21 independent copies */
template <class Ctx> KF_HD static inline void kf_warp_copy_pred(Ctx &c, int lane) {
    if (lane < 6)       { int a, b; kf_uidx(lane, &a, &b);      c.Au[a*3+b] = c.Ap[a*3+b]; c.Au[b*3+a] = c.Ap[b*3+a]; }
    else if (lane < 15) { const int i = lane - 6;                c.Cu[i] = c.Cp[i]; }
    else if (lane < 21) { int a, b; kf_uidx(lane - 15, &a, &b); c.Du[a*3+b] = c.Dp[a*3+b]; c.Du[b*3+a] = c.Dp[b*3+a]; }
}
/* last phase: write the final state and the full 6x6 covariance back to global memory */
template <class Ctx> KF_HD static inline void kf_warp_store(const Ctx &c, Track *t, int lane) {
    if (lane < 6) t->x[lane] = c.x[lane];
    for (int idx = lane; idx < 36; idx += KF_WARP) {
        const int r = idx / 6, cc = idx % 6; float v;
        if (r < 3 && cc < 3)  v = c.Au[r*3+cc];
        else if (r < 3)       v = c.Cu[r*3+cc-3];
        else if (cc < 3)      v = c.Cu[cc*3+(r-3)];              /* lower-left block = transpose of C */
        else                  v = c.Du[(r-3)*3+cc-3];
        t->P[idx] = v;
    }
}

/* a "filter" without a warp implementation (the ORIGINAL filters): keeps the templates compiling */
struct NoWarp {
    struct Ctx { int d; };
    enum { NPHASE = 0 };
    KF_HD static void phase(Ctx &, Track *, const Meas *, const KFParams *, int, int) {}
};

/* Host emulation of the warp execution: all lanes run phase k, then phase k+1 ... exactly the order a real warp
 * with __syncwarp() between phases guarantees. order_seed != 0 runs the lanes of every phase in a RANDOM order:
 * if the result changes, a phase has a hidden dependency between lanes (a bug). Used by the tests and by the CUDA shim. */
template <class W> static void kf_warp_emulate(Track *T, const Meas *M, int n, KFParams p, unsigned order_seed = 0) {
    for (int i = 0; i < n; i++) {
        typename W::Ctx c; memset(&c, 0xFF, sizeof c);       /* poison: floats = NaN, so a read of a value no earlier phase wrote shows up */
        unsigned s = order_seed * 2654435761u + (unsigned)i + 1u;
        for (int k = 0; k < (int)W::NPHASE; k++) {
            int ord[KF_WARP]; for (int l = 0; l < KF_WARP; l++) ord[l] = l;
            if (order_seed) for (int l = KF_WARP - 1; l > 0; l--) { s = s * 1664525u + 1013904223u; const int j = (int)((s >> 8) % (unsigned)(l + 1)); const int t = ord[l]; ord[l] = ord[j]; ord[j] = t; }
            for (int l = 0; l < KF_WARP; l++) W::phase(c, &T[i], &M[i], &p, ord[l], k);
        }
    }
}
#endif
