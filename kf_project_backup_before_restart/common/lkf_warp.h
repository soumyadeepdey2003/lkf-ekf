/* lkf_warp.h - UPGRADED LKF in the WARP-PER-TRACK mapping: 32 lanes cooperate on ONE track, 8 phases.
 * It computes exactly the same equations as lkf_light_step (lkf_lightweight.h), but distributes the independent scalar
 * equations of every stage over the lanes. Phase table (what each lane does):
 *   0 load      : lanes copy the track into shared scratch
 *   1 predict   : 24 lanes -> 3 state entries, 6 unique A-, 9 C-, 6 unique D- (independent)
 *   2 innovation: lanes 0-2 -> y ; lanes 3-8 -> the 6 unique entries of S = A- + R
 *   3 factor    : lane 0 only -> choose the solver case and compute the factors (the stage that must finish first)
 *   4 solve     : lanes 0-5 -> one gain column each (the 6 independent right-hand sides) + that state entry's correction
 *   5 Joseph    : 63 independent intermediate products (lam, eta, e, n, phi, O, J: 7 x 9) over the 32 lanes
 *   6 Joseph    : 21 lanes -> the 6 unique A+, 9 C+, 6 unique D+ entries
 *   7 store     : lanes write state and covariance back */
#ifndef LKF_WARP_H
#define LKF_WARP_H
#include "lkf_lightweight.h"
#include "kf_warp_common.h"

struct LkfWarp {
    enum { NPHASE = 8 };
    struct Ctx {
        float x[6];
        float A[9], C[9], D[9];          /* input covariance blocks            */
        float Ap[9], Cp[9], Dp[9];       /* predicted blocks (A-, C-, D-)       */
        float S[9], y[3], K[18], fac[8]; /* innovation cov., innovation, gain, solver factors */
        float lam[9], eta[9], e[9], n[9], phi[9], O[9], J[9];   /* Joseph intermediates */
        float Au[9], Cu[9], Du[9];       /* updated blocks (A+, C+, D+)         */
        int mode, r, ok;                 /* solver case, isolated index (block case), "update possible" */
    };

    /* ---- phase 3 (lane 0): detect the structure of S and compute what the six column solves share ---- */
    KF_HD static void factor(Ctx &c, int valid) {
        c.ok = 0; c.mode = 0; c.r = 0;
        if (!valid) return;
        const float *S = c.S;
        c.mode = lkf_detect(S, &c.r);
        if (c.mode == LKF_MODE_SPD) {                                         /* Cholesky; if it fails use Gauss-Jordan */
            const float l11 = (S[0] > 0.f) ? sqrtf(S[0]) : 0.f;
            if (l11 > 0.f) {
                const float i11 = 1.f / l11, l21 = S[3] * i11, l31 = S[6] * i11, d2 = KF_SUB(S[4], l21 * l21);
                if (d2 > 0.f) {
                    const float l22 = sqrtf(d2), i22 = 1.f / l22, l32 = KF_SUB(S[7], l31 * l21) * i22;
                    const float d3 = KF_SUB(KF_SUB(S[8], l31 * l31), l32 * l32);
                    if (d3 > 0.f) {
                        const float l33 = sqrtf(d3);
                        c.fac[0] = l21; c.fac[1] = l31; c.fac[2] = l32; c.fac[3] = i11; c.fac[4] = i22; c.fac[5] = 1.f / l33;
                        c.ok = 1; return;
                    }
                }
            }
            c.mode = LKF_MODE_GJ;
        }
        switch (c.mode) {
        case LKF_MODE_IDENTITY: c.ok = 1; break;
        case LKF_MODE_DIAG:
        case LKF_MODE_TRI_LOWER:
        case LKF_MODE_TRI_UPPER:
            c.ok = (fabsf(S[0]) >= KF_PIVOT_EPS && fabsf(S[4]) >= KF_PIVOT_EPS && fabsf(S[8]) >= KF_PIVOT_EPS);
            if (c.ok) { c.fac[0] = 1.f / S[0]; c.fac[1] = 1.f / S[4]; c.fac[2] = 1.f / S[8]; }
            break;
        case LKF_MODE_BLOCK: {
            const int r = c.r, p = (r == 0) ? 1 : 0, q = (r == 2) ? 1 : 2;
            const float delta = KF_SUB(S[p*3+p] * S[q*3+q], S[p*3+q] * S[q*3+p]), srr = S[r*3+r];
            c.ok = (fabsf(delta) >= KF_PIVOT_EPS && fabsf(srr) >= KF_PIVOT_EPS);
            if (c.ok) { c.fac[0] = 1.f / delta; c.fac[1] = 1.f / srr; }
            break; }
        default: {                                                            /* general: is S nonsingular? (dry run) */
            float z[9] = {0}, Kt[18];
            c.ok = lkf_gain_gauss_jordan(S, z, z, Kt);
            break; }
        }
    }

    /* ---- phase 4 (one lane per column j): solve S k = b_j for one right-hand side ---- */
    KF_HD static void solve_col(const Ctx &c, int j, float k[3]) {
        const float *S = c.S, *f = c.fac;
        const float b1 = kf_B(c.Ap, c.Cp, 0, j), b2 = kf_B(c.Ap, c.Cp, 1, j), b3 = kf_B(c.Ap, c.Cp, 2, j);
        switch (c.mode) {
        case LKF_MODE_IDENTITY: k[0] = b1; k[1] = b2; k[2] = b3; break;
        case LKF_MODE_DIAG:     k[0] = b1 * f[0]; k[1] = b2 * f[1]; k[2] = b3 * f[2]; break;
        case LKF_MODE_BLOCK: {
            const int r = c.r, p = (r == 0) ? 1 : 0, q = (r == 2) ? 1 : 2;
            const float b[3] = { b1, b2, b3 };
            const float spp = S[p*3+p], sqq = S[q*3+q], spq = S[p*3+q], sqp = S[q*3+p];
            k[p] = KF_SUB(sqq * b[p], spq * b[q]) * f[0];
            k[q] = KF_SUB(spp * b[q], sqp * b[p]) * f[0];
            k[r] = b[r] * f[1];
            break; }
        case LKF_MODE_TRI_LOWER:
            k[0] = b1 * f[0]; k[1] = (b2 + (-S[3] * k[0])) * f[1]; k[2] = (b3 + (-S[6] * k[0]) + (-S[7] * k[1])) * f[2]; break;
        case LKF_MODE_TRI_UPPER:
            k[2] = b3 * f[2]; k[1] = (b2 + (-S[5] * k[2])) * f[1]; k[0] = (b1 + (-S[1] * k[1]) + (-S[2] * k[2])) * f[0]; break;
        case LKF_MODE_SPD: {                                                  /* fac = l21 l31 l32 1/l11 1/l22 1/l33 */
            const float w1 = b1 * f[3], w2 = KF_SUB(b2, f[0] * w1) * f[4], w3 = KF_SUB(KF_SUB(b3, f[1] * w1), f[2] * w2) * f[5];
            k[2] = w3 * f[5]; k[1] = KF_SUB(w2, f[2] * k[2]) * f[4]; k[0] = KF_SUB(KF_SUB(w1, f[0] * k[1]), f[1] * k[2]) * f[3];
            break; }
        default: {                                                            /* Gauss-Jordan on this one column */
            float W[9], kv[3] = { b1, b2, b3 };
            for (int i = 0; i < 9; i++) W[i] = S[i];
            for (int s = 0; s < 3; s++) {
                int pk = s; float best = fabsf(W[s*3+s]);
                for (int r = s + 1; r < 3; r++) if (fabsf(W[r*3+s]) > best) { best = fabsf(W[r*3+s]); pk = r; }
                if (pk != s) { for (int m = 0; m < 3; m++) { float t = W[s*3+m]; W[s*3+m] = W[pk*3+m]; W[pk*3+m] = t; }
                               float t = kv[s]; kv[s] = kv[pk]; kv[pk] = t; }
                const float inv = 1.f / W[s*3+s];
                for (int m = 0; m < 3; m++) W[s*3+m] *= inv;
                kv[s] *= inv;
                for (int r = 0; r < 3; r++) if (r != s) {
                    const float fr = W[r*3+s];
                    for (int m = 0; m < 3; m++) W[r*3+m] += -fr * W[s*3+m];
                    kv[r] += -fr * kv[s]; W[r*3+s] = 0.f;
                }
            }
            k[0] = kv[0]; k[1] = kv[1]; k[2] = kv[2];
            break; }
        }
    }

    /* ---- phase 5: ONE of the 63 intermediate products (w = 0..62: 7 kinds x 9 entries) ---- */
    KF_HD static void joseph_item(Ctx &c, const float *R, int w) {
        const int kind = w / 9, idx = w % 9, a = idx / 3, b = idx % 3;
        const float *U = c.K, *V = c.K + 9;
        float s = 0.f;
        switch (kind) {
        case 0: for (int q = 0; q < 3; q++) s += KF_SUB((a == q ? 1.f : 0.f), U[a*3+q]) * c.Ap[q*3+b]; c.lam[idx] = s; break;   /* M A- */
        case 1: for (int q = 0; q < 3; q++) s += U[a*3+q] * R[q*3+b]; c.eta[idx] = s; break;                                     /* U R  */
        case 2: s = c.Cp[idx]; for (int q = 0; q < 3; q++) s = KF_SUB(s, c.Ap[a*3+q] * V[b*3+q]); c.e[idx] = s; break;           /* C- - A- V^T */
        case 3: for (int q = 0; q < 3; q++) s += V[a*3+q] * c.Ap[q*3+b]; c.n[idx] = s; break;                                    /* V A- */
        case 4: for (int q = 0; q < 3; q++) s += V[a*3+q] * R[q*3+b]; c.phi[idx] = s; break;                                     /* V R  */
        case 5: for (int q = 0; q < 3; q++) s += c.Cp[q*3+a] * V[b*3+q]; c.O[idx] = s; break;                                    /* C-^T V^T */
        default: for (int q = 0; q < 3; q++) s += V[a*3+q] * c.Cp[q*3+b]; c.J[idx] = s; break;                                   /* V C- */
        }
    }
    /* ---- phase 6: ONE of the 21 result entries ---- */
    KF_HD static void joseph_out(Ctx &c, const float *R, int lane) {
        (void)R;
        const float *U = c.K, *V = c.K + 9;
        if (lane < 6) {                                                       /* unique A+ */
            int a, b; kf_uidx(lane, &a, &b); float s = 0.f;
            for (int q = 0; q < 3; q++) s += c.lam[a*3+q] * KF_SUB((b == q ? 1.f : 0.f), U[b*3+q]) + c.eta[a*3+q] * U[b*3+q];
            c.Au[a*3+b] = s; c.Au[b*3+a] = s;
        } else if (lane < 15) {                                               /* all 9 C+ */
            const int i = lane - 6, a = i / 3, b = i % 3; float s = 0.f;
            for (int q = 0; q < 3; q++) s += KF_SUB((a == q ? 1.f : 0.f), U[a*3+q]) * c.e[q*3+b] + c.eta[a*3+q] * V[b*3+q];
            c.Cu[i] = s;
        } else if (lane < 21) {                                               /* unique D+ */
            int a, b; kf_uidx(lane - 15, &a, &b);
            float s = KF_SUB(KF_SUB(c.Dp[a*3+b], c.O[a*3+b]), c.J[a*3+b]);
            for (int q = 0; q < 3; q++) s += c.n[a*3+q] * V[b*3+q] + c.phi[a*3+q] * V[b*3+q];
            c.Du[a*3+b] = s; c.Du[b*3+a] = s;
        }
    }

    KF_HD static void phase(Ctx &c, Track *t, const Meas *m, const KFParams *p, int lane, int k) {
        switch (k) {
        case 0: kf_warp_load(c, t, lane); break;
        case 1: kf_warp_predict(c, p, lane); break;
        case 2:
            if (lane < 3) c.y[lane] = m->z[lane] + (-c.x[lane]);
            else if (lane < 9) { int a, b; kf_uidx(lane - 3, &a, &b); const float v = c.Ap[a*3+b] + p->R[a*3+b]; c.S[a*3+b] = v; c.S[b*3+a] = v; }
            break;
        case 3: if (lane == 0) factor(c, m->valid); break;
        case 4:
            if (lane < 6 && c.ok) {
                float kk[3]; solve_col(c, lane, kk);
                c.K[lane*3] = kk[0]; c.K[lane*3+1] = kk[1]; c.K[lane*3+2] = kk[2];
                c.x[lane] += kk[0] * c.y[0] + kk[1] * c.y[1] + kk[2] * c.y[2];
            }
            break;
        case 5: if (c.ok) for (int w = lane; w < 63; w += KF_WARP) joseph_item(c, p->R, w); break;
        case 6: if (lane < 21) { if (c.ok) joseph_out(c, p->R, lane); else kf_warp_copy_pred(c, lane); } break;
        default: kf_warp_store(c, t, lane); break;
        }
    }
};
#endif
