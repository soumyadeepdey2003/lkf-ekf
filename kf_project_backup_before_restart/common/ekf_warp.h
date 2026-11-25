/* ekf_warp.h - UPGRADED EKF in the WARP-PER-TRACK mapping: 32 lanes cooperate on ONE track, 12 phases.
 * Same equations as ekf_light_step (ekf_lightweight.h). What is parallel and what is NOT (paper Sec. 3.6.2):
 *   - CORDIC iterations and the reciprocals depend on the previous iteration -> ONE lane does them (phase 2);
 *     parallelism for this stage exists ACROSS tracks (other warps), not inside it.
 *   - everything else is independent scalar work spread over the lanes.
 * Phase table:
 *   0 load | 1 predict (24 lanes) | 2 CORDIC + reciprocals (lane 0) | 3 Jacobian 9 lanes + innovation 3 lanes
 *   4 T = Hp [A- C-] (18 lanes) | 5 S, 6 unique entries | 6 factor (lane 0) | 7 solve: 6 lanes = 6 gain columns + state update
 *   8 M and N (18 lanes) | 9 Joseph intermediates (63 items) | 10 Joseph results (21 lanes) | 11 store */
#ifndef EKF_WARP_H
#define EKF_WARP_H
#include "ekf_lightweight.h"
#include "kf_warp_common.h"

enum { EKF_MODE_IDENTITY = 0, EKF_MODE_DIAG, EKF_MODE_BLOCK, EKF_MODE_LDL, EKF_MODE_GJ };

struct EkfWarp {
    enum { NPHASE = 12 };
    struct Ctx {
        float x[6];
        float A[9], C[9], D[9], Ap[9], Cp[9], Dp[9];
        float rxy, rho, th, ph, irho, ixy2, irho2, irho2xy;     /* CORDIC results and reciprocal products */
        float Hp[9], nu[3], TL[9], TR[9], S[9], K[18], fac[8];
        float M[9], N[9];
        float al[9], ga[9], e[9], om[9], ps[9], chi[9], xi[9];
        float Au[9], Cu[9], Du[9];
        int mode, r, ok;
    };

    /* ---- phase 6 (lane 0): structure of S (same test order as ekf_light_gain) and the shared factors ---- */
    KF_HD static void factor(Ctx &c) {
        const float *S = c.S;
        c.mode = EKF_MODE_GJ; c.r = 0;
        if (S[1] == 0.f && S[2] == 0.f && S[5] == 0.f) c.mode = (S[0] == 1.f && S[4] == 1.f && S[8] == 1.f) ? EKF_MODE_IDENTITY : EKF_MODE_DIAG;
        else if (S[2] == 0.f && S[5] == 0.f) { c.mode = EKF_MODE_BLOCK; c.r = 2; }
        else if (S[1] == 0.f && S[5] == 0.f) { c.mode = EKF_MODE_BLOCK; c.r = 1; }
        else if (S[1] == 0.f && S[2] == 0.f) { c.mode = EKF_MODE_BLOCK; c.r = 0; }
        else c.mode = EKF_MODE_LDL;
        if (c.mode == EKF_MODE_IDENTITY) { c.ok = 1; return; }
        if (c.mode == EKF_MODE_DIAG) {
            c.ok = (fabsf(S[0]) >= 1e-30f && fabsf(S[4]) >= 1e-30f && fabsf(S[8]) >= 1e-30f);
            if (c.ok) { c.fac[0] = kf_recip(S[0]); c.fac[1] = kf_recip(S[4]); c.fac[2] = kf_recip(S[8]); }
            return;
        }
        if (c.mode == EKF_MODE_BLOCK) {
            const int r = c.r, p = (r == 0) ? 1 : 0, q = (r == 2) ? 1 : 2;
            const float spp = S[p*3+p], sqq = S[q*3+q], spq = S[p*3+q], sqp = S[q*3+p], srr = S[r*3+r];
            const float delta = spp * sqq + (-spq * sqp);
            c.ok = !(fabsf(delta) <= KF_PIVOT_REL * fabsf(spp * sqq) || fabsf(srr) < 1e-30f);
            if (c.ok) { c.fac[0] = kf_recip(delta); c.fac[1] = kf_recip(srr); }
            return;
        }
        if (c.mode == EKF_MODE_LDL) {                                  /* S = L D L^T, no square roots */
            const float z1 = S[0];
            if (!(z1 <= KF_PIVOT_REL * S[0])) {
                const float b1 = kf_recip(z1), l21 = S[3] * b1, l31 = S[6] * b1;
                const float z2 = S[4] + (-l21 * l21 * z1);
                if (!(z2 <= KF_PIVOT_REL * S[4])) {
                    const float b2 = kf_recip(z2), l32 = (S[7] + (-l31 * l21 * z1)) * b2;
                    const float z3 = S[8] + (-l31 * l31 * z1) + (-l32 * l32 * z2);
                    if (!(z3 <= KF_PIVOT_REL * S[8])) {
                        c.fac[0] = l21; c.fac[1] = l31; c.fac[2] = l32; c.fac[3] = b1; c.fac[4] = b2; c.fac[5] = kf_recip(z3);
                        c.ok = 1; return;
                    }
                }
            }
            c.mode = EKF_MODE_GJ;
        }
        float z[9] = {0}, Kt[18];                                      /* general: nonsingular? (dry run) */
        c.ok = ekf_gain_gj(S, z, z, Kt);
    }

    /* ---- phase 7 (one lane per gain column) ---- */
    KF_HD static void solve_col(const Ctx &c, int j, float k[3]) {
        const float *S = c.S, *f = c.fac;
        const float t1 = kf_B(c.TL, c.TR, 0, j), t2 = kf_B(c.TL, c.TR, 1, j), t3 = kf_B(c.TL, c.TR, 2, j);
        switch (c.mode) {
        case EKF_MODE_IDENTITY: k[0] = t1; k[1] = t2; k[2] = t3; break;
        case EKF_MODE_DIAG:     k[0] = t1 * f[0]; k[1] = t2 * f[1]; k[2] = t3 * f[2]; break;
        case EKF_MODE_BLOCK: {
            const int r = c.r, p = (r == 0) ? 1 : 0, q = (r == 2) ? 1 : 2;
            const float t[3] = { t1, t2, t3 };
            const float spp = S[p*3+p], sqq = S[q*3+q], spq = S[p*3+q], sqp = S[q*3+p];
            k[p] = (sqq * t[p] + (-spq * t[q])) * f[0];
            k[q] = (spp * t[q] + (-sqp * t[p])) * f[0];
            k[r] = t[r] * f[1];
            break; }
        case EKF_MODE_LDL: {                                           /* fac = l21 l31 l32 b1 b2 b3 */
            const float w1 = t1, w2 = t2 + (-f[0] * w1), w3 = t3 + (-f[1] * w1) + (-f[2] * w2);
            const float v1 = w1 * f[3], v2 = w2 * f[4], v3 = w3 * f[5];
            k[2] = v3; k[1] = v2 + (-f[2] * k[2]); k[0] = v1 + (-f[0] * k[1]) + (-f[1] * k[2]);
            break; }
        default: {                                                     /* Gauss-Jordan with RECIP on this one column */
            float W[9], kv[3] = { t1, t2, t3 }, smax = 0.f;
            for (int i = 0; i < 9; i++) { W[i] = S[i]; if (fabsf(S[i]) > smax) smax = fabsf(S[i]); }
            for (int s = 0; s < 3; s++) {
                int pk = s; float best = fabsf(W[s*3+s]);
                for (int r = s + 1; r < 3; r++) if (fabsf(W[r*3+s]) > best) { best = fabsf(W[r*3+s]); pk = r; }
                if (pk != s) { for (int m = 0; m < 3; m++) { float t = W[s*3+m]; W[s*3+m] = W[pk*3+m]; W[pk*3+m] = t; }
                               float t = kv[s]; kv[s] = kv[pk]; kv[pk] = t; }
                const float beta = kf_recip(W[s*3+s]);
                for (int m = 0; m < 3; m++) W[s*3+m] *= beta;
                kv[s] *= beta;
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

    /* ---- phase 9: ONE of the 63 intermediate products (7 kinds x 9 entries) ---- */
    KF_HD static void joseph_item(Ctx &c, const float *R, int w) {
        const int kind = w / 9, idx = w % 9, a = idx / 3, b = idx % 3;
        const float *U = c.K, *V = c.K + 9;
        float s = 0.f;
        switch (kind) {
        case 0: for (int q = 0; q < 3; q++) s += c.M[a*3+q] * c.Ap[q*3+b]; c.al[idx] = s; break;                       /* alpha = M A-  */
        case 1: for (int q = 0; q < 3; q++) s += U[a*3+q] * R[q*3+b]; c.ga[idx] = s; break;                            /* gamma = U R   */
        case 2: s = c.Cp[idx]; for (int q = 0; q < 3; q++) s += c.Ap[a*3+q] * c.N[b*3+q]; c.e[idx] = s; break;         /* e = C- + A- N^T */
        case 3: for (int q = 0; q < 3; q++) s += c.N[a*3+q] * c.Ap[q*3+b]; c.om[idx] = s; break;                       /* omega = N A-  */
        case 4: for (int q = 0; q < 3; q++) s += V[a*3+q] * R[q*3+b]; c.ps[idx] = s; break;                            /* psi = V R     */
        case 5: for (int q = 0; q < 3; q++) s += c.Cp[q*3+a] * c.N[b*3+q]; c.chi[idx] = s; break;                      /* chi = C-^T N^T */
        default: for (int q = 0; q < 3; q++) s += c.N[a*3+q] * c.Cp[q*3+b]; c.xi[idx] = s; break;                      /* xi = N C-     */
        }
    }
    /* ---- phase 10: ONE of the 21 result entries ---- */
    KF_HD static void joseph_out(Ctx &c, int lane) {
        const float *U = c.K, *V = c.K + 9;
        if (lane < 6) {
            int a, b; kf_uidx(lane, &a, &b); float s = 0.f;
            for (int q = 0; q < 3; q++) s += c.al[a*3+q] * c.M[b*3+q] + c.ga[a*3+q] * U[b*3+q];
            c.Au[a*3+b] = s; c.Au[b*3+a] = s;
        } else if (lane < 15) {
            const int i = lane - 6, a = i / 3, b = i % 3; float s = 0.f;
            for (int q = 0; q < 3; q++) s += c.M[a*3+q] * c.e[q*3+b] + c.ga[a*3+q] * V[b*3+q];
            c.Cu[i] = s;
        } else if (lane < 21) {
            int a, b; kf_uidx(lane - 15, &a, &b);
            float s = c.chi[a*3+b] + c.xi[a*3+b] + c.Dp[a*3+b];
            for (int q = 0; q < 3; q++) s += c.om[a*3+q] * c.N[b*3+q] + c.ps[a*3+q] * V[b*3+q];
            c.Du[a*3+b] = s; c.Du[b*3+a] = s;
        }
    }

    KF_HD static void phase(Ctx &c, Track *t, const Meas *m, const KFParams *p, int lane, int k) {
        switch (k) {
        case 0: kf_warp_load(c, t, lane); break;
        case 1: kf_warp_predict(c, p, lane); break;
        case 2:                                                           /* sequential stage: one lane */
            if (lane == 0) {
                c.ok = 0;
                if (m->valid) {
                    ekf_light_meas(c.x[0], c.x[1], c.x[2], &c.rxy, &c.rho, &c.th, &c.ph);
                    if (c.rxy >= KF_RANGE_EPS && c.rho >= KF_RANGE_EPS) {
                        const float ixy = kf_recip(c.rxy); c.irho = kf_recip(c.rho);
                        c.ixy2 = ixy * ixy; c.irho2 = c.irho * c.irho; c.irho2xy = c.irho2 * ixy; c.ok = 1;
                    }
                }
            }
            break;
        case 3:
            if (c.ok) {
                const float x = c.x[0], y = c.x[1], z = c.x[2];
                switch (lane) {
                case 0: c.Hp[0] = x * c.irho; break;               case 1: c.Hp[1] = y * c.irho; break;     case 2: c.Hp[2] = z * c.irho; break;
                case 3: c.Hp[3] = (-y) * c.ixy2; break;            case 4: c.Hp[4] = x * c.ixy2; break;     case 5: c.Hp[5] = 0.f; break;
                case 6: c.Hp[6] = (-x * z) * c.irho2xy; break;     case 7: c.Hp[7] = (-y * z) * c.irho2xy; break;
                case 8: c.Hp[8] = c.rxy * c.irho2; break;
                case 9:  c.nu[0] = m->z[0] + (-c.rho); break;
                case 10: c.nu[1] = kf_wrap_pi_light(m->z[1] + (-c.th)); break;
                case 11: c.nu[2] = m->z[2] + (-c.ph); break;
                default: break;
                }
            }
            break;
        case 4:
            if (c.ok && lane < 18) {
                const int i = (lane < 9) ? lane : lane - 9, a = i / 3, j = i % 3; float s = 0.f;
                if (lane < 9) { for (int q = 0; q < 3; q++) s += c.Hp[a*3+q] * c.Ap[q*3+j]; c.TL[i] = s; }
                else          { for (int q = 0; q < 3; q++) s += c.Hp[a*3+q] * c.Cp[q*3+j]; c.TR[i] = s; }
            }
            break;
        case 5:
            if (c.ok && lane < 6) {
                int a, b; kf_uidx(lane, &a, &b); float s = p->R[a*3+b];
                for (int q = 0; q < 3; q++) s += c.TL[a*3+q] * c.Hp[b*3+q];
                c.S[a*3+b] = s; c.S[b*3+a] = s;
            }
            break;
        case 6: if (lane == 0 && c.ok) factor(c); break;
        case 7:
            if (c.ok && lane < 6) {
                float kk[3]; solve_col(c, lane, kk);
                c.K[lane*3] = kk[0]; c.K[lane*3+1] = kk[1]; c.K[lane*3+2] = kk[2];
                c.x[lane] += kk[0] * c.nu[0] + kk[1] * c.nu[1] + kk[2] * c.nu[2];
            }
            break;
        case 8:
            if (c.ok && lane < 18) {
                const int i = (lane < 9) ? lane : lane - 9, a = i / 3, b = i % 3;
                const float *U = c.K, *V = c.K + 9;
                if (lane < 9) { float m_ = (a == b) ? 1.f : 0.f; for (int q = 0; q < 3; q++) m_ += -U[a*3+q] * c.Hp[q*3+b]; c.M[i] = m_; }
                else          { float n_ = 0.f;                  for (int q = 0; q < 3; q++) n_ += -V[a*3+q] * c.Hp[q*3+b]; c.N[i] = n_; }
            }
            break;
        case 9: if (c.ok) for (int w = lane; w < 63; w += KF_WARP) joseph_item(c, p->R, w); break;
        case 10: if (lane < 21) { if (c.ok) joseph_out(c, lane); else kf_warp_copy_pred(c, lane); } break;
        default: kf_warp_store(c, t, lane); break;
        }
    }
};
#endif
