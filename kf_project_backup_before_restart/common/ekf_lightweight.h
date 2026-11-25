/* ekf_lightweight.h - UPGRADED EKF (paper Sec. 3.4): scalar equations + CORDIC + Newton reciprocal + structure-aware gain.
 * Replaces sqrt/atan2/sin/cos by CORDIC shift-add loops and every division by RECIP (table + Newton refinement).
 * NOTE: these replacements target FPGA / fixed-point hardware. On a GPU or CPU the native sqrtf/atan2f/division are
 * usually FASTER - the benchmarks will tell you, and that is a legitimate result for the paper. */
#ifndef EKF_LIGHTWEIGHT_H
#define EKF_LIGHTWEIGHT_H
#include "lkf_lightweight.h"
#include "ekf_conventional.h"

#ifndef KF_NC
#define KF_NC 24        /* number of CORDIC stages (accuracy ~ 2^-(NC-1) rad) */
#endif
#ifndef KF_NR
#define KF_NR 3         /* Newton iterations of RECIP (table error 3% -> ~1e-12 after 3) */
#endif
#define KF_PIVOT_REL 1e-6f
#define KF_CORDIC_KAPPA 0.607252955f   /* 1 / prod sqrt(1+2^-2n), generated offline */

/* phase constants atan(2^-n), generated offline (n >= 12: equal to 2^-n at float precision) */
KF_HD static inline float kf_cordic_theta(int n) {
    switch (n) {
    case 0: return 0.785398185f;
    case 1: return 0.463647604f;
    case 2: return 0.244978666f;
    case 3: return 0.124354996f;
    case 4: return 0.062418811f;
    case 5: return 0.0312398337f;
    case 6: return 0.0156237287f;
    case 7: return 0.00781234121f;
    case 8: return 0.00390623021f;
    case 9: return 0.00195312256f;
    case 10: return 0.000976562209f;
    case 11: return 0.000488281221f;
    default: return ldexpf(1.0f, -n);
    }
}
/* reciprocal start values: 16-entry table over the mantissa interval [0.5, 1), generated offline */
KF_HD static inline float kf_lut_recip(float mu) {
    int i = (int)((mu - 0.5f) * 32.f);
    if (i < 0) i = 0;
    if (i > 15) i = 15;
    switch (i) {
    case 0: return 1.939394f;
    case 1: return 1.82857144f;
    case 2: return 1.72972977f;
    case 3: return 1.64102566f;
    case 4: return 1.56097555f;
    case 5: return 1.48837209f;
    case 6: return 1.42222226f;
    case 7: return 1.36170208f;
    case 8: return 1.30612242f;
    case 9: return 1.25490201f;
    case 10: return 1.20754719f;
    case 11: return 1.16363633f;
    case 12: return 1.12280703f;
    case 13: return 1.08474576f;
    case 14: return 1.04918027f;
    case 15: return 1.01587307f;
    default: return 1.f;
    }
}
/* RECIP(a) ~ 1/a : a = sign * mu * 2^e (0.5<=mu<1); u <- u(2 - mu u); caller must ensure a != 0 */
KF_HD static inline float kf_recip(float a) {
    int e; const float mu = frexpf(fabsf(a), &e);
    float u = kf_lut_recip(mu);
    for (int q = 0; q < KF_NR; q++) u = u * (2.f + (-mu * u));
    u = ldexpf(u, -e);
    return (a < 0.f) ? -u : u;
}
/* CORDIC vectoring: (a,b) -> magnitude and angle atan2(b,a). ldexpf(q,-n) is the arithmetic shift ASR(q,n). */
KF_HD static inline void kf_cordic_vec(float a, float b, float *mag, float *ang) {
    float X, Y, Z;
    if (a >= 0.f)      { X = a;  Y = b;  Z = 0.f; }
    else if (b >= 0.f) { X = -a; Y = -b; Z = KF_PI; }
    else               { X = -a; Y = -b; Z = -KF_PI; }
    for (int n = 0; n < KF_NC; n++) {
        const float d = (Y >= 0.f) ? 1.f : -1.f, sx = ldexpf(X, -n), sy = ldexpf(Y, -n);
        const float Xn = X + d * sy, Yn = Y + (-d * sx);
        Z += d * kf_cordic_theta(n); X = Xn; Y = Yn;
    }
    *mag = KF_CORDIC_KAPPA * X; *ang = Z;
}
/* CORDIC rotation: alpha -> (cos alpha, sin alpha) */
KF_HD static inline void kf_cordic_rot(float alpha, float *c, float *s) {
    float ar = alpha, sg = 1.f;
    if (alpha > 0.5f * KF_PI)       { ar = alpha - KF_PI; sg = -1.f; }
    else if (alpha < -0.5f * KF_PI) { ar = alpha + KF_PI; sg = -1.f; }
    float X = KF_CORDIC_KAPPA, Y = 0.f, Z = ar;
    for (int n = 0; n < KF_NC; n++) {
        const float d = (Z >= 0.f) ? 1.f : -1.f, sx = ldexpf(X, -n), sy = ldexpf(Y, -n);
        const float Xn = X + (-d * sy), Yn = Y + d * sx;
        Z += -d * kf_cordic_theta(n); X = Xn; Y = Yn;
    }
    *c = sg * X; *s = sg * Y;
}
/* Stage 10: new track from the first (range, azimuth, elevation) measurement using CORDIC rotation */
KF_HD static inline void ekf_light_init(Track *t, const Meas *m) {
    float cth, sth, cph, sph;
    kf_cordic_rot(m->z[1], &cth, &sth); kf_cordic_rot(m->z[2], &cph, &sph);
    const float rh = m->z[0] * cph;
    t->x[0] = rh * cth; t->x[1] = rh * sth; t->x[2] = m->z[0] * sph;
    t->x[3] = t->x[4] = t->x[5] = 0.f;
    for (int i = 0; i < 36; i++) t->P[i] = 0.f;
    for (int i = 0; i < 6; i++) t->P[i * 7] = KF_P0;
}

/* paper Eq. 308: the azimuth difference is wrapped into [-pi, pi) with additions of +-2*pi */
KF_HD static inline float kf_wrap_pi_light(float a) {
    if (a >= KF_PI) return a + (-2.f * KF_PI);
    if (a < -KF_PI) return a + 2.f * KF_PI;
    return a;
}
/* Stage 3: predicted measurement by two CORDIC vectoring operations */
KF_HD static inline void ekf_light_meas(float x, float y, float z, float *rxy, float *rho, float *th, float *ph) {
    kf_cordic_vec(x, y, rxy, th);
    kf_cordic_vec(*rxy, z, rho, ph);
}
/* Stage 4: position part of the Jacobian (3x3, row-major) using reusable reciprocals; 0 if geometry is singular */
KF_HD static inline int ekf_light_jacobian(float x, float y, float z, float rxy, float rho, float Hp[9]) {
    if (rxy < KF_RANGE_EPS || rho < KF_RANGE_EPS) return 0;
    const float ixy = kf_recip(rxy), irho = kf_recip(rho);
    const float ixy2 = ixy * ixy, irho2 = irho * irho, irho2xy = irho2 * ixy;
    Hp[0] = x * irho;          Hp[1] = y * irho;          Hp[2] = z * irho;
    Hp[3] = (-y) * ixy2;       Hp[4] = x * ixy2;          Hp[5] = 0.f;
    Hp[6] = (-x * z) * irho2xy; Hp[7] = (-y * z) * irho2xy; Hp[8] = rxy * irho2;
    return 1;
}
/* Stage 6: T = Hp * [A- C-]  stored as two 3x3 halves TL, TR  (so that T[m][j] = kf_B(TL,TR,m,j)) */
KF_HD static inline void ekf_light_T(const float Hp[9], const float A[9], const float C[9], float TL[9], float TR[9]) {
    for (int a = 0; a < 3; a++) for (int j = 0; j < 3; j++) {
        float l = 0.f, r = 0.f;
        for (int q = 0; q < 3; q++) { l += Hp[a*3+q] * A[q*3+j]; r += Hp[a*3+q] * C[q*3+j]; }
        TL[a*3+j] = l; TR[a*3+j] = r;
    }
}
KF_HD static inline void ekf_light_S(const float Hp[9], const float TL[9], const float R[9], float S[9]) {
    for (int a = 0; a < 3; a++) for (int b = a; b < 3; b++) {      /* 6 unique elements, then mirror */
        float s = R[a*3+b];
        for (int q = 0; q < 3; q++) s += TL[a*3+q] * Hp[b*3+q];
        S[a*3+b] = s; S[b*3+a] = s;
    }
}

/* ===== Stage 7: gain, 5 cases (identity, diagonal, block, LDL^T, Gauss-Jordan), all divisions via RECIP ===== */
KF_HD static inline int ekf_gain_diagonal(const float S[9], const float TL[9], const float TR[9], float K[18]) {
    float beta[3];
    for (int m = 0; m < 3; m++) { if (fabsf(S[m*3+m]) < 1e-30f) return 0; beta[m] = kf_recip(S[m*3+m]); }
    for (int j = 0; j < 6; j++) for (int m = 0; m < 3; m++) K[j*3+m] = kf_B(TL, TR, m, j) * beta[m];
    return 1;
}
KF_HD static inline int ekf_gain_block(const float S[9], int r, const float TL[9], const float TR[9], float K[18]) {
    const int p = (r == 0) ? 1 : 0, q = (r == 2) ? 1 : 2;
    const float spp = S[p*3+p], sqq = S[q*3+q], spq = S[p*3+q], sqp = S[q*3+p], srr = S[r*3+r];
    const float delta = spp * sqq + (-spq * sqp);
    if (fabsf(delta) <= KF_PIVOT_REL * fabsf(spp * sqq) || fabsf(srr) < 1e-30f) return 0;
    const float bd = kf_recip(delta), br = kf_recip(srr);
    for (int j = 0; j < 6; j++) {
        const float tp = kf_B(TL, TR, p, j), tq = kf_B(TL, TR, q, j), tr = kf_B(TL, TR, r, j);
        K[j*3+p] = (sqq * tp + (-spq * tq)) * bd;
        K[j*3+q] = (spp * tq + (-sqp * tp)) * bd;
        K[j*3+r] = tr * br;
    }
    return 1;
}
/* symmetric positive definite: S = L D L^T (no square roots) */
KF_HD static inline int ekf_gain_ldl(const float S[9], const float TL[9], const float TR[9], float K[18]) {
    const float z1 = S[0];                                       if (z1 <= KF_PIVOT_REL * S[0]) return 0;
    const float b1 = kf_recip(z1), l21 = S[3] * b1, l31 = S[6] * b1;
    const float z2 = S[4] + (-l21 * l21 * z1);                   if (z2 <= KF_PIVOT_REL * S[4]) return 0;
    const float b2 = kf_recip(z2), l32 = (S[7] + (-l31 * l21 * z1)) * b2;
    const float z3 = S[8] + (-l31 * l31 * z1) + (-l32 * l32 * z2); if (z3 <= KF_PIVOT_REL * S[8]) return 0;
    const float b3 = kf_recip(z3);
    for (int j = 0; j < 6; j++) {
        const float t1 = kf_B(TL, TR, 0, j), t2 = kf_B(TL, TR, 1, j), t3 = kf_B(TL, TR, 2, j);
        const float w1 = t1, w2 = t2 + (-l21 * w1), w3 = t3 + (-l31 * w1) + (-l32 * w2);
        const float v1 = w1 * b1, v2 = w2 * b2, v3 = w3 * b3;
        const float k3 = v3, k2 = v2 + (-l32 * k3), k1 = v1 + (-l21 * k2) + (-l31 * k3);
        K[j*3] = k1; K[j*3+1] = k2; K[j*3+2] = k3;
    }
    return 1;
}
/* general nonsingular: pivoted Gauss-Jordan with RECIP */
KF_HD static inline int ekf_gain_gj(const float S[9], const float TL[9], const float TR[9], float K[18]) {
    float W[9], smax = 0.f;
    for (int i = 0; i < 9; i++) { W[i] = S[i]; if (fabsf(S[i]) > smax) smax = fabsf(S[i]); }
    for (int j = 0; j < 6; j++) for (int m = 0; m < 3; m++) K[j*3+m] = kf_B(TL, TR, m, j);
    for (int k = 0; k < 3; k++) {
        int pk = k; float best = fabsf(W[k*3+k]);
        for (int r = k + 1; r < 3; r++) if (fabsf(W[r*3+k]) > best) { best = fabsf(W[r*3+k]); pk = r; }
        if (best <= KF_PIVOT_REL * smax) return 0;
        if (pk != k) {
            for (int m = 0; m < 3; m++) { float t = W[k*3+m]; W[k*3+m] = W[pk*3+m]; W[pk*3+m] = t; }
            for (int j = 0; j < 6; j++) { float t = K[j*3+k]; K[j*3+k] = K[j*3+pk]; K[j*3+pk] = t; }
        }
        const float beta = kf_recip(W[k*3+k]);
        for (int m = 0; m < 3; m++) W[k*3+m] *= beta;
        for (int j = 0; j < 6; j++) K[j*3+k] *= beta;
        for (int r = 0; r < 3; r++) if (r != k) {
            const float f = W[r*3+k];
            for (int m = 0; m < 3; m++) W[r*3+m] += -f * W[k*3+m];
            for (int j = 0; j < 6; j++) K[j*3+r] += -f * K[j*3+k];
            W[r*3+k] = 0.f;
        }
    }
    return 1;
}
KF_HD static inline int ekf_light_gain(const float S[9], const float TL[9], const float TR[9], float K[18]) {
    if (S[1] == 0.f && S[2] == 0.f && S[5] == 0.f) {
        if (S[0] == 1.f && S[4] == 1.f && S[8] == 1.f) return lkf_gain_identity(TL, TR, K);
        return ekf_gain_diagonal(S, TL, TR, K);
    }
    if (S[2] == 0.f && S[5] == 0.f) return ekf_gain_block(S, 2, TL, TR, K);
    if (S[1] == 0.f && S[5] == 0.f) return ekf_gain_block(S, 1, TL, TR, K);
    if (S[1] == 0.f && S[2] == 0.f) return ekf_gain_block(S, 0, TL, TR, K);
    if (ekf_gain_ldl(S, TL, TR, K)) return 1;
    return ekf_gain_gj(S, TL, TR, K);
}

/* Stage 9: Joseph covariance update with the EKF Jacobian (paper Eqs. 353-368) */
KF_HD static inline void ekf_light_joseph(float A[9], float C[9], float D[9], const float K[18], const float Hp[9], const float R[9]) {
    const float *U = K, *V = K + 9;
    float M[9], N[9], al[9], ga[9], e[9], om[9], ps[9], chi[9], xi[9];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
        float m = (a == b) ? 1.f : 0.f, n = 0.f;
        for (int q = 0; q < 3; q++) { m += -U[a*3+q] * Hp[q*3+b]; n += -V[a*3+q] * Hp[q*3+b]; }
        M[a*3+b] = m; N[a*3+b] = n;
    }
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
        float s_al = 0, s_ga = 0, s_e = C[a*3+b], s_om = 0, s_ps = 0, s_chi = 0, s_xi = 0;
        for (int q = 0; q < 3; q++) {
            s_al += M[a*3+q] * A[q*3+b];     s_ga += U[a*3+q] * R[q*3+b];
            s_e  += A[a*3+q] * N[b*3+q];     s_om += N[a*3+q] * A[q*3+b];
            s_ps += V[a*3+q] * R[q*3+b];     s_chi += C[q*3+a] * N[b*3+q];
            s_xi += N[a*3+q] * C[q*3+b];
        }
        al[a*3+b] = s_al; ga[a*3+b] = s_ga; e[a*3+b] = s_e; om[a*3+b] = s_om; ps[a*3+b] = s_ps; chi[a*3+b] = s_chi; xi[a*3+b] = s_xi;
    }
    float An[9], Cn[9], Dn[9];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
        float c = 0;
        for (int q = 0; q < 3; q++) c += M[a*3+q] * e[q*3+b] + ga[a*3+q] * V[b*3+q];
        Cn[a*3+b] = c;
    }
    for (int a = 0; a < 3; a++) for (int b = a; b < 3; b++) {
        float aa = 0, dd = chi[a*3+b] + xi[a*3+b] + D[a*3+b];
        for (int q = 0; q < 3; q++) {
            aa += al[a*3+q] * M[b*3+q] + ga[a*3+q] * U[b*3+q];
            dd += om[a*3+q] * N[b*3+q] + ps[a*3+q] * V[b*3+q];
        }
        An[a*3+b] = aa; An[b*3+a] = aa; Dn[a*3+b] = dd; Dn[b*3+a] = dd;
    }
    for (int i = 0; i < 9; i++) { A[i] = An[i]; C[i] = Cn[i]; D[i] = Dn[i]; }
}

/* UPGRADED EKF: full step for one object (prediction is identical to the LKF one: same constant-velocity model) */
KF_HD static inline void ekf_light_step(Track *t, const Meas *m, const KFParams *p) {
    float A[9], C[9], D[9];
    kf_unpack(t, A, C, D);
    lkf_light_predict(t->x, A, C, D, p->dt, p->sa2);
    float K[18], TL[9], TR[9], S[9], Hp[9], nu[3];
    int ok = m->valid;
    if (ok) {
        float rxy, rho, th, ph;
        ekf_light_meas(t->x[0], t->x[1], t->x[2], &rxy, &rho, &th, &ph);
        ok = ekf_light_jacobian(t->x[0], t->x[1], t->x[2], rxy, rho, Hp);
        if (ok) {
            nu[0] = m->z[0] + (-rho); nu[1] = kf_wrap_pi_light(m->z[1] + (-th)); nu[2] = m->z[2] + (-ph);
            ekf_light_T(Hp, A, C, TL, TR);
            ekf_light_S(Hp, TL, p->R, S);
            ok = ekf_light_gain(S, TL, TR, K);
        }
    }
    if (ok) {
        for (int j = 0; j < 6; j++) t->x[j] += K[j*3]*nu[0] + K[j*3+1]*nu[1] + K[j*3+2]*nu[2];
        ekf_light_joseph(A, C, D, K, Hp, p->R);
    }
    kf_pack(t, A, C, D);
}
#endif
