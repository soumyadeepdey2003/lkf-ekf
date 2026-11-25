/* lkf_lightweight.h - UPGRADED LKF (paper Sec. 3.2), one object, one timestep.
 * The three ideas of the paper are visible in the code:
 *   (1) scalar equations: every output number has its own short equation (no general matrix products),
 *   (2) arithmetic reformulation: subtraction = addition of a negative (KF_SUB), 2*x = x + x, division = multiply by a reciprocal,
 *   (3) reuse of intermediate results: dt^2, the q-coefficients, reciprocals and the Joseph products are computed ONCE.
 * Because the scalar equations are independent of each other they can also be spread over several GPU threads: see lkf_warp.h. */
#ifndef LKF_LIGHTWEIGHT_H
#define LKF_LIGHTWEIGHT_H
#include "kf_types.h"

/* ---- Stages 1-2: state and covariance prediction (Eqs. 5-30). Blocks A,C,D are 3x3 row-major, updated in place ---- */
KF_HD static inline void lkf_light_predict(float *x, float A[9], float C[9], float D[9], float dt, float sa2) {
    const float dt2 = dt * dt;                                            /* reused 6 times */
    const float qpp = sa2 * dt2 * dt2 * 0.25f, qpv = sa2 * dt2 * dt * 0.5f, qvv = sa2 * dt2;   /* Eqs. 12-14 */
    for (int a = 0; a < 3; a++) x[a] += dt * x[a + 3];                    /* Eqs. 5-7; velocity unchanged (Eqs. 8-10) */
    float An[9], Cn[9], Dn[9];
    for (int a = 0; a < 3; a++) for (int b = a; b < 3; b++) {             /* only the 6 UNIQUE entries of A and D */
        float v;
        if (a == b) { const float t = dt * C[a*3+a]; v = A[a*3+a] + t + t + dt2 * D[a*3+a] + qpp; }   /* Eqs. 15-17: 2*dt*c as t + t */
        else        { v = A[a*3+b] + dt * C[a*3+b] + dt * C[b*3+a] + dt2 * D[a*3+b]; }                /* Eqs. 18-20 */
        An[a*3+b] = v; An[b*3+a] = v;                                     /* Eq. 21: symmetry */
        float d = D[a*3+b]; if (a == b) d += qvv;                         /* Eqs. 27-29 */
        Dn[a*3+b] = d; Dn[b*3+a] = d;
    }
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {             /* all 9 entries of C (not symmetric), Eqs. 23-25 */
        float c = C[a*3+b] + dt * D[a*3+b]; if (a == b) c += qpv;
        Cn[a*3+b] = c;
    }
    for (int i = 0; i < 9; i++) { A[i] = An[i]; C[i] = Cn[i]; D[i] = Dn[i]; }
}

/* ===== Stage 5: Kalman gain without S^-1 (paper Sec. 3.2.5). Solve S k_j = b_j, j = 0..5, with B = [A- C-] (3x6).
 * K is 6x3 row-major (row j = solution k_j). Every solver returns 1 on success, 0 if S is unusable. ===== */
KF_HD static inline float kf_B(const float A[9], const float C[9], int m, int j) {      /* B[m][j] read from the two blocks */
    return (j < 3) ? A[m*3 + j] : C[m*3 + j - 3];
}
/* Case 1 (Eq. 45): S = I  ->  K = B^T */
KF_HD static inline int lkf_gain_identity(const float A[9], const float C[9], float K[18]) {
    for (int j = 0; j < 6; j++) for (int m = 0; m < 3; m++) K[j*3+m] = kf_B(A, C, m, j);
    return 1;
}
/* Case 2 (Eqs. 46-47): S diagonal -> three reciprocals, reused for all six columns */
KF_HD static inline int lkf_gain_diagonal(const float S[9], const float A[9], const float C[9], float K[18]) {
    float rho[3];
    for (int m = 0; m < 3; m++) { if (fabsf(S[m*3+m]) < KF_PIVOT_EPS) return 0; rho[m] = 1.f / S[m*3+m]; }
    for (int j = 0; j < 6; j++) for (int m = 0; m < 3; m++) K[j*3+m] = kf_B(A, C, m, j) * rho[m];
    return 1;
}
/* Case 3 (Eqs. 48-53): block-diagonal, coupled 2x2 block (p,q) and one independent scalar r */
KF_HD static inline int lkf_gain_block(const float S[9], int r, const float A[9], const float C[9], float K[18]) {
    const int p = (r == 0) ? 1 : 0, q = (r == 2) ? 1 : 2;
    const float spp = S[p*3+p], sqq = S[q*3+q], spq = S[p*3+q], sqp = S[q*3+p], srr = S[r*3+r];
    const float delta = KF_SUB(spp * sqq, spq * sqp);                                    /* Eq. 49 */
    if (fabsf(delta) < KF_PIVOT_EPS || fabsf(srr) < KF_PIVOT_EPS) return 0;
    const float id = 1.f / delta, ir = 1.f / srr;                                        /* reciprocals, computed once */
    for (int j = 0; j < 6; j++) {
        const float bp = kf_B(A, C, p, j), bq = kf_B(A, C, q, j), br = kf_B(A, C, r, j);
        K[j*3+p] = KF_SUB(sqq * bp, spq * bq) * id;                                      /* Eq. 50 */
        K[j*3+q] = KF_SUB(spp * bq, sqp * bp) * id;                                      /* Eq. 51 */
        K[j*3+r] = br * ir;                                                              /* Eq. 52 */
    }
    return 1;
}
/* Case 4a (Eqs. 54-56): S lower-triangular -> forward substitution */
KF_HD static inline int lkf_gain_tri_lower(const float S[9], const float A[9], const float C[9], float K[18]) {
    if (fabsf(S[0]) < KF_PIVOT_EPS || fabsf(S[4]) < KF_PIVOT_EPS || fabsf(S[8]) < KF_PIVOT_EPS) return 0;
    const float r1 = 1.f / S[0], r2 = 1.f / S[4], r3 = 1.f / S[8];
    for (int j = 0; j < 6; j++) {
        const float k1 = kf_B(A, C, 0, j) * r1;
        const float k2 = (kf_B(A, C, 1, j) + (-S[3] * k1)) * r2;
        const float k3 = (kf_B(A, C, 2, j) + (-S[6] * k1) + (-S[7] * k2)) * r3;
        K[j*3] = k1; K[j*3+1] = k2; K[j*3+2] = k3;
    }
    return 1;
}
/* Case 4b (Eqs. 58-60): S upper-triangular -> backward substitution */
KF_HD static inline int lkf_gain_tri_upper(const float S[9], const float A[9], const float C[9], float K[18]) {
    if (fabsf(S[0]) < KF_PIVOT_EPS || fabsf(S[4]) < KF_PIVOT_EPS || fabsf(S[8]) < KF_PIVOT_EPS) return 0;
    const float r1 = 1.f / S[0], r2 = 1.f / S[4], r3 = 1.f / S[8];
    for (int j = 0; j < 6; j++) {
        const float k3 = kf_B(A, C, 2, j) * r3;
        const float k2 = (kf_B(A, C, 1, j) + (-S[5] * k3)) * r2;
        const float k1 = (kf_B(A, C, 0, j) + (-S[1] * k2) + (-S[2] * k3)) * r1;
        K[j*3] = k1; K[j*3+1] = k2; K[j*3+2] = k3;
    }
    return 1;
}
/* Case 5 (Eqs. 62-72): symmetric positive-definite -> Cholesky S = L L^T, forward then backward substitution */
KF_HD static inline int lkf_gain_spd(const float S[9], const float A[9], const float C[9], float K[18]) {
    if (S[0] <= 0.f) return 0;
    const float l11 = sqrtf(S[0]), i11 = 1.f / l11;
    const float l21 = S[3] * i11, l31 = S[6] * i11;
    const float d2 = KF_SUB(S[4], l21 * l21);                  if (d2 <= 0.f) return 0;
    const float l22 = sqrtf(d2), i22 = 1.f / l22;
    const float l32 = KF_SUB(S[7], l31 * l21) * i22;
    const float d3 = KF_SUB(KF_SUB(S[8], l31 * l31), l32 * l32);   if (d3 <= 0.f) return 0;
    const float l33 = sqrtf(d3), i33 = 1.f / l33;
    for (int j = 0; j < 6; j++) {
        const float b1 = kf_B(A, C, 0, j), b2 = kf_B(A, C, 1, j), b3 = kf_B(A, C, 2, j);
        const float w1 = b1 * i11, w2 = KF_SUB(b2, l21 * w1) * i22, w3 = KF_SUB(KF_SUB(b3, l31 * w1), l32 * w2) * i33;   /* Eqs. 67-69 */
        const float k3 = w3 * i33, k2 = KF_SUB(w2, l32 * k3) * i22, k1 = KF_SUB(KF_SUB(w1, l21 * k2), l31 * k3) * i11;   /* Eqs. 70-72 */
        K[j*3] = k1; K[j*3+1] = k2; K[j*3+2] = k3;
    }
    return 1;
}
/* Case 6 (Eqs. 74-80): general nonsingular -> pivoted Gauss-Jordan */
KF_HD static inline int lkf_gain_gauss_jordan(const float S[9], const float A[9], const float C[9], float K[18]) {
    float W[9];
    for (int i = 0; i < 9; i++) W[i] = S[i];
    for (int j = 0; j < 6; j++) for (int m = 0; m < 3; m++) K[j*3+m] = kf_B(A, C, m, j);     /* Eq. 74 */
    for (int k = 0; k < 3; k++) {
        int pk = k; float best = fabsf(W[k*3+k]);                                              /* Eq. 75: partial pivoting */
        for (int r = k + 1; r < 3; r++) if (fabsf(W[r*3+k]) > best) { best = fabsf(W[r*3+k]); pk = r; }
        if (best < KF_PIVOT_EPS) return 0;
        if (pk != k) {
            for (int m = 0; m < 3; m++) { float t = W[k*3+m]; W[k*3+m] = W[pk*3+m]; W[pk*3+m] = t; }
            for (int j = 0; j < 6; j++) { float t = K[j*3+k]; K[j*3+k] = K[j*3+pk]; K[j*3+pk] = t; }
        }
        const float inv = 1.f / W[k*3+k];                                                      /* Eqs. 76-77 */
        for (int m = 0; m < 3; m++) W[k*3+m] *= inv;
        for (int j = 0; j < 6; j++) K[j*3+k] *= inv;
        for (int r = 0; r < 3; r++) if (r != k) {                                              /* Eqs. 78-79 */
            const float f = W[r*3+k];
            for (int m = 0; m < 3; m++) W[r*3+m] += -f * W[k*3+m];
            for (int j = 0; j < 6; j++) K[j*3+r] += -f * K[j*3+k];
            W[r*3+k] = 0.f;
        }
    }
    return 1;
}

/* ---- structure detection (Eq. 43): tests in the paper's order; a failed test passes on to the next ---- */
enum { LKF_MODE_IDENTITY = 0, LKF_MODE_DIAG, LKF_MODE_BLOCK, LKF_MODE_TRI_LOWER, LKF_MODE_TRI_UPPER, LKF_MODE_SPD, LKF_MODE_GJ };
/* returns the case number; *r receives the isolated index for the block case */
KF_HD static inline int lkf_detect(const float S[9], int *r) {
    const int lower_zero = (S[1] == 0.f && S[2] == 0.f && S[5] == 0.f);      /* nothing above the diagonal -> lower triangular */
    const int upper_zero = (S[3] == 0.f && S[6] == 0.f && S[7] == 0.f);      /* nothing below the diagonal -> upper triangular */
    if (lower_zero && upper_zero) return (S[0] == 1.f && S[4] == 1.f && S[8] == 1.f) ? LKF_MODE_IDENTITY : LKF_MODE_DIAG;
    for (int i = 0; i < 3; i++) {                                            /* Eq. 48: s_pr = s_rp = s_qr = s_rq = 0 */
        const int p = (i == 0) ? 1 : 0, q = (i == 2) ? 1 : 2;
        if (S[p*3+i] == 0.f && S[i*3+p] == 0.f && S[q*3+i] == 0.f && S[i*3+q] == 0.f) { *r = i; return LKF_MODE_BLOCK; }
    }
    if (lower_zero) return LKF_MODE_TRI_LOWER;
    if (upper_zero) return LKF_MODE_TRI_UPPER;
    if (S[1] == S[3] && S[2] == S[6] && S[5] == S[7]) return LKF_MODE_SPD;   /* Cholesky needs an exactly symmetric S */
    return LKF_MODE_GJ;
}
KF_HD static inline int lkf_light_gain(const float S[9], const float A[9], const float C[9], float K[18]) {
    int r = 0; const int mode = lkf_detect(S, &r);
    switch (mode) {
    case LKF_MODE_IDENTITY:  return lkf_gain_identity(A, C, K);
    case LKF_MODE_DIAG:      return lkf_gain_diagonal(S, A, C, K);
    case LKF_MODE_BLOCK:     return lkf_gain_block(S, r, A, C, K);
    case LKF_MODE_TRI_LOWER: return lkf_gain_tri_lower(S, A, C, K);
    case LKF_MODE_TRI_UPPER: return lkf_gain_tri_upper(S, A, C, K);
    case LKF_MODE_SPD:       if (lkf_gain_spd(S, A, C, K)) return 1;       /* Cholesky failed (not positive definite): */
                             return lkf_gain_gauss_jordan(S, A, C, K);      /* use the general solver                   */
    default:                 return lkf_gain_gauss_jordan(S, A, C, K);
    }
}

/* ---- Stage 7: Joseph covariance correction (Eqs. 89-109) ---- */
KF_HD static inline void lkf_light_joseph(float A[9], float C[9], float D[9], const float K[18], const float R[9]) {
    const float *U = K, *V = K + 9;                                        /* position rows / velocity rows of the gain */
    float M[9], lam[9], eta[9], e[9], n[9], phi[9], O[9], J[9];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) M[a*3+b] = KF_SUB((a == b ? 1.f : 0.f), U[a*3+b]);   /* Eq. 89 */
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {              /* the seven intermediate products, Eqs. 98-104 */
        float l = 0, h = 0, ee = C[a*3+b], nn = 0, ph = 0, o = 0, jj = 0;
        for (int q = 0; q < 3; q++) {
            l  += M[a*3+q] * A[q*3+b];
            h  += U[a*3+q] * R[q*3+b];
            ee  = KF_SUB(ee, A[a*3+q] * V[b*3+q]);
            nn += V[a*3+q] * A[q*3+b];
            ph += V[a*3+q] * R[q*3+b];
            o  += C[q*3+a] * V[b*3+q];
            jj += V[a*3+q] * C[q*3+b];
        }
        lam[a*3+b] = l; eta[a*3+b] = h; e[a*3+b] = ee; n[a*3+b] = nn; phi[a*3+b] = ph; O[a*3+b] = o; J[a*3+b] = jj;
    }
    float An[9], Cn[9], Dn[9];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {              /* Eq. 106: all 9 entries of C+ */
        float c = 0;
        for (int q = 0; q < 3; q++) c += M[a*3+q] * e[q*3+b] + eta[a*3+q] * V[b*3+q];
        Cn[a*3+b] = c;
    }
    for (int a = 0; a < 3; a++) for (int b = a; b < 3; b++) {              /* Eqs. 105, 107: unique entries only, then mirror */
        float aa = 0, dd = KF_SUB(KF_SUB(D[a*3+b], O[a*3+b]), J[a*3+b]);
        for (int q = 0; q < 3; q++) {
            aa += lam[a*3+q] * M[b*3+q] + eta[a*3+q] * U[b*3+q];
            dd += n[a*3+q] * V[b*3+q] + phi[a*3+q] * V[b*3+q];
        }
        An[a*3+b] = aa; An[b*3+a] = aa; Dn[a*3+b] = dd; Dn[b*3+a] = dd;
    }
    for (int i = 0; i < 9; i++) { A[i] = An[i]; C[i] = Cn[i]; D[i] = Dn[i]; }
}

KF_HD static inline void kf_unpack(const Track *t, float A[9], float C[9], float D[9]) {
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
        A[a*3+b] = t->P[a*6+b]; C[a*3+b] = t->P[a*6+3+b]; D[a*3+b] = t->P[(3+a)*6+3+b];
    }
}
KF_HD static inline void kf_pack(Track *t, const float A[9], const float C[9], const float D[9]) {
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
        t->P[a*6+b] = A[a*3+b]; t->P[a*6+3+b] = C[a*3+b];
        t->P[(3+b)*6+a] = C[a*3+b]; t->P[(3+a)*6+3+b] = D[a*3+b];
    }
}
/* UPGRADED LKF: full step for one object (this is the "thread-per-track" mapping: one thread does all stages) */
KF_HD static inline void lkf_light_step(Track *t, const Meas *m, const KFParams *p) {
    float A[9], C[9], D[9];
    kf_unpack(t, A, C, D);
    lkf_light_predict(t->x, A, C, D, p->dt, p->sa2);
    float K[18], S[9];
    int ok = m->valid;
    if (ok) {
        for (int i = 0; i < 9; i++) S[i] = A[i] + p->R[i];                                   /* Eqs. 35-37 */
        ok = lkf_light_gain(S, A, C, K);
    }
    if (ok) {
        const float y[3] = { m->z[0] + (-t->x[0]), m->z[1] + (-t->x[1]), m->z[2] + (-t->x[2]) };   /* Eqs. 31-33 */
        for (int j = 0; j < 6; j++) t->x[j] += K[j*3]*y[0] + K[j*3+1]*y[1] + K[j*3+2]*y[2];         /* Eqs. 82-87 */
        lkf_light_joseph(A, C, D, K, p->R);
    }
    kf_pack(t, A, C, D);
}
#endif
