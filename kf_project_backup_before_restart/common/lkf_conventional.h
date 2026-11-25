/* lkf_conventional.h - ORIGINAL matrix-form Kalman math (shared by the original LKF and the original EKF).
 * predict -> (innovation, H given by the caller) -> S, K = P H^T S^-1, state update, Joseph covariance. */
#ifndef LKF_CONVENTIONAL_H
#define LKF_CONVENTIONAL_H
#include "kf_types.h"

KF_HD static inline void kf_mm(const float *A, const float *B, float *C, int n, int m, int p) {   /* C = A*B */
    for (int i = 0; i < n; i++) for (int j = 0; j < p; j++) {
        float s = 0.f;
        for (int k = 0; k < m; k++) s += A[i * m + k] * B[k * p + j];
        C[i * p + j] = s;
    }
}
KF_HD static inline void kf_tr(const float *A, float *AT, int n, int m) {                          /* transpose */
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) AT[j * n + i] = A[i * m + j];
}
KF_HD static inline int kf_inv3(const float S[9], float Si[9]) {                                    /* 3x3 inverse */
    float c00 = S[4]*S[8]-S[5]*S[7], c01 = S[5]*S[6]-S[3]*S[8], c02 = S[3]*S[7]-S[4]*S[6];
    float det = S[0]*c00 + S[1]*c01 + S[2]*c02;
    if (fabsf(det) < 1e-30f) return 0;
    float id = 1.f / det;
    Si[0]=c00*id; Si[1]=(S[2]*S[7]-S[1]*S[8])*id; Si[2]=(S[1]*S[5]-S[2]*S[4])*id;
    Si[3]=c01*id; Si[4]=(S[0]*S[8]-S[2]*S[6])*id; Si[5]=(S[2]*S[3]-S[0]*S[5])*id;
    Si[6]=c02*id; Si[7]=(S[1]*S[6]-S[0]*S[7])*id; Si[8]=(S[0]*S[4]-S[1]*S[3])*id;
    return 1;
}

/* x- = F x ;  P- = F P F^T + Q   (constant-velocity model, isotropic acceleration noise) */
KF_HD static inline void kf_conv_predict(const Track *t, const KFParams *p, float xm[6], float Pm[36]) {
    const float dt = p->dt, s2 = p->sa2;
    float F[36] = {0}, Q[36] = {0}, FP[36], FT[36];
    for (int i = 0; i < 6; i++) F[i * 6 + i] = 1.f;
    for (int i = 0; i < 3; i++) F[i * 6 + i + 3] = dt;
    const float qpp = s2*dt*dt*dt*dt/4.f, qpv = s2*dt*dt*dt/2.f, qvv = s2*dt*dt;
    for (int i = 0; i < 3; i++) { Q[i*6+i] = qpp; Q[i*6+i+3] = qpv; Q[(i+3)*6+i] = qpv; Q[(i+3)*6+i+3] = qvv; }
    kf_mm(F, t->x, xm, 6, 6, 1);
    kf_mm(F, t->P, FP, 6, 6, 6); kf_tr(F, FT, 6, 6); kf_mm(FP, FT, Pm, 6, 6, 6);
    for (int i = 0; i < 36; i++) Pm[i] += Q[i];
}
KF_HD static inline void kf_conv_commit(Track *t, const float xm[6], const float Pm[36]) {
    for (int i = 0; i < 6; i++) t->x[i] = xm[i];
    for (int i = 0; i < 36; i++) t->P[i] = Pm[i];
}
/* Update with an arbitrary 3x6 measurement matrix H and innovation nu. Returns 0 (and leaves t untouched) if S is singular. */
KF_HD static inline int kf_conv_update(Track *t, const float xm[6], const float Pm[36], const float H[18],
                                       const float nu[3], const KFParams *p) {
    float HP[18], HT[18], S[9], Si[9], PHT[18], K[18];
    kf_mm(H, Pm, HP, 3, 6, 6); kf_tr(H, HT, 3, 6); kf_mm(HP, HT, S, 3, 6, 3);       /* S = H P H^T + R */
    for (int i = 0; i < 9; i++) S[i] += p->R[i];
    if (!kf_inv3(S, Si)) return 0;
    kf_mm(Pm, HT, PHT, 6, 6, 3); kf_mm(PHT, Si, K, 6, 3, 3);                         /* K = P H^T S^-1 */
    for (int i = 0; i < 6; i++) t->x[i] = xm[i] + K[i*3]*nu[0] + K[i*3+1]*nu[1] + K[i*3+2]*nu[2];
    float KH[36], IKH[36], IKHT[36], T1[36], T2[36], KR[18], KT[18], T3[36];         /* Joseph form */
    kf_mm(K, H, KH, 6, 3, 6);
    for (int i = 0; i < 36; i++) IKH[i] = ((i / 6 == i % 6) ? 1.f : 0.f) - KH[i];
    kf_tr(IKH, IKHT, 6, 6); kf_mm(IKH, Pm, T1, 6, 6, 6); kf_mm(T1, IKHT, T2, 6, 6, 6);
    kf_mm(K, p->R, KR, 6, 3, 3); kf_tr(K, KT, 6, 3); kf_mm(KR, KT, T3, 6, 3, 6);
    for (int i = 0; i < 36; i++) t->P[i] = T2[i] + T3[i];
    return 1;
}

/* ORIGINAL LKF: H = [I 0], innovation = z - x-(position) */
KF_HD static inline void lkf_conv_step(Track *t, const Meas *m, const KFParams *p) {
    float xm[6], Pm[36];
    kf_conv_predict(t, p, xm, Pm);
    if (m->valid) {
        float H[18] = {0}, nu[3];
        for (int i = 0; i < 3; i++) { H[i * 6 + i] = 1.f; nu[i] = m->z[i] - xm[i]; }
        if (kf_conv_update(t, xm, Pm, H, nu, p)) return;
    }
    kf_conv_commit(t, xm, Pm);
}
#endif
