/* ekf_conventional.h - ORIGINAL EKF (paper Sec. 3.5): sensor at the origin measures (range, azimuth, elevation). */
#ifndef EKF_CONVENTIONAL_H
#define EKF_CONVENTIONAL_H
#include "lkf_conventional.h"

KF_HD static inline float kf_wrap_pi(float a) {          /* angle difference back into [-pi, pi) */
    if (a >= KF_PI) a -= 2.f * KF_PI; else if (a < -KF_PI) a += 2.f * KF_PI;
    return a;
}
/* spherical -> Cartesian for a NEW track (conventional: sinf / cosf) */
KF_HD static inline void ekf_conv_init(Track *t, const Meas *m) {
    const float rho = m->z[0], th = m->z[1], ph = m->z[2], rh = rho * cosf(ph);
    t->x[0] = rh * cosf(th); t->x[1] = rh * sinf(th); t->x[2] = rho * sinf(ph);
    t->x[3] = t->x[4] = t->x[5] = 0.f;
    for (int i = 0; i < 36; i++) t->P[i] = 0.f;
    for (int i = 0; i < 6; i++) t->P[i * 7] = KF_P0;
}
KF_HD static inline void ekf_conv_step(Track *t, const Meas *m, const KFParams *p) {
    float xm[6], Pm[36];
    kf_conv_predict(t, p, xm, Pm);
    if (m->valid) {
        const float x = xm[0], y = xm[1], z = xm[2];
        const float rxy2 = x*x + y*y, rxy = sqrtf(rxy2), rho2 = rxy2 + z*z, rho = sqrtf(rho2);
        if (rxy >= KF_RANGE_EPS && rho >= KF_RANGE_EPS) {            /* angles undefined at the origin axis */
            float H[18] = {0}, nu[3];
            H[0] = x / rho;  H[1] = y / rho;  H[2] = z / rho;                       /* d range     */
            H[6] = -y / rxy2; H[7] = x / rxy2;                                      /* d azimuth   */
            H[12] = -x*z / (rho2 * rxy); H[13] = -y*z / (rho2 * rxy); H[14] = rxy / rho2;   /* d elevation */
            nu[0] = m->z[0] - rho;
            nu[1] = kf_wrap_pi(m->z[1] - atan2f(y, x));
            nu[2] = m->z[2] - atan2f(z, rxy);
            if (kf_conv_update(t, xm, Pm, H, nu, p)) return;
        }
    }
    kf_conv_commit(t, xm, Pm);
}
#endif
