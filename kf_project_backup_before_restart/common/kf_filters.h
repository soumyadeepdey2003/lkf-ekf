/* kf_filters.h - the four filters, each described by the same small "policy" struct:
 *   name/tag : text for reports      kind : measurement type (LINEAR or SPHERICAL)
 *   step()   : one object, one timestep (runs on CPU and GPU)      init() : create a track from the first measurement
 *   has_warp / Warp : the UPGRADED filters also have a warp-per-track GPU mapping (32 lanes cooperate on one track) */
#ifndef KF_FILTERS_H
#define KF_FILTERS_H
#include "lkf_lightweight.h"
#include "ekf_lightweight.h"      /* also pulls in lkf_conventional.h and ekf_conventional.h */
#include "lkf_warp.h"
#include "ekf_warp.h"

static inline void lkf_init(Track *t, const Meas *m) {
    for (int k = 0; k < 3; k++) { t->x[k] = m->z[k]; t->x[k + 3] = 0.f; }
    for (int i = 0; i < 36; i++) t->P[i] = 0.f;
    for (int i = 0; i < 6; i++) t->P[i * 7] = KF_P0;
}
struct LkfOrig { enum { kind = KF_LINEAR, has_warp = 0 }; typedef NoWarp Warp;
    static const char *name() { return "LKF original"; }  static const char *tag() { return "lkf_orig"; }
    KF_HD static void step(Track *t, const Meas *m, const KFParams *p) { lkf_conv_step(t, m, p); }
    static void init(Track *t, const Meas *m) { lkf_init(t, m); } };
struct LkfNew { enum { kind = KF_LINEAR, has_warp = 1 }; typedef LkfWarp Warp;
    static const char *name() { return "LKF upgraded"; }  static const char *tag() { return "lkf_new"; }
    KF_HD static void step(Track *t, const Meas *m, const KFParams *p) { lkf_light_step(t, m, p); }
    static void init(Track *t, const Meas *m) { lkf_init(t, m); } };
struct EkfOrig { enum { kind = KF_SPHERICAL, has_warp = 0 }; typedef NoWarp Warp;
    static const char *name() { return "EKF original"; }  static const char *tag() { return "ekf_orig"; }
    KF_HD static void step(Track *t, const Meas *m, const KFParams *p) { ekf_conv_step(t, m, p); }
    static void init(Track *t, const Meas *m) { ekf_conv_init(t, m); } };
struct EkfNew { enum { kind = KF_SPHERICAL, has_warp = 1 }; typedef EkfWarp Warp;
    static const char *name() { return "EKF upgraded"; }  static const char *tag() { return "ekf_new"; }
    KF_HD static void step(Track *t, const Meas *m, const KFParams *p) { ekf_light_step(t, m, p); }
    static void init(Track *t, const Meas *m) { ekf_light_init(t, m); } };
#endif
