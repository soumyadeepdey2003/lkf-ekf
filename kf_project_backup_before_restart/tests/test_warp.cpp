/* test_warp.cpp - the WARP-PER-TRACK mapping must give the same result as the thread-per-track step, and its phases must be
 * race-free. The lanes of every phase are run in several RANDOM orders: any hidden dependency between lanes would change the result. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kf_filters.h"

static int bad = 0;
static unsigned st = 7u;
static float rnd() { st = st * 1664525u + 1013904223u; return ((st >> 8) & 0xFFFF) / 65535.f - 0.5f; }
static void verdict(const char *name, double err, double tol) {
    printf("%s %-62s err = %.2e (tol %.0e)\n", err <= tol ? "pass" : "FAIL", name, err, tol);
    if (!(err <= tol)) bad++;
}
static void random_track(Track *a, float scale_z) {
    float G[36];
    for (int k = 0; k < 36; k++) G[k] = 0.3f * rnd();
    for (int r = 0; r < 6; r++) for (int c = 0; c < 6; c++) { float s = (r == c) ? 1.f : 0.f; for (int q = 0; q < 6; q++) s += G[r*6+q] * G[c*6+q]; a->P[r*6+c] = s; }
    for (int k = 0; k < 3; k++) a->x[k] = 100.f * rnd();
    a->x[2] *= scale_z;
    for (int k = 3; k < 6; k++) a->x[k] = 8.f * rnd();
}
static double cell(double a, double b) { if (!isfinite(a) || !isfinite(b)) return 1e9; return fabs(a - b) / (1.0 + fabs(a)); }   /* NaN counts as a failure */
static double diff(const Track &a, const Track &b) {
    double e = 0;
    for (int k = 0; k < 6; k++) e = fmax(e, cell(a.x[k], b.x[k]));
    for (int k = 0; k < 36; k++) e = fmax(e, cell(a.P[k], b.P[k]));
    return e;
}
template <class F> static void compare_steps(const char *name, bool spherical, int n_tracks) {
    KFParams p; memset(&p, 0, sizeof p); p.dt = 0.1f; p.sa2 = 0.05f;
    if (spherical) { p.R[0] = .25f; p.R[4] = 4e-6f; p.R[8] = 4e-6f; } else { p.R[0] = p.R[4] = p.R[8] = .25f; }
    double e_ordered = 0, e_random = 0; int used = 0, noupd = 0;
    for (int i = 0; i < n_tracks; i++) {
        Track a; random_track(&a, .4f);
        if (spherical && a.x[0]*a.x[0] + a.x[1]*a.x[1] < 400) continue;
        Meas m; m.valid = (i % 7 != 3); if (!m.valid) noupd++;                           /* some objects not detected */
        const float px = a.x[0] + .1f*a.x[3], py = a.x[1] + .1f*a.x[4], pz = a.x[2] + .1f*a.x[5], rxy = hypotf(px, py);
        if (spherical) { m.z[0] = hypotf(rxy, pz) + .5f*rnd(); m.z[1] = atan2f(py, px) + .004f*rnd(); m.z[2] = atan2f(pz, rxy) + .004f*rnd(); }
        else           { m.z[0] = px + .5f*rnd(); m.z[1] = py + .5f*rnd(); m.z[2] = pz + .5f*rnd(); }
        Track ref = a; F::step(&ref, &m, &p);                                             /* thread-per-track reference */
        Track w0 = a; kf_warp_emulate<typename F::Warp>(&w0, &m, 1, p, 0);               /* warp, lanes in natural order */
        e_ordered = fmax(e_ordered, diff(ref, w0));
        for (unsigned seed = 1; seed <= 4; seed++) {                                      /* warp, lanes in random order */
            Track w = a; kf_warp_emulate<typename F::Warp>(&w, &m, 1, p, seed * 17u + (unsigned)i);
            e_random = fmax(e_random, diff(ref, w));
        }
        used++;
    }
    char buf[128];
    snprintf(buf, sizeof buf, "%s: warp vs thread-per-track (%d tracks, %d without measurement)", name, used, noupd);
    verdict(buf, e_ordered, 1e-5);
    snprintf(buf, sizeof buf, "%s: warp, lanes in RANDOM order (race detector)", name);
    verdict(buf, e_random, 1e-5);
}
int main() {
    compare_steps<LkfNew>("LKF upgraded", false, 3000);
    compare_steps<EkfNew>("EKF upgraded", true, 3000);

    /* solver level: every LKF structure case through the warp factor()/solve_col() must equal lkf_light_gain() */
    {
        const float A[9] = { 2,.3f,.1f, .3f,1.5f,.2f, .1f,.2f,1 }, C[9] = { .2f,.1f,0, .05f,.3f,.1f, 0,.1f,.4f };
        struct { const char *n; float S[9]; } cs[] = {
            { "identity", { 1,0,0, 0,1,0, 0,0,1 } }, { "diagonal", { 2,0,0, 0,3,0, 0,0,4 } },
            { "block r=2", { 2,.5f,0, .5f,3,0, 0,0,4 } }, { "block r=1", { 2,0,.5f, 0,3,0, .5f,0,4 } }, { "block r=0", { 2,0,0, 0,3,.5f, 0,.5f,4 } },
            { "lower triangular", { 2,0,0, .5f,3,0, .3f,.2f,4 } }, { "upper triangular", { 2,.5f,.3f, 0,3,.2f, 0,0,4 } },
            { "SPD (Cholesky)", { 4,1,.5f, 1,3,.2f, .5f,.2f,2 } }, { "indefinite (Gauss-Jordan)", { 1,2,.5f, 2,1,.3f, .5f,.3f,-1 } },
            { "needs pivoting", { 0,1,0, 1,0,0, 0,0,2 } }, { "non-symmetric dense", { 4,1,.5f, .7f,3,.2f, .5f,.9f,2 } } };
        for (auto &t : cs) {
            LkfWarp::Ctx c; memcpy(c.S, t.S, sizeof c.S); memcpy(c.Ap, A, sizeof c.Ap); memcpy(c.Cp, C, sizeof c.Cp);
            LkfWarp::factor(c, 1);
            float K[18]; const int ok = lkf_light_gain(t.S, A, C, K);
            double e = 0; if (ok != c.ok) e = 1e9;
            if (ok && c.ok) for (int j = 0; j < 6; j++) { float k[3]; LkfWarp::solve_col(c, j, k); for (int m = 0; m < 3; m++) e = fmax(e, fabs(k[m] - K[j*3+m])); }
            char b[96]; snprintf(b, sizeof b, "LKF warp solver case: %s", t.n); verdict(b, e, 1e-5);
        }
    }
    /* same for the EKF solver cases (RECIP + LDL^T) */
    {
        const float TL[9] = { 2,.3f,.1f, .3f,1.5f,.2f, .1f,.2f,1 }, TR[9] = { .2f,.1f,0, .05f,.3f,.1f, 0,.1f,.4f };
        struct { const char *n; float S[9]; } cs[] = {
            { "identity", { 1,0,0, 0,1,0, 0,0,1 } }, { "diagonal", { 2,0,0, 0,3,0, 0,0,4 } },
            { "block r=2", { 2,.5f,0, .5f,3,0, 0,0,4 } }, { "block r=1", { 2,0,.5f, 0,3,0, .5f,0,4 } }, { "block r=0", { 2,0,0, 0,3,.5f, 0,.5f,4 } },
            { "SPD (LDL^T)", { 4,1,.5f, 1,3,.2f, .5f,.2f,2 } }, { "indefinite (Gauss-Jordan)", { 1,2,.5f, 2,1,.3f, .5f,.3f,-1 } }, { "needs pivoting", { 0,1,0, 1,0,0, 0,0,2 } } };
        for (auto &t : cs) {
            EkfWarp::Ctx c; memcpy(c.S, t.S, sizeof c.S); memcpy(c.TL, TL, sizeof c.TL); memcpy(c.TR, TR, sizeof c.TR);
            EkfWarp::factor(c);
            float K[18]; const int ok = ekf_light_gain(t.S, TL, TR, K);
            double e = 0; if (ok != c.ok) e = 1e9;
            if (ok && c.ok) for (int j = 0; j < 6; j++) { float k[3]; EkfWarp::solve_col(c, j, k); for (int m = 0; m < 3; m++) e = fmax(e, fabs(k[m] - K[j*3+m])); }
            char b[96]; snprintf(b, sizeof b, "EKF warp solver case: %s", t.n); verdict(b, e, 2e-5);
        }
    }
    /* structured covariance through the FULL step: diagonal S and block S arise naturally from structured P and R */
    {
        KFParams p; memset(&p, 0, sizeof p); p.dt = 0.1f; p.sa2 = 0.05f;
        double e = 0;
        for (int variant = 0; variant < 2; variant++) {
            memset(p.R, 0, sizeof p.R); p.R[0] = .5f; p.R[4] = .4f; p.R[8] = .3f; if (variant) { p.R[1] = p.R[3] = .1f; }   /* variant 1: block-diagonal S */
            Track a; memset(&a, 0, sizeof a); for (int i = 0; i < 6; i++) { a.P[i*7] = 1.f + .2f * i; a.x[i] = (float)(i + 1); }
            Meas m = { 1, { 1.5f, 2.5f, 3.5f } };
            Track r = a, w = a; lkf_light_step(&r, &m, &p); kf_warp_emulate<LkfWarp>(&w, &m, 1, p, 5u);
            e = fmax(e, diff(r, w));
        }
        verdict("LKF warp, full step with diagonal S and block-diagonal S", e, 1e-5);
    }
    printf(bad ? "\n%d FAILED\n" : "\nWARP MAPPING: ALL CHECKS OK\n", bad);
    return bad != 0;
}
