/* test_ekf.cpp - checks the EKF building blocks and the full steps against the original (conventional) formulas. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kf_filters.h"

static int bad = 0;
static unsigned st = 99u;
static float rnd() { st = st * 1664525u + 1013904223u; return ((st >> 8) & 0xFFFF) / 65535.f - 0.5f; }
static void verdict(const char *name, double err, double tol) {
    printf("%s %-46s err = %.2e (tol %.0e)\n", err < tol ? "pass" : "FAIL", name, err, tol);
    if (!(err < tol)) bad++;
}
int main() {
    /* 1. CORDIC vectoring, all quadrants */
    double e_mag = 0, e_ang = 0;
    for (int i = 0; i < 20000; i++) {
        float a = 200.f * rnd(), b = 200.f * rnd(), mag, ang;
        kf_cordic_vec(a, b, &mag, &ang);
        e_mag = fmax(e_mag, fabs(mag - hypot(a, b)) / hypot(a, b));
        double d = fabs(ang - atan2((double)b, (double)a)); if (d > 3.14159) d = fabs(d - 2 * 3.14159265358979);  /* +-pi seam */
        e_ang = fmax(e_ang, d);
    }
    verdict("CORDIC vectoring: magnitude (relative)", e_mag, 1e-6);
    verdict("CORDIC vectoring: angle (rad)", e_ang, 1e-5);
    /* 2. CORDIC rotation */
    double e_cs = 0;
    for (int i = 0; i < 20000; i++) {
        float al = 2.f * 3.14159f * rnd(), c, s; kf_cordic_rot(al, &c, &s);
        e_cs = fmax(e_cs, fmax(fabs(c - cos((double)al)), fabs(s - sin((double)al))));
    }
    verdict("CORDIC rotation: sin/cos", e_cs, 1e-5);
    /* 3. RECIP */
    double e_r = 0;
    for (int i = 0; i < 20000; i++) {
        float a = powf(10.f, 6.f * rnd()) * (rnd() > 0 ? 1.f : -1.f);        /* magnitudes 1e-3 .. 1e3, both signs */
        e_r = fmax(e_r, fabs(kf_recip(a) * (double)a - 1.0));
    }
    verdict("RECIP: |a * RECIP(a) - 1|", e_r, 5e-7);
    /* 4. original Jacobian formulas vs finite differences of h(x) (double precision) */
    double e_j = 0;
    for (int i = 0; i < 200; i++) {
        double x = 80 * rnd(), y = 80 * rnd(), z = 40 * rnd(); if (x * x + y * y < 100) continue;
        float xm[6] = { (float)x, (float)y, (float)z, 0, 0, 0 };
        /* build H exactly as ekf_conv_step does */
        float rxy2 = xm[0]*xm[0] + xm[1]*xm[1], rxy = sqrtf(rxy2), rho2 = rxy2 + xm[2]*xm[2], rho = sqrtf(rho2);
        float H[9] = { xm[0]/rho, xm[1]/rho, xm[2]/rho, -xm[1]/rxy2, xm[0]/rxy2, 0, -xm[0]*xm[2]/(rho2*rxy), -xm[1]*xm[2]/(rho2*rxy), rxy/rho2 };
        auto h = [](double X, double Y, double Z, double o[3]) { double r = sqrt(X*X + Y*Y); o[0] = sqrt(r*r + Z*Z); o[1] = atan2(Y, X); o[2] = atan2(Z, r); };
        double d = 1e-4, p[3], q[3], pos[3] = { x, y, z };
        for (int c = 0; c < 3; c++) {
            double a[3] = { pos[0], pos[1], pos[2] }, b[3] = { pos[0], pos[1], pos[2] }; a[c] += d; b[c] -= d;
            h(a[0], a[1], a[2], p); h(b[0], b[1], b[2], q);
            for (int r = 0; r < 3; r++) e_j = fmax(e_j, fabs((p[r] - q[r]) / (2 * d) - H[r * 3 + c]));
        }
    }
    verdict("EKF Jacobian (paper formulas) vs finite difference", e_j, 1e-4);
    /* 5. EKF gain cases vs explicit inverse */
    {
        const float TL[9] = { 2,.3f,.1f, .3f,1.5f,.2f, .1f,.2f,1 }, TR[9] = { .2f,.1f,0, .05f,.3f,.1f, 0,.1f,.4f };
        struct { const char *n; float S[9]; int ok; } cs[] = {
            { "identity", { 1,0,0, 0,1,0, 0,0,1 }, 1 }, { "diagonal", { 2,0,0, 0,3,0, 0,0,4 }, 1 },
            { "block r=2", { 2,.5f,0, .5f,3,0, 0,0,4 }, 1 }, { "block r=1", { 2,0,.5f, 0,3,0, .5f,0,4 }, 1 },
            { "block r=0", { 2,0,0, 0,3,.5f, 0,.5f,4 }, 1 }, { "SPD -> LDL^T", { 4,1,.5f, 1,3,.2f, .5f,.2f,2 }, 1 },
            { "indefinite -> Gauss-Jordan", { 1,2,.5f, 2,1,.3f, .5f,.3f,-1 }, 1 }, { "needs pivoting", { 0,1,0, 1,0,0, 0,0,2 }, 1 },
            { "singular", { 1,1,0, 1,1,0, 0,0,1 }, 0 } };
        for (auto &c : cs) {
            float K[18], Si[9]; int ok = ekf_light_gain(c.S, TL, TR, K);
            if (ok != c.ok) { printf("FAIL gain %-30s ok=%d expected %d\n", c.n, ok, c.ok); bad++; continue; }
            if (!ok) { printf("pass gain %-30s rejected as expected\n", c.n); continue; }
            kf_inv3(c.S, Si); double err = 0;
            for (int j = 0; j < 6; j++) for (int m = 0; m < 3; m++) {
                float v = 0; for (int q = 0; q < 3; q++) v += Si[m*3+q] * kf_B(TL, TR, q, j);
                err = fmax(err, fabs(v - K[j*3+m]));
            }
            char nm[64]; snprintf(nm, 64, "EKF gain: %s", c.n); verdict(nm, err, 2e-5);
        }
    }
    /* 6. full steps: upgraded vs original, random realistic tracks */
    KFParams p; memset(&p, 0, sizeof p); p.dt = 0.1f; p.sa2 = 0.05f; p.R[0] = .25f; p.R[4] = 4e-6f; p.R[8] = 4e-6f;
    double ex = 0, eP = 0, e_l = 0, e_lP = 0, e_i = 0; int n_upd = 0;
    for (int i = 0; i < 3000; i++) {
        Track a; float G[36];
        for (int k = 0; k < 36; k++) G[k] = 0.3f * rnd();
        for (int r = 0; r < 6; r++) for (int c = 0; c < 6; c++) { float s = (r == c) ? 1.f : 0.f; for (int q = 0; q < 6; q++) s += G[r*6+q]*G[c*6+q]; a.P[r*6+c] = s; }
        for (int k = 0; k < 3; k++) a.x[k] = 100.f * rnd();
        a.x[2] *= .4f;
        for (int k = 3; k < 6; k++) a.x[k] = 8.f * rnd();
        if (a.x[0]*a.x[0] + a.x[1]*a.x[1] < 400) continue;
        Track b = a; Meas m; m.valid = 1;
        float px = a.x[0] + .1f*a.x[3], py = a.x[1] + .1f*a.x[4], pz = a.x[2] + .1f*a.x[5];
        float rxy = hypotf(px, py);
        m.z[0] = hypotf(rxy, pz) + .5f*rnd(); m.z[1] = atan2f(py, px) + .004f*rnd(); m.z[2] = atan2f(pz, rxy) + .004f*rnd();
        ekf_conv_step(&a, &m, &p); ekf_light_step(&b, &m, &p); n_upd++;
        for (int k = 0; k < 6; k++) ex = fmax(ex, fabs(a.x[k] - b.x[k]) / (1.0 + fabs(a.x[k])));
        double dn = 0, nn = 1e-9; for (int k = 0; k < 36; k++) { dn += (a.P[k]-b.P[k])*(a.P[k]-b.P[k]); nn += a.P[k]*a.P[k]; }
        eP = fmax(eP, sqrt(dn / nn));
        /* LKF pair on the same track with a Cartesian measurement */
        Track c = a, d = a; Meas ml; ml.valid = 1; for (int k = 0; k < 3; k++) ml.z[k] = a.x[k] + .5f*rnd();
        KFParams pl = p; for (int k = 0; k < 9; k++) pl.R[k] = 0; pl.R[0] = pl.R[4] = pl.R[8] = .25f;
        lkf_conv_step(&c, &ml, &pl); lkf_light_step(&d, &ml, &pl);
        for (int k = 0; k < 6; k++) e_l = fmax(e_l, fabs(c.x[k] - d.x[k]) / (1.0 + fabs(c.x[k])));
        dn = 0; nn = 1e-9; for (int k = 0; k < 36; k++) { dn += (c.P[k]-d.P[k])*(c.P[k]-d.P[k]); nn += c.P[k]*c.P[k]; }
        e_lP = fmax(e_lP, sqrt(dn / nn));
        /* init */
        Meas mi; mi.valid = 1; mi.z[0] = 20.f + 100.f*fabsf(rnd()); mi.z[1] = 6.2f*rnd(); mi.z[2] = 3.f*rnd();
        Track i1, i2; ekf_conv_init(&i1, &mi); ekf_light_init(&i2, &mi);
        for (int k = 0; k < 3; k++) e_i = fmax(e_i, fabs(i1.x[k] - i2.x[k]));
    }
    printf("(%d random tracks)\n", n_upd);
    verdict("EKF step: state, upgraded vs original", ex, 2e-4);
    verdict("EKF step: covariance (relative Frobenius)", eP, 1e-3);
    verdict("LKF step: state, upgraded vs original", e_l, 1e-4);
    verdict("LKF step: covariance (relative Frobenius)", e_lP, 1e-4);
    verdict("EKF init (CORDIC rotation vs sin/cos), metres", e_i, 1e-3);
    printf(bad ? "\n%d FAILED\n" : "\nALL EKF/LKF CHECKS OK\n", bad);
    return bad != 0;
}
