/* test_gain.c - unit test: every Kalman-gain solver case must equal K = B^T S^-1 (reference by explicit inverse). */
#include <stdio.h>
#include "kf_types.h"
#include "lkf_conventional.h"
#include "lkf_lightweight.h"

static int bad = 0;
static void check(const char *name, const float S[9], const float A[9], const float C[9], int expect_ok) {
    float K[18], Si[9];
    int ok = lkf_light_gain(S, A, C, K);
    if (ok != expect_ok) { printf("FAIL %-22s ok=%d expected %d\n", name, ok, expect_ok); bad++; return; }
    if (!ok) { printf("pass %-22s (correctly rejected)\n", name); return; }
    kf_inv3(S, Si);
    double err = 0;
    for (int j = 0; j < 6; j++) for (int m = 0; m < 3; m++) {
        float v = 0; for (int q = 0; q < 3; q++) v += Si[m*3+q] * kf_B(A, C, q, j);   /* (S^-1 B)[m][j] */
        double d = fabs(v - K[j*3+m]); if (d > err) err = d;
    }
    printf("%s %-22s max|K-ref| = %.2e\n", err < 1e-4 ? "pass" : "FAIL", name, err);
    if (err >= 1e-4) bad++;
}
int main(void) {
    const float A[9] = { 2,.3f,.1f, .3f,1.5f,.2f, .1f,.2f,1 }, C[9] = { .2f,.1f,0, .05f,.3f,.1f, 0,.1f,.4f };
    const float I[9]  = { 1,0,0, 0,1,0, 0,0,1 };
    const float Dg[9] = { 2,0,0, 0,3,0, 0,0,4 };
    const float B2[9] = { 2,.5f,0, .5f,3,0, 0,0,4 };    /* isolated r = 2 */
    const float B1[9] = { 2,0,.5f, 0,3,0, .5f,0,4 };    /* isolated r = 1 */
    const float B0[9] = { 2,0,0, 0,3,.5f, 0,.5f,4 };    /* isolated r = 0 */
    const float SP[9] = { 4,1,.5f, 1,3,.2f, .5f,.2f,2 };/* dense SPD -> Cholesky */
    const float IN[9] = { 1,2,.5f, 2,1,.3f, .5f,.3f,-1 };/* symmetric, NOT SPD -> Gauss-Jordan */
    const float PV[9] = { 0,1,0, 1,0,0, 0,0,2 };         /* needs row pivoting (zero on diagonal) */
    const float SG[9] = { 1,1,0, 1,1,0, 0,0,1 };         /* singular -> must be rejected */
    const float TL[9] = { 2,0,0, .5f,3,0, .3f,.2f,4 };   /* lower triangular (Case 4a) */
    const float TU[9] = { 2,.5f,.3f, 0,3,.2f, 0,0,4 };   /* upper triangular (Case 4b) */
    const float NS_[9] = { 4,1,.5f, 0.7f,3,.2f, .5f,.9f,2 };  /* NOT symmetric, dense -> must go to Gauss-Jordan, not Cholesky */
    check("identity", I, A, C, 1);       check("diagonal", Dg, A, C, 1);
    check("block r=2", B2, A, C, 1);     check("block r=1", B1, A, C, 1);     check("block r=0", B0, A, C, 1);
    check("lower triangular", TL, A, C, 1); check("upper triangular", TU, A, C, 1);
    check("non-symmetric dense (G-J)", NS_, A, C, 1);
    check("SPD (Cholesky)", SP, A, C, 1);check("indefinite (G-J)", IN, A, C, 1);
    check("pivoting needed (G-J)", PV, A, C, 1); check("singular", SG, A, C, 0);
    float K[18]; int g = lkf_gain_gauss_jordan(SP, A, C, K); float K2[18]; lkf_gain_spd(SP, A, C, K2);
    double e = 0; for (int i = 0; i < 18; i++) e = fmax(e, fabs(K[i] - K2[i]));
    printf("%s Gauss-Jordan == Cholesky on SPD (%.2e)\n", (g && e < 1e-4) ? "pass" : "FAIL", e); if (!g || e >= 1e-4) bad++;
    printf(bad ? "\n%d FAILED\n" : "\nALL GAIN CASES OK\n", bad);
    return bad != 0;
}
