/* lkf_fpga_hls.cpp - EXTRA file (not one of the 17): Vitis HLS top function for the upgraded LKF.
 * It is a functional starting point and a C-simulation golden model. For real area / timing:
 *   - switch to fixed point (ap_fixed), - use the square-root-free LDL^T + RECIP path (as in the EKF header),
 *   - partition the small 3x3 arrays, - choose pipeline II and the number of parallel lanes.
 * The EKF version can be added the same way with ekf_light_step (CORDIC + RECIP are already shift-add based). */
#include "../common/lkf_lightweight.h"

extern "C" void lkf_fpga_top(Track *T, const Meas *M, int n, KFParams p) {
#pragma HLS INTERFACE m_axi port=T offset=slave bundle=gmem0 depth=1024
#pragma HLS INTERFACE m_axi port=M offset=slave bundle=gmem1 depth=1024
#pragma HLS INTERFACE s_axilite port=n
#pragma HLS INTERFACE s_axilite port=p
#pragma HLS INTERFACE s_axilite port=return
    for (int i = 0; i < n; i++) {
        Track t = T[i]; Meas m = M[i];
        lkf_light_step(&t, &m, &p);
        T[i] = t;
    }
}
