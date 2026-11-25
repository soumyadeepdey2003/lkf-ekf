/* ekf_orig_gpu.cu
 * FILTER : ORIGINAL (matrix-form) extended Kalman filter
 * ROLE   : GPU ONLY: one CUDA thread per track. Reports end-to-end time (copies included) and kernel-only time, plus accuracy.
 * The measurements come from detection/detection.h (file 17). All heavy lifting is in common/kf_gpu_runtime.cuh. */
#include "../common/kf_gpu_runtime.cuh"

int main(int argc, char **argv) { return kf_main_gpu<EkfOrig>(argc, argv); }
