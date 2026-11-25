/* lkf_orig_hetero.cu
 * FILTER : ORIGINAL (matrix-form) linear Kalman filter
 * ROLE   : ADAPTIVE CPU/GPU: a controller picks the processor each frame from a cost model with a switching margin; compared with always-CPU and always-GPU on the same rising-then-falling object count.
 * The measurements come from detection/detection.h (file 17). All heavy lifting is in common/kf_adaptive.cuh. */
#include "../common/kf_adaptive.cuh"

int main(int argc, char **argv) { return kf_main_hetero<LkfOrig>(argc, argv); }
