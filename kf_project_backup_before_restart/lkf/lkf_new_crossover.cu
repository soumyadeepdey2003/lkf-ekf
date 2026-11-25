/* lkf_new_crossover.cu
 * FILTER : UPGRADED (scalar, structure-aware gain) linear Kalman filter
 * ROLE   : CROSSOVER FINDER ("saddle point"): times CPU and GPU for N = 1,2,4,... objects and reports the break-even N* where the GPU becomes faster, the two switching thresholds, and a CSV.
 * The measurements come from detection/detection.h (file 17). All heavy lifting is in common/kf_adaptive.cuh. */
#include "../common/kf_adaptive.cuh"

int main(int argc, char **argv) { return kf_main_crossover<LkfNew>(argc, argv); }
