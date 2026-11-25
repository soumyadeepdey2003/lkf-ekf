/* lkf_new_cpu.cpp
 * FILTER : UPGRADED (scalar, structure-aware gain) linear Kalman filter
 * ROLE   : CPU ONLY: all tracks are filtered by OpenMP CPU threads. Reports ms/frame, updates/s and accuracy for each object count.
 * The measurements come from detection/detection.h (file 17). All heavy lifting is in common/kf_runtime.h. */
#include "../common/kf_runtime.h"

int main(int argc, char **argv) { return kf_main_cpu<LkfNew>(argc, argv); }
