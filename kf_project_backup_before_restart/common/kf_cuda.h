/* kf_cuda.h - includes the real CUDA runtime, or (for testing the logic on a machine without a GPU)
 * a tiny CPU "shim" that imitates it.  Build the shim with -DKF_CUDA_SHIM. */
#ifndef KF_CUDA_H
#define KF_CUDA_H
#ifdef KF_CUDA_SHIM
#include "kf_cuda_shim.h"
#else
#include <cuda_runtime.h>
#define KF_LAUNCH(K, G, B, ...) K<<<(G), (B)>>>(__VA_ARGS__)
#define KF_WARP_LAUNCH(W, G, B, ...) kf_warp_kernel<W><<<(G), (B)>>>(__VA_ARGS__)
#endif
#endif
