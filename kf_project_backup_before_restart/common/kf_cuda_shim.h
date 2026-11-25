/* kf_cuda_shim.h - TESTING ONLY. Lets the .cu programs compile with g++ and run on the CPU (one "GPU thread" at a time),
 * so the control logic (copies, launches, adaptive controller, crossover analysis) can be tested without a GPU.
 * Timings from the shim are meaningless. */
#ifndef KF_CUDA_SHIM_H
#define KF_CUDA_SHIM_H
#include <stdlib.h>
#include <string.h>
#include <chrono>
#define __global__
typedef int cudaError_t;
enum { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
struct KfShimEvent { double t; };
typedef KfShimEvent *cudaEvent_t;
struct KfShimDim { int x; };
static KfShimDim threadIdx, blockIdx, blockDim;
static inline const char *cudaGetErrorString(cudaError_t) { return "shim"; }
static inline cudaError_t cudaGetLastError() { return 0; }
static inline cudaError_t cudaMalloc(void **p, size_t b) { *p = malloc(b); return *p ? 0 : 1; }
static inline cudaError_t cudaMallocManaged(void **p, size_t b) { *p = malloc(b); return *p ? 0 : 1; }
static inline cudaError_t cudaFree(void *p) { free(p); return 0; }
static inline cudaError_t cudaMemcpy(void *d, const void *s, size_t b, cudaMemcpyKind) { memcpy(d, s, b); return 0; }
static inline cudaError_t cudaDeviceSynchronize() { return 0; }
static inline double kf_shim_now() { using namespace std::chrono; return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count(); }
static inline cudaError_t cudaEventCreate(cudaEvent_t *e) { *e = new KfShimEvent{0}; return 0; }
static inline cudaError_t cudaEventRecord(cudaEvent_t e) { e->t = kf_shim_now(); return 0; }
static inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return 0; }
static inline cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t a, cudaEvent_t b) { *ms = (float)(b->t - a->t); return 0; }
#define KF_LAUNCH(K, G, B, ...) do { blockDim.x = (B); for (int _b = 0; _b < (G); _b++) { blockIdx.x = _b; \
        for (int _t = 0; _t < (B); _t++) { threadIdx.x = _t; K(__VA_ARGS__); } } } while (0)
#define KF_WARP_LAUNCH(W, G, B, ...) do { (void)(G); (void)(B); kf_warp_emulate<W>(__VA_ARGS__); } while (0)   /* host emulation of warps + phases */
#endif
