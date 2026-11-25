/* kf_gpu_runtime.cuh - GPU backend + the GPU-only program role (called by the 4 *_gpu.cu files).
 * Mapping: ONE CUDA THREAD = ONE TRACK (simple first version). Time is reported two ways:
 *   kernel_ms : only the kernel (cudaEvent)        end-to-end ms : copies + launch + kernel + synchronisation
 * Build with -DKF_UNIFIED_MEMORY on the Jetson Orin Nano: no copies, CPU and GPU share the same DRAM. */
#ifndef KF_GPU_RUNTIME_CUH
#define KF_GPU_RUNTIME_CUH
#include "kf_cuda.h"
#include "kf_runtime.h"

#define KF_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(3); } } while (0)

template <class F> __global__ void kf_kernel(Track *T, const Meas *M, int n, KFParams p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;      /* my track number */
    if (i < n) F::step(&T[i], &M[i], &p);
}

/* WARP-PER-TRACK kernel (paper Sec. 3.6.2): one warp (32 threads) per track. The lanes of the warp share the scalar equations of
 * every stage (see W::phase) and __syncwarp() separates the stages. One block = KF_WARPS_PER_BLOCK warps = 4 tracks. */
#ifndef KF_CUDA_SHIM
template <class W> __global__ void kf_warp_kernel(Track *T, const Meas *M, int n, KFParams p) {
    __shared__ typename W::Ctx sh[KF_WARPS_PER_BLOCK];            /* each warp's private scratch space, in fast shared memory */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int i = blockIdx.x * KF_WARPS_PER_BLOCK + warp;         /* the track of this warp */
    if (i >= n) return;                                           /* whole warp leaves together */
    typename W::Ctx &c = sh[warp];
    for (int k = 0; k < (int)W::NPHASE; k++) { W::phase(c, &T[i], &M[i], &p, lane, k); __syncwarp(); }
}
#endif

struct GpuBuf { Track *dT = 0; Meas *dM = 0; int cap = 0; cudaEvent_t e0, e1; bool ready = false; int mapping = 0; /* 0 thread-per-track, 1 warp-per-track */ };

/* start the right kernel for the chosen mapping (T, M are device pointers, or managed memory) */
template <class F> static void gpu_launch(const GpuBuf *b, Track *T, const Meas *M, int n, const KFParams *p) {
    if (b->mapping == 1 && F::has_warp) {
        const int B = KF_WARP * KF_WARPS_PER_BLOCK, G = (n + KF_WARPS_PER_BLOCK - 1) / KF_WARPS_PER_BLOCK;
        KF_WARP_LAUNCH(typename F::Warp, G, B, T, M, n, *p);
    } else {
        const int B = 128, G = (n + B - 1) / B;
        KF_LAUNCH(kf_kernel<F>, G, B, T, M, n, *p);
    }
}
template <class F> static int kf_resolve_mapping(const KfArgs &a) {
    int m = (a.mapping < 0) ? (F::has_warp ? 1 : 0) : a.mapping;
    if (m == 1 && !F::has_warp) { fprintf(stderr, "note: %s has no warp mapping (only the upgraded filters do): using thread-per-track\n", F::name()); m = 0; }
    return m;
}
static inline const char *kf_mapping_name(int m) { return m ? "warp-per-track (32 lanes cooperate on one track)" : "thread-per-track"; }

static inline void gpu_init(GpuBuf *b) { KF_CK(cudaEventCreate(&b->e0)); KF_CK(cudaEventCreate(&b->e1)); b->ready = true; }

/* process one batch on the GPU. T (tracks) is updated in place on the host. kernel_ms may be NULL. */
template <class F> static void gpu_step(GpuBuf *b, Track *T, const Meas *M, int n, const KFParams *p, double *kernel_ms) {
    if (n <= 0) { if (kernel_ms) *kernel_ms = 0; return; }
    if (!b->ready) gpu_init(b);
#ifdef KF_UNIFIED_MEMORY
    KF_CK(cudaEventRecord(b->e0));
    gpu_launch<F>(b, T, M, n, p);                             /* T and M are managed memory: no copies */
    KF_CK(cudaEventRecord(b->e1));
    KF_CK(cudaDeviceSynchronize());
#else
    if (n > b->cap) {                                          /* device buffers are kept between frames */
        if (b->dT) { cudaFree(b->dT); cudaFree(b->dM); }
        KF_CK(cudaMalloc((void **)&b->dT, sizeof(Track) * (size_t)n)); KF_CK(cudaMalloc((void **)&b->dM, sizeof(Meas) * (size_t)n)); b->cap = n;
    }
    KF_CK(cudaMemcpy(b->dT, T, sizeof(Track) * (size_t)n, cudaMemcpyHostToDevice));
    KF_CK(cudaMemcpy(b->dM, M, sizeof(Meas) * (size_t)n, cudaMemcpyHostToDevice));
    KF_CK(cudaEventRecord(b->e0));
    gpu_launch<F>(b, b->dT, b->dM, n, p);
    KF_CK(cudaEventRecord(b->e1));
    KF_CK(cudaMemcpy(T, b->dT, sizeof(Track) * (size_t)n, cudaMemcpyDeviceToHost));   /* also waits for the kernel */
#endif
    KF_CK(cudaGetLastError());
    KF_CK(cudaEventSynchronize(b->e1));
    if (kernel_ms) { float ms = 0; KF_CK(cudaEventElapsedTime(&ms, b->e0, b->e1)); *kernel_ms = ms; }
}
/* first CUDA call creates the context (can take 100s of ms): do it once, untimed */
template <class F> static void gpu_warmup(GpuBuf *b, const KfArgs &a) {
    Scenario s; if (!scenario_open<F>(&s, 1024, a)) return;
    for (int i = 0; i < 3; i++) gpu_step<F>(b, s.T, s.M, s.n, &s.p, 0);
    scenario_close(&s);
}

/* ---------------- PROGRAM ROLE 2: GPU-only run ---------------- */
template <class F> static int kf_main_gpu(int argc, char **argv) {
    KfArgs a = kf_parse(argc, argv, { 1, 10, 100, 1000, 10000, 100000 });
    GpuBuf gb; gb.mapping = kf_resolve_mapping<F>(a);
#ifdef KF_UNIFIED_MEMORY
    const char *mode = "unified memory (Jetson)";
#else
    const char *mode = "discrete GPU, explicit copies";
#endif
    printf("== %s | GPU-only | %s | mapping: %s | %d frames per size ==\n", F::name(), mode, kf_mapping_name(gb.mapping), a.frames);
    gpu_warmup<F>(&gb, a);
    printf("%8s %14s %12s %14s %10s %10s %6s\n", "N", "e2e_ms/frame", "kernel_ms", "updates/s", "RMSE_pos", "RMSE_vel", "bad");
    KfTimingLog tl(F::tag(), F::name(), "gpu", gb.mapping ? "warp" : "thread");   /* file 18 */
    for (int n : a.sizes) {
        Scenario s; if (!scenario_open<F>(&s, n, a)) { fprintf(stderr, "cannot open scenario N=%d\n", n); return 1; }
        double total = 0, ktotal = 0; Acc acc; std::vector<double> e2e, ker;
        for (int f = 0; f < a.frames; f++) {
            if (!det_next(&s.src, s.M, 0)) break;
            double k = 0; const double t0 = kf_now_ms(); gpu_step<F>(&gb, s.T, s.M, n, &s.p, &k);
            const double dt_ms = kf_now_ms() - t0; total += dt_ms; ktotal += k; e2e.push_back(dt_ms); ker.push_back(k);
            if (f >= a.frames / 2) acc_add(&acc, s);
        }
        tl.record("gpu_e2e", n, e2e); tl.record("gpu_kernel", n, ker, false);   /* kernel time is a PART of the end-to-end time */
        char ep[32], ev[32]; acc_text(acc, ep, ev, 32);
        const double ms = total / a.frames;
        printf("%8d %14.5f %12.5f %14.0f %10s %10s %6ld\n", n, ms, ktotal / a.frames, ms > 0 ? n / ms * 1e3 : 0.0, ep, ev, acc.bad);
        scenario_close(&s);
    }
    tl.finish(a.timing_csv);
    return 0;
}
#endif
