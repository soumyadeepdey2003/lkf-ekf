/* kf_adaptive.cuh - the two programs that need BOTH processors:
 *   ROLE 3  kf_main_hetero    : adaptive CPU/GPU controller (paper Sec. 3.6.3) on a rising-then-falling object count
 *   ROLE 4  kf_main_crossover : measures CPU vs GPU end-to-end time for increasing N and finds the break-even point N*
 * Both processors always receive the SAME data; the whole batch goes to ONE processor per frame. */
#ifndef KF_ADAPTIVE_CUH
#define KF_ADAPTIVE_CUH
#include "kf_gpu_runtime.cuh"

/* ================= the controller ================= */
struct HeteroState {
    double cpu_per_track_ms = 1e-3, gpu_fixed_ms = 0.2, gpu_per_track_ms = 1e-4;   /* cost model: T_cpu = a*N ; T_gpu = f + b*N */
    double margin = 0.10;       /* switching margin delta: move only if the other side is (1-delta) times cheaper */
    double alpha = 0.2;         /* smoothing of the runtime correction */
    double tauD = 0.25;         /* reassess only if N changed by more than this fraction (workload-change threshold) */
    double cpu_corr = 0.0, gpu_corr = 0.0;   /* smoothed relative residuals (observed-predicted)/predicted */
    int last = 0, last_n = -1;  /* previous processor (0 = CPU, 1 = GPU) and N at the last assessment */
};
static inline double hetero_cost_cpu(const HeteroState &s, int n) { return s.cpu_per_track_ms * n * (1.0 + s.cpu_corr); }
static inline double hetero_cost_gpu(const HeteroState &s, int n) { return (s.gpu_fixed_ms + s.gpu_per_track_ms * n) * (1.0 + s.gpu_corr); }
static inline int hetero_choose(HeteroState *s, int n) {          /* returns 1 = GPU, 0 = CPU */
    if (s->last_n >= 0 && fabs((double)n - s->last_n) / (s->last_n + 1.0) <= s->tauD) return s->last;   /* no reassessment */
    const double tc = hetero_cost_cpu(*s, n), tg = hetero_cost_gpu(*s, n);
    int g = s->last;                                               /* hysteresis: keep previous unless clearly worse */
    if (tg < (1.0 - s->margin) * tc) g = 1; else if (tc < (1.0 - s->margin) * tg) g = 0;
    s->last_n = n; s->last = g;
    return g;
}
static inline void hetero_observe(HeteroState *s, int gpu, int n, double obs_ms) {
    const double pred = gpu ? hetero_cost_gpu(*s, n) / (1.0 + s->gpu_corr) : hetero_cost_cpu(*s, n) / (1.0 + s->cpu_corr);
    if (pred <= 0) return;
    double r = (obs_ms - pred) / pred; if (r > 5) r = 5; if (r < -0.9) r = -0.9;      /* one odd sample cannot wreck the model */
    double &c = gpu ? s->gpu_corr : s->cpu_corr; c = (1.0 - s->alpha) * c + s->alpha * r;
}

/* measure end-to-end time of CPU and GPU for one N: median of `reps` runs, tracks restored before every run (untimed) */
template <class F> static void measure_pair(GpuBuf *gb, const KfArgs &a, int n, int reps, double *cpu_ms, double *gpu_ms, double *gpu_kernel_ms, KfTimingLog *tl = 0) {
    Scenario s; if (!scenario_open<F>(&s, n, a)) { *cpu_ms = *gpu_ms = *gpu_kernel_ms = -1; return; }
    det_next(&s.src, s.M, 0);
    Track *snap = (Track *)malloc(sizeof(Track) * (size_t)n); memcpy(snap, s.T, sizeof(Track) * (size_t)n);
    std::vector<double> tc, tg, tk;
    for (int r = -2; r < reps; r++) {                              /* two warm-up rounds */
        memcpy(s.T, snap, sizeof(Track) * (size_t)n);
        double t0 = kf_now_ms(); cpu_step<F>(s.T, s.M, n, &s.p); const double c = kf_now_ms() - t0;
        memcpy(s.T, snap, sizeof(Track) * (size_t)n);
        double k = 0; t0 = kf_now_ms(); gpu_step<F>(gb, s.T, s.M, n, &s.p, &k); const double g = kf_now_ms() - t0;
        if (r >= 0) { tc.push_back(c); tg.push_back(g); tk.push_back(k); }
    }
    *cpu_ms = kf_median(tc); *gpu_ms = kf_median(tg); *gpu_kernel_ms = kf_median(tk);
    if (tl) { tl->record("cpu", n, tc); tl->record("gpu_e2e", n, tg); tl->record("gpu_kernel", n, tk, false); }   /* file 18 */
    free(snap); scenario_close(&s);
}
/* initial cost model from two probes (N=1 and N=nb) */
template <class F> static void hetero_calibrate(HeteroState *h, GpuBuf *gb, const KfArgs &a, int nb) {
    double c1, g1, k1, cb, gbm, kb;
    measure_pair<F>(gb, a, 1, 15, &c1, &g1, &k1); measure_pair<F>(gb, a, nb, 9, &cb, &gbm, &kb);
    h->cpu_per_track_ms = cb / nb;
    h->gpu_fixed_ms = g1;
    h->gpu_per_track_ms = (gbm - g1) / (nb - 1); if (h->gpu_per_track_ms < 1e-9) h->gpu_per_track_ms = 1e-9;
}

/* ================= ROLE 3: adaptive run ================= */
struct SeqResult { double total_ms = 0; Acc acc; std::vector<double> level_ms; std::vector<int> level_gpu; };

/* strategy: 0 = always CPU, 1 = always GPU, 2 = adaptive. Identical data for every strategy (same seeds). */
template <class F> static SeqResult run_sequence(int strategy, const std::vector<int> &levels, const KfArgs &a, HeteroState hs, GpuBuf *gb,
                                                 KfTimingLog *tl = 0, const char *series = "") {
    SeqResult r;
    for (int n : levels) {
        Scenario s; if (!scenario_open<F>(&s, n, a)) { fprintf(stderr, "cannot open scenario N=%d\n", n); exit(1); }
        double lvl = 0; int gpu_frames = 0; std::vector<double> frame_ms, ovh_ms;
        for (int f = 0; f < a.frames; f++) {
            if (!det_next(&s.src, s.M, 0)) break;
            const double t0 = kf_now_ms();                           /* decision overhead is INSIDE the timed region */
            const int g = (strategy == 2) ? hetero_choose(&hs, n) : strategy;
            const double t1 = kf_now_ms();
            if (g) gpu_step<F>(gb, s.T, s.M, n, &s.p, 0); else cpu_step<F>(s.T, s.M, n, &s.p);
            const double t2 = kf_now_ms();
            if (strategy == 2) hetero_observe(&hs, g, n, t2 - t0);
            const double t3 = kf_now_ms();
            lvl += t3 - t0; gpu_frames += g; frame_ms.push_back(t3 - t0);
            if (strategy == 2) ovh_ms.push_back((t1 - t0) + (t3 - t2));      /* what the controller itself costs */
            if (f >= a.frames / 2) acc_add(&r.acc, s);
        }
        if (tl) { tl->record(series, n, frame_ms); if (strategy == 2) tl->record("adaptive_ctrl", n, ovh_ms, false); }
        r.total_ms += lvl; r.level_ms.push_back(lvl / a.frames); r.level_gpu.push_back(gpu_frames * 2 > a.frames);
        scenario_close(&s);
    }
    return r;
}
template <class F> static int kf_main_hetero(int argc, char **argv) {
    KfArgs a = kf_parse(argc, argv, { 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536 });
    if (!a.stream.empty()) { fprintf(stderr, "hetero/crossover use the synthetic scene (they need to change N); use *_cpu / *_gpu for a stream\n"); return 2; }
    std::vector<int> levels = a.sizes;                               /* rising ... then falling */
    for (int i = (int)a.sizes.size() - 2; i >= 0; i--) levels.push_back(a.sizes[i]);
    GpuBuf gb; gb.mapping = kf_resolve_mapping<F>(a); gpu_warmup<F>(&gb, a);
    KfTimingLog tl(F::tag(), F::name(), "hetero", gb.mapping ? "warp" : "thread");     /* file 18 */
    HeteroState hs; hs.margin = a.margin;
    hetero_calibrate<F>(&hs, &gb, a, a.sizes.back() < 16384 ? a.sizes.back() : 16384);
    printf("== %s | ADAPTIVE CPU/GPU | GPU mapping: %s | margin=%.2f | frames/level=%d | N: ", F::name(), kf_mapping_name(gb.mapping), hs.margin, a.frames);
    for (int n : levels) printf("%d ", n);
    printf("==\ncalibrated cost model: T_cpu = %.3e ms * N ; T_gpu = %.4f ms + %.3e ms * N\n", hs.cpu_per_track_ms, hs.gpu_fixed_ms, hs.gpu_per_track_ms);
    SeqResult rc = run_sequence<F>(0, levels, a, hs, &gb, &tl, "cpu_fixed"), rg = run_sequence<F>(1, levels, a, hs, &gb, &tl, "gpu_fixed"), ra = run_sequence<F>(2, levels, a, hs, &gb, &tl, "adaptive");
    printf("\n%8s %5s | %12s %12s %12s   (ms/frame per level; level index %s)\n", "N", "dir", "CPU fixed", "GPU fixed", "ADAPTIVE", "rising first");
    const int up = (int)a.sizes.size();
    for (size_t i = 0; i < levels.size(); i++)
        printf("%8d %5s | %12.5f %12.5f %12.5f   -> adaptive used %s\n", levels[i], (int)i < up ? "up" : "down", rc.level_ms[i], rg.level_ms[i], ra.level_ms[i], ra.level_gpu[i] ? "GPU" : "CPU");
    int sw_up = -1, sw_dn = -1;
    for (int i = 0; i < up; i++) if (ra.level_gpu[i]) { sw_up = levels[i]; break; }
    for (size_t i = up; i < levels.size(); i++) if (!ra.level_gpu[i]) { sw_dn = levels[i]; break; }
    printf("\nTOTAL time over the whole sequence: CPU-only %.2f ms | GPU-only %.2f ms | ADAPTIVE %.2f ms\n", rc.total_ms, rg.total_ms, ra.total_ms);
    const double best = std::min(rc.total_ms, rg.total_ms);
    printf("S_adaptive = min(T_cpu, T_gpu) / T_adaptive = %.3f   (>1: adaptive beat BOTH fixed choices)\n", best / ra.total_ms);
    printf("adaptive switched CPU->GPU at N = %d (rising) and GPU->CPU at N = %d (falling)   (-1 = never)\n", sw_up, sw_dn);
    char ep[32], ev[32]; acc_text(ra.acc, ep, ev, 32); printf("accuracy under adaptive execution: RMSE_pos = %s  RMSE_vel = %s  (bad tracks: %ld)\n", ep, ev, ra.acc.bad);
    tl.finish(a.timing_csv);
    return 0;
}

/* ================= ROLE 4: crossover ("saddle point") finder ================= */
template <class F> static int kf_main_crossover(int argc, char **argv) {
    KfArgs a = kf_parse(argc, argv, {});
    if (!a.stream.empty()) { fprintf(stderr, "crossover needs the synthetic scene\n"); return 2; }
    if (a.sizes.empty()) for (int n = 1; n <= a.maxn; n *= 2) a.sizes.push_back(n);
    GpuBuf gb; gb.mapping = kf_resolve_mapping<F>(a); gpu_warmup<F>(&gb, a);
    printf("== %s | CPU vs GPU CROSSOVER (end-to-end, median of repeats) | GPU mapping: %s | margin=%.2f ==\n", F::name(), kf_mapping_name(gb.mapping), a.margin);
    printf("%8s %12s %12s %12s %9s\n", "N", "CPU ms", "GPU e2e ms", "GPU kern ms", "GPU/CPU");
    std::vector<double> N, tc, tg;
    KfTimingLog tl(F::tag(), F::name(), "crossover", gb.mapping ? "warp" : "thread");  /* file 18 */
    char csvname[96]; snprintf(csvname, sizeof csvname, "crossover_%s.csv", F::tag());
    FILE *csv = fopen(csvname, "w"); if (csv) fprintf(csv, "N,cpu_ms,gpu_e2e_ms,gpu_kernel_ms\n");
    for (int n : a.sizes) {
        double c, g, k; const int reps = n <= 1024 ? 41 : (n <= 16384 ? 21 : 9);
        measure_pair<F>(&gb, a, n, reps, &c, &g, &k, &tl);
        if (c < 0) return 1;
        printf("%8d %12.5f %12.5f %12.5f %9.3f\n", n, c, g, k, g / c);
        if (csv) fprintf(csv, "%d,%.6f,%.6f,%.6f\n", n, c, g, k);
        N.push_back(n); tc.push_back(c); tg.push_back(g);
    }
    if (csv) fclose(csv);
    const int m = (int)N.size();
    /* break-even N*: first N from which the GPU stays faster for every larger tested N */
    int kc = -1;
    for (int i = 0; i < m && kc < 0; i++) { bool stay = true; for (int j = i; j < m; j++) if (tg[j] >= tc[j]) stay = false; if (stay) kc = i; }
    printf("\n");
    if (kc < 0) {
        bool never = true; for (int i = 0; i < m; i++) if (tg[i] < tc[i]) never = false;
        printf("RESULT: %s\n", never ? "GPU was never faster inside the tested range - NO crossover found (not extrapolated)."
                                      : "curves intersect more than once / GPU not consistently faster at the top of the range - NO single crossover; see CSV.");
    } else if (kc == 0) {
        printf("RESULT: GPU is already faster at the smallest tested N = %d (crossover lies below the tested range).\n", (int)N[0]);
    } else {
        const double d0 = tc[kc-1] - tg[kc-1], d1 = tc[kc] - tg[kc];               /* d = CPU - GPU : negative then positive */
        const double nstar = N[kc-1] + (N[kc] - N[kc-1]) * (-d0) / (d1 - d0);
        printf("RESULT: break-even N* ~ %.0f objects (between N=%d and N=%d, linear interpolation).\n", nstar, (int)N[kc-1], (int)N[kc]);
        printf("        below N* the CPU is faster (GPU set-up/launch/transfer dominates), above it the GPU is faster.\n");
    }
    /* operational thresholds with the switching margin (hysteresis): rising N and falling N switch at different points */
    int n_up = -1, n_dn = -1;
    for (int i = 0; i < m; i++) if (tg[i] < (1.0 - a.margin) * tc[i]) { n_up = (int)N[i]; break; }
    for (int i = m - 1; i >= 0; i--) if (tc[i] < (1.0 - a.margin) * tg[i]) { n_dn = (int)N[i]; break; }
    printf("operational thresholds (margin %.2f): CPU->GPU when rising at N >= %d ; GPU->CPU when falling at N <= %d   (-1 = not reached)\n", a.margin, n_up, n_dn);
    /* linear cost-model fit that the adaptive controller starts from */
    double sxx = 0, sxy = 0; for (int i = 0; i < m; i++) { sxx += N[i] * N[i]; sxy += N[i] * tc[i]; }
    double mx = 0, my = 0; for (int i = 0; i < m; i++) { mx += N[i]; my += tg[i]; } mx /= m; my /= m;
    double vx = 0, cxy = 0; for (int i = 0; i < m; i++) { vx += (N[i] - mx) * (N[i] - mx); cxy += (N[i] - mx) * (tg[i] - my); }
    const double b = vx > 0 ? cxy / vx : 0, f = my - b * mx;
    printf("fitted cost model: T_cpu ~ %.3e ms * N ; T_gpu ~ %.4f ms + %.3e ms * N   (CSV written to %s)\n", sxy / sxx, f, b, csvname);
    tl.finish(a.timing_csv);
    return 0;
}
#endif
