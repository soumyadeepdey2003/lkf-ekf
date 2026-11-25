/* kf_runtime.h - shared plumbing used by all 16 programs (timing, command line, one "scenario", accuracy, CPU runner).
 * The 16 program files only say WHICH filter and WHICH role; the repeated work lives here once, so it is tested once. */
#ifndef KF_RUNTIME_H
#define KF_RUNTIME_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef KF_UNIFIED_MEMORY            /* Jetson: CPU and GPU share DRAM -> allocate "managed" memory, no copies */
#include "kf_cuda.h"
#endif
#include "kf_filters.h"
#include "../detection/detection.h"
#include "../timing/kf_timing.h"

static inline double kf_now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
static inline void *kf_alloc(size_t b) {
#ifdef KF_UNIFIED_MEMORY
    void *p = 0; cudaMallocManaged(&p, b); return p;
#else
    return malloc(b);
#endif
}
static inline void kf_release(void *p) {
#ifdef KF_UNIFIED_MEMORY
    cudaFree(p);
#else
    free(p);
#endif
}
static inline double kf_median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

/* ---------------- command line ---------------- */
struct KfArgs {
    std::vector<int> sizes;      /* object counts to test                       (--sizes 1,10,100)         */
    int frames = 50;             /* frames per size                             (--frames 50)              */
    unsigned seed = 1;           /*                                             (--seed 1)                 */
    double margin = 0.10;        /* switching margin delta of the controller    (--margin 0.1)             */
    int maxn = 131072;           /* largest N of the crossover sweep            (--maxn 131072)            */
    std::string stream;          /* real detector output file                   (--stream f.bin --objects K)*/
    int objects = 0;
    std::string timing_csv = "timing_results.csv";   /* where the timing report appends its rows (--csv none = do not write) */
    int mapping = -1;            /* GPU mapping: -1 auto (warp for upgraded filters), 0 thread-per-track, 1 warp-per-track (--mapping thread|warp) */
};
static inline KfArgs kf_parse(int argc, char **argv, const std::vector<int> &default_sizes) {
    KfArgs a; a.sizes = default_sizes;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i]; const char *v = (i + 1 < argc) ? argv[i + 1] : "";
        if (k == "--frames") { a.frames = atoi(v); i++; }
        else if (k == "--seed") { a.seed = (unsigned)atoi(v); i++; }
        else if (k == "--margin") { a.margin = atof(v); i++; }
        else if (k == "--maxn") { a.maxn = atoi(v); i++; }
        else if (k == "--stream") { a.stream = v; i++; }
        else if (k == "--objects") { a.objects = atoi(v); i++; }
        else if (k == "--csv") { a.timing_csv = v; i++; }
        else if (k == "--mapping") { std::string m = v; a.mapping = (m == "thread") ? 0 : (m == "warp") ? 1 : -1; i++; }
        else if (k == "--sizes") { a.sizes.clear(); char *s = strdup(v); for (char *t = strtok(s, ","); t; t = strtok(0, ",")) a.sizes.push_back(atoi(t)); free(s); i++; }
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); exit(2); }
    }
    if (!a.stream.empty()) { if (a.objects <= 0) { fprintf(stderr, "--stream needs --objects K\n"); exit(2); } a.sizes.assign(1, a.objects); }
    if (a.frames < 2) a.frames = 2;
    return a;
}

/* ---------------- one scenario = tracks + measurements + the detection source ---------------- */
struct Scenario { int n = 0; Track *T = 0; Meas *M = 0; DetSource src; KFParams p; };

template <class F> static bool scenario_open(Scenario *s, int n, const KfArgs &a) {
    s->n = n;
    if (!a.stream.empty()) { if (!det_open_stream(&s->src, a.stream.c_str(), F::kind, n)) return false; }
    else if (!det_open_synthetic(&s->src, n, F::kind, a.seed)) return false;
    det_fill_params(&s->src, &s->p);
    s->T = (Track *)kf_alloc(sizeof(Track) * (size_t)n); s->M = (Meas *)kf_alloc(sizeof(Meas) * (size_t)n);
    if (!s->T || !s->M) return false;
    if (!det_next(&s->src, s->M, 1)) return false;                    /* first detections create the tracks */
    for (int i = 0; i < n; i++) F::init(&s->T[i], &s->M[i]);
    return true;
}
static inline void scenario_close(Scenario *s) { kf_release(s->T); kf_release(s->M); det_close(&s->src); s->T = 0; s->M = 0; }

/* ---------------- accuracy bookkeeping ---------------- */
struct Acc { double sp = 0, sv = 0; long cnt = 0, bad = 0; bool truth = false; };
static inline void acc_add(Acc *c, const Scenario &s) {
    if (!s.src.has_truth) return;
    c->truth = true; det_accumulate_error(&s.src, s.T, &c->sp, &c->sv, &c->bad); c->cnt += s.n;
}
static inline void acc_text(const Acc &c, char *pos, char *vel, size_t len) {
    if (!c.truth || c.cnt - c.bad <= 0) { snprintf(pos, len, "n/a"); snprintf(vel, len, "n/a"); }
    else { snprintf(pos, len, "%.4f", sqrt(c.sp / (double)(c.cnt - c.bad))); snprintf(vel, len, "%.4f", sqrt(c.sv / (double)(c.cnt - c.bad))); }
}

/* ---------------- CPU backend: one OpenMP thread group over the tracks ---------------- */
template <class F> static void cpu_step(Track *T, const Meas *M, int n, const KFParams *p) {
#pragma omp parallel for schedule(static) if (n >= 256)
    for (int i = 0; i < n; i++) F::step(&T[i], &M[i], p);
}
static inline int cpu_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* ---------------- PROGRAM ROLE 1: CPU run (called by the 4 *_cpu.cpp files) ---------------- */
template <class F> static int kf_main_cpu(int argc, char **argv) {
    KfArgs a = kf_parse(argc, argv, { 1, 10, 100, 1000, 10000, 100000 });
    printf("== %s | CPU-only | %d OpenMP thread(s) | %d frames per size | timing = whole CPU routine ==\n", F::name(), cpu_threads(), a.frames);
    printf("%8s %12s %14s %10s %10s %6s\n", "N", "ms/frame", "updates/s", "RMSE_pos", "RMSE_vel", "bad");
    KfTimingLog tl(F::tag(), F::name(), "cpu", "cpu_omp");                /* file 18: collects every frame time */
    for (int n : a.sizes) {
        Scenario s; if (!scenario_open<F>(&s, n, a)) { fprintf(stderr, "cannot open scenario N=%d\n", n); return 1; }
        double total = 0; Acc acc; std::vector<double> samples;
        for (int f = 0; f < a.frames; f++) {
            if (!det_next(&s.src, s.M, 0)) break;                       /* detection (not timed) */
            const double t0 = kf_now_ms(); cpu_step<F>(s.T, s.M, n, &s.p); const double dt_ms = kf_now_ms() - t0;
            total += dt_ms; samples.push_back(dt_ms);
            if (f >= a.frames / 2) acc_add(&acc, s);
        }
        tl.record("cpu", n, samples);
        char ep[32], ev[32]; acc_text(acc, ep, ev, 32);
        const double ms = total / a.frames;
        printf("%8d %12.5f %14.0f %10s %10s %6ld\n", n, ms, ms > 0 ? n / ms * 1e3 : 0.0, ep, ev, acc.bad);
        scenario_close(&s);
    }
    tl.finish(a.timing_csv);                                            /* file 18 prints the report and appends the CSV */
    return 0;
}
#endif
