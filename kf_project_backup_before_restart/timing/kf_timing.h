/* kf_timing.h - FILE 18: the timing reporter. Every one of the 16 programs calls it automatically at the end of its run.
 *
 * WHAT IT DOES
 *   1. Collects the timing samples (one number per frame, or per repeat) that the program measured, per series and per N.
 *   2. Prints a standard TIMING REPORT: mean, median, min, max, 95th percentile, standard deviation, updates per second,
 *      and the total wall-clock time of the whole program (so you see how much time went into the filter and how much into set-up).
 *   3. Appends one row per (series, N) to timing_results.csv, so after you have run several programs ONE file contains the
 *      timing of all of them.
 *   4. The same file also builds a small report tool:   g++ -x c++ -DKF_TIMING_REPORT_MAIN timing/kf_timing.h -o kf_timing_report
 *      It reads timing_results.csv and prints, for every original/upgraded pair, the algorithmic speed-up
 *      S_algorithm = T_original / T_upgraded and the latency reduction  dT(%) = 100 (1 - T_upgraded / T_original)   (paper Sec. 4.9).
 * Only the C++ standard library is used, so it works on a PC, on the Jetson and in the tests. */
#ifndef KF_TIMING_H
#define KF_TIMING_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

static inline double kft_now_ms() {                       /* a clock that never goes backwards, in milliseconds */
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct KfStats { int n = 0; double mean = 0, median = 0, minv = 0, maxv = 0, p95 = 0, sd = 0; };
static inline KfStats kft_stats(std::vector<double> v) {
    KfStats s; s.n = (int)v.size(); if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0; for (double x : v) sum += x;
    s.mean = sum / v.size(); s.minv = v.front(); s.maxv = v.back(); s.median = v[v.size() / 2];
    s.p95 = v[std::min(v.size() - 1, (size_t)(0.95 * (v.size() - 1) + 0.5))];
    double q = 0; for (double x : v) q += (x - s.mean) * (x - s.mean);
    s.sd = v.size() > 1 ? sqrt(q / (v.size() - 1)) : 0.0;
    return s;
}

struct KfTimingRow { std::string series; int n; KfStats st; };
/* objects updated per second from the median time; meaningless for the controller-overhead series, so it shows 0 there */
static inline double kft_rate(const KfTimingRow &r) { return (r.st.median > 0 && r.series != "adaptive_ctrl") ? r.n / r.st.median * 1e3 : 0.0; }

struct KfTimingLog {
    std::string tag, name, role, mapping;                  /* which program this is */
    double t_start;                                        /* program start: for the total wall time */
    double timed_ms = 0;                                   /* time spent inside the timed calls */
    std::vector<KfTimingRow> rows;

    KfTimingLog(const char *tag_, const char *name_, const char *role_, const char *mapping_)
        : tag(tag_), name(name_), role(role_), mapping(mapping_), t_start(kft_now_ms()) {}

    /* record the samples (ms) of one series at one object count. counts_in_total = false for a SUB-part of another series
     * (for example the kernel time is already inside the end-to-end time, so it must not be added twice). */
    void record(const char *series, int n, const std::vector<double> &samples_ms, bool counts_in_total = true) {
        KfTimingRow r; r.series = series; r.n = n; r.st = kft_stats(samples_ms); rows.push_back(r);
        if (counts_in_total) for (double x : samples_ms) timed_ms += x;
    }
    void print() const {
        printf("\n---- TIMING REPORT: %s  (%s | role: %s | mapping: %s) ----\n", tag.c_str(), name.c_str(), role.c_str(), mapping.c_str());
        printf("%-16s %8s %8s %11s %11s %11s %11s %11s %10s %14s\n", "series", "N", "samples", "mean ms", "median ms", "min ms", "max ms", "p95 ms", "sd ms", "updates/s");
        for (const KfTimingRow &r : rows)
            printf("%-16s %8d %8d %11.5f %11.5f %11.5f %11.5f %11.5f %10.5f %14.0f\n", r.series.c_str(), r.n, r.st.n, r.st.mean, r.st.median,
                   r.st.minv, r.st.maxv, r.st.p95, r.st.sd, kft_rate(r));
        const double wall = kft_now_ms() - t_start;
        printf("whole program: %.3f s wall-clock | %.3f s (%.1f%%) inside the timed calls | the rest is scenario creation, warm-up, calibration, printing\n",
               wall / 1e3, timed_ms / 1e3, wall > 0 ? 100.0 * timed_ms / wall : 0.0);
    }
    /* append to the shared CSV (a header is written when the file is new). csv_path "none" disables it. */
    void save(const std::string &csv_path) const {
        if (csv_path.empty() || csv_path == "none") return;
        FILE *chk = fopen(csv_path.c_str(), "r"); const bool fresh = (chk == 0); if (chk) fclose(chk);
        FILE *f = fopen(csv_path.c_str(), "a"); if (!f) { fprintf(stderr, "cannot write %s\n", csv_path.c_str()); return; }
        if (fresh) fprintf(f, "timestamp_utc,program,filter,role,mapping,series,N,samples,mean_ms,median_ms,min_ms,max_ms,p95_ms,sd_ms,updates_per_s,program_wall_s\n");
        char ts[32]; time_t t = time(0); strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
        const double wall = (kft_now_ms() - t_start) / 1e3;
        std::string flt = name; for (char &c : flt) if (c == ' ' || c == ',') c = '_';
        for (const KfTimingRow &r : rows)
            fprintf(f, "%s,%s,%s,%s,%s,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.0f,%.3f\n", ts, tag.c_str(), flt.c_str(), role.c_str(), mapping.c_str(),
                    r.series.c_str(), r.n, r.st.n, r.st.mean, r.st.median, r.st.minv, r.st.maxv, r.st.p95, r.st.sd,
                    kft_rate(r), wall);
        fclose(f);
        printf("timing rows appended to %s\n", csv_path.c_str());
    }
    /* THE ONE CALL every program makes at the end */
    void finish(const std::string &csv_path) const { print(); save(csv_path); }
};

/* ================= report tool: original versus upgraded from the CSV =================
 * build:  g++ -x c++ -DKF_TIMING_REPORT_MAIN timing/kf_timing.h -o kf_timing_report      run: ./kf_timing_report [timing_results.csv] */
#ifdef KF_TIMING_REPORT_MAIN
struct KftRow { std::string program, mapping, role, series; int n; double median; };
static std::vector<std::string> kft_split(const std::string &s) {
    std::vector<std::string> out; std::string cur;
    for (char c : s) { if (c == ',') { out.push_back(cur); cur.clear(); } else if (c != '\r' && c != '\n') cur += c; }
    out.push_back(cur); return out;
}
int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "timing_results.csv";
    FILE *f = fopen(path, "r"); if (!f) { fprintf(stderr, "cannot open %s (run some programs first)\n", path); return 1; }
    std::map<std::string, KftRow> last;                    /* the newest row wins for each (program, mapping, series, N) */
    char line[2048]; bool header = true;
    while (fgets(line, sizeof line, f)) {
        if (header) { header = false; continue; }
        std::vector<std::string> c = kft_split(line); if (c.size() < 16) continue;
        KftRow r; r.program = c[1]; r.role = c[3]; r.mapping = c[4]; r.series = c[5]; r.n = atoi(c[6].c_str()); r.median = atof(c[9].c_str());
        last[r.program + "|" + r.mapping + "|" + r.series + "|" + c[6]] = r;
    }
    fclose(f);
    printf("%-5s %-10s %-10s %8s %14s %14s %-8s %12s %10s\n", "filt", "role", "series", "N", "original ms", "upgraded ms", "mapping", "S_algorithm", "dT %");
    int printed = 0;
    for (auto &kv : last) {
        const KftRow &nw = kv.second; if (nw.program.size() < 4 || nw.program.compare(nw.program.size() - 4, 4, "_new") != 0) continue;
        const std::string orig = nw.program.substr(0, nw.program.size() - 4) + "_orig";
        const KftRow *o = 0;
        for (auto &kv2 : last) if (kv2.second.program == orig && kv2.second.role == nw.role && kv2.second.series == nw.series && kv2.second.n == nw.n) { o = &kv2.second; break; }
        if (!o || nw.median <= 0 || o->median <= 0) continue;
        printf("%-5s %-10s %-10s %8d %14.5f %14.5f %-8s %12.3f %10.1f\n", nw.program.substr(0, 3).c_str(), nw.role.c_str(), nw.series.c_str(), nw.n,
               o->median, nw.median, nw.mapping.c_str(), o->median / nw.median, 100.0 * (1.0 - nw.median / o->median));
        printed++;
    }
    if (!printed) printf("(no original/upgraded pair found yet: run the *_orig and the *_new program of the same role with the same sizes)\n");
    printf("S_algorithm > 1 means the upgraded version is faster; dT %% is the latency reduction (paper Sec. 4.9).\n");
    return 0;
}
#endif
#endif
