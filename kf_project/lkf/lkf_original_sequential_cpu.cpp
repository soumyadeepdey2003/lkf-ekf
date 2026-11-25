
#include "../common/lkf_conventional.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

struct Options {
    std::vector<int> sizes{1, 10, 100, 1000};
    int frames = 1000;
    int warmup = 50;
    std::string csv = "lkf_original_sequential_cpu.csv";
};

static bool parseSizes(
    const std::string& text,
    std::vector<int>& sizes)
{
    sizes.clear();
    std::size_t start = 0;

    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);

        const std::string token = text.substr(
            start,
            comma == std::string::npos
                ? std::string::npos
                : comma - start
        );

        char* end = nullptr;
        const long value = std::strtol(token.c_str(), &end, 10);

        if (end == token.c_str() ||
            *end != '\0' ||
            value <= 0 ||
            value > 10000000)
            return false;

        sizes.push_back(static_cast<int>(value));

        if (comma == std::string::npos)
            break;

        start = comma + 1;
    }

    return !sizes.empty();
}

static bool parseArgs(int argc, char** argv, Options& o)
{
    for (int i = 1; i < argc; ++i) {

        const std::string arg = argv[i];

        if (arg == "--sizes" && i + 1 < argc) {
            if (!parseSizes(argv[++i], o.sizes))
                return false;
        }
        else if (arg == "--frames" && i + 1 < argc) {
            o.frames = std::atoi(argv[++i]);
        }
        else if (arg == "--warmup" && i + 1 < argc) {
            o.warmup = std::atoi(argv[++i]);
        }
        else if (arg == "--csv" && i + 1 < argc) {
            o.csv = argv[++i];
        }
        else if (arg == "--help") {
            std::cout
                << "--sizes 1,10,100 "
                << "--frames N "
                << "--warmup N "
                << "--csv file.csv|none\n";

            std::exit(0);
        }
        else {
            return false;
        }
    }

    return o.frames > 0 && o.warmup >= 0;
}

/* Filter configuration */
static KFParams makeParams()
{
    KFParams p{};

    p.dt = 0.1f;
    p.sa2 = 0.5f;

    p.R[0] = 4.0f;
    p.R[4] = 4.0f;
    p.R[8] = 4.0f;

    return p;
}

/*
 * Deterministic synthetic measurements.
 * Measurement generation is outside the timed filter region.
 */
static void makeMeasurement(
    int objectId,
    int frame,
    Meas& m)
{
    const float t = 0.1f * static_cast<float>(frame);
    const float id = static_cast<float>(objectId);

    m.valid = 1;

    m.z[0] =
        8.0f * id +
        12.0f * t +
        0.35f * std::sin(1.71f * frame + 0.37f * id);

    m.z[1] =
        3.0f * id +
        5.0f * t +
        0.35f * std::cos(1.13f * frame + 0.29f * id);

    m.z[2] =
        1.5f * id +
        2.0f * t +
        0.25f * std::sin(0.91f * frame + 0.41f * id);
}

static void fillMeasurements(
    int frame,
    std::vector<Meas>& measurements)
{
    for (std::size_t i = 0; i < measurements.size(); ++i) {
        makeMeasurement(
            static_cast<int>(i),
            frame,
            measurements[i]
        );
    }
}

/* Initialize one object's state and covariance. */
static void initializeTrack(
    Track& track,
    const Meas& measurement)
{
    for (int i = 0; i < 3; ++i) {
        track.x[i] = measurement.z[i];
        track.x[i + 3] = 0.0f;
    }

    for (int i = 0; i < NS * NS; ++i)
        track.P[i] = 0.0f;

    for (int i = 0; i < NS; ++i)
        track.P[i * NS + i] = KF_P0;
}

/* Prevent the compiler from making the final results irrelevant. */
static double calculateChecksum(
    const std::vector<Track>& tracks)
{
    double sum = 0.0;

    for (const Track& track : tracks) {

        for (int i = 0; i < NS; ++i)
            sum += track.x[i] * (1.0 + 0.001 * i);

        for (int i = 0; i < NS * NS; ++i)
            sum += track.P[i] * 0.0001;
    }

    return sum;
}

/*
 * Sequential execution:
 * Every object is updated by the same CPU thread.
 */
static void updateAllSequential(
    std::vector<Track>& tracks,
    const std::vector<Meas>& measurements,
    const KFParams& params)
{
    for (std::size_t i = 0; i < tracks.size(); ++i) {

        lkf_conv_step(
            &tracks[i],
            &measurements[i],
            &params
        );
    }
}

int main(int argc, char** argv)
{
    Options options;

    if (!parseArgs(argc, argv, options)) {

        std::cerr
            << "Usage: lkf_original_sequential_cpu "
            << "--sizes 1,10,100 "
            << "--frames N "
            << "--warmup N "
            << "--csv file.csv|none\n";

        return 2;
    }

    const KFParams params = makeParams();

    std::cout
        << "Original LKF | Sequential CPU\n";

    std::cout
        << "Objects, Mean ms/frame, Median ms/frame, "
        << "Updates/sec, Checksum\n";

    std::ofstream csv;

    if (options.csv != "none") {

        const bool newFile =
            !std::ifstream(options.csv).good();

        csv.open(options.csv, std::ios::app);

        if (!csv) {
            std::cerr << "Cannot open CSV output.\n";
            return 1;
        }

        if (newFile) {
            csv
                << "implementation,objects,frames,"
                << "mean_ms_per_frame,median_ms_per_frame,"
                << "updates_per_second,checksum\n";
        }
    }

    for (int count : options.sizes) {

        std::vector<Track> tracks(count);
        std::vector<Meas> measurements(count);

        fillMeasurements(0, measurements);

        for (int i = 0; i < count; ++i) {
            initializeTrack(
                tracks[i],
                measurements[i]
            );
        }

        /* Warm-up: excluded from timing. */
        for (int f = 1; f <= options.warmup; ++f) {

            fillMeasurements(f, measurements);

            updateAllSequential(
                tracks,
                measurements,
                params
            );
        }

        std::vector<double> times;
        times.reserve(options.frames);

        /* Actual benchmark. */
        for (int f = 1; f <= options.frames; ++f) {

            fillMeasurements(f, measurements);

            const auto begin =
                std::chrono::steady_clock::now();

            updateAllSequential(
                tracks,
                measurements,
                params
            );

            const auto end =
                std::chrono::steady_clock::now();

            const double elapsed =
                std::chrono::duration<double, std::milli>(
                    end - begin
                ).count();

            times.push_back(elapsed);
        }

        const double mean =
            std::accumulate(
                times.begin(),
                times.end(),
                0.0
            ) / times.size();

        std::sort(times.begin(), times.end());

        const double median =
            times[times.size() / 2];

        const double throughput =
            mean > 0.0
                ? count * 1000.0 / mean
                : 0.0;

        const double checksum =
            calculateChecksum(tracks);

        std::cout
            << count << ", "
            << std::fixed << std::setprecision(6)
            << mean << ", "
            << median << ", "
            << std::setprecision(2)
            << throughput << ", "
            << std::setprecision(9)
            << checksum << '\n';

        if (csv) {

            csv
                << "original_lkf_sequential_cpu,"
                << count << ','
                << options.frames << ','
                << std::setprecision(12)
                << mean << ','
                << median << ','
                << throughput << ','
                << checksum << '\n';
        }
    }

    return 0;
}
