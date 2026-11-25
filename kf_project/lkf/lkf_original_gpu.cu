
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <new>

#include "../common/lkf_conventional.h"

// ============================================================
// GPU DEVICE FUNCTION: Inverse of a 3 x 3 matrix
// ============================================================

__device__ bool gpu_inverse3(
    const double A[9],
    double invA[9])
{
    const double a = A[0], b = A[1], c = A[2];
    const double d = A[3], e = A[4], f = A[5];
    const double g = A[6], h = A[7], i = A[8];

    const double det =
        a * (e * i - f * h)
      - b * (d * i - f * g)
      + c * (d * h - e * g);

    if (!isfinite(det) || fabs(det) < 1e-12)
        return false;

    const double s = 1.0 / det;

    invA[0] =  (e * i - f * h) * s;
    invA[1] =  (c * h - b * i) * s;
    invA[2] =  (b * f - c * e) * s;

    invA[3] =  (f * g - d * i) * s;
    invA[4] =  (a * i - c * g) * s;
    invA[5] =  (c * d - a * f) * s;

    invA[6] =  (d * h - e * g) * s;
    invA[7] =  (b * g - a * h) * s;
    invA[8] =  (a * e - b * d) * s;

    return true;
}


// ============================================================
// GPU KERNEL: Conventional Linear Kalman Filter
//
// One CUDA thread processes one independent tracking object.
// ============================================================

__global__ void lkf_original_gpu_kernel(
    Track* tracks,
    const Meas* measurements,
    KFParams params,
    int number_of_tracks,
    int* status)
{
    const int id =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= number_of_tracks)
        return;

    Track track = tracks[id];
    const Meas measurement = measurements[id];

    const double dt = params.dt;
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;

    // ========================================================
    // STEP 1: STATE PREDICTION
    // xp = F * x
    // ========================================================

    double xp[NS];

    for (int axis = 0; axis < 3; ++axis)
    {
        xp[axis] =
            track.x[axis] +
            dt * track.x[axis + 3];

        xp[axis + 3] =
            track.x[axis + 3];
    }


    // ========================================================
    // STEP 2: COVARIANCE PREDICTION
    // Pp = F * P * F^T + Q
    // ========================================================

    double FP[NS * NS] = {};
    double Pp[NS * NS] = {};

    // FP = F * P

    for (int r = 0; r < NS; ++r)
    {
        for (int c = 0; c < NS; ++c)
        {
            FP[r * NS + c] =
                track.P[r * NS + c];

            if (r < 3)
            {
                FP[r * NS + c] +=
                    dt * track.P[(r + 3) * NS + c];
            }
        }
    }

    // Pp = FP * F^T

    for (int r = 0; r < NS; ++r)
    {
        for (int c = 0; c < NS; ++c)
        {
            Pp[r * NS + c] =
                FP[r * NS + c];

            if (c < 3)
            {
                Pp[r * NS + c] +=
                    dt * FP[r * NS + c + 3];
            }
        }
    }

    // White acceleration process noise

    const double q00 =
        params.sa2 * dt4 / 4.0;

    const double q01 =
        params.sa2 * dt3 / 2.0;

    const double q11 =
        params.sa2 * dt2;

    for (int axis = 0; axis < 3; ++axis)
    {
        const int p = axis;
        const int v = axis + 3;

        Pp[p * NS + p] += q00;
        Pp[p * NS + v] += q01;
        Pp[v * NS + p] += q01;
        Pp[v * NS + v] += q11;
    }


    // ========================================================
    // STEP 3: HANDLE INVALID MEASUREMENT
    // ========================================================

    if (!measurement.valid)
    {
        for (int i = 0; i < NS; ++i)
            track.x[i] = xp[i];

        for (int i = 0; i < NS * NS; ++i)
            track.P[i] = Pp[i];

        tracks[id] = track;
        status[id] = 1;

        return;
    }


    // ========================================================
    // STEP 4: MEASUREMENT INNOVATION
    // innovation = z - H * xp
    // ========================================================

    double innovation[NZ];

    for (int i = 0; i < NZ; ++i)
    {
        innovation[i] =
            measurement.z[i] - xp[i];
    }


    // ========================================================
    // STEP 5: INNOVATION COVARIANCE
    // S = H * Pp * H^T + R
    // ========================================================

    double S[9];

    for (int r = 0; r < NZ; ++r)
    {
        for (int c = 0; c < NZ; ++c)
        {
            S[r * NZ + c] =
                Pp[r * NS + c] +
                params.R[r * NZ + c];
        }
    }


    // ========================================================
    // STEP 6: INVERT INNOVATION COVARIANCE
    // ========================================================

    double Sinv[9];

    if (!gpu_inverse3(S, Sinv))
    {
        status[id] = 0;
        return;
    }


    // ========================================================
    // STEP 7: KALMAN GAIN
    // K = Pp * H^T * inverse(S)
    // ========================================================

    double K[NS * NZ] = {};

    for (int r = 0; r < NS; ++r)
    {
        for (int c = 0; c < NZ; ++c)
        {
            for (int j = 0; j < NZ; ++j)
            {
                K[r * NZ + c] +=
                    Pp[r * NS + j] *
                    Sinv[j * NZ + c];
            }
        }
    }


    // ========================================================
    // STEP 8: STATE CORRECTION
    // x = xp + K * innovation
    // ========================================================

    double corrected_x[NS];

    for (int r = 0; r < NS; ++r)
    {
        corrected_x[r] = xp[r];

        for (int c = 0; c < NZ; ++c)
        {
            corrected_x[r] +=
                K[r * NZ + c] *
                innovation[c];
        }
    }


    // ========================================================
    // STEP 9: JOSEPH-FORM COVARIANCE CORRECTION
    // P = (I-KH)Pp(I-KH)^T + KRK^T
    // ========================================================

    double A[NS * NS] = {};
    double AP[NS * NS] = {};
    double newP[NS * NS] = {};

    // A = I - K*H

    for (int r = 0; r < NS; ++r)
    {
        for (int c = 0; c < NS; ++c)
        {
            A[r * NS + c] =
                (r == c) ? 1.0 : 0.0;

            if (c < NZ)
            {
                A[r * NS + c] -=
                    K[r * NZ + c];
            }
        }
    }

    // AP = A * Pp

    for (int r = 0; r < NS; ++r)
    {
        for (int c = 0; c < NS; ++c)
        {
            for (int j = 0; j < NS; ++j)
            {
                AP[r * NS + c] +=
                    A[r * NS + j] *
                    Pp[j * NS + c];
            }
        }
    }

    // First term: AP * A^T

    for (int r = 0; r < NS; ++r)
    {
        for (int c = 0; c < NS; ++c)
        {
            for (int j = 0; j < NS; ++j)
            {
                newP[r * NS + c] +=
                    AP[r * NS + j] *
                    A[c * NS + j];
            }
        }
    }

    // Second term: K * R

    double KR[NS * NZ] = {};

    for (int r = 0; r < NS; ++r)
    {
        for (int c = 0; c < NZ; ++c)
        {
            for (int j = 0; j < NZ; ++j)
            {
                KR[r * NZ + c] +=
                    K[r * NZ + j] *
                    params.R[j * NZ + c];
            }
        }
    }

    // Add KR * K^T

    for (int r = 0; r < NS; ++r)
    {
        for (int c = 0; c < NS; ++c)
        {
            for (int j = 0; j < NZ; ++j)
            {
                newP[r * NS + c] +=
                    KR[r * NZ + j] *
                    K[c * NZ + j];
            }
        }
    }


    // ========================================================
    // STEP 10: REMOVE NUMERICAL ASYMMETRY
    // ========================================================

    for (int r = 0; r < NS; ++r)
    {
        for (int c = r + 1; c < NS; ++c)
        {
            const double value =
                0.5 * (
                    newP[r * NS + c] +
                    newP[c * NS + r]
                );

            newP[r * NS + c] = value;
            newP[c * NS + r] = value;
        }
    }


    // ========================================================
    // STEP 11: STORE CORRECTED STATE AND COVARIANCE
    // ========================================================

    for (int i = 0; i < NS; ++i)
        track.x[i] = corrected_x[i];

    for (int i = 0; i < NS * NS; ++i)
        track.P[i] = newP[i];

    tracks[id] = track;
    status[id] = 1;
}


// ============================================================
// HOST FUNCTION: GPU-ONLY EXECUTION
//
// Uses a cleanup lambda instead of goto.
// ============================================================

bool lkf_original_gpu_step(
    Track* tracks,
    const Meas* measurements,
    int number_of_tracks,
    const KFParams& params)
{
    if (!tracks || !measurements || number_of_tracks <= 0)
    {
        std::printf("LKF GPU Error: Invalid input arguments.\n");
        return false;
    }

    Track* d_tracks = nullptr;
    Meas* d_measurements = nullptr;
    int* d_status = nullptr;
    int* h_status = nullptr;

    // --------------------------------------------------------
    // MEMORY SIZES
    // --------------------------------------------------------

    const size_t track_bytes =
        sizeof(Track) * static_cast<size_t>(number_of_tracks);

    const size_t measurement_bytes =
        sizeof(Meas) * static_cast<size_t>(number_of_tracks);

    const size_t status_bytes =
        sizeof(int) * static_cast<size_t>(number_of_tracks);

    // --------------------------------------------------------
    // RESOURCE CLEANUP
    // --------------------------------------------------------

    auto cleanup = [&]()
    {
        if (h_status)
        {
            delete[] h_status;
            h_status = nullptr;
        }

        if (d_tracks)
        {
            cudaFree(d_tracks);
            d_tracks = nullptr;
        }

        if (d_measurements)
        {
            cudaFree(d_measurements);
            d_measurements = nullptr;
        }

        if (d_status)
        {
            cudaFree(d_status);
            d_status = nullptr;
        }
    };

    // --------------------------------------------------------
    // STEP 1: ALLOCATE GPU MEMORY
    // --------------------------------------------------------

    cudaError_t error;

    error = cudaMalloc(
        reinterpret_cast<void**>(&d_tracks),
        track_bytes);

    if (error != cudaSuccess)
    {
        std::printf(
            "CUDA Error allocating tracks: %s\n",
            cudaGetErrorString(error));

        cleanup();
        return false;
    }

    error = cudaMalloc(
        reinterpret_cast<void**>(&d_measurements),
        measurement_bytes);

    if (error != cudaSuccess)
    {
        std::printf(
            "CUDA Error allocating measurements: %s\n",
            cudaGetErrorString(error));

        cleanup();
        return false;
    }

    error = cudaMalloc(
        reinterpret_cast<void**>(&d_status),
        status_bytes);

    if (error != cudaSuccess)
    {
        std::printf(
            "CUDA Error allocating status: %s\n",
            cudaGetErrorString(error));

        cleanup();
        return false;
    }

    // --------------------------------------------------------
    // STEP 2: ALLOCATE CPU STATUS MEMORY
    // --------------------------------------------------------

    h_status = new (std::nothrow) int[number_of_tracks];

    if (!h_status)
    {
        std::printf(
            "CPU Error: Unable to allocate status memory.\n");

        cleanup();
        return false;
    }

    // --------------------------------------------------------
    // STEP 3: CPU TO GPU DATA TRANSFER
    // --------------------------------------------------------

    error = cudaMemcpy(
        d_tracks,
        tracks,
        track_bytes,
        cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        std::printf(
            "CUDA Error copying tracks: %s\n",
            cudaGetErrorString(error));

        cleanup();
        return false;
    }

    error = cudaMemcpy(
        d_measurements,
        measurements,
        measurement_bytes,
        cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        std::printf(
            "CUDA Error copying measurements: %s\n",
            cudaGetErrorString(error));

        cleanup();
        return false;
    }

    // --------------------------------------------------------
    // STEP 4: CONFIGURE CUDA EXECUTION
    // --------------------------------------------------------

    const int threads_per_block = 128;

    const int blocks =
        (number_of_tracks + threads_per_block - 1)
        / threads_per_block;

    // Clear any previous CUDA runtime error.
    cudaGetLastError();

    // --------------------------------------------------------
    // STEP 5: LAUNCH GPU KERNEL
    // --------------------------------------------------------

    lkf_original_gpu_kernel<<<blocks, threads_per_block>>>(
        d_tracks,
        d_measurements,
        params,
        number_of_tracks,
        d_status);

    // Check immediate kernel launch errors.

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        std::printf(
            "CUDA Kernel Launch Error: %s\n",
            cudaGetErrorString(error));

        cleanup();
        return false;
    }

    // --------------------------------------------------------
    // STEP 6: WAIT FOR GPU EXECUTION
    // --------------------------------------------------------

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        std::printf(
            "CUDA Kernel Execution Error: %s\n",
            cudaGetErrorString(error));

        cleanup();
        return false;
    }

    // --------------------------------------------------------
    // STEP 7: GPU TO CPU DATA TRANSFER
    // --------------------------------------------------------

    error = cudaMemcpy(
        tracks,
        d_tracks,
        track_bytes,
        cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        std::printf(
            "CUDA Error copying corrected tracks: %s\n",
            cudaGetErrorString(error));

        cleanup();
        return false;
    }

    error = cudaMemcpy(
        h_status,
        d_status,
        status_bytes,
        cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        std::printf(
            "CUDA Error copying status: %s\n",
            cudaGetErrorString(error));

        cleanup();
        return false;
    }

    // --------------------------------------------------------
    // STEP 8: VERIFY ALL TRACKING OBJECTS
    // --------------------------------------------------------

    for (int i = 0; i < number_of_tracks; ++i)
    {
        if (h_status[i] != 1)
        {
            std::printf(
                "LKF GPU Error: Track %d failed.\n",
                i);

            cleanup();
            return false;
        }
    }

    // --------------------------------------------------------
    // STEP 9: RELEASE RESOURCES
    // --------------------------------------------------------

    cleanup();

    return true;
}
