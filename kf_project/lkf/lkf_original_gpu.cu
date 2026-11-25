
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <new>

#include "../common/lkf_conventional.h"


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

   

    double xp[NS];

    for (int axis = 0; axis < 3; ++axis)
    {
        xp[axis] =
            track.x[axis] +
            dt * track.x[axis + 3];

        xp[axis + 3] =
            track.x[axis + 3];
    }


    

    double FP[NS * NS] = {};
    double Pp[NS * NS] = {};


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


   

    double innovation[NZ];

    for (int i = 0; i < NZ; ++i)
    {
        innovation[i] =
            measurement.z[i] - xp[i];
    }


  
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


   

    double Sinv[9];

    if (!gpu_inverse3(S, Sinv))
    {
        status[id] = 0;
        return;
    }


  
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


  

    double A[NS * NS] = {};
    double AP[NS * NS] = {};
    double newP[NS * NS] = {};


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


   

    for (int i = 0; i < NS; ++i)
        track.x[i] = corrected_x[i];

    for (int i = 0; i < NS * NS; ++i)
        track.P[i] = newP[i];

    tracks[id] = track;
    status[id] = 1;
}




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

   

    const size_t track_bytes =
        sizeof(Track) * static_cast<size_t>(number_of_tracks);

    const size_t measurement_bytes =
        sizeof(Meas) * static_cast<size_t>(number_of_tracks);

    const size_t status_bytes =
        sizeof(int) * static_cast<size_t>(number_of_tracks);

   

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

   

    h_status = new (std::nothrow) int[number_of_tracks];

    if (!h_status)
    {
        std::printf(
            "CPU Error: Unable to allocate status memory.\n");

        cleanup();
        return false;
    }

   

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

    

    const int threads_per_block = 128;

    const int blocks =
        (number_of_tracks + threads_per_block - 1)
        / threads_per_block;

    cudaGetLastError();

   
    lkf_original_gpu_kernel<<<blocks, threads_per_block>>>(
        d_tracks,
        d_measurements,
        params,
        number_of_tracks,
        d_status);


    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        std::printf(
            "CUDA Kernel Launch Error: %s\n",
            cudaGetErrorString(error));

        cleanup();
        return false;
    }

   

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        std::printf(
            "CUDA Kernel Execution Error: %s\n",
            cudaGetErrorString(error));

        cleanup();
        return false;
    }

   

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

  

    cleanup();

    return true;
}
