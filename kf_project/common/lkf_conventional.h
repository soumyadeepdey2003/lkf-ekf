
#ifndef LKF_CONVENTIONAL_H
#define LKF_CONVENTIONAL_H

#include <cmath>
#include <algorithm>

static constexpr int NS = 6;
static constexpr int NZ = 3;
static constexpr double KF_P0 = 1.0;

// ------------------------------------------------------------
// Track state and covariance
// ------------------------------------------------------------

struct Track
{
    double x[NS] = {};
    double P[NS * NS] = {};
};

// ------------------------------------------------------------
// Position measurement
// ------------------------------------------------------------

struct Meas
{
    bool valid = true;
    double z[NZ] = {};
};

// ------------------------------------------------------------
// Conventional LKF parameters
// ------------------------------------------------------------

struct KFParams
{
    double dt = 0.1;
    double sa2 = 1.0;

    // Measurement noise covariance (3 x 3)
    double R[9] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
};

// ------------------------------------------------------------
// Invert a 3 x 3 matrix using the adjugate method
// ------------------------------------------------------------

inline bool lkf_inverse3(const double A[9], double invA[9])
{
    const double a = A[0], b = A[1], c = A[2];
    const double d = A[3], e = A[4], f = A[5];
    const double g = A[6], h = A[7], i = A[8];

    const double det =
        a * (e * i - f * h)
      - b * (d * i - f * g)
      + c * (d * h - e * g);

    if (!std::isfinite(det) || std::abs(det) < 1e-12)
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

// ------------------------------------------------------------
// Conventional Linear Kalman Filter: one complete time step
// ------------------------------------------------------------

inline bool lkf_conv_step(
    Track* track,
    const Meas* measurement,
    const KFParams* params)
{
    if (!track || !measurement || !params)
        return false;

    const double dt = params->dt;
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;

    // ========================================================
    // STEP 1: State prediction
    //
    // x(k|k-1) = F * x(k-1|k-1)
    // ========================================================

    double xp[NS];

    for (int axis = 0; axis < 3; ++axis)
    {
        xp[axis] =
            track->x[axis] + dt * track->x[axis + 3];

        xp[axis + 3] = track->x[axis + 3];
    }

    // ========================================================
    // STEP 2: Covariance prediction
    //
    // P(k|k-1) = F * P * F^T + Q
    // ========================================================

    double FP[NS * NS] = {};
    double Pp[NS * NS] = {};

    // FP = F * P
    for (int r = 0; r < NS; ++r)
    {
        for (int c = 0; c < NS; ++c)
        {
            FP[r * NS + c] = track->P[r * NS + c];

            if (r < 3)
                FP[r * NS + c] +=
                    dt * track->P[(r + 3) * NS + c];
        }
    }

    // Pp = FP * F^T
    for (int r = 0; r < NS; ++r)
    {
        for (int c = 0; c < NS; ++c)
        {
            Pp[r * NS + c] = FP[r * NS + c];

            if (c < 3)
                Pp[r * NS + c] +=
                    dt * FP[r * NS + c + 3];
        }
    }

    // White-acceleration process noise covariance
    const double q00 = params->sa2 * dt4 / 4.0;
    const double q01 = params->sa2 * dt3 / 2.0;
    const double q11 = params->sa2 * dt2;

    for (int axis = 0; axis < 3; ++axis)
    {
        const int p = axis;
        const int v = axis + 3;

        Pp[p * NS + p] += q00;
        Pp[p * NS + v] += q01;
        Pp[v * NS + p] += q01;
        Pp[v * NS + v] += q11;
    }

    // If there is no valid measurement, retain prediction.
    if (!measurement->valid)
    {
        for (int i = 0; i < NS; ++i)
            track->x[i] = xp[i];

        for (int i = 0; i < NS * NS; ++i)
            track->P[i] = Pp[i];

        return true;
    }

    // ========================================================
    // STEP 3: Measurement innovation
    //
    // y = z - H * xp
    // ========================================================

    double innovation[NZ];

    for (int i = 0; i < NZ; ++i)
    {
        innovation[i] = measurement->z[i] - xp[i];
    }

    // ========================================================
    // STEP 4: Innovation covariance
    //
    // S = H * Pp * H^T + R
    // ========================================================

    double S[9];

    for (int r = 0; r < NZ; ++r)
    {
        for (int c = 0; c < NZ; ++c)
        {
            S[r * NZ + c] =
                Pp[r * NS + c] + params->R[r * NZ + c];
        }
    }

    // ========================================================
    // STEP 5: Invert innovation covariance
    // ========================================================

    double Sinv[9];

    if (!lkf_inverse3(S, Sinv))
        return false;

    // ========================================================
    // STEP 6: Kalman gain
    //
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
                    Pp[r * NS + j] * Sinv[j * NZ + c];
            }
        }
    }

    // ========================================================
    // STEP 7: State correction
    //
    // x(k|k) = xp + K * innovation
    // ========================================================

    double corrected_x[NS];

    for (int r = 0; r < NS; ++r)
    {
        corrected_x[r] = xp[r];

        for (int c = 0; c < NZ; ++c)
        {
            corrected_x[r] +=
                K[r * NZ + c] * innovation[c];
        }
    }

    // ========================================================
    // STEP 8: Joseph-form covariance correction
    //
    // P = (I-KH)Pp(I-KH)^T + KRK^T
    // ========================================================

    double A[NS * NS] = {};
    double AP[NS * NS] = {};
    double newP[NS * NS] = {};

    // A = I - K*H
    // H selects the first three state variables.
    for (int r = 0; r < NS; ++r)
    {
        for (int c = 0; c < NS; ++c)
        {
            A[r * NS + c] = (r == c) ? 1.0 : 0.0;

            if (c < NZ)
                A[r * NS + c] -= K[r * NZ + c];
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
                    A[r * NS + j] * Pp[j * NS + c];
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
                    AP[r * NS + j] * A[c * NS + j];
            }
        }
    }

    // Second term: K * R * K^T
    double KR[NS * NZ] = {};

    for (int r = 0; r < NS; ++r)
    {
        for (int c = 0; c < NZ; ++c)
        {
            for (int j = 0; j < NZ; ++j)
            {
                KR[r * NZ + c] +=
                    K[r * NZ + j] * params->R[j * NZ + c];
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
                    KR[r * NZ + j] * K[c * NZ + j];
            }
        }
    }

    // Remove small numerical asymmetry.
    for (int r = 0; r < NS; ++r)
    {
        for (int c = r + 1; c < NS; ++c)
        {
            const double value =
                0.5 * (newP[r * NS + c] + newP[c * NS + r]);

            newP[r * NS + c] = value;
            newP[c * NS + r] = value;
        }
    }

    // ========================================================
    // STEP 9: Store corrected state and covariance
    // ========================================================

    for (int i = 0; i < NS; ++i)
        track->x[i] = corrected_x[i];

    for (int i = 0; i < NS * NS; ++i)
        track->P[i] = newP[i];

    return true;
}

#endif // LKF_CONVENTIONAL_H
