
#ifndef LKF_CONVENTIONAL_H
#define LKF_CONVENTIONAL_H

#include <cmath>
#include <algorithm>

static constexpr int NS = 6;
static constexpr int NZ = 3;
static constexpr double KF_P0 = 1.0;



struct Track
{
    double x[NS] = {};
    double P[NS * NS] = {};
};



struct Meas
{
    bool valid = true;
    double z[NZ] = {};
};



struct KFParams
{
    double dt = 0.1;
    double sa2 = 1.0;

    double R[9] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
};



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

  

    double xp[NS];

    for (int axis = 0; axis < 3; ++axis)
    {
        xp[axis] =
            track->x[axis] + dt * track->x[axis + 3];

        xp[axis + 3] = track->x[axis + 3];
    }

  

    double FP[NS * NS] = {};
    double Pp[NS * NS] = {};

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

    if (!measurement->valid)
    {
        for (int i = 0; i < NS; ++i)
            track->x[i] = xp[i];

        for (int i = 0; i < NS * NS; ++i)
            track->P[i] = Pp[i];

        return true;
    }

   

    double innovation[NZ];

    for (int i = 0; i < NZ; ++i)
    {
        innovation[i] = measurement->z[i] - xp[i];
    }

    

    double S[9];

    for (int r = 0; r < NZ; ++r)
    {
        for (int c = 0; c < NZ; ++c)
        {
            S[r * NZ + c] =
                Pp[r * NS + c] + params->R[r * NZ + c];
        }
    }

   

    double Sinv[9];

    if (!lkf_inverse3(S, Sinv))
        return false;

   

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

  

    double A[NS * NS] = {};
    double AP[NS * NS] = {};
    double newP[NS * NS] = {};

    
    for (int r = 0; r < NS; ++r)
    {
        for (int c = 0; c < NS; ++c)
        {
            A[r * NS + c] = (r == c) ? 1.0 : 0.0;

            if (c < NZ)
                A[r * NS + c] -= K[r * NZ + c];
        }
    }

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

   

    for (int i = 0; i < NS; ++i)
        track->x[i] = corrected_x[i];

    for (int i = 0; i < NS * NS; ++i)
        track->P[i] = newP[i];

    return true;
}

#endif 
