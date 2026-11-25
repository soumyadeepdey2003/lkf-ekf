
#ifndef KF_LKF_LIGHTWEIGHT_H
#define KF_LKF_LIGHTWEIGHT_H

#include <cmath>
#include <algorithm>

namespace kf
{

// ============================================================
// 1. DIMENSIONS AND CONFIGURATION
// ============================================================

constexpr int STATE_DIM = 6;
constexpr int MEAS_DIM = 3;
constexpr double DEFAULT_TOLERANCE = 1.0e-12;

// ============================================================
// 2. DATA STRUCTURES
// ============================================================

// State = [x, y, z, vx, vy, vz]

struct State
{
    double x{}, y{}, z{};
    double vx{}, vy{}, vz{};
};

// Covariance block representation:
//
// P = | A    C   |
//     | C^T  D   |
//
// A: position covariance (3x3)
// C: position-velocity covariance (3x3)
// D: velocity covariance (3x3)

struct Covariance
{
    double A[3][3]{};
    double C[3][3]{};
    double D[3][3]{};
};

struct Measurement
{
    double x{}, y{}, z{};
};

struct MeasurementNoise
{
    double R[3][3]{};
};

enum class SolverType
{
    Identity,
    Diagonal,
    Triangular,
    Cholesky,
    GaussJordan,
    Invalid
};

struct LKFResult
{
    State state{};
    Covariance covariance{};
    double gain[6][3]{};
    SolverType solver = SolverType::Invalid;
    bool success = false;
};

// ============================================================
// 3. BASIC VALIDATION FUNCTIONS
// ============================================================

namespace detail
{

inline bool finite(double value)
{
    return std::isfinite(value);
}

inline bool approximatelyZero(double value, double tolerance)
{
    return std::abs(value) <= tolerance;
}

inline bool approximatelyEqual(
    double a, double b, double tolerance)
{
    return std::abs(a - b) <= tolerance;
}

inline bool validMatrix3(
    const double M[3][3])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (!finite(M[i][j]))
                return false;

    return true;
}

inline bool symmetricMatrix3(
    const double M[3][3],
    double tolerance)
{
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (!approximatelyEqual(
                    M[i][j], M[j][i], tolerance))
                return false;

    return true;
}

inline bool isIdentity(
    const double S[3][3],
    double tolerance)
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            double expected = (i == j) ? 1.0 : 0.0;

            if (!approximatelyEqual(
                    S[i][j], expected, tolerance))
                return false;
        }

    return true;
}

inline bool isDiagonal(
    const double S[3][3],
    double tolerance)
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (i != j &&
                !approximatelyZero(S[i][j], tolerance))
                return false;

    return true;
}

inline bool isTriangular(
    const double S[3][3],
    double tolerance,
    bool& lower)
{
    bool upper = true;
    bool lowerCheck = true;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            if (i < j &&
                !approximatelyZero(S[i][j], tolerance))
                lowerCheck = false;

            if (i > j &&
                !approximatelyZero(S[i][j], tolerance))
                upper = false;
        }

    if (upper)
    {
        lower = false;
        return true;
    }

    if (lowerCheck)
    {
        lower = true;
        return true;
    }

    return false;
}

// ============================================================
// 4. SPECIALIZED LINEAR SYSTEM SOLVERS
// ============================================================

// Solve:
//
// S * X = B
//
// S: 3x3
// B: 3x6
// X: 3x6
//
// We solve directly instead of explicitly calculating S inverse.

inline bool solveDiagonal(
    const double S[3][3],
    double B[3][6],
    double tolerance)
{
    for (int i = 0; i < 3; ++i)
    {
        if (std::abs(S[i][i]) <= tolerance)
            return false;

        const double reciprocal = 1.0 / S[i][i];

        for (int j = 0; j < 6; ++j)
            B[i][j] *= reciprocal;
    }

    return true;
}

inline bool solveTriangular(
    const double S[3][3],
    double B[3][6],
    bool lower,
    double tolerance)
{
    for (int col = 0; col < 6; ++col)
    {
        if (lower)
        {
            for (int i = 0; i < 3; ++i)
            {
                double value = B[i][col];

                for (int k = 0; k < i; ++k)
                    value -= S[i][k] * B[k][col];

                if (std::abs(S[i][i]) <= tolerance)
                    return false;

                B[i][col] = value / S[i][i];
            }
        }
        else
        {
            for (int i = 2; i >= 0; --i)
            {
                double value = B[i][col];

                for (int k = i + 1; k < 3; ++k)
                    value -= S[i][k] * B[k][col];

                if (std::abs(S[i][i]) <= tolerance)
                    return false;

                B[i][col] = value / S[i][i];
            }
        }
    }

    return true;
}

// Cholesky decomposition: S = L * L^T

inline bool cholesky(
    const double S[3][3],
    double L[3][3])
{
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            double sum = S[i][j];

            for (int k = 0; k < j; ++k)
                sum -= L[i][k] * L[j][k];

            if (i == j)
            {
                if (!(sum > 0.0) || !finite(sum))
                    return false;

                L[i][j] = std::sqrt(sum);
            }
            else
            {
                L[i][j] = sum / L[j][j];
            }
        }
    }

    return true;
}

// Solve L * L^T * X = B.

inline bool solveCholesky(
    const double L[3][3],
    double B[3][6],
    double tolerance)
{
    // Forward substitution: L * Y = B

    if (!solveTriangular(L, B, true, tolerance))
        return false;

    // Backward substitution: L^T * X = Y

    for (int col = 0; col < 6; ++col)
    {
        for (int i = 2; i >= 0; --i)
        {
            double value = B[i][col];

            for (int k = i + 1; k < 3; ++k)
                value -= L[k][i] * B[k][col];

            if (std::abs(L[i][i]) <= tolerance)
                return false;

            B[i][col] = value / L[i][i];
        }
    }

    return true;
}

// Gauss-Jordan with partial pivoting.

inline bool solveGaussJordan(
    double S[3][3],
    double B[3][6],
    double tolerance)
{
    for (int pivot = 0; pivot < 3; ++pivot)
    {
        int pivotRow = pivot;
        double largest = std::abs(S[pivot][pivot]);

        for (int row = pivot + 1; row < 3; ++row)
        {
            double candidate = std::abs(S[row][pivot]);

            if (candidate > largest)
            {
                largest = candidate;
                pivotRow = row;
            }
        }

        if (largest <= tolerance || !finite(largest))
            return false;

        if (pivotRow != pivot)
        {
            for (int col = 0; col < 3; ++col)
                std::swap(
                    S[pivot][col],
                    S[pivotRow][col]);

            for (int col = 0; col < 6; ++col)
                std::swap(
                    B[pivot][col],
                    B[pivotRow][col]);
        }

        const double divisor = S[pivot][pivot];

        for (int col = 0; col < 3; ++col)
            S[pivot][col] /= divisor;

        for (int col = 0; col < 6; ++col)
            B[pivot][col] /= divisor;

        for (int row = 0; row < 3; ++row)
        {
            if (row == pivot)
                continue;

            const double factor = S[row][pivot];

            for (int col = 0; col < 3; ++col)
                S[row][col] -= factor * S[pivot][col];

            for (int col = 0; col < 6; ++col)
                B[row][col] -= factor * B[pivot][col];

            S[row][pivot] = 0.0;
        }
    }

    return true;
}

// ============================================================
// 5. ADAPTIVE SOLVER SELECTION
// ============================================================

inline bool calculateGain(
    const double A[3][3],
    const double C[3][3],
    const double innovationS[3][3],
    double K[6][3],
    SolverType& selectedSolver,
    double tolerance)
{
    double S[3][3]{};
    double B[3][6]{};

    // Copy innovation covariance.

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            S[i][j] = innovationS[i][j];

    // P * H^T = [ A ; C^T ]
    //
    // Therefore:
    //
    // B = (P * H^T)^T = [ A^T  C ]
    //
    // A is symmetric, so A^T = A.

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            B[i][j] = A[i][j];
            B[i][j + 3] = C[i][j];
        }

    bool solved = false;

    // Case 1: Identity matrix.

    if (isIdentity(S, tolerance))
    {
        selectedSolver = SolverType::Identity;
        solved = true;
    }

    // Case 2: Diagonal matrix.

    else if (isDiagonal(S, tolerance))
    {
        selectedSolver = SolverType::Diagonal;

        solved = solveDiagonal(S, B, tolerance);
    }

    // Case 3: Triangular matrix.

    if (!solved)
    {
        bool lower = false;

        if (isTriangular(S, tolerance, lower))
        {
            selectedSolver = SolverType::Triangular;

            solved = solveTriangular(
                S, B, lower, tolerance);
        }
    }

    // Case 4: Symmetric positive-definite matrix.

    if (!solved)
    {
        double L[3][3]{};

        if (cholesky(S, L))
        {
            selectedSolver = SolverType::Cholesky;

            solved = solveCholesky(
                L, B, tolerance);
        }
    }

    // Case 5: General matrix fallback.

    if (!solved)
    {
        selectedSolver = SolverType::GaussJordan;

        solved = solveGaussJordan(
            S, B, tolerance);
    }

    if (!solved)
    {
        selectedSolver = SolverType::Invalid;
        return false;
    }

    // Transpose the solved result to obtain K.

    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 3; ++j)
            K[i][j] = B[j][i];

    return true;
}

} // namespace detail

// ============================================================
// 6. MAIN LIGHTWEIGHT LKF FUNCTION
// ============================================================

inline bool updateLKF(
    const State& previousState,
    const Covariance& previousCovariance,
    const Measurement& measurement,
    const MeasurementNoise& measurementNoise,
    double deltaTime,
    double accelerationNoiseVariance,
    LKFResult& result,
    double tolerance = DEFAULT_TOLERANCE)
{
    // --------------------------------------------------------
    // STEP 1: VALIDATE INPUTS
    // --------------------------------------------------------

    result = LKFResult{};

    if (!detail::finite(deltaTime) ||
        !detail::finite(accelerationNoiseVariance) ||
        !detail::finite(tolerance) ||
        deltaTime <= 0.0 ||
        accelerationNoiseVariance < 0.0 ||
        tolerance <= 0.0)
        return false;

    const double stateValues[6] =
    {
        previousState.x, previousState.y, previousState.z,
        previousState.vx, previousState.vy, previousState.vz
    };

    const double measurementValues[3] =
    {
        measurement.x, measurement.y, measurement.z
    };

    for (double value : stateValues)
        if (!detail::finite(value))
            return false;

    for (double value : measurementValues)
        if (!detail::finite(value))
            return false;

    if (!detail::validMatrix3(previousCovariance.A) ||
        !detail::validMatrix3(previousCovariance.C) ||
        !detail::validMatrix3(previousCovariance.D) ||
        !detail::validMatrix3(measurementNoise.R))
        return false;

    if (!detail::symmetricMatrix3(
            previousCovariance.A, tolerance) ||
        !detail::symmetricMatrix3(
            previousCovariance.D, tolerance) ||
        !detail::symmetricMatrix3(
            measurementNoise.R, tolerance))
        return false;

    // --------------------------------------------------------
    // STEP 2: STATE PREDICTION
    // --------------------------------------------------------

    State predictedState{};

    predictedState.x =
        previousState.x + deltaTime * previousState.vx;

    predictedState.y =
        previousState.y + deltaTime * previousState.vy;

    predictedState.z =
        previousState.z + deltaTime * previousState.vz;

    predictedState.vx = previousState.vx;
    predictedState.vy = previousState.vy;
    predictedState.vz = previousState.vz;

    // --------------------------------------------------------
    // STEP 3: PROCESS NOISE CALCULATION
    // --------------------------------------------------------

    const double dt2 = deltaTime * deltaTime;
    const double dt3 = dt2 * deltaTime;
    const double dt4 = dt2 * dt2;

    const double q = accelerationNoiseVariance;

    const double qpp = q * dt4 / 4.0;
    const double qpv = q * dt3 / 2.0;
    const double qvv = q * dt2;

    // --------------------------------------------------------
    // STEP 4: COVARIANCE PREDICTION
    // --------------------------------------------------------

    Covariance predictedCovariance{};

    // Position covariance A'.

    for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
        {
            double value = previousCovariance.A[i][j];

            value += deltaTime *
                (previousCovariance.C[i][j] +
                 previousCovariance.C[j][i]);

            value += dt2 * previousCovariance.D[i][j];

            if (i == j)
                value += qpp;

            predictedCovariance.A[i][j] = value;
            predictedCovariance.A[j][i] = value;
        }

    // Position-velocity covariance C'.

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            predictedCovariance.C[i][j] =
                previousCovariance.C[i][j] +
                deltaTime * previousCovariance.D[i][j];

            if (i == j)
                predictedCovariance.C[i][j] += qpv;
        }

    // Velocity covariance D'.

    for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
        {
            double value = previousCovariance.D[i][j];

            if (i == j)
                value += qvv;

            predictedCovariance.D[i][j] = value;
            predictedCovariance.D[j][i] = value;
        }

    // --------------------------------------------------------
    // STEP 5: INNOVATION
    // --------------------------------------------------------

    const double innovation[3] =
    {
        measurement.x - predictedState.x,
        measurement.y - predictedState.y,
        measurement.z - predictedState.z
    };

    // --------------------------------------------------------
    // STEP 6: INNOVATION COVARIANCE
    // --------------------------------------------------------

    double S[3][3]{};

    for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
        {
            double value =
                predictedCovariance.A[i][j] +
                measurementNoise.R[i][j];

            S[i][j] = value;
            S[j][i] = value;
        }

    if (!detail::validMatrix3(S))
        return false;

    // --------------------------------------------------------
    // STEP 7: KALMAN GAIN
    // --------------------------------------------------------

    double K[6][3]{};

    SolverType solver = SolverType::Invalid;

    if (!detail::calculateGain(
            predictedCovariance.A,
            predictedCovariance.C,
            S,
            K,
            solver,
            tolerance))
        return false;

    // --------------------------------------------------------
    // STEP 8: STATE CORRECTION
    // --------------------------------------------------------

    double predicted[6] =
    {
        predictedState.x,
        predictedState.y,
        predictedState.z,
        predictedState.vx,
        predictedState.vy,
        predictedState.vz
    };

    double corrected[6]{};

    for (int i = 0; i < 6; ++i)
    {
        double correction = 0.0;

        for (int j = 0; j < 3; ++j)
            correction += K[i][j] * innovation[j];

        corrected[i] = predicted[i] + correction;
    }

    State correctedState
    {
        corrected[0], corrected[1], corrected[2],
        corrected[3], corrected[4], corrected[5]
    };

    // --------------------------------------------------------
    // STEP 9: JOSEPH-FORM COVARIANCE CORRECTION
    // --------------------------------------------------------

    // P_corrected =
    //
    // (I - K*H) * P_predicted * (I - K*H)^T
    //                 + K * R * K^T
    //
    // Use fixed-size arrays and reusable intermediate matrices.

    double P[6][6]{};

    // Reconstruct full 6x6 covariance.

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            P[i][j] = predictedCovariance.A[i][j];

            P[i][j + 3] = predictedCovariance.C[i][j];

            P[i + 3][j] = predictedCovariance.C[j][i];

            P[i + 3][j + 3] = predictedCovariance.D[i][j];
        }

    // Construct F = I - K*H.
    //
    // H = [ I  0 ]
    //
    // Therefore only the first three columns of K
    // contribute to K*H.

    double F[6][6]{};

    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            F[i][j] =
                (i == j ? 1.0 : 0.0) -
                (j < 3 ? K[i][j] : 0.0);

    // Intermediate = F * P.

    double intermediate[6][6]{};

    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            for (int m = 0; m < 6; ++m)
                intermediate[i][j] +=
                    F[i][m] * P[m][j];

    // Corrected P = intermediate * F^T + K*R*K^T.

    double correctedP[6][6]{};

    for (int i = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j)
        {
            double value = 0.0;

            for (int m = 0; m < 6; ++m)
                value += intermediate[i][m] * F[j][m];

            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    value +=
                        K[i][a] *
                        measurementNoise.R[a][b] *
                        K[j][b];

            if (!detail::finite(value))
                return false;

            correctedP[i][j] = value;
            correctedP[j][i] = value;
        }

    // --------------------------------------------------------
    // STEP 10: EXTRACT COVARIANCE BLOCKS
    // --------------------------------------------------------

    Covariance correctedCovariance{};

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            correctedCovariance.A[i][j] =
                correctedP[i][j];

            correctedCovariance.C[i][j] =
                correctedP[i][j + 3];

            correctedCovariance.D[i][j] =
                correctedP[i + 3][j + 3];
        }

    // --------------------------------------------------------
    // STEP 11: FINAL RESULT VALIDATION
    // --------------------------------------------------------

    for (double value : corrected)
        if (!detail::finite(value))
            return false;

    if (!detail::validMatrix3(correctedCovariance.A) ||
        !detail::validMatrix3(correctedCovariance.C) ||
        !detail::validMatrix3(correctedCovariance.D))
        return false;

    // --------------------------------------------------------
    // STEP 12: STORE RESULTS
    // --------------------------------------------------------

    result.state = correctedState;
    result.covariance = correctedCovariance;

    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 3; ++j)
            result.gain[i][j] = K[i][j];

    result.solver = solver;
    result.success = true;

    return true;
}

} // namespace kf

#endif
