/* kf_types.h - shared data types. Same layout is used by C, CUDA, HLS and Python (ctypes). */
#ifndef KF_TYPES_H
#define KF_TYPES_H
#include <math.h>
#include <stddef.h>

#ifdef __CUDACC__
#define KF_HD __host__ __device__
#else
#define KF_HD
#endif

#define NS 6   /* state size: x y z vx vy vz        */
#define NM 3   /* measurement size: LKF = x y z     */
#define KF_PI 3.14159265358979323846f
#define KF_P0 10.0f           /* initial diagonal covariance of a new track */
#define KF_RANGE_EPS 1e-3f    /* EKF: below this horizontal distance / range the angles are undefined */
enum { KF_LINEAR = 0, KF_SPHERICAL = 1 };   /* measurement type: (x,y,z) or (range,azimuth,elevation) */
/* Paper Eq.156/257: subtraction is carried out as addition of a negative:  A - B = A + (-B).
 * Compile with -DKF_NO_NEGSUM to switch this transformation OFF (ablation: plain subtraction). */
#ifdef KF_NO_NEGSUM
#define KF_SUB(a, b) ((a) - (b))
#else
#define KF_SUB(a, b) ((a) + (-(b)))
#endif
#define KF_PIVOT_EPS 1e-12f   /* pivots/determinants smaller than this = "numerically singular" */

typedef struct { float x[NS]; float P[NS * NS]; } Track;      /* one object's filter state   */
typedef struct { int valid; float z[NM]; } Meas;              /* valid = 0 -> predict only   */
typedef struct { float dt, sa2; float R[NM * NM]; } KFParams; /* sampling time, accel. noise */

#endif
