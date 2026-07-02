/* fdmforce C backend: random-Fourier force/potential evaluation hot loop.
 *
 * Evaluates, for N positions and M modes,
 *     F_d(x_i) = sum_j [ zr_{jd} cos(k_j . x_i) - zi_{jd} sin(k_j . x_i) ]
 * where z_{jd} = a_{jd} b_j (complex) is precomputed on the Python side, so this
 * routine is agnostic to the amplitude/phase model.  OpenMP-parallel over i.
 */
#define _GNU_SOURCE
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline void mysincos(double x, double *s, double *c) {
#if defined(__APPLE__)
    __sincos(x, s, c);
#elif defined(__GNUC__)
    sincos(x, s, c);
#else
    *s = sin(x);
    *c = cos(x);
#endif
}

void fdm_force(long N, long M,
               const double *px, const double *py, const double *pz,
               const double *kx, const double *ky, const double *kz,
               const double *zrx, const double *zry, const double *zrz,
               const double *zix, const double *ziy, const double *ziz,
               double *Fx, double *Fy, double *Fz) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < N; i++) {
        const double xi = px[i], yi = py[i], zi = pz[i];
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (long j = 0; j < M; j++) {
            const double p = kx[j] * xi + ky[j] * yi + kz[j] * zi;
            double s, c;
            mysincos(p, &s, &c);
            ax += zrx[j] * c - zix[j] * s;
            ay += zry[j] * c - ziy[j] * s;
            az += zrz[j] * c - ziz[j] * s;
        }
        Fx[i] = ax;
        Fy[i] = ay;
        Fz[i] = az;
    }
}

/* Scalar potential: Phi(x_i) = sum_j [ pr_j cos(k_j.x_i) - pi_j sin(k_j.x_i) ]. */
void fdm_potential(long N, long M,
                   const double *px, const double *py, const double *pz,
                   const double *kx, const double *ky, const double *kz,
                   const double *pr, const double *pim,
                   double *Phi) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < N; i++) {
        const double xi = px[i], yi = py[i], zi = pz[i];
        double acc = 0.0;
        for (long j = 0; j < M; j++) {
            const double p = kx[j] * xi + ky[j] * yi + kz[j] * zi;
            double s, c;
            mysincos(p, &s, &c);
            acc += pr[j] * c - pim[j] * s;
        }
        Phi[i] = acc;
    }
}

/* Reports whether the library was built with OpenMP (for diagnostics). */
int fdm_has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}
