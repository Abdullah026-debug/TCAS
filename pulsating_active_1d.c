/* pulsating_active_1d.c
 * Adapted from Davide Marenduzzo's Phase-Field code.
  Also warning that the numerics on the bilaplacian wasnt dealt in a super careful way!!!
 *
 * 1D coupled PDE solver:
 *   dL/dt   = M_L  d^2/dX^2 [ k(L-l(f)) - kappaL d^2L/dX^2 ]   [Model B]
 *   dphi/dt = omega - M_phi [ -k(L-l(f)) l'(f) + U'(f) ]        [Model A + drive]
 *
 * with  l(f) = ell0 + ellbar*cos(f),   U(f) = -v*cos(m*f)
 *
 * Usage: ./pulsating1d omega ellbar v kappaL dt ML Mphi L0 seed
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Pi 3.141592653589793

/* ── grid ── */
#define Nx       512
#define Nmax     1000000
#define stepskip 2000

/* ── parameters (defaults; overridden by argv) ── */
double dt     = 0.05;
double dx     = 1.0;
double ML     = 0.1;
double Mphi   = 0.5;
double ksp    = 1.0;    /* spring constant k                  */
double kappaL = 0.5;    /* L-gradient coefficient kappa_L     */
double ell0   = 1.0;    /* mean rest length l_0               */
double ellbar = 0.5;    /* rest-length modulation amplitude   */
double vpot   = 0.3;    /* tilt-potential amplitude v         */
int    mwind  = 1;      /* winding number m                   */
double omega  = 0.2;    /* drive omega = M_phi * Dmu/(2*pi)   */
double L0     = 1.0;    /* initial mean strain                */
double phi0   = 0.0;    /* initial mean phase                 */
int    seed   = 42;

/* ── fields ── */
double L[Nx], Lnew[Nx], muL[Nx];
double f[Nx],  fnew[Nx];   /* f = phi */

/* ── helpers ── */
static inline double ell_f(double p)  { return ell0 + ellbar * cos(p); }
static inline double dell_f(double p) { return -ellbar * sin(p); }           /* d ell/dphi */
static inline double dU_f(double p)   { return vpot * mwind * sin(mwind*p); }/* dU/dphi   */

void initialize(void);
void update(void);
void write_snap(FILE *fp, int n);

/* ── main ── */
int main(int argc, char **argv)
{
    if (argc == 10) {
        omega  = atof(argv[1]);
        ellbar = atof(argv[2]);
        vpot   = atof(argv[3]);
        kappaL = atof(argv[4]);
        dt     = atof(argv[5]);
        ML     = atof(argv[6]);
        Mphi   = atof(argv[7]);
        L0     = atof(argv[8]);
        seed   = atoi(argv[9]);
    } else {
        fprintf(stderr, "Usage: ./pulsating1d omega ellbar v kappaL dt ML Mphi L0 seed\n");
        fprintf(stderr, "Running with defaults.\n");
    }

    srand48(seed);
    initialize();

    /* stability check for the bilaplacian term */
    double dt_max = dx*dx*dx*dx / (8.0 * ML * kappaL);
    if (dt > dt_max)
        fprintf(stderr, "WARNING: dt=%.4f > stability limit %.4f (bilaplacian)\n",
                dt, dt_max);

    FILE *fp = fopen("spacetime_1d.dat", "w");
    fprintf(fp, "# t X L phi\n");

    for (int n = 1; n <= Nmax; n++) {
        update();
        if (n % stepskip == 0 || n == 1)
            write_snap(fp, n);
    }

    fclose(fp);
    return 0;
}

/* ── physics ── */
void initialize(void)
{
    for (int i = 0; i < Nx; i++) {
        L[i] = L0   + 0.05 * (1.0 - 2.0 * drand48());
        f[i] = phi0 + 0.30 * (1.0 - 2.0 * drand48());
    }
}

void update(void)
{
    /* Step 1: chemical potential  muL = k(L-ell) - kappaL * d^2L/dX^2 */
    for (int i = 0; i < Nx; i++) {
        int ip = (i == Nx-1) ? 0 : i+1;
        int im = (i == 0)    ? Nx-1 : i-1;
        double d2L = (L[ip] + L[im] - 2.0*L[i]) / (dx*dx);
        muL[i] = ksp * (L[i] - ell_f(f[i])) - kappaL * d2L;
    }

    /* Step 2: forward-Euler update */
    for (int i = 0; i < Nx; i++) {
        int ip = (i == Nx-1) ? 0 : i+1;
        int im = (i == 0)    ? Nx-1 : i-1;

        /* Model B:  dL/dt = ML * d^2 muL / dX^2 */
        double d2muL = (muL[ip] + muL[im] - 2.0*muL[i]) / (dx*dx);
        Lnew[i] = L[i] + dt * ML * d2muL;

        /* Model A + drive:  dphi/dt = omega - Mphi * dF/dphi
         * dF/dphi = -k(L-ell) * ell'(phi) + U'(phi) */
        double sm   = L[i] - ell_f(f[i]);
        double dFdf = -ksp * sm * dell_f(f[i]) + dU_f(f[i]);
        fnew[i] = f[i] + dt * (omega - Mphi * dFdf);
    }

    for (int i = 0; i < Nx; i++) { L[i] = Lnew[i]; f[i] = fnew[i]; }
}

/* ── output ── */
void write_snap(FILE *fp, int n)
{
    double t = n * dt;
    for (int i = 0; i < Nx; i++)
        fprintf(fp, "%.4f %.4f %16.10f %16.10f\n", t, i*dx, L[i], f[i]);
    fprintf(fp, "\n");
    fflush(fp);
    fprintf(stderr, "t=%7.1f   L_mid=%8.5f   phi_mid=%8.5f\n",
            t, L[Nx/2], f[Nx/2]);
}
