/* pulsating_active_2d.c
 * Adapted from Davide Marenduzzo's Phase-Field code see (https://www2.ph.ed.ac.uk/~dmarendu/phasefield2025.html)
 Also warning that the numerics on the bilaplacian wasnt dealt in a super careful way!!!
 * 2D coupled PDE solver:
 *   dL/dt   = M_L  nabla^2 [ k(L-l(f)) - kappaL nabla^2 L ]    [Model B]
 *   dphi/dt = omega - M_phi [ -k(L-l(f)) l'(f) + U'(f) ]       [Model A + drive]
 *
 * with  l(f) = ell0 + ellbar*cos(f),   U(f) = -v*cos(m*f)
 * Periodic BCs on Nx x Ny grid.
 *
 * Usage: ./pulsating2d omega ellbar v kappaL dt ML Mphi L0 seed
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Pi 3.141592653589793

/* ── grid ── */
#define Nx        200
#define Ny        200
#define Nmax      60000
#define stepskip  120

/* ── parameters (defaults; overridden by argv) ── */
double dt     = 0.05;
double dx     = 1.0;
double ML     = 0.1;
double Mphi   = 0.5;
double ksp    = 1.0;
double kappaL = 0.5;
double ell0   = 1.0;
double ellbar = 0.5;
double vpot   = 0.3;
int    mwind  = 1;
double omega  = 0.2;
double L0     = 1.0;
double phi0   = 0.0;
int    seed   = 42;

/* ── fields ── */
double L[Nx][Ny], Lnew[Nx][Ny], muL[Nx][Ny];
double f[Nx][Ny], fnew[Nx][Ny];

static inline double ell_f(double p)  { return ell0 + ellbar * cos(p); }
static inline double dell_f(double p) { return -ellbar * sin(p); }
static inline double dU_f(double p)   { return vpot * mwind * sin(mwind*p); }

void initialize(void);
void update(void);
void write_snap(int n);

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
        fprintf(stderr, "Usage: ./pulsating2d omega ellbar v kappaL dt ML Mphi L0 seed\n");
        fprintf(stderr, "Running with defaults.\n");
    }

    srand48(seed);
    initialize();

    double dt_max = dx*dx*dx*dx / (16.0 * ML * kappaL);
    if (dt > dt_max)
        fprintf(stderr, "WARNING: dt=%.4f > stability limit %.4f (2D bilaplacian)\n",
                dt, dt_max);

    for (int n = 1; n <= Nmax; n++) {
        update();

        if (n % stepskip == 0 || n == 1) {
            write_snap(n);
            fprintf(stderr, "t=%7.1f   L_ctr=%8.5f   phi_ctr=%8.5f\n",
                    n*dt, L[Nx/2][Ny/2], f[Nx/2][Ny/2]);
        }
    }

    return 0;
}

/* ── physics ── */
void initialize(void)
{
    for (int i = 0; i < Nx; i++)
        for (int j = 0; j < Ny; j++) {
            L[i][j] = L0   + 0.05 * (1.0 - 2.0 * drand48());
            f[i][j] = phi0 + 0.30 * (1.0 - 2.0 * drand48());
        }
}

void update(void)
{
    /* Step 1: muL = k(L-ell) - kappaL * nabla^2 L */
    for (int i = 0; i < Nx; i++) {
        int ip = (i==Nx-1)?0:i+1, im = (i==0)?Nx-1:i-1;
        for (int j = 0; j < Ny; j++) {
            int jp = (j==Ny-1)?0:j+1, jm = (j==0)?Ny-1:j-1;
            double lapL = (L[ip][j]+L[im][j]+L[i][jp]+L[i][jm]-4.0*L[i][j]) / (dx*dx);
            muL[i][j] = ksp*(L[i][j] - ell_f(f[i][j])) - kappaL*lapL;
        }
    }

    /* Step 2: forward-Euler */
    for (int i = 0; i < Nx; i++) {
        int ip = (i==Nx-1)?0:i+1, im = (i==0)?Nx-1:i-1;
        for (int j = 0; j < Ny; j++) {
            int jp = (j==Ny-1)?0:j+1, jm = (j==0)?Ny-1:j-1;

            /* Model B */
            double lapmu = (muL[ip][j]+muL[im][j]+muL[i][jp]+muL[i][jm]-4.0*muL[i][j]) / (dx*dx);
            Lnew[i][j] = L[i][j] + dt*ML*lapmu;

            /* Model A + drive */
            double sm   = L[i][j] - ell_f(f[i][j]);
            double dFdf = -ksp*sm*dell_f(f[i][j]) + dU_f(f[i][j]);
            fnew[i][j] = f[i][j] + dt*(omega - Mphi*dFdf);
        }
    }

    for (int i = 0; i < Nx; i++)
        for (int j = 0; j < Ny; j++) {
            L[i][j] = Lnew[i][j];
            f[i][j] = fnew[i][j];
        }
}

/* ── output: compact format  "# t Nx Ny" then L phi per line ── */
void write_snap(int n)
{
    char fname[64];
    sprintf(fname, "snap2d_%07d.dat", n);
    FILE *fp = fopen(fname, "w");
    fprintf(fp, "# %.4f %d %d\n", n*dt, Nx, Ny);
    for (int i = 0; i < Nx; i++)
        for (int j = 0; j < Ny; j++)
            fprintf(fp, "%.8f %.8f\n", L[i][j], f[i][j]);
    fclose(fp);
}
