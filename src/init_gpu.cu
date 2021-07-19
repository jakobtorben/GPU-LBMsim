#include <cmath>
#include <cuda.h>
#include <stdio.h>

#include "init_gpu.hpp"

__device__ __forceinline__ size_t f_index(unsigned int Nx, unsigned int Ny, unsigned int x, unsigned int y, unsigned int a)
{
    return ((Ny*a + y)*Nx + x);
    //return (x + Nx*y)*9 + a;
}

// this will later read in a predefined mask
__global__ void read_geometry(int Nx, int Ny, bool* solid_node)
{
    unsigned int y = blockIdx.y;
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("blockidx %d blockdimx %d threadidx %d blockidxy %d\n ", blockIdx.x, blockDim.x, threadIdx.x, blockIdx.y);

	// define geometry
	const int cx = Nx/3, cy = Ny/2;
	const int radius = Ny/8;

    int cord = x + Nx*y;
    float dx = std::abs(cx - (long int)x);
    float dy = std::abs(cy - (long int)y);
    float dist = std::sqrt(dx*dx + dy*dy);
    solid_node[cord] = (dist < radius) ? 1 : 0;
}

// apply initial conditions - flow to the right
__global__ void initialise(int Nx, int Ny, int Q, float ux0, float* f, float* ftemp, float* rho_arr, float* ux_arr, float* uy_arr, bool* solid_node)
{
    unsigned int y = blockIdx.y;
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    int cord = x + Nx*y;
    // set density to 1.0 to keep as much precision as possible during calculation
    rho_arr[cord] = 1.;
    ux_arr[cord] = ux0;
    uy_arr[cord] = 0.;

	float c2 = 9./2.;

    if (!solid_node[cord])
    {
        float w_rho0 = 4./9.  * rho_arr[cord];
        float w_rho1 = 1./9.  * rho_arr[cord];
        float w_rho2 = 1./36. * rho_arr[cord];

        float uxij = ux_arr[cord];
        float uyij = uy_arr[cord];

        float uxsq = uxij * uxij;
        float uysq = uyij * uyij;
        float usq = uxsq + uysq;

        float uxuy5 = uxij + uyij;
        float uxuy6 = -uxij + uyij;
        float uxuy7 = -uxij - uyij;
        float uxuy8 = uxij - uyij;

        float c = 1 - 1.5*usq;

        f[f_index(Nx, Ny, x, y, 0)] = w_rho0*(c                            );
        f[f_index(Nx, Ny, x, y, 1)] = w_rho1*(c + 3.*uxij  + c2*uxsq       );
        f[f_index(Nx, Ny, x, y, 2)] = w_rho1*(c + 3.*uyij  + c2*uysq       );
        f[f_index(Nx, Ny, x, y, 3)] = w_rho1*(c - 3.*uxij  + c2*uxsq       );
        f[f_index(Nx, Ny, x, y, 4)] = w_rho1*(c - 3.*uyij  + c2*uysq       );
        f[f_index(Nx, Ny, x, y, 5)] = w_rho2*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
        f[f_index(Nx, Ny, x, y, 6)] = w_rho2*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
        f[f_index(Nx, Ny, x, y, 7)] = w_rho2*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
        f[f_index(Nx, Ny, x, y, 8)] = w_rho2*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);

        // copy values to ftemp
        ftemp[f_index(Nx, Ny, x, y, 0)] = f[f_index(Nx, Ny, x, y, 0)];
        ftemp[f_index(Nx, Ny, x, y, 1)] = f[f_index(Nx, Ny, x, y, 1)];
        ftemp[f_index(Nx, Ny, x, y, 2)] = f[f_index(Nx, Ny, x, y, 2)];
        ftemp[f_index(Nx, Ny, x, y, 3)] = f[f_index(Nx, Ny, x, y, 3)];
        ftemp[f_index(Nx, Ny, x, y, 4)] = f[f_index(Nx, Ny, x, y, 4)];
        ftemp[f_index(Nx, Ny, x, y, 5)] = f[f_index(Nx, Ny, x, y, 5)];
        ftemp[f_index(Nx, Ny, x, y, 6)] = f[f_index(Nx, Ny, x, y, 6)];
        ftemp[f_index(Nx, Ny, x, y, 7)] = f[f_index(Nx, Ny, x, y, 7)];
        ftemp[f_index(Nx, Ny, x, y, 8)] = f[f_index(Nx, Ny, x, y, 8)];

    }
    else
        // set distributions to zero at solids
        for (int a = 0; a < Q; a++)
        {
            f[f_index(Nx, Ny, x, y, a)] = 0;
            ftemp[f_index(Nx, Ny, x, y, a)] = 0;
        }
}