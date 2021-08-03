#include <cmath>
#include <cuda.h>
#include <stdio.h>

#include "init_gpu.hpp"

__device__ __forceinline__ size_t f_index(int Nx, int Ny, int x, int y, int a)
{
    return ((Ny*a + y)*Nx + x);
    //return (x + Nx*y)*9 + a;
}

// this will later read in a predefined mask
__global__ void read_geometry(int Nx, int Ny, bool* solid_node)
{
    int y = blockIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("blockidx %d blockdimx %d threadidx %d blockidxy %d\n ", blockIdx.x, blockDim.x, threadIdx.x, blockIdx.y);

	// define geometry
	const int cx = Nx/4, cy = Ny/2;
	const int radius = Ny/16;

    int cord = x + Nx*y;
    float dx = std::abs(cx - (long int)x);
    float dy = std::abs(cy - (long int)y);
    float dist = std::sqrt(dx*dx + dy*dy);
    if (( x > (cx - radius)) && (x < (cx + radius)) && (y > (cy - radius)) && (y < (cy + radius)))
        solid_node[cord] = 0;
    else
        solid_node[cord] = 0;
    // solid_node[cord] = (dist < radius) ? 1 : 0;
}

__global__ void read_geometry_lid(int Nx, int Ny, bool* solid_node)
{
    int y = blockIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("blockidx %d blockdimx %d threadidx %d blockidxy %d\n ", blockIdx.x, blockDim.x, threadIdx.x, blockIdx.y);

    int cord = x + Nx*y;
    
    // define geometry
	//const int cx = Nx/4, cy = Ny/2;
	//const int radius = Ny/16;
    // square
    //if (( x > (cx - radius)) && (x < (cx + radius)) && (y > (cy - radius)) && (y < (cy + radius)))
    //    solid_node[cord] = 0;
    // else
    //    solid_node[cord] = 0;

    // cylinder
    //float dx = std::abs(cx - (long int)x);
    //float dy = std::abs(cy - (long int)y);
    //float dist = std::sqrt(dx*dx + dy*dy);
    // solid_node[cord] = (dist < radius) ? 1 : 0;
    
    if ((x == 0) && (y < Ny-1)) // west wall
        solid_node[cord] = 1;
    else if ((x == Nx-1) && (y < Ny-1))
        solid_node[cord] = 1;  // east wall
    else if (y == 0)
        solid_node[cord] = 1;  // south wall
    //else if ((y == Ny-1) && (x > 0) && (x < Nx-1))
    //    solid_node[cord] = 1;  // north wall
    else 
        solid_node[cord] = 0;
}

// apply initial conditions - flow to the right
__global__ void initialise(int Nx, int Ny, int Q, float ux0, float* f, float* rho_arr, float* ux_arr, float* uy_arr, bool* solid_node)
{
    int y = blockIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;

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
    }
    else
        // set distributions to zero at solids
        for (int a = 0; a < Q; a++)
        {
            f[f_index(Nx, Ny, x, y, a)] = 0;
        }
}

// apply initial conditions - lid driven cavity
__global__ void initialise_lid(int Nx, int Ny, int Q, float u0, float* f, float* rho_arr, float* ux_arr, float* uy_arr)
{
    int y = blockIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    int cord = x + Nx*y;
    // set density to 1.0 to keep as much precision as possible during calculation
    rho_arr[cord] = 1.;
    uy_arr[cord] = 0.;
    if (y == Ny - 1)
        ux_arr[cord] = u0;
    else
        ux_arr[cord] = 0;

	float c2 = 9./2.;

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
}