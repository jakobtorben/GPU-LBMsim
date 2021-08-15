#include <cmath>
#include <cuda.h>

#include "init_gpu.hpp"
#include "core_gpu.hpp"


__global__ void define_geometry(int Nx, int Ny, bool* solid_node)
{
    int y = blockIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    // set fixed walls to solid boundaries
    if (x == 0)
        solid_node[arr_idx(Nx, x, y)] = 1;  // west wall
    else if (x == Nx-1)
        solid_node[arr_idx(Nx, x, y)] = 1;  // east wall
    else if (y == 0)
        solid_node[arr_idx(Nx, x, y)] = 1;  // south wall
    else 
        solid_node[arr_idx(Nx, x, y)] = 0;
    
}

// Set inital distributions to equilibrium values for - lid driven cavity
__global__ void initialise_lid(int Nx, int Ny, int Q, float u_lid, float* f, float* rho_arr, float* ux_arr, float* uy_arr)
{
    int y = blockIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    // set density to 1.0 to keep as much precision as possible during calculation
    rho_arr[arr_idx(Nx, x, y)] = 1.;
    uy_arr[arr_idx(Nx, x, y)] = 0.;
    if (y == Ny - 1)
        ux_arr[arr_idx(Nx, x, y)] = u_lid;
    else
        ux_arr[arr_idx(Nx, x, y)] = 0;
    
    float ux = ux_arr[arr_idx(Nx, x, y)];
    float uy = uy_arr[arr_idx(Nx, x, y)];
    float rho = rho_arr[arr_idx(Nx, x, y)];

    float uxsq = ux * ux;
    float uysq = uy * uy;
    float usq = uxsq + uysq;

    float uxuy5 =  ux + uy;
    float uxuy6 = -ux + uy;
    float uxuy7 = -ux - uy;
    float uxuy8 =  ux - uy;

    float c = 1 - 1.5*usq;
    float w0 = 4./9., w1 = 1./9., w2 = 1./36.;
    float c2 = 9./2.;
    float w_rho0 = w0 * rho;
    float w_rho1 = w1 * rho;
    float w_rho2 = w2 * rho;

    f[f_idx_gpu(Nx, Ny, x, y, 0)] = w_rho0*(c                           );
    f[f_idx_gpu(Nx, Ny, x, y, 1)] = w_rho1*(c + 3.*ux  + c2*uxsq        );
    f[f_idx_gpu(Nx, Ny, x, y, 2)] = w_rho1*(c + 3.*uy  + c2*uysq        );
    f[f_idx_gpu(Nx, Ny, x, y, 3)] = w_rho1*(c - 3.*ux  + c2*uxsq        );
    f[f_idx_gpu(Nx, Ny, x, y, 4)] = w_rho1*(c - 3.*uy  + c2*uysq        );
    f[f_idx_gpu(Nx, Ny, x, y, 5)] = w_rho2*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
    f[f_idx_gpu(Nx, Ny, x, y, 6)] = w_rho2*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
    f[f_idx_gpu(Nx, Ny, x, y, 7)] = w_rho2*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
    f[f_idx_gpu(Nx, Ny, x, y, 8)] = w_rho2*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);
}