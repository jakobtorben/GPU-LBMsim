/* Defines constants, helper types and methods for core GPU code
 * 
 * Filename: core_gpu.hpp
 * Author: Jakob Torben
 * Created: 13.07.2021
 * Last modified: 26.08.2021
 * 
 * This code is provided under the MIT license. See LICENSE.txt.
 */

#pragma once

// define weights for feq calculation to avoid uunecessary divisions
__constant__ float w0 = 4./9., w1 = 1./9., w2 = 1./36.;
__constant__ float c2 = 9./2.;

const int Q = 9; // number of velocities

// helper types to differentiate compile time settings
template <bool val=true> struct use_LES {};
template <> struct use_LES<false> {};

template <bool val=true> struct use_MRT {};
template <> struct use_MRT<false> {};

/* 
 * define transformation matrices for MRT collision operator
 * obtained through Gram-Schmidt procedure from
 * Lallemand, P. & Luo, L.-S. (2000), ‘Theory of the lattice Boltzmann method:
 * Dispersion, dissipation,isotropy, Galilean invariance, and stability’,Phys. Rev. E61(6), 6546–6562.
 * m = M * f
 * stored in constant memory on device, which is cached in the constant cache
 */
__constant__ int M[Q*Q]={
  1, 1, 1, 1, 1, 1, 1, 1, 1,
 -4,-1,-1,-1,-1, 2, 2, 2, 2,
  4,-2,-2,-2,-2, 1, 1, 1, 1,
  0, 1, 0,-1, 0, 1,-1,-1, 1,
  0,-2, 0, 2, 0, 1,-1,-1, 1,
  0, 0, 1, 0,-1, 1, 1,-1,-1,
  0, 0,-2, 0, 2, 1, 1,-1,-1,
  0, 1,-1, 1,-1, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 1,-1, 1,-1
};

// f = m * Minv
__constant__ float Minv[Q*Q] = {
  1./9., -1./9. ,  1./9.,   0.,     0.,      0.,     0.,     0.,    0.,
  1./9., -1./36., -1./18.,  1./6., -1./6.,   0.,     0.,     1./4., 0.  ,
  1./9., -1./36., -1./18.,  0.,     0.,      1./6., -1./6., -1./4., 0.  ,
  1./9., -1./36., -1./18., -1./6,   1./6,    0.,     0.,     1./4., 0.  ,
  1./9., -1./36., -1./18.,  0.,     0.,     -1./6.,  1./6., -1./4., 0.  ,
  1./9.,  1./18.,  1./36.,  1./6.,  1./12.,  1./6.,  1./12., 0.,    1./4.,
  1./9.,  1./18.,  1./36., -1./6., -1./12.,  1./6.,  1./12., 0.,   -1./4.,
  1./9.,  1./18.,  1./36., -1./6., -1./12., -1./6., -1./12., 0.,    1./4.,
  1./9.,  1./18.,  1./36.,  1./6.,  1./12., -1./6., -1./12., 0.,   -1./4.
};


// function declarations
__global__ void stream_collide_gpu(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u_lid, float* f,
                                   bool* solid_node, float tau, float omega, bool save, use_LES<false>, use_MRT<false>);
__global__ void stream_collide_gpu(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u_lid, float* f,
                                   bool* solid_node, float tau, float omega, bool save, use_LES<true>, use_MRT<false>);
__global__ void stream_collide_gpu(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u_lid, float* f,
                                   bool* solid_node, float tau, float omega, bool save, use_LES<false>, use_MRT<true>);
__global__ void stream_collide_gpu(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u_lid, float* f,
                                   bool* solid_node, float tau, float omega, bool save, use_LES<true>, use_MRT<true>);

/**
 * Inline helper function to find index of 3D array using strucutre of array
 * data layout to have coalesced memory access on GPU.
 *
 * @param[in] Nx, Ny domain size
 * @param[in] x, y grid coordinates
 * @param[in] a distriburion direction
 * @return 1D array index
 */
__device__ __forceinline__ size_t f_idx_gpu(int Nx, int Ny, int x, int y, int a)
{
    return ((Ny*a + y)*Nx + x);
}

/**
 * Inline helper function to find index of 2D array.
 *
 * @param[in] Nx, Ny domain size
 * @param[in] x, y grid coordinates
 * @return 1D array index
 */
__device__ __forceinline__ size_t arr_idx(int Nx, int x, int y)
{
    return x + Nx*y;
}