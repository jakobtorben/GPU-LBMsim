#pragma once

// define constants
const int Q = 9;

// helper types to differentiate compile time settings
template <bool val=true> struct use_LES {};
template <> struct use_LES<false> {};

template <bool val=true> struct use_MRT {};
template <> struct use_MRT<false> {};

// define transformation matrices for MRT collision operator
// obtained through Gram-Schmidt procedure
// insert citation
// m = M * f
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
__global__ void stream_collide_gpu_lid(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float ux0, float* f, bool* solid_node, float tau, bool save, use_LES<false>, use_MRT<false>);
__global__ void stream_collide_gpu_lid(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u0, float* f, bool* solid_node, float tau, bool save, use_LES<true>, use_MRT<false>);
__global__ void stream_collide_gpu_lid(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u0, float* f, bool* solid_node, float tau, bool save, use_LES<false>, use_MRT<true>);
__global__ void stream_collide_gpu_lid(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u0, float* f, bool* solid_node, float tau, bool save, use_LES<true>, use_MRT<true>);