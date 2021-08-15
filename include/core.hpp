#pragma once

#include <cstddef>
#include <math.h>

// define constants
const int Q = 9;

// define weights for feq calculation to avoid uunecessary divisions
const float w0 = 4./9., w1 = 1./9., w2 = 1./36.;
const float c2 = 9./2.;

// helper types to differentiate compile time settings
template <bool val=true> struct use_LES {};
template <> struct use_LES<false> {};

template <bool val=true> struct use_MRT {};
template <> struct use_MRT<false> {};

// define transformation matrices for MRT collision operator
// obtained through Gram-Schmidt procedure
// insert citation
// m = M * f
const int M[Q*Q]={
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
const float Minv[Q*Q] = {
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
void stream_collide_gpu_lid(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u_lid, float* f, bool* solid_node, float tau, float omega, bool save, use_LES<false>, use_MRT<false>);
void stream_collide_gpu_lid(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u_lid, float* f, bool* solid_node, float tau, float omega, bool save, use_LES<true>, use_MRT<false>);
void stream_collide_gpu_lid(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u_lid, float* f, bool* solid_node, float tau, float omega, bool save, use_LES<false>, use_MRT<true>);
void stream_collide_gpu_lid(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u_lid, float* f, bool* solid_node, float tau, float omega, bool save, use_LES<true>, use_MRT<true>);

inline size_t f_idx_cpu(int Nx, int Ny, int x, int y, int a)
{
    // use array of structures memory layout to have
    // coalesced access and utilise CPU cache
    return (x + Nx*y)*Q + a;
}

inline size_t arr_idx(int Nx, int x, int y)
{
    return x + Nx*y;
}