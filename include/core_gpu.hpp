#pragma once

// helper types to differentiate compile time settings
template <bool val=true> struct is_periodic {};
template <> struct is_periodic<false> {};

template <bool val=true> struct use_LES {};
template <> struct use_LES<false> {};

__global__ void stream_collide_gpu(unsigned int Nx, unsigned int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float ux0, float* f, bool* solid_node, float tau, bool save, is_periodic<false>, use_LES<false>);
__global__ void stream_collide_gpu(unsigned int Nx, unsigned int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float ux0, float* f, bool* solid_node, float tau, bool save, is_periodic<true>, use_LES<false>);
__global__ void stream_collide_gpu(unsigned int Nx, unsigned int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float ux0, float* f, bool* solid_node, float tau, bool save, is_periodic<false>, use_LES<true>);
__global__ void stream_collide_gpu(unsigned int Nx, unsigned int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float ux0, float* f, bool* solid_node, float tau, bool save, is_periodic<true>, use_LES<true>);