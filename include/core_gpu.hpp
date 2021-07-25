#pragma once

template <bool val=true> struct is_periodic {};
template <> struct is_periodic<false> {};

__global__ void stream_collide_gpu(unsigned int Nx, unsigned int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float ux0, float* f, bool* solid_node, float tau, bool save, is_periodic<false>);
__global__ void stream_collide_gpu(unsigned int Nx, unsigned int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float ux0, float* f, bool* solid_node, float tau, bool save, is_periodic<true>);