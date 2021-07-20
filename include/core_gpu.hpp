#pragma once

__global__ void stream_collide_periodic_gpu(unsigned int Nx, unsigned int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float* f, float* ftemp, bool* solid_node, float tau, bool save);
__global__ void stream_collide_gpu(unsigned int Nx, unsigned int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float ux0, float* f, float* ftemp, bool* solid_node, float tau, bool save);