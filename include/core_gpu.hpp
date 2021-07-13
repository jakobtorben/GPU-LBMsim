#pragma once

__global__ void stream_periodic_gpu(int Nx, int Ny, int Q, float* ftemp, float* f, bool* solid_node);
__global__ void stream_gpu(int Nx, int Ny, int Q, float* ftemp, float* f, bool* solid_node);
void boundary_gpu(int Nx, int Ny, int Q, float ux0, float* ftemp, float* f, bool* solid_node);
__global__ void collide_gpu(int Nx, int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float* f, float* ftemp, bool* solid_node, float tau, bool save);