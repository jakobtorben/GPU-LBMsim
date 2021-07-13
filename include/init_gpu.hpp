#pragma once

__global__ void read_geometry(int Nx, int Ny, bool* solid_node);

__global__ void initialise(int Nx, int Ny, int Q, float ux0, float* f, float* ftemp, float* rho_arr, float* ux_arr, float* uy_arr, bool* solid_node);