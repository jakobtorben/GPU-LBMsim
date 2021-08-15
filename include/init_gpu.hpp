#pragma once


__global__ void define_geometry(int Nx, int Ny, bool* solid_node);
__global__ void initialise_lid(int Nx, int Ny, int Q, float u_lid, float* f, float* rho_arr, float* ux_arr, float* uy_arr);