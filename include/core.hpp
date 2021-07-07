#pragma once

void stream_periodic(int Nx, int Ny, int Q, float* ftemp, float* f, bool* solid_node);
void stream(int Nx, int Ny, int Q, float* ftemp, float* f, bool* solid_node);
void boundary(int Nx, int Ny, int Q, float ux0, float* ftemp, float* f, bool* solid_node);
void calc_eq(int Nx, int Ny, int Q, float* rho, float* u_x, float* u_y, bool* solid_node, float* result);
void collide(int Nx, int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float* f, float* ftemp, bool* solid_node, float tau, bool save);