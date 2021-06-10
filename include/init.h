#pragma once

void read_geometry(int Nx, int Ny, bool* solid_node);

void initialise(int Nx, int Ny, int Q, float* f, float* ftemp, float* rho, float* u_x, float* u_y);