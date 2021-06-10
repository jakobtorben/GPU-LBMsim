#pragma once

void stream(int Nx, int Ny, int Q, float* ftemp, float* f, bool* solid_node);
void calc_macro_quant(int Nx, int Ny, int Q,
	float* u_x, float* u_y,
	float* rho, float* ftemp, bool* solid_node,
	const int* ex, const int* ey);
void calc_eq(int Nx, int Ny, int Q, float* rho, float* u_x, float* u_y, bool* solid_node, float* result);
void collide(int Nx, int Ny, int Q, float* f, float* ftemp, float* feq, bool* solid_node, float tau);