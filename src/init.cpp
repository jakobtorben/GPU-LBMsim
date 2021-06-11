#include <cmath>

#include "init.hpp"


// this will later read in a predefined mask
void read_geometry(int Nx, int Ny, bool* solid_node)
{
	// define geometry
	const int cx = 150, cy = 150;
	const int radius = 50;
	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			int pos = i + Nx*j;
			float dx = std::abs(cx - i);
			float dy = std::abs(cy - j);
			float dist = std::sqrt(dx*dx + dy*dy);
			solid_node[pos] = (dist < radius) ? 1 : 0;
			if (j == 0 || j == Ny-1)
				solid_node[pos] = 1;
		}
}

// apply initial conditions - flow to the rigth
void initialise(int Nx, int Ny, int Q, float ux0, float* f, float* ftemp, float* rho, float* u_x, float* u_y)
{
	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			int pos = i + Nx*j;
			rho[pos] = 1.;
			u_x[pos] = ux0;
			u_y[pos] = 0.;
		}

	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			int pos = i + Nx*j;
			for (int a = 0; a < Q; a++)
			{
				f[Q*pos + a] = (a == 1) ? 2 : 1;
				ftemp[pos*Q + a] = f[pos*Q + a];
			}
		}
}