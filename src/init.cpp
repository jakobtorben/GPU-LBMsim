#include <cmath>

#include "init.hpp"


// this will later read in a predefined mask
void read_geometry(int Nx, int Ny, bool* solid_node)
{
	// define geometry
	const int cx = Nx/3, cy = Ny/2;
	const int radius = Ny/8;
	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			int ij = i + Nx*j;
			float dx = std::abs(cx - i);
			float dy = std::abs(cy - j);
			float dist = std::sqrt(dx*dx + dy*dy);
			solid_node[ij] = (dist < radius) ? 1 : 0;
			if (j == 0 || j == Ny-1)
				solid_node[ij] = 1;
		}
}

// apply initial conditions - flow to the rigth
void initialise(int Nx, int Ny, int Q, float ux0, float* f, float* ftemp, float* rho, float* u_x, float* u_y)
{
	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			int ij = i + Nx*j;
            // set density to 1.0 to keep as much precision as possible during calculation
			rho[ij] = 1.;
			u_x[ij] = ux0;
			u_y[ij] = 0.;
		}

	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			int ij = i + Nx*j;
			for (int a = 0; a < Q; a++)
			{
                // set higher number of particles travelling to the right
				f[Q*ij + a] = (a == 1) ? 2 : 1;
				ftemp[ij*Q + a] = f[ij*Q + a];
			}
		}
}