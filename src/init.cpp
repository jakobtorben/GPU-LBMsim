#include <cmath>

#include "init.hpp"


// this will later read xn a predefined mask
void read_geometry(int Nx, int Ny, bool* solid_node)
{
	// define geometry
	const int cx = Nx/3, cy = Ny/2;
	const int radius = Ny/8;
	for (int y = 0; y < Ny; y++)
		for (int x = 0; x < Nx; x++)
		{
			int cord = x + Nx*y;
			float dx = std::abs(cx - x);
			float dy = std::abs(cy - y);
			float dist = std::sqrt(dx*dx + dy*dy);
			solid_node[cord] = (dist < radius) ? 1 : 0;
		}
}

// apply initial conditions - flow to the rigth
void initialise(int Nx, int Ny, int Q, float ux0, float* f, float* ftemp, float* rho, float* u_x, float* u_y)
{
	for (int y = 0; y < Ny; y++)
		for (int x = 0; x < Nx; x++)
		{
			int cord = x + Nx*y;
            // set density to 1.0 to keep as much precision as possible during calculation
			rho[cord] = 1.;
			u_x[cord] = ux0;
			u_y[cord] = 0.;
		}

	for (int y = 0; y < Ny; y++)
		for (int x = 0; x < Nx; x++)
		{
			int cord = x + Nx*y;
			for (int a = 0; a < Q; a++)
			{
                // set higher number of particles travelling to the right
				f[Q*cord + a] = (a == 1) ? 2 : 1;
				ftemp[cord*Q + a] = f[cord*Q + a];
			}
		}
}