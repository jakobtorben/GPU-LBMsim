#include <cmath>

#include "init.hpp"


// this will later read in a predefined mask
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
void initialise(int Nx, int Ny, int Q, float ux0, float* f, float* ftemp, float* rho_arr, float* ux_arr, float* uy_arr, bool* solid_node)
{
	for (int y = 0; y < Ny; y++)
		for (int x = 0; x < Nx; x++)
		{
			int cord = x + Nx*y;
			// set density to 1.0 to keep as much precision as possible during calculation
			rho_arr[cord] = 1.;
			ux_arr[cord] = ux0;
			uy_arr[cord] = 0.;
		}

	float c2 = 9./2.;
	for (int y = 0; y < Ny; y++)
		for (int x = 0; x < Nx; x++)
		{
			int cord = x + Nx*y;
			if (!solid_node[cord])
			{
				float w_rho0 = 4./9.  * rho_arr[cord];
				float w_rho1 = 1./9.  * rho_arr[cord];
				float w_rho2 = 1./36. * rho_arr[cord];

				float uxij = ux_arr[cord];
				float uyij = uy_arr[cord];

				float uxsq = uxij * uxij;
				float uysq = uyij * uyij;
				float usq = uxsq + uysq;

				float uxuy5 = uxij + uyij;
				float uxuy6 = -uxij + uyij;
				float uxuy7 = -uxij - uyij;
				float uxuy8 = uxij - uyij;

				float c = 1 - 1.5*usq;

				f[Q*cord    ] = w_rho0*(c                            );
				f[Q*cord + 1] = w_rho1*(c + 3.*uxij  + c2*uxsq       );
				f[Q*cord + 2] = w_rho1*(c + 3.*uyij  + c2*uysq       );
				f[Q*cord + 3] = w_rho1*(c - 3.*uxij  + c2*uxsq       );
				f[Q*cord + 4] = w_rho1*(c - 3.*uyij  + c2*uysq       );
				f[Q*cord + 5] = w_rho2*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
				f[Q*cord + 6] = w_rho2*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
				f[Q*cord + 7] = w_rho2*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
				f[Q*cord + 8] = w_rho2*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);

				// copy values to ftemp
				ftemp[Q*cord    ] = f[Q*cord    ];
				ftemp[Q*cord + 1] = f[Q*cord + 1];
				ftemp[Q*cord + 2] = f[Q*cord + 2];
				ftemp[Q*cord + 3] = f[Q*cord + 3];
				ftemp[Q*cord + 4] = f[Q*cord + 4];
				ftemp[Q*cord + 5] = f[Q*cord + 5];
				ftemp[Q*cord + 6] = f[Q*cord + 6];
				ftemp[Q*cord + 7] = f[Q*cord + 7];
				ftemp[Q*cord + 8] = f[Q*cord + 8];

			}
			else
				// set distributions to zero at solids
				for (int a = 0; a < Q; a++)
				{
					f[Q*cord + a] = 0;
					ftemp[Q*cord + a] = 0;
				}
		}



}