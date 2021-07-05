#include "core.hpp"
#include <iostream>  // delete later

// streaming step - periodic boundary conditions
void stream_perdiodic(int Nx, int Ny, int Q, float* ftemp, float* f, bool* solid_node)
{
	for (int y = 0; y < Ny; y++)
	{
		int yn = (y>0   ) ? (y-1) : (Ny-1);
		int yp = (y<Ny-1) ? (y+1) : (0   );

		for (int x = 0; x < Nx; x++)
		{
			int cord = x + Nx*y;
			int xn = (x>0   ) ? (x-1) : (Nx-1);
			int xp = (x<Nx-1) ? (x+1) : (0   );
			// can later skip this for interiour nodes
			ftemp[Q*(x  + Nx*y     )] = f[Q*cord  ];
			ftemp[Q*(xp + Nx*y)  + 1] = f[Q*cord + 1];
			ftemp[Q*(x  + Nx*yp) + 2] = f[Q*cord + 2];
			ftemp[Q*(xn + Nx*y)  + 3] = f[Q*cord + 3];
			ftemp[Q*(x  + Nx*yn) + 4] = f[Q*cord + 4];
			ftemp[Q*(xp + Nx*yp) + 5] = f[Q*cord + 5];
			ftemp[Q*(xn + Nx*yp) + 6] = f[Q*cord + 6];
			ftemp[Q*(xn + Nx*yn) + 7] = f[Q*cord + 7];
			ftemp[Q*(xp + Nx*yn) + 8] = f[Q*cord + 8];
		}
	}
}

// streaming step - without periodic boundary condittions
void stream(int Nx, int Ny, int Q, float* ftemp, float* f, bool* solid_node)
{
	for (int y = 0; y < Ny; y++)
	{
		// don't stream beyond boundary nodes
		int yn = (y>0) ? (y-1) : -1;
		int yp = (y<Ny-1) ? (y+1) : -1;

		for (int x = 0; x < Nx; x++)
		{
			// TODO: skip this for interiour nodes
			int cord = x + Nx*y;
			int xn = (x>0) ? (x-1) : -1;
			int xp = (x<Nx-1) ? (x+1) : -1;

				                      ftemp[Q*(x  + Nx*y)     ] = f[Q*cord    ];
			if (xp != -1            ) ftemp[Q*(xp + Nx*y)  + 1] = f[Q*cord + 1];
			if (yp != -1            ) ftemp[Q*(x  + Nx*yp) + 2] = f[Q*cord + 2];
			if (xn != -1            ) ftemp[Q*(xn + Nx*y)  + 3] = f[Q*cord + 3];
			if (yn != -1            ) ftemp[Q*(x  + Nx*yn) + 4] = f[Q*cord + 4];
			if (xp != -1 && yp != -1) ftemp[Q*(xp + Nx*yp) + 5] = f[Q*cord + 5];
			if (xn != -1 && yp != -1) ftemp[Q*(xn + Nx*yp) + 6] = f[Q*cord + 6];
			if (xn != -1 && yn != -1) ftemp[Q*(xn + Nx*yn) + 7] = f[Q*cord + 7];
			if (xp != -1 && yn != -1) ftemp[Q*(xp + Nx*yn) + 8] = f[Q*cord + 8];
		}
	}
}

void boundary(int Nx, int Ny, int Q, float ux0, float* ftemp, float* f, bool* solid_node)
{
	// velocity BCs on west-side (inlet) using Zou and He.
	int x = 0;
	for (int y = 1; y < Ny - 1; y++)
	{
		int cord = x + Nx*y;
		float rho0 = (ftemp[Q*cord + 0] + ftemp[Q*cord + 2] + ftemp[Q*cord + 4]
			+ 2.*(ftemp[Q*cord + 3] + ftemp[Q*cord + 7] + ftemp[Q*cord + 6])) / (1. - ux0);
		float ru = rho0*ux0;
		ftemp[Q*cord + 1] = ftemp[Q*cord + 3] + (2./3.)*ru;
		ftemp[Q*cord + 5] = ftemp[Q*cord + 7] + (1./6.)*ru - 0.5*(ftemp[Q*cord + 2]-ftemp[Q*cord + 4]);
		ftemp[Q*cord + 8] = ftemp[Q*cord + 6] + (1./6.)*ru - 0.5*(ftemp[Q*cord + 4]-ftemp[Q*cord + 2]);
	}

	// BCs at east-side (outlet) using extrapolation from previous node (Nx-2) xn x-dirn
	x = Nx-1;
	for (int y = 0; y < Ny; y++)
	{
		int cord = x + Nx*y;
		ftemp[Q*cord + 0] = ftemp[Q*cord - Q + 0];
		ftemp[Q*cord + 1] = ftemp[Q*cord - Q + 1];
		ftemp[Q*cord + 2] = ftemp[Q*cord - Q + 2];
		ftemp[Q*cord + 3] = ftemp[Q*cord - Q + 3];
		ftemp[Q*cord + 4] = ftemp[Q*cord - Q + 4];
		ftemp[Q*cord + 5] = ftemp[Q*cord - Q + 5];
		ftemp[Q*cord + 6] = ftemp[Q*cord - Q + 6];
		ftemp[Q*cord + 7] = ftemp[Q*cord - Q + 7];
		ftemp[Q*cord + 8] = ftemp[Q*cord - Q + 8];
	}

	// bounceback at top wall
	int y  = Ny - 1;
	for (int x = 1; x < Nx - 1; x++)
	{
		int cord = x + Nx*y;
		ftemp[Q*cord + 4] = ftemp[Q*cord + 2];
		ftemp[Q*cord + 7] = ftemp[Q*cord + 5];
		ftemp[Q*cord + 8] = ftemp[Q*cord + 6];
	}

	// bounceback at bottom wall
	y = 0;
	for (int x = 1; x < Nx - 1; x++)
	{
		int cord = x + Nx*y;
		ftemp[Q*cord + 2] = ftemp[Q*cord + 4];
		ftemp[Q*cord + 5] = ftemp[Q*cord + 7];
		ftemp[Q*cord + 6] = ftemp[Q*cord + 8];
	}

	// corners need special treatment as we have extra unknown.
	// Treatment based on Zou & He (1997), for further details see
	// palabos-forum.unige.ch/t/corner-nodes-2d-channel-boundary-condition-zou-he/577/5

	// corner of south-west inlet
	int cord = 0 + Nx*1; // extrapolate density from neighbour node
	float loc_rho = 0.0;
	for (int a = 0; a < Q; a++)
		loc_rho += ftemp[Q*cord + a];
	cord = 0 + Nx*0;
	ftemp[Q*cord + 1] = ftemp[Q*cord + 3];
	ftemp[Q*cord + 2] = ftemp[Q*cord + 4];
	ftemp[Q*cord + 5] = ftemp[Q*cord + 7];
	// f6 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f5 + f7))
	ftemp[Q*cord + 6] = 0.5*(loc_rho - ftemp[Q*cord]) - (ftemp[Q*cord + 1] + ftemp[Q*cord + 2] + ftemp[Q*cord + 5]);
	ftemp[Q*cord + 8] = ftemp[Q*cord + 6];


	// 	corner of south-east outlet
	cord = (Nx - 1) + Nx*1; //extrapolate neighbour density
	loc_rho = 0.0;
	for (int a = 0; a < Q; a++)
		loc_rho += ftemp[Q*cord + a];
	cord = (Nx-1) + Nx*0;
	ftemp[Q*cord + 2] = ftemp[Q*cord + 4];
	ftemp[Q*cord + 3] = ftemp[Q*cord + 1];
	ftemp[Q*cord + 6] = ftemp[Q*cord + 8];
	// f5 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f6 + f8))
	ftemp[Q*cord + 5] = 0.5*(loc_rho - ftemp[Q*cord]) - (ftemp[Q*cord + 2] + ftemp[Q*cord + 3] + ftemp[Q*cord + 6]);
	ftemp[Q*cord + 7] = ftemp[Q*cord + 5];


	// corner of north-west inlet
	cord = 0 + Nx*(Ny - 2);  // extrapolate neighbour density
	loc_rho = 0.0;
	for (int a = 0; a < Q; a++)
		loc_rho += ftemp[Q*cord + a];
	cord = 0 + Nx*(Ny - 1);
	ftemp[Q*cord + 1] = ftemp[Q*cord + 3];
	ftemp[Q*cord + 4] = ftemp[Q*cord + 2];
	ftemp[Q*cord + 8] = ftemp[Q*cord + 6];
	// f5 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f6 + f8))
	ftemp[Q*cord + 5] = 0.5*(loc_rho - ftemp[Q*cord]) - (ftemp[Q*cord + 2] + ftemp[Q*cord + 3] + ftemp[Q*cord + 6]);
	ftemp[Q*cord + 7] = ftemp[Q*cord + 5];


	// corner of north-east outlet
	cord = (Nx - 1) + Nx*(Ny - 2);  // extrapolate neighbour density
	loc_rho = 0.0;
	for (int a = 0; a < Q; a++)
		loc_rho += ftemp[Q*cord + a];
	cord = (Nx - 1) + Nx*(Ny - 1);
	ftemp[Q*cord + 3] = ftemp[Q*cord + 1];
	ftemp[Q*cord + 4] = ftemp[Q*cord + 2];
	ftemp[Q*cord + 7] = ftemp[Q*cord + 5];
	// f6 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f5 + f7))
	ftemp[Q*cord + 6] = 0.5*(loc_rho - ftemp[Q*cord]) - (ftemp[Q*cord + 3] + ftemp[Q*cord + 4] + ftemp[Q*cord + 7]);
	ftemp[Q*cord + 8] = ftemp[Q*cord + 6];


	// Apply standard bounceback at all inner solids (on-grid)
	for (int y = 1; y < Ny-1; y++)
		for (int x = 1; x < Nx-1; x++)
		{
			int cord = x + Nx*y;
			if (solid_node[cord])
			{
				f[Q*cord + 1] = ftemp[Q*cord + 3];
				f[Q*cord + 2] = ftemp[Q*cord + 4];
				f[Q*cord + 3] = ftemp[Q*cord + 1];
				f[Q*cord + 4] = ftemp[Q*cord + 2];
				f[Q*cord + 5] = ftemp[Q*cord + 7];
				f[Q*cord + 6] = ftemp[Q*cord + 8];
				f[Q*cord + 7] = ftemp[Q*cord + 5];
				f[Q*cord + 8] = ftemp[Q*cord + 6];
			}
		}
}

// calculate macroscopic quantities
void calc_macro_quant(int Nx, int Ny, int Q,
					  float* u_x, float* u_y,
					  float* rho, float* ftemp, bool* solid_node,
					  const int* ex, const int* ey)
{
	for (int y = 0; y < Ny; y++)
		for (int x = 0; x < Nx; x++)
		{
			int cord = x + Nx*y;
			u_x[cord] = 0.0;
			u_y[cord] = 0.0;
			rho[cord] = 0.0;
			if (!solid_node[cord])
			{
				for (int a = 0; a < Q; a++)
				{
					u_x[cord] += ex[a] * ftemp[Q*cord + a];
					u_y[cord] += ey[a] * ftemp[Q*cord + a];
					rho[cord] +=         ftemp[Q*cord + a];
				}
				u_x[cord] /= rho[cord];
				u_y[cord] /= rho[cord];
			}
		}
}

// calculate equilibrium distribution feq
void calc_eq(int Nx, int Ny, int Q, float* rho, float* u_x, float* u_y, bool* solid_node, float* result)
{
	float c1 = 3.;
	float c2 = 9./2.;
	float c3 = 3./2.;
	for (int y = 0; y < Ny; y++)
		for (int x = 0; x < Nx; x++)
		{
			int cord = x + Nx*y;
			if (!solid_node[cord])
			{
				float rhoij = rho[cord];
				float w_rho0 = 4./9.  * rhoij;
				float w_rho1 = 1./9.  * rhoij;
				float w_rho2 = 1./36. * rhoij;

				float uxij = u_x[cord];
				float uyij = u_y[cord];

				float uxsq = uxij * uxij;
				float uysq = uyij * uyij;
				float usq =  uxsq + uysq;

				float uxuy5 =  uxij + uyij;
				float uxuy6 = -uxij + uyij;
				float uxuy7 = -uxij - uyij;
				float uxuy8 =  uxij - uyij;

				float c4 = c3*usq;

				// note that c = 1
				result[Q*cord]     = w_rho0*(1.                             - c4);
				result[Q*cord + 1] = w_rho1*(1. + c1*uxij  + c2*uxsq        - c4);
				result[Q*cord + 2] = w_rho1*(1. + c1*uyij  + c2*uysq        - c4);
				result[Q*cord + 3] = w_rho1*(1. - c1*uxij  + c2*uxsq        - c4);
				result[Q*cord + 4] = w_rho1*(1. - c1*uyij  + c2*uysq        - c4);
				result[Q*cord + 5] = w_rho2*(1. + c1*uxuy5 + c2*uxuy5*uxuy5 - c4);
				result[Q*cord + 6] = w_rho2*(1. + c1*uxuy6 + c2*uxuy6*uxuy6 - c4);
				result[Q*cord + 7] = w_rho2*(1. + c1*uxuy7 + c2*uxuy7*uxuy7 - c4);
				result[Q*cord + 8] = w_rho2*(1. + c1*uxuy8 + c2*uxuy8*uxuy8 - c4);
			}
		}
} 

// collision step
void collide(int Nx, int Ny, int Q, float* f, float* ftemp, float* feq, bool* solid_node, float tau)
{
	for (int y = 0; y < Ny; y++)
		for (int x = 0; x < Nx; x++)
		{
			int cord = x + Nx*y;
			if (!solid_node[cord])
			{
				for (int a = 0; a < Q; a++)
				{
					int xya = Q*cord + a;
					f[xya] = ftemp[xya] - (ftemp[xya] - feq[xya]) / tau;
				}
			}
		}
}