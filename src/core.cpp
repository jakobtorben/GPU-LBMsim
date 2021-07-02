#include "core.hpp"
#include <iostream>  // delete later

// streaming step - periodic boundary conditions
void stream_perdiodic(int Nx, int Ny, int Q, float* ftemp, float* f, bool* solid_node)
{
	for (int j = 0; j < Ny; j++)
	{
		int jn = (j>0   ) ? (j-1) : (Ny-1);
		int jp = (j<Ny-1) ? (j+1) : (0   );

		for (int i = 0; i < Nx; i++)
		{
			int ij = i + Nx*j;
			int in = (i>0   ) ? (i-1) : (Nx-1);
			int ip = (i<Nx-1) ? (i+1) : (0   );
			// can later skip this for interiour nodes
			ftemp[Q*(i  + Nx*j     )] = f[Q*ij    ];
			ftemp[Q*(ip + Nx*j)  + 1] = f[Q*ij + 1];
			ftemp[Q*(i  + Nx*jp) + 2] = f[Q*ij + 2];
			ftemp[Q*(in + Nx*j)  + 3] = f[Q*ij + 3];
			ftemp[Q*(i  + Nx*jn) + 4] = f[Q*ij + 4];
			ftemp[Q*(ip + Nx*jp) + 5] = f[Q*ij + 5];
			ftemp[Q*(in + Nx*jp) + 6] = f[Q*ij + 6];
			ftemp[Q*(in + Nx*jn) + 7] = f[Q*ij + 7];
			ftemp[Q*(ip + Nx*jn) + 8] = f[Q*ij + 8];
		}
	}
}

// streaming step - without periodic boundary condittions
void stream(int Nx, int Ny, int Q, float* ftemp, float* f, bool* solid_node)
{
	for (int j = 0; j < Ny; j++)
	{
		// don't stream beyond boundary nodes
		int jn = (j>0) ? (j-1) : -1;
		int jp = (j<Ny-1) ? (j+1) : -1;

		for (int i = 0; i < Nx; i++)
		{
			// TODO: skip this for interiour nodes
			int ij = i + Nx*j;
			int in = (i>0) ? (i-1) : -1;
			int ip = (i<Nx-1) ? (i+1) : -1;

				                        ftemp[Q*(i  + Nx*j)     ] = f[Q*ij    ];
			if (ip != -1            ) ftemp[Q*(ip + Nx*j)  + 1] = f[Q*ij + 1];
			if (jp != -1            ) ftemp[Q*(i  + Nx*jp) + 2] = f[Q*ij + 2];
			if (in != -1            ) ftemp[Q*(in + Nx*j)  + 3] = f[Q*ij + 3];
			if (jn != -1            ) ftemp[Q*(i  + Nx*jn) + 4] = f[Q*ij + 4];
			if (ip != -1 && jp != -1) ftemp[Q*(ip + Nx*jp) + 5] = f[Q*ij + 5];
			if (in != -1 && jp != -1) ftemp[Q*(in + Nx*jp) + 6] = f[Q*ij + 6];
			if (in != -1 && jn != -1) ftemp[Q*(in + Nx*jn) + 7] = f[Q*ij + 7];
			if (ip != -1 && jn != -1) ftemp[Q*(ip + Nx*jn) + 8] = f[Q*ij + 8];
		}
	}
}

void boundary(int Nx, int Ny, int Q, float ux0, float* ftemp, float* f, bool* solid_node)
{
	// velocity BCs on west-side (inlet) using Zou and He.
	int i = 0;
	for (int j = 1; j < Ny - 1; j++)
	{
		int ij = i + Nx*j;
		float rho0 = (ftemp[Q*ij + 0] + ftemp[Q*ij + 2] + ftemp[Q*ij + 4]
			+ 2.*(ftemp[Q*ij + 3] + ftemp[Q*ij + 7] + ftemp[Q*ij + 6])) / (1. - ux0);
		float ru = rho0*ux0;
		ftemp[Q*ij + 1] = ftemp[Q*ij + 3] + (2./3.)*ru;
		ftemp[Q*ij + 5] = ftemp[Q*ij + 7] + (1./6.)*ru - 0.5*(ftemp[Q*ij + 2]-ftemp[Q*ij + 4]);
		ftemp[Q*ij + 8] = ftemp[Q*ij + 6] + (1./6.)*ru - 0.5*(ftemp[Q*ij + 4]-ftemp[Q*ij + 2]);
	}

	// BCs at east-side (outlet) using extrapolation from previous node (Nx-2) in x-dirn
	i = Nx-1;
	for (int j = 0; j < Ny; j++)
	{
		int ij = i + Nx*j;
		ftemp[Q*ij + 0] = ftemp[Q*ij - Q + 0];
		ftemp[Q*ij + 1] = ftemp[Q*ij - Q + 1];
		ftemp[Q*ij + 2] = ftemp[Q*ij - Q + 2];
		ftemp[Q*ij + 3] = ftemp[Q*ij - Q + 3];
		ftemp[Q*ij + 4] = ftemp[Q*ij - Q + 4];
		ftemp[Q*ij + 5] = ftemp[Q*ij - Q + 5];
		ftemp[Q*ij + 6] = ftemp[Q*ij - Q + 6];
		ftemp[Q*ij + 7] = ftemp[Q*ij - Q + 7];
		ftemp[Q*ij + 8] = ftemp[Q*ij - Q + 8];
	}

	// bounceback at top wall
	int j  = Ny - 1;
	for (int i = 1; i < Nx - 1; i++)
	{
		int ij = i + Nx*j;
		ftemp[Q*ij + 4] = ftemp[Q*ij + 2];
		ftemp[Q*ij + 7] = ftemp[Q*ij + 5];
		ftemp[Q*ij + 8] = ftemp[Q*ij + 6];
	}

	// bounceback at bottom wall
	j = 0;
	for (int i = 1; i < Nx - 1; i++)
	{
		int ij = i + Nx*j;
		ftemp[Q*ij + 2] = ftemp[Q*ij + 4];
		ftemp[Q*ij + 5] = ftemp[Q*ij + 7];
		ftemp[Q*ij + 6] = ftemp[Q*ij + 8];
	}

	// corners need special treatment as we have extra unknown.
	// Treatment based on Zou & He (1997), for further details see
	// palabos-forum.unige.ch/t/corner-nodes-2d-channel-boundary-condition-zou-he/577/5

	// corner of south-west inlet
	int ij = 0 + Nx*1; // extrapolate density from neighbour node
	float loc_rho = 0.0;
	for (int a = 0; a < Q; a++)
		loc_rho += ftemp[Q*ij + a];
	ij = 0 + Nx*0;
	ftemp[Q*ij + 1] = ftemp[Q*ij + 3];
	ftemp[Q*ij + 2] = ftemp[Q*ij + 4];
	ftemp[Q*ij + 5] = ftemp[Q*ij + 7];
	// f6 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f5 + f7))
	ftemp[Q*ij + 6] = 0.5*(loc_rho - ftemp[Q*ij]) - (ftemp[Q*ij + 1] + ftemp[Q*ij + 2] + ftemp[Q*ij + 5]);
	ftemp[Q*ij + 8] = ftemp[Q*ij + 6];


	// 	corner of south-east outlet
	ij = (Nx - 1) + Nx*1; //extrapolate neighbour density
	loc_rho = 0.0;
	for (int a = 0; a < Q; a++)
		loc_rho += ftemp[Q*ij + a];
	ij = (Nx-1) + Nx*0;
	ftemp[Q*ij + 2] = ftemp[Q*ij + 4];
	ftemp[Q*ij + 3] = ftemp[Q*ij + 1];
	ftemp[Q*ij + 6] = ftemp[Q*ij + 8];
	// f5 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f6 + f8))
	ftemp[Q*ij + 5] = 0.5*(loc_rho - ftemp[Q*ij]) - (ftemp[Q*ij + 2] + ftemp[Q*ij + 3] + ftemp[Q*ij + 6]);
	ftemp[Q*ij + 7] = ftemp[Q*ij + 5];


	// corner of north-west inlet
	ij = 0 + Nx*(Ny - 2);  // extrapolate neighbour density
	loc_rho = 0.0;
	for (int a = 0; a < Q; a++)
		loc_rho += ftemp[Q*ij + a];
	ij = 0 + Nx*(Ny - 1);
	ftemp[Q*ij + 1] = ftemp[Q*ij + 3];
	ftemp[Q*ij + 4] = ftemp[Q*ij + 2];
	ftemp[Q*ij + 8] = ftemp[Q*ij + 6];
	// f5 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f6 + f8))
	ftemp[Q*ij + 5] = 0.5*(loc_rho - ftemp[Q*ij]) - (ftemp[Q*ij + 2] + ftemp[Q*ij + 3] + ftemp[Q*ij + 6]);
	ftemp[Q*ij + 7] = ftemp[Q*ij + 5];


	// corner of north-east outlet
	ij = (Nx - 1) + Nx*(Ny - 2);  // extrapolate neighbour density
	loc_rho = 0.0;
	for (int a = 0; a < Q; a++)
		loc_rho += ftemp[Q*ij + a];
	ij = (Nx - 1) + Nx*(Ny - 1);
	ftemp[Q*ij + 3] = ftemp[Q*ij + 1];
	ftemp[Q*ij + 4] = ftemp[Q*ij + 2];
	ftemp[Q*ij + 7] = ftemp[Q*ij + 5];
	// f6 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f5 + f7))
	ftemp[Q*ij + 6] = 0.5*(loc_rho - ftemp[Q*ij]) - (ftemp[Q*ij + 3] + ftemp[Q*ij + 4] + ftemp[Q*ij + 7]);
	ftemp[Q*ij + 8] = ftemp[Q*ij + 6];


	// Apply standard bounceback at all inner solids (on-grid)
	for (int j = 1; j < Ny-1; j++)
		for (int i = 1; i < Nx-1; i++)
		{
			int ij = i + Nx*j;
			if (solid_node[ij])
			{
				f[Q*ij + 1] = ftemp[Q*ij + 3];
				f[Q*ij + 2] = ftemp[Q*ij + 4];
				f[Q*ij + 3] = ftemp[Q*ij + 1];
				f[Q*ij + 4] = ftemp[Q*ij + 2];
				f[Q*ij + 5] = ftemp[Q*ij + 7];
				f[Q*ij + 6] = ftemp[Q*ij + 8];
				f[Q*ij + 7] = ftemp[Q*ij + 5];
				f[Q*ij + 8] = ftemp[Q*ij + 6];
			}
		}
}

// calculate macroscopic quantities
void calc_macro_quant(int Nx, int Ny, int Q,
					  float* u_x, float* u_y,
					  float* rho, float* ftemp, bool* solid_node,
					  const int* ex, const int* ey)
{
	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			int ij = i + Nx*j;
			u_x[ij] = 0.0;
			u_y[ij] = 0.0;
			rho[ij] = 0.0;
			if (!solid_node[ij])
			{
				for (int a = 0; a < Q; a++)
				{
					u_x[ij] += ex[a] * ftemp[Q*ij + a];
					u_y[ij] += ey[a] * ftemp[Q*ij + a];
					rho[ij] +=         ftemp[Q*ij + a];
				}
				u_x[ij] /= rho[ij];
				u_y[ij] /= rho[ij];
			}
		}
}

// calculate equilibrium distribution feq
void calc_eq(int Nx, int Ny, int Q, float* rho, float* u_x, float* u_y, bool* solid_node, float* result)
{
	float c1 = 3.;
	float c2 = 9./2.;
	float c3 = 3./2.;
	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			int ij = i + Nx*j;
			if (!solid_node[ij])
			{
				float rhoij = rho[ij];
				float w_rho0 = 4./9.  * rhoij;
				float w_rho1 = 1./9.  * rhoij;
				float w_rho2 = 1./36. * rhoij;

				float uxij = u_x[ij];
				float uyij = u_y[ij];

				float uxsq = uxij * uxij;
				float uysq = uyij * uyij;
				float usq =  uxsq + uysq;

				float uxuy5 =  uxij + uyij;
				float uxuy6 = -uxij + uyij;
				float uxuy7 = -uxij - uyij;
				float uxuy8 =  uxij - uyij;

				float c4 = c3*usq;

				// note that c = 1
				result[Q*ij]     = w_rho0*(1.                             - c4);
				result[Q*ij + 1] = w_rho1*(1. + c1*uxij  + c2*uxsq        - c4);
				result[Q*ij + 2] = w_rho1*(1. + c1*uyij  + c2*uysq        - c4);
				result[Q*ij + 3] = w_rho1*(1. - c1*uxij  + c2*uxsq        - c4);
				result[Q*ij + 4] = w_rho1*(1. - c1*uyij  + c2*uysq        - c4);
				result[Q*ij + 5] = w_rho2*(1. + c1*uxuy5 + c2*uxuy5*uxuy5 - c4);
				result[Q*ij + 6] = w_rho2*(1. + c1*uxuy6 + c2*uxuy6*uxuy6 - c4);
				result[Q*ij + 7] = w_rho2*(1. + c1*uxuy7 + c2*uxuy7*uxuy7 - c4);
				result[Q*ij + 8] = w_rho2*(1. + c1*uxuy8 + c2*uxuy8*uxuy8 - c4);
			}
		}
} 

// collision step
void collide(int Nx, int Ny, int Q, float* f, float* ftemp, float* feq, bool* solid_node, float tau)
{
	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			int ij = i + Nx*j;
			if (!solid_node[ij])
			{
				for (int a = 0; a < Q; a++)
				{
					int ija = Q*ij + a;
					f[ija] = ftemp[ija] - (ftemp[ija] - feq[ija]) / tau;
				}
			}
		}
}