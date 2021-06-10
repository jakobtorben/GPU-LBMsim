#include "core.hpp"

// streaming step - periodic boundary conditions
void stream(int Nx, int Ny, int Q, float* ftemp, float* f, bool* solid_node)
{
	for (int j = 0; j < Ny; j++)
	{
		int jn = (j>0   ) ? (j-1) : (Ny-1);
		int jp = (j<Ny-1) ? (j+1) : (0   );

		for (int i = 0; i < Nx; i++)
		{
			int pos = i + Nx*j;
			if (!solid_node[pos])
			{
				int in = (i>0   ) ? (i-1) : (Nx-1);
				int ip = (i<Nx-1) ? (i+1) : (0   );

				ftemp[Q*(i  + Nx*j     )] = f[Q*pos    ];
				ftemp[Q*(ip + Nx*j)  + 1] = f[Q*pos + 1];
				ftemp[Q*(i  + Nx*jp) + 2] = f[Q*pos + 2];
				ftemp[Q*(in + Nx*j)  + 3] = f[Q*pos + 3];
				ftemp[Q*(i  + Nx*jn) + 4] = f[Q*pos + 4];
				ftemp[Q*(ip + Nx*jp) + 5] = f[Q*pos + 5];
				ftemp[Q*(in + Nx*jp) + 6] = f[Q*pos + 6];
				ftemp[Q*(in + Nx*jn) + 7] = f[Q*pos + 7];
				ftemp[Q*(ip + Nx*jn) + 8] = f[Q*pos + 8];
			}
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
			int pos = i + Nx*j;
			u_x[pos] = 0.0;
			u_y[pos] = 0.0;
			rho[pos] = 0.0;
			if (!solid_node[pos])
			{
				for (int a = 0; a < Q; a++)
				{
					u_x[pos] += ex[a] * ftemp[Q*pos + a];
					u_y[pos] += ey[a] * ftemp[Q*pos + a];
					rho[pos] +=         ftemp[Q*pos + a];
				}
				u_x[pos] /= rho[pos];
				u_y[pos] /= rho[pos];
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
			int pos = i + Nx*j;
			if (!solid_node[pos])
			{
				float rhoij = rho[pos];
				float w_rho0 = 4./9.  * rhoij;
				float w_rho1 = 1./9.  * rhoij;
				float w_rho2 = 1./36. * rhoij;

				float uxij = u_x[pos];
				float uyij = u_y[pos];

				float uxsq = uxij * uxij;
				float uysq = uyij * uyij;
				float usq =  uxsq + uysq;

				float uxuy5 =  uxij + uyij;
				float uxuy6 = -uxij + uyij;
				float uxuy7 = -uxij - uyij;
				float uxuy8 =  uxij - uyij;

				float c4 = c3*usq;

				// note that c = 1
				result[Q*pos]     = w_rho0*(1.                             - c4);
				result[Q*pos + 1] = w_rho1*(1. + c1*uxij  + c2*uxsq        - c4);
				result[Q*pos + 2] = w_rho1*(1. + c1*uyij  + c2*uysq        - c4);
				result[Q*pos + 3] = w_rho1*(1. - c1*uxij  + c2*uxsq        - c4);
				result[Q*pos + 4] = w_rho1*(1. - c1*uyij  + c2*uysq        - c4);
				result[Q*pos + 5] = w_rho2*(1. + c1*uxuy5 + c2*uxuy5*uxuy5 - c4);
				result[Q*pos + 6] = w_rho2*(1. + c1*uxuy6 + c2*uxuy6*uxuy6 - c4);
				result[Q*pos + 7] = w_rho2*(1. + c1*uxuy7 + c2*uxuy7*uxuy7 - c4);
				result[Q*pos + 8] = w_rho2*(1. + c1*uxuy8 + c2*uxuy8*uxuy8 - c4);
			}
		}
}

// collision step
void collide(int Nx, int Ny, int Q, float* f, float* ftemp, float* feq, bool* solid_node, float tau)
{
	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			int pos = i + Nx*j;
			if (!solid_node[pos])
			{
				for (int a = 0; a < Q; a++)
				{
					int ija = Q*pos + a;
					f[ija] = ftemp[ija] - (ftemp[ija] - feq[ija]) / tau;
				}
			}
			else
			{
				// bounceback boundary conditions
				f[Q*pos + 1] = ftemp[Q*pos + 3];
				f[Q*pos + 2] = ftemp[Q*pos + 4];
				f[Q*pos + 3] = ftemp[Q*pos + 1];
				f[Q*pos + 4] = ftemp[Q*pos + 2];
				f[Q*pos + 5] = ftemp[Q*pos + 7];
				f[Q*pos + 6] = ftemp[Q*pos + 8];
				f[Q*pos + 7] = ftemp[Q*pos + 5];
				f[Q*pos + 8] = ftemp[Q*pos + 6];
			}
		}
}