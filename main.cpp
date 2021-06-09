#include <iostream>
#include <math.h>

#include "include/utils.h"

using namespace std;


// define constants
const int Nx = 300, Ny = 150;	// grid size
const int Q = 9;			// number of velocity components 
const float tau = 0.6;			// collision timescale
const float ux0 = 1.;				// inital speed in x direction
//const float cs2 = 1./3.;	// speed of sound**2 D2Q9
const int iterations = 5000;		// number of iteratinos to run
// dx = 1, dt = 1, c = 1 assumed throughout

// define geometry
const int cx = 75, cy = 75;
const int radius = 50;

// velocity components
const int ex[Q] = { 0, 1, 0, -1,  0, 1, -1, -1,  1 };
const int ey[Q] = { 0, 0, 1,  0, -1, 1,  1, -1, -1 };

// allocate grid
float* f          = new float[Nx * Ny * Q];
float* ftemp      = new float[Nx * Ny * Q];
float* feq        = new float[Nx * Ny * Q];

bool*  solid_node = new bool [Nx * Ny];
float* u_x        = new float[Nx * Ny];
float* u_y        = new float[Nx * Ny];
float* rho        = new float[Nx * Ny];

// define functions
void calc_eq(float* rho, float* u_x, float* u_y, float* result)
{
	// calculate equilibrium distribution feq
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
				float usq = uxsq + uysq;

				float uxuy5 = uxij + uyij;
				float uxuy6 = -uxij + uyij;
				float uxuy7 = -uxij - uyij;
				float uxuy8 = uxij - uyij;

				float c4 = c3*usq;

				// note that c = 1
				result[Q*pos    ] = w_rho0*(1.                             - c4);
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


int main()
{
	// define geometry
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

	// apply initial conditions - flow to the rigth
	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++)
		{
			int pos = i + Nx*j;
			rho[pos] = 1.;
			u_x[pos] = 1.;
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

	// simulation main loop
	int it = 0, out_cnt = 0;
	while (it < iterations)
	{

		// streaming step - periodic boundary conditions
		for (int j = 0; j < Ny; j++)
		{
			int jn = (j>0) ? (j-1) : (Ny-1);
			int jp = (j<Ny-1) ? (j+1) : (0);

			for (int i = 0; i < Nx; i++)
			{
				int pos = i + Nx*j;
				if (!solid_node[pos])
				{
					int in = (i>0) ? (i-1) : (Nx-1);
					int ip = (i<Nx-1) ? (i+1) : (0);

					ftemp[Q*(i  + Nx*j )    ] = f[Q*pos    ];
					ftemp[Q*(ip + Nx*j ) + 1] = f[Q*pos + 1];
					ftemp[Q*(i  + Nx*jp) + 2] = f[Q*pos + 2];
					ftemp[Q*(in + Nx*j ) + 3] = f[Q*pos + 3];
					ftemp[Q*(i  + Nx*jn) + 4] = f[Q*pos + 4];
					ftemp[Q*(ip + Nx*jp) + 5] = f[Q*pos + 5];
					ftemp[Q*(in + Nx*jp) + 6] = f[Q*pos + 6];
					ftemp[Q*(in + Nx*jn) + 7] = f[Q*pos + 7];
					ftemp[Q*(ip + Nx*jn) + 8] = f[Q*pos + 8];
				}
			}
		}

		// calculate macroscopic quantities
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
		
		// calc equilibrium function
		calc_eq(rho, u_x, u_y, feq);

		// collision step
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

		if (it % 50 == 0)
		{
			cout << "iteration: " << it << "\toutput: " << out_cnt << endl;
			grid_to_file(out_cnt, u_x, u_y, Nx, Ny);
			out_cnt++;
		}
		it++;
	}

	delete[] f;
	delete[] ftemp;
	delete[] feq;
	delete[] solid_node;
	delete[] u_x;
	delete[] u_y;
	delete[] rho;

}