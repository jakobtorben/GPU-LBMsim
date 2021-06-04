#include <iostream>
#include <math.h>

#include "include/utils.h"

using namespace std;

#define PI 3.14159265



// define constants
const int Nx = 100, Ny = 100;	// grid size
const int Q = 9;			// number of velocity components 
float tau = 0.6;			// collision timescale
int iterations = 10;		// number of iteratinos to run
// dx = 1, dt = 1, c_s = 1 assumed throughout

// define geometry
int cx = 50, cy = 50;
int radius = 20;

// velocity components
int ex[Q] = { 0, 1, 0, -1,  0, 1, -1, -1,  1 };
int ey[Q] = { 0, 0, 1,  0, -1, 1,  1, -1, -1 };

// allocate grid

float* f          = new float[Nx * Ny * Q];
float* ftemp      = new float[Nx * Ny * Q];
float* feq        = new float[Nx * Ny * Q];

bool*  solid_node = new bool [Nx * Ny];
float* u_x        = new float[Nx * Ny];
float* u_y        = new float[Nx * Ny];
float* rho        = new float[Nx * Ny];


int main()
{
	// define geometry - cylinder with r = 5
	for (int i = 0; i < Nx; i++)
		for (int j = 0; j < Ny; j++)
		{
			float dx = std::abs(cx - i);
			float dy = std::abs(cy - j);
			float dist = std::sqrt(dx*dx + dy*dy);
			solid_node[i + Ny*j] = (dist < radius) ? 1 : 0;
		}


	// apply initial conditions - flow to the rigth
	for (int i = 0; i < Nx; i++)
		for (int j = 0; j < Ny; j++)
			for (int a = 0; a < 9; a++)
			{
				f[i + Ny*(j + Q*a)] = (a != 1) ? 1.0 + 0.01*rand() : 2.0 * (1.0 + 0.2*cos(2*PI*i/Nx*4));
				ftemp[i + Ny*(j + Q*a)] = 0.0;  // f[i + Ny*(j + Q*a)];
			}


	// simulation main loop
	int it = 0, out_cnt = 0;
	while (it < iterations)
	{
		// calculate macroscopic quantities
		for (int i = 0; i < Nx; i++)
			for (int j = 0; j < Ny; j++)
			{
				u_x[i + Ny*j] = 0.0;
				u_y[i + Ny*j] = 0.0;
				rho[i + Ny*j] = 0.0;
				for (int a = 0; a < Q; a++)
				{
					u_x[i + Ny*j] += ex[a] * f[i + Ny*(j + Q*a)];
					u_y[i + Ny*j] += ey[a] * f[i + Ny*(j + Q*a)];
					rho[i + Ny*j] +=         f[i + Ny*(j + Q*a)];
				}
				u_x[i + Ny*j] /= rho[i + Ny*j];
				u_y[i + Ny*j] /= rho[i + Ny*j];
			}
		
		// streaming step - periodic boundary conditions
		for (int i = 0; i < Nx; i++)
		{
			int in = (i>0   ) ? (i-1) : (Nx-1);
			int ip = (i<Nx-1) ? (i+1) : (0   );

			for (int j = 0; j < Ny; j++)
			{
				if (!solid_node[i + Ny*j])
				{
					int jn = (j>0) ? (j-1) : (Ny-1);
					int jp = (j<Ny-1) ? (j+1) : (Ny-1);

					ftemp[i  + Ny*(j     )] = f[i + Ny*(j    )];
					ftemp[ip + Ny*(j  + 1)] = f[i + Ny*(j + 1)];
					ftemp[i  + Ny*(jp + 2)] = f[i + Ny*(j + 2)];
					ftemp[in + Ny*(j  + 3)] = f[i + Ny*(j + 3)];
					ftemp[i  + Ny*(jn + 4)] = f[i + Ny*(j + 4)];
					ftemp[ip + Ny*(jp + 5)] = f[i + Ny*(j + 5)];
					ftemp[in + Ny*(jp + 6)] = f[i + Ny*(j + 6)];
					ftemp[in + Ny*(jn + 7)] = f[i + Ny*(j + 7)];
					ftemp[ip + Ny*(jn + 8)] = f[i + Ny*(j + 8)];
				}
				else
				{
					// bounceback boundary conditions
					float temp;
					temp = f[i + Ny*(j + 1)]; f[i + Ny*(j + 1)] = f[i + Ny*(j + 3)]; f[i + Ny*(j + 3)] = temp;
					temp = f[i + Ny*(j + 2)]; f[i + Ny*(j + 2)] = f[i + Ny*(j + 4)]; f[i + Ny*(j + 4)] = temp;
					temp = f[i + Ny*(j + 5)]; f[i + Ny*(j + 5)] = f[i + Ny*(j + 7)]; f[i + Ny*(j + 7)] = temp;
					temp = f[i + Ny*(j + 6)]; f[i + Ny*(j + 6)] = f[i + Ny*(j + 8)]; f[i + Ny*(j + 8)] = temp;
				}
			}
		}

		// calculate equilibrium distribution feq
		float c1 = 3.0;
		float c2 = 9./2.;
		float c3 = 3./2.;
		for (int i = 0; i < Nx; i++)
			for (int j = 0; j < Ny; j++)
			{
				if (!solid_node[i + Ny*j])
				{
					float rhoij = rho[i + Ny*j];
					float w_rho0 = 4./9.  * rhoij;
					float w_rho1 = 1./9.  * rhoij;
					float w_rho2 = 1./36. * rhoij;

					float uxij = u_x[i + Ny*j];
					float uyij = u_y[i + Ny*j];

					float uxsq = uxij * uxij;
					float uysq = uyij * uyij;
					float usq = uxsq + uysq;

					float uxuy5 =  uxij + uyij;
					float uxuy6 = -uxij + uyij;
					float uxuy7 = -uxij - uyij;
					float uxuy8 =  uxij - uyij;

					float c4 = c3*usq;

					feq[i + Ny*(j)      ] = w_rho0*(1.                             - c4);
					feq[i + Ny*(j + Q*1)] = w_rho1*(1. + c1*uxij  + c2*uxsq        - c4);
					feq[i + Ny*(j + Q*2)] = w_rho1*(1. + c1*uyij  + c2*uysq        - c4);
					feq[i + Ny*(j + Q*3)] = w_rho1*(1. - c1*uxij  + c2*uxsq        - c4);
					feq[i + Ny*(j + Q*4)] = w_rho1*(1. - c1*uyij  + c2*uysq        - c4);
					feq[i + Ny*(j + Q*5)] = w_rho2*(1. + c1*uxuy5 + c2*uxuy5*uxuy5 - c4);
					feq[i + Ny*(j + Q*6)] = w_rho2*(1. + c1*uxuy6 + c2*uxuy6*uxuy6 - c4);
					feq[i + Ny*(j + Q*7)] = w_rho2*(1. + c1*uxuy7 + c2*uxuy7*uxuy7 - c4);
					feq[i + Ny*(j + Q*8)] = w_rho2*(1. + c1*uxuy8 + c2*uxuy8*uxuy8 - c4);
				}
			}

		// collision step
		for (int i = 0; i < Nx; i++)
			for (int j = 0; j < Ny; j++)
				if (!solid_node[i + Ny*j])
					for (int a = 0; a < 9; a++)
					{
						int ija = i + Ny*(j + Q*a);
						f[ija] = ftemp[ija] - (ftemp[ija] - feq[ija]) / tau;
					}

		if (it % 1 == 0)
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