#include <iostream>
#include <math.h>

#include "core.h"
#include "utils.h"
#include "init.h"

using namespace std;


// define constants
const int Nx = 300, Ny = 150;	// grid size
const int Q = 9;			// number of velocity components 
const float tau = 0.6;			// collision timescale
const float ux0 = 1.;				// inital speed in x direction
//const float cs2 = 1./3.;	// speed of sound**2 D2Q9
const int iterations = 5000;		// number of iteratinos to run
// dx = 1, dt = 1, c = 1 assumed throughout



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


int main()
{
	// defines geometry
	read_geometry(Nx, Ny, solid_node);

	// apply initial conditions - flow to the rigth
	initialise(Nx, Ny, Q, f, ftemp, rho, u_x, u_y);


	// simulation main loop
	int it = 0, out_cnt = 0;
	while (it < iterations)
	{

		// streaming step - periodic boundary conditions
		stream(Nx, Ny, Q, ftemp, f, solid_node);

		// calculate macroscopic quantities
		calc_macro_quant(Nx, Ny, Q, u_x, u_y, rho, ftemp, solid_node, ex, ey);
		
		// calc equilibrium function
		calc_eq(Nx, Ny, Q, rho, u_x, u_y, solid_node, feq);

		// collision step
		collide(Nx, Ny, Q, f, ftemp, feq, solid_node, tau);


		if (it % 50 == 0)
		{
			cout << "iteration: " << it << "\toutput: " << out_cnt << endl;
			write_to_file(out_cnt, u_x, u_y, Nx, Ny);
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