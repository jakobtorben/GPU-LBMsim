#include <iostream>
#include <math.h>

#include "core.hpp"
#include "utils.hpp"
#include "init.hpp"

using namespace std;


// define constants
// dx = 1, dt = 1, c = 1 assumed throughout
const int Nx = 600, Ny = 300;	// grid size
const int Q = 9;			    // number of velocity components
const float reynolds = 100.;
const float kin_visc = 0.015;				// Kinematic viscosity
const float ux0 = reynolds*kin_visc / float(Ny-1); // inital speed in x direction
const float cs = sqrt(1./3.);	    // speed of sound**2 D2Q9
const float mach = ux0 / cs;
const float tau = (3. * kin_visc + 0.5); // collision timescale	
const int iterations = 15000;	// number of iteratinos to run



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

	// print constants
	cout << "Reynolds number: " << reynolds << endl;
	cout << "kinematic viscosity: " << kin_visc << endl;
	cout << "ux0: " << ux0 << endl;
	cout << "mach number: " << mach << endl;
	cout << "tau : " << tau << endl;


	// defines geometry
	read_geometry(Nx, Ny, solid_node);

	// apply initial conditions - flow to the rigth
	initialise(Nx, Ny, Q, ux0, f, ftemp, rho, u_x, u_y);


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


		if (it > 10000 && it % 100 == 0)
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