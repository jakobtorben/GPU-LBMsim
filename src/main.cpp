#include <iostream>
#include <string>
#include <math.h>
#include <chrono>

#include "core.hpp"
#include "utils.hpp"
#include "init.hpp"

using namespace std;



int main(int argc, char* argv[])
{
    // Read simulation inputs from file
    string inputfile;
	input_struct input;
    inputfile = argv[1];
	read_input(inputfile, input);

	const int Q = 9;			    // number of velocity components
	const int Nx = input.Nx;		// grid size x-direction
	const int Ny = input.Ny;		// grid size y-direction
    float cs = sqrt(1./3.);			// speed of sound**2 D2Q9
	// inital speed in x direction
	float ux0 = input.reynolds*input.kin_visc / float(Ny/4-1);  // Ny/4 is diameter of cylinder
    float mach = ux0 / cs;			// mach number
    float tau = (3. * input.kin_visc + 0.5); // collision timescale	

    // velocity components
    const int ex[Q] = { 0, 1, 0, -1,  0, 1, -1, -1,  1 };
    const int ey[Q] = { 0, 0, 1,  0, -1, 1,  1, -1, -1 };

	// print constants
	cout << "Nx: " << Nx << " Ny: " << Ny << endl;
	cout << "Reynolds number: " << input.reynolds << endl;
	cout << "kinematic viscosity: " << input.kin_visc << endl;
	cout << "ux0: " << ux0 << endl;
	cout << "mach number: " << mach << endl;
	cout << "tau : " << tau << endl;

    // allocate grid
    float* f          = new float[Nx * Ny * Q];
    float* ftemp      = new float[Nx * Ny * Q];
    float* feq        = new float[Nx * Ny * Q];

    bool*  solid_node = new bool [Nx * Ny];
    float* u_x        = new float[Nx * Ny];
    float* u_y        = new float[Nx * Ny];
    float* rho        = new float[Nx * Ny];


	// defines geometry
	read_geometry(Nx, Ny, solid_node);

	// apply initial conditions - flow to the rigth
	initialise(Nx, Ny, Q, ux0, f, ftemp, rho, u_x, u_y);

	// simulation main loop
	cout << "Running simulation...\n";
	auto start = std::chrono::system_clock::now();
	int it = 0, out_cnt = 0;
	while (it < input.iterations)
	{

		// streaming step
		stream(Nx, Ny, Q, ftemp, f, solid_node);

		// enforces bounadry conditions
		boundary(Nx, Ny, Q, ux0, ftemp, f, solid_node);

		// calculate macroscopic quantities
		calc_macro_quant(Nx, Ny, Q, u_x, u_y, rho, ftemp, solid_node, ex, ey);
		
		// calc equilibrium function
		calc_eq(Nx, Ny, Q, rho, u_x, u_y, solid_node, feq);

		// collision step
		collide(Nx, Ny, Q, f, ftemp, feq, solid_node, tau);

		// write to file
		if (input.save && it > input.iterations*0.8 && it % 100 == 0)
		{
			cout << "iteration: " << it << "\toutput: " << out_cnt << endl;
			write_to_file(out_cnt, u_x, u_y, Nx, Ny);
			out_cnt++;
		}
		it++;
	}

	timings(start, input);

	delete[] f;
	delete[] ftemp;
	delete[] feq;
	delete[] solid_node;
	delete[] u_x;
	delete[] u_y;
	delete[] rho;

}