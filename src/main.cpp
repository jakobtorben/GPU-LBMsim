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

    // compile time options
    #ifndef LES
        #define LES 0
    #endif
    #ifndef MRT
        #define MRT 1
    #endif
    constexpr bool les = LES;
    constexpr bool mrt = MRT;

	const int Q = 9;			    // number of velocity components
	const int Nx = input.Nx;		// grid size x-direction
	const int Ny = input.Ny;		// grid size y-direction
    float cs = sqrt(1./3.);			// speed of sound**2 D2Q9
    float mach = 0.1;               // mach number
	float u_lid =  mach * cs;         // lid speed
    float kin_visc = u_lid * float(Nx-1) / input.reynolds; // Nx is length of slididng lid	
    float tau = (3. * kin_visc + 0.5); // collision timescale
    float omega = 1/tau;

	// print parameters
	cout << "Nx: " << Nx << " Ny: " << Ny << "\n";
	cout << "Boundary conditions: Lid driven cavity\n";
    cout << "Collision operator: ";
    if (mrt) cout << "MRT";
    else cout << "SRT";
    if (les) cout << "-LES\n";
    else cout << "\n";
    cout << "Reynolds number: " << input.reynolds << "\n";
	cout << "kinematic viscosity: " << kin_visc << "\n";
	cout << "u_lid: " << u_lid << "\n";
	cout << "mach number: " << mach << "\n";
	cout << "tau : " << tau << "\n\n";

    // allocate memory
    const size_t arr_size = sizeof(float)*Nx*Ny;
    const size_t f_size = sizeof(float)*Nx*Ny*Q;
    float* f          = new float[f_size];
    bool*  solid_node = new bool [arr_size];
    float* ux_arr        = new float[arr_size];
    float* uy_arr        = new float[arr_size];
    float* rho_arr        = new float[arr_size];


	// defines geometry
	define_geometry(Nx, Ny, solid_node);

	// apply initial conditions - lid moving to the right
	initialise_lid(Nx, Ny, Q, u_lid, f, rho_arr, ux_arr, uy_arr);

	// simulation main loop
	cout << "Running simulation...\n";
	auto start = std::chrono::system_clock::now();
	int it = 0, out_cnt = 0;
	bool save = input.save;
	while (it < input.iterations)
	{
		save = input.save && (it > input.printstart) && (it % input.printstep == 0);

        // streaming and collision step combined to one kernel
        stream_collide_gpu_lid(Nx, Ny, rho_arr, ux_arr, uy_arr, u_lid, f, solid_node, tau, omega, save, use_LES<les>(), use_MRT<mrt>());

		// write to file
		if (save)
		{
			cout << "iteration: " << it << "\toutput: " << out_cnt << endl;
			write_to_file(out_cnt, ux_arr, uy_arr, Nx, Ny);
			out_cnt++;
		}
		it++;
	}

	timings(start, input);

	delete[] f;
	delete[] solid_node;
	delete[] ux_arr;
	delete[] uy_arr;
	delete[] rho_arr;

}