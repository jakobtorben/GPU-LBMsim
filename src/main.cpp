  
/* Driver code for setting up and running simulation on CPU
 * 
 * Filename: main.cpp
 * Author: Jakob Torben
 * Created: 04.06.2021
 * Last modified: 26.08.2021
 * 
 * This code is provided under the MIT license. See LICENSE.txt.
 */

#include <iostream>
#include <string>
#include <chrono>

#include "init.hpp"
#include "core.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char* argv[])
{
    // Read simulation inputs from file
	input_struct input;
    string inputfile = argv[1];
	read_input(inputfile, input);

    // initialise simulation parameters
    float u_lid, tau, omega;
    const int Nx = input.Nx;
    const int Ny = input.Ny;
    string fname;
    initialise_simulation(input, fname, u_lid, tau, omega);

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
	initialise_distributions(Nx, Ny, u_lid, f, rho_arr, ux_arr, uy_arr);

	// simulation main loop
	cout << "Running simulation...\n";
	auto start = std::chrono::system_clock::now();
	int out_cnt = 0;
	bool save = input.save;
	for (int it = 0; it < input.iterations; it++)
	{
		save = input.save && (it > input.printstart) && (it % input.printstep == 0);

        // streaming and collision step combined to one kernel
        stream_collide_gpu(Nx, Ny, rho_arr, ux_arr, uy_arr, u_lid, f, solid_node,
                           tau, omega, save, use_LES<les>(), use_MRT<mrt>());

		// write to file
		if (save)
		{
			cout << "iteration: " << it << "\toutput: " << out_cnt << endl;
			write_to_file(fname, out_cnt, ux_arr, uy_arr, u_lid, Nx, Ny);
			out_cnt++;
		}
	}

	timings(start, input);

	delete[] f;
	delete[] solid_node;
	delete[] ux_arr;
	delete[] uy_arr;
	delete[] rho_arr;

}