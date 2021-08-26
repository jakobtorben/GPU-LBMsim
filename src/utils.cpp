/* Utility methods used to run simulations.
 * 
 * Filename: utils.cpp
 * Author: Jakob Torben
 * Created: 04.06.2021
 * Last modified: 26.08.2021
 * 
 * This code is provided under the MIT license. See LICENSE.txt.
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <string>

#include "utils.hpp"
#include "core.hpp"

/**
 * Save velocities in physical units to a VTK datafile.
 * 
 * @param[in] fname filename
 * @param[in] out_cnt output number
 * @param[in] ux_arr, uy_arr arrays storing the velocities
 * @param[in] u_lid, lid velocity
 * @param[in] Nx, Ny domain size
 */
void write_to_file(std::string fname, int out_cnt, float* u_x, float* u_y, float u_lid, int Nx, int Ny)
{
    // transform from lattice units to physical units
    // physical domain lenght = 1
    // reference velocity u_lid = 1
    float dx = 1. / (Nx - 1);
    float dy = 1. / (Ny - 1);

	std::stringstream outputname;
	std::fstream f1;
	outputname << "./out/" << fname << "_it" << out_cnt << ".vtk";
	f1.open(outputname.str().c_str(), std::ios_base::out);

	// write header
	f1 << "# vtk DataFile Version 5.1\n";
	f1 << "Lattice_Boltzmann_fluid_flow\n";
	f1 << "ASCII\n";
	f1 << "DATASET RECTILINEAR_GRID\n";
	f1 << "DIMENSIONS " << Nx << " " << Ny << " 1 \n";
	f1 << "X_COORDINATES " << Nx << " float\n";
	for (int x = 0; x < Nx; x++)
		f1 << x*dx << " ";
	f1 << "\nY_COORDINATES " << Ny << " float\n";
	for (int y = 0; y < Ny; y++)
		f1 << y*dy << " ";
	f1 << "\nZ_COORDINATES 1 float\n";
	f1 << "0\n";
	f1 << "POINT_DATA " << Nx*Ny << "\n\n";
	f1 << "VECTORS VecVelocity float\n";

    // write velocities in physical units
	for (int y = 0; y < Ny; y++)
		for (int x = 0; x < Nx; x++)
			f1 << u_x[arr_idx(Nx, x, y)]/u_lid << " " << u_y[arr_idx(Nx, x, y)]/u_lid << " 0.0 \n";
	f1.close();
}

/**
 * Read simulation inputs from inputfile and store to input struct
 *
 * @param[in] fname filename of inputfile
 * @param[out] input struct stroing the parameters
 */
void read_input(std::string fname, input_struct& input)
{
	std::fstream file(fname, std::ios_base::in);
	std::string variable;  // to read varaible name
	std::cout << "Reading inputs\n";
	while (file >> variable)
	{
		if (variable == "Nx")         file >> input.Nx;
		if (variable == "Ny")         file >> input.Ny;
		if (variable == "reynolds")   file >> input.reynolds;
		if (variable == "iterations") file >> input.iterations;
		if (variable == "printstart") file >> input.printstart;
        if (variable == "printstep")  file >> input.printstep;
		if (variable == "save")		  file >> input.save;
	}
	file.close();
}

/**
 * Record time of simulation and calculate performance metrics.
 *
 * @param[in] start starttime of simulation
 * @param[in] input parameters from inputfile
 */
void timings(std::chrono::time_point<std::chrono::system_clock> start, input_struct input)
{
	auto end = std::chrono::system_clock::now();
	double runtime = std::chrono::duration_cast<
		std::chrono::duration<double>>(end - start).count();
	
	size_t node_updates = input.iterations * size_t(input.Nx * input.Ny);
	
	// calculate million lattice updates per second
	double updates = 1e-6 * node_updates / runtime;

	// calculate memory bandwidth xn GiB/s
	double GB = 1e9;
	// bandwidth has one read and one write for each distributuion value
	double bandwidth = node_updates * 2 * Q * sizeof(float) / (runtime*GB);

	std::cout << "\nElapsed runtime (s): " << runtime << '\n';
	std::cout << "Lattice updates per second (Mlups): " << updates << "\n";
	std::cout << "Memory bandwidth (GB/s): " << bandwidth << '\n';
}