#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <chrono>
#include <string>

#include "utils.hpp"

/***********************************************************
*  **Function**: grid_to_file\n
*  **Description**: Saves the grid to a VTK datafile
*  **Param**: int out
************************************************************/
void write_to_file(int it, float* u_x, float* u_y, int Nx, int Ny)
{
	std::stringstream fname;
	std::fstream f1;
	fname << "./out/" << "output" << "_" << it << ".vtk";
	f1.open(fname.str().c_str(), std::ios_base::out);

	// write header
	f1 << "# vtk DataFile Version 5.1\n";
	f1 << "Lattice_Boltzmann_fluid_flow\n";
	f1 << "ASCII\n";
	f1 << "DATASET RECTILINEAR_GRID\n";
	f1 << "DIMENSIONS " << Nx << " " << Ny << " 1 \n";
	f1 << "X_COORDINATES " << Nx << " float\n";
	for (int x = 0; x < Nx; x++)
		f1 << x << " ";
	f1 << "\nY_COORDINATES " << Ny << " float\n";
	for (int y = 0; y < Ny; y++)
		f1 << y << " ";
	f1 << "\nZ_COORDINATES 1 float\n";
	f1 << "0\n";
	f1 << "POINT_DATA " << Nx*Ny << "\n\n";
	f1 << "VECTORS VecVelocity float\n";

	for (int y = 0; y < Ny; y++)
	{
		for (int x = 0; x < Nx; x++)
		{
			int cord = x + Nx*y;
			f1 << u_x[cord] << " " << u_y[cord] << " 0.0 \n";
		}
	}
	f1.close();
}

// read simulation inputs from file
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

void timings(std::chrono::time_point<std::chrono::system_clock> start, input_struct input)
{
	auto end = std::chrono::system_clock::now();
	double runtime = std::chrono::duration_cast<
		std::chrono::duration<double>>(end - start).count();
	
	size_t node_updates = input.iterations * size_t(input.Nx * input.Ny);
	
	// calculate million lattice updates per second
	double updates = 1e-6 * node_updates / runtime;

	// calculate memory bandwidth xn GiB/s
	int Q = 9;
	double GiB = 1024. * 1024. * 1024.;
	// bandwidth has one read and one write for each distributuion value
	double bandwidth = node_updates * 2 * Q * sizeof(float) / (runtime*GiB);

	std::cout << "Elapsed runtime (s): " << runtime << '\n';
	std::cout << "Lattice updates per second (Mlups): " << updates << "\n";
	std::cout << "Memory bandwidth (GiB/s): " << bandwidth << '\n';
}
