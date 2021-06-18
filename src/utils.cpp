#include <sstream>
#include <fstream>
#include <math.h>

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
	for (int i = 0; i < Nx; i++)
		f1 << i << " ";
	f1 << "\nY_COORDINATES " << Ny << " float\n";
	for (int j = 0; j < Ny; j++)
		f1 << j << " ";
	f1 << "\nZ_COORDINATES 1 float\n";
	f1 << "0\n";
	f1 << "POINT_DATA " << Nx*Ny << "\n\n";
	f1 << "VECTORS VecVelocity float\n";

	for (int j = 0; j < Ny; j++)
	{
		for (int i = 0; i < Nx; i++)
		{
			int pos = i + Nx*j;
			f1 << u_x[pos] << " " << u_y[pos] << " 0.0 \n";
		}
	}
	f1.close();
}