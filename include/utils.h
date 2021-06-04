#pragma once

#include <sstream>
#include <fstream>
#include <math.h>


/***********************************************************
*  **Function**: grid_to_file\n
*  **Description**: Saves the grid to a datafile
*  **Param**: int out
************************************************************/
void grid_to_file(int it, float* u_x, float* u_y, int rows, int columns)
{
	std::stringstream fname;
	std::fstream f1;
	fname << "./out/" << "output" << "_" << it << ".dat";
	f1.open(fname.str().c_str(), std::ios_base::out);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			float vel = std::sqrt(u_x[i + columns*j]*u_x[i + columns*j] + u_y[i + columns*j]*u_y[i + columns*j]);
			f1 << vel  << '\t';
		}
		f1 << '\n';
	}
	f1.close();
}
