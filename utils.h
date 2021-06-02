#pragma once

#include <sstream>
#include <fstream>

/***********************************************************
*  **Function**: setup_continuous_array\n
*  **Description**: Sets up arrays that will be used for
*                   for grids, using a double pointer that
*                   points to a 1D array.
*  **Param**: double**& array_2d, double*& array_1d, int m, int n
************************************************************/
void setup_continuous_array(double**& array_2d, double*& array_1d, int m, int n)
{
	array_1d = new double[m * n];
	array_2d = new double* [m];

	for (int i = 0; i < m; i++)
	{
		array_2d[i] = &array_1d[i * n];						//index = i*n + j
		for (int j = 0; j < n; j++)  // remove later when setup ghost cells
			array_2d[i][j] = 0;
	}
}

/***********************************************************
*  **Function**: free_continuous_array\n
*  **Description**: Deletes dynamically allocated arrays
*  **Param**: double**& array_2d, double*& array_1d
************************************************************/
void free_continuous_array(double**& array_2d, double*& array_1d)
{
	delete[] array_1d;
	delete[] array_2d;
}


/***********************************************************
*  **Function**: grid_to_file\n
*  **Description**: Saves the grid to a datafile
*  **Param**: int out
************************************************************/
void grid_to_file(int it, double** grid, int rows, int columns)
{
	std::stringstream fname;
	std::fstream f1;
	fname << "./out/" << "output" << "_" << it << ".dat";
	f1.open(fname.str().c_str(), std::ios_base::out);
	for (int i = 1; i < rows + 1; i++)
	{
		for (int j = 1; j < columns + 1; j++)
			f1 << grid[i][j] << '\t';
		f1 << '\n';
	}
	f1.close();
}