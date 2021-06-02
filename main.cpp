#include <iostream>

#include "LBM.h"
#include "utils.h"


using namespace std;
double** grid, * grid_1d;
double** old_grid, * old_grid_1d;
double** new_grid, * new_grid_1d;


int rows, columns;							// size of whole domain

// define constants
int imax = 5, jmax = 5;
double t_max = 10.0;
double t = 0.0, t_out = 0.0, dt_out = 0.04, dt;
double y_max = 10.0, x_max = 10.0, dx, dy;

int main()
{
	// allocate grid padded with ghost cells
	setup_continuous_array(grid, grid_1d, rows + 2, columns + 2);
	setup_continuous_array(old_grid, old_grid_1d, rows + 2, columns + 2);
	setup_continuous_array(new_grid, new_grid_1d, rows + 2, columns + 2);

	dx = x_max / ((double)imax - 1);
	dy = y_max / ((double)jmax - 1);

	dt = 0.1 * min(dx, dy);

	int out_cnt = 0, it = 0;

	while (t < t_max)
	{
		iteration();



		t += dt;
		std::swap(old_grid, new_grid);
		std::swap(old_grid, grid);


		if (t_out <= t)
		{
			cout << "output: " << out_cnt << "\tt: " << t << "\titeration: " << it << endl;
			grid_to_file(out_cnt, new_grid, rows, columns);
			out_cnt++;
			t_out += dt_out;
		}
		it++;
	}


	free_continuous_array(grid, grid_1d);
	free_continuous_array(old_grid, old_grid_1d);
	free_continuous_array(new_grid, new_grid_1d);


}