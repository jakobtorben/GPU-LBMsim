#pragma once

// define struct to store simulations inputs
struct input_struct
{
	int Nx;
	int Ny;
	float reynolds;
	float kin_visc;
	int iterations;
	int printstep;
	bool save;

};

__host__ void write_to_file(int it, float* u_x, float* u_y, int Nx, int Ny);
__host__ void read_input(std::string fname, input_struct& input);
__hosy__ void timings(std::chrono::time_point<std::chrono::system_clock> start, input_struct input);