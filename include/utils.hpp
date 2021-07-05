#pragma once

// define struct to store simulations inputs
struct input_struct
{
	int Nx;
	int Ny;
	float reynolds;
	float kin_visc;
	int iterations;
	bool save;

};

void write_to_file(int it, float* u_x, float* u_y, int Nx, int Ny);
void read_input(std::string fname, input_struct& input);
void timings(std::chrono::time_point<std::chrono::system_clock> start, input_struct input);