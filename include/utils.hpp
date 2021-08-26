/* Defines struct to store input, and declare utils methods.
 * 
 * Filename: utils.hpp
 * Author: Jakob Torben
 * Created: 04.06.2021
 * Last modified: 26.08.2021
 * 
 * This code is provided under the MIT license. See LICENSE.txt.
 */

#pragma once

#include <chrono>
#include <string>

/** 
 * Struct to store simulations inputs
 */
struct input_struct
{
	int Nx;
	int Ny;
	float reynolds;
	int iterations;
    int printstart;
	int printstep;
	bool save;
};

// function declarations
void write_to_file(std::string fname, int it, float* u_x, float* u_y, float u_lid, int Nx, int Ny);
void read_input(std::string fname, input_struct& input);
void timings(std::chrono::time_point<std::chrono::system_clock> start, input_struct input);