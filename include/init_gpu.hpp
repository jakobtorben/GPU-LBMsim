  /* Defninig options and methods for initialising the simulation on the GPU.
 * 
 * Filename: init_gpu.hpp
 * Author: Jakob Torben
 * Created: 13.07.2021
 * Last modified: 26.08.2021
 * 
 * This code is provided under the MIT license. See LICENSE.txt.
 */

#pragma once

#include "utils.hpp"

#ifndef LES
    #define LES 1
#endif
#ifndef MRT
    #define MRT 1
#endif

// compile time options
constexpr bool les = LES;
constexpr bool mrt = MRT;

// set threads to nVidia's warp size to run all threads concurrently 
const int num_threads = 32;

void initialise_simulation(input_struct input, std::string& fname, float& u_lid, float& tau, float& omega);
void set_up_GPU();
__global__ void define_geometry(int Nx, int Ny, bool* solid_node);
__global__ void initialise_distributions(int Nx, int Ny, float u_lid, float* f, float* rho_arr, float* ux_arr, float* uy_arr);