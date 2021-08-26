/* Defninig options and methods for initialising the simulation on the CPU.
 * 
 * Filename: init.hpp
 * Author: Jakob Torben
 * Created: 04.06.2021
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

void initialise_simulation(input_struct input, std::string& fname, float& u_lid, float& tau, float& omega);
void define_geometry(int Nx, int Ny, bool* solid_node);
void initialise_distributions(int Nx, int Ny, float u_lid, float* f, float* rho_arr, float* ux_arr, float* uy_arr);

