/* Mehtods for initialising the simulation on the CPU.
 * 
 * Filename: init.cpp
 * Author: Jakob Torben
 * Created: 04.06.2021
 * Last modified: 26.08.2021
 * 
 * This code is provided under the MIT license. See LICENSE.txt.
 */

#include <iostream>
#include <sstream>
#include <cmath>

#include "init.hpp"
#include "core.hpp"

/**
 * Initialisation and non dimensionalisation of simulation parameters.
 * Using lattice units dx = dy = dt = 1, and lattice velocity c = dx/dy = 1
 * Mach set to constant 0.1 to ensure stability, Reynolds number controls simulation
 *
 * @param[in] input parameters from inputfile
 * @param[out] fname filename to save the outputs to
 * @param[out] u_lid lid velocity
 * @param[out] tau relaxation time
 * @param[out] omega inverse relaxation time
 */
void initialise_simulation(input_struct input, std::string& fname, float& u_lid, float& tau, float& omega)
{
    float cs = std::sqrt(1./3.);			// lattice speed of sound D2Q9
    float mach = 0.1;               // lattice mach number
	u_lid =  mach * cs;         // lid speed
    float kin_visc = u_lid * float(input.Nx-1) / input.reynolds; // Nx-1 is length of slididng lid	
    tau = (3. * kin_visc + 0.5); // relaxation time, ensures consistency with Navier-Stokes eq
    omega = 1/tau;

    // define output filename
    std::stringstream stream;
    stream << "LBM_sim_";
    if (mrt) stream << "MRT"; else stream << "SRT";
    if (les) stream << "_LES";
    stream << "_Nx" << input.Nx << "_Ny" << input.Ny << "_Re"<< input.reynolds;
    fname = stream.str();

    // print simulation parameters
	std::cout << "Nx: " << input.Nx << " Ny: " << input.Ny << "\n";
	std::cout << "Boundary conditions: Lid driven cavity\n";
    std::cout << "Collision operator: ";
    if (mrt) std::cout << "MRT"; else std::cout << "SRT";
    if (les) std::cout << "-LES\n"; else std::cout << "\n";
    std::cout << "Reynolds number: " << input.reynolds << "\n";
	std::cout << "kinematic viscosity: " << kin_visc << "\n";
	std::cout << "u_lid: " << u_lid << "\n";
	std::cout << "mach number: " << mach << "\n";
	std::cout << "tau : " << tau << "\n\n";
}

/**
 * Define the geometry of the fixed walls.
 *
 * @param[in] Nx, Ny domain size
 * @param[out] solid_node array storing the position of solid nodes
 */
void define_geometry(int Nx, int Ny, bool* solid_node)
{

	for (int y = 0; y < Ny; y++)
		for (int x = 0; x < Nx; x++)
		{
            // set fixed walls to solid boundaries
            if (x == 0)
                solid_node[arr_idx(Nx, x, y)] = 1;  // west wall
            else if (x == Nx-1) 
                solid_node[arr_idx(Nx, x, y)] = 1;  // east wall
            else if (y == 0)
                solid_node[arr_idx(Nx, x, y)] = 1;  // south wall
            else 
                solid_node[arr_idx(Nx, x, y)] = 0;
		}
}

/**
 * Set inital distributions to equilibrium values for lid driven cavity
 *
 * @param[in] Nx, Ny domain size
 * @param[in] u_lid, lid velocity
 * @param[out] f array storing the distributions
 * @param[out] rho_arr array storing the density
 * @param[out] ux_arr, uy_arr arrays storing the velocities
 */
void initialise_distributions(int Nx, int Ny, float u_lid, float* f, float* rho_arr, float* ux_arr, float* uy_arr)
{
	for (int y = 0; y < Ny; y++)
		for (int x = 0; x < Nx; x++)
		{
            // set density to 1.0 to keep as much precision as possible during calculation
            rho_arr[arr_idx(Nx, x, y)] = 1.;
            uy_arr[arr_idx(Nx, x, y)] = 0.;
            if (y == Ny - 1)
                ux_arr[arr_idx(Nx, x, y)] = u_lid;
            else
                ux_arr[arr_idx(Nx, x, y)] = 0;
            
            float ux = ux_arr[arr_idx(Nx, x, y)];
            float uy = uy_arr[arr_idx(Nx, x, y)];
            float rho = rho_arr[arr_idx(Nx, x, y)];

            float uxsq = ux * ux;
            float uysq = uy * uy;
            float usq = uxsq + uysq;

            float uxuy5 =  ux + uy;
            float uxuy6 = -ux + uy;
            float uxuy7 = -ux - uy;
            float uxuy8 =  ux - uy;

            float c = 1 - 1.5*usq;
            float w_rho0 = w0 * rho;
            float w_rho1 = w1 * rho;
            float w_rho2 = w2 * rho;

            f[f_idx_cpu(Nx, x, y, 0)] = w_rho0*(c                           );
            f[f_idx_cpu(Nx, x, y, 1)] = w_rho1*(c + 3.*ux  + c2*uxsq        );
            f[f_idx_cpu(Nx, x, y, 2)] = w_rho1*(c + 3.*uy  + c2*uysq        );
            f[f_idx_cpu(Nx, x, y, 3)] = w_rho1*(c - 3.*ux  + c2*uxsq        );
            f[f_idx_cpu(Nx, x, y, 4)] = w_rho1*(c - 3.*uy  + c2*uysq        );
            f[f_idx_cpu(Nx, x, y, 5)] = w_rho2*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
            f[f_idx_cpu(Nx, x, y, 6)] = w_rho2*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
            f[f_idx_cpu(Nx, x, y, 7)] = w_rho2*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
            f[f_idx_cpu(Nx, x, y, 8)] = w_rho2*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);
	    }
}