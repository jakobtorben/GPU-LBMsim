/* CPU numerical implementation for streaming, boundary conditions 
 * and collisions, for the different models.
 * 
 * Filename: core.cpp
 * Author: Jakob Torben
 * Created: 04.06.2021
 * Last modified: 26.08.2021
 * 
 * This code is provided under the MIT license. See LICENSE.txt.
 */

#include <iostream>
#include <cmath>

#include "core.hpp"

// D2Q9 streaming direction scheme
// 6 2 5
// 3 0 1
// 7 4 8


/**
 * Stream and collide CPU function for the SRT model.
 *
 * @param[in] Nx, Ny domain size
 * @param[out] rho_arr array storing the density
 * @param[out] ux_arr, uy_arr arrays storing the velocities
 * @param[in] u_lid, lid velocity
 * @param[out] f array storing the distributions
 * @param[in] solid_node array storing the position of solid nodes
 * @param[in] tau relaxation time
 * @param[in] omega inverse relaxation time
 * @param[in] save bool for determinig if saving at current iteration
 * @param[in] use_LES<false> LES helper type set to false
 * @param[in] use_MRT<false> MRT helper type set to false
 */
void stream_collide_gpu(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u_lid, float* f,
                        bool* solid_node, float tau, float omega, bool save, use_LES<false>, use_MRT<false>)
{	
    float one_omega = 1 - omega; // 1 - 1/tau

	for (int y = 0; y < Ny; y++)
	{
		// don't stream beyond boundary nodes
		int yn = (y>0) ? y-1 : -1;
		int yp = (y<Ny-1) ? y+1 : -1;

		for (int x = 0; x < Nx; x++)
		{
			int xn = (x>0) ? x-1 : -1;
			int xp = (x<Nx-1) ? x+1 : -1;

            float f0=-1, f1=-1, f2=-1, f3=-1, f4=-1, f5=-1, f6=-1, f7=-1, f8=-1;

                                      f0 = f[f_idx_cpu(Nx, x,  y,  0)];
            if (xn != -1            ) f1 = f[f_idx_cpu(Nx, xn, y,  1)];
            if (yn != -1            ) f2 = f[f_idx_cpu(Nx, x,  yn, 2)];
            if (xp != -1            ) f3 = f[f_idx_cpu(Nx, xp, y,  3)];
            if (yp != -1            ) f4 = f[f_idx_cpu(Nx, x,  yp, 4)];
            if (xn != -1 && yn != -1) f5 = f[f_idx_cpu(Nx, xn, yn, 5)];
            if (xp != -1 && yn != -1) f6 = f[f_idx_cpu(Nx, xp, yn, 6)];
            if (xp != -1 && yp != -1) f7 = f[f_idx_cpu(Nx, xp, yp, 7)];
            if (xn != -1 && yp != -1) f8 = f[f_idx_cpu(Nx, xn, yp, 8)];

            // bounceback on west wall
            if (x == 0)
            {
                f1 = f3;
                f5 = f7;
                f8 = f6;
            }

            // bounceback at east wall
            if (x == Nx-1)
            {
                f3 = f1;
                f7 = f5;
                f6 = f8;
            }

            // bounceback at south wall
            if (y == 0)
            {
                f2 = f4;
                f5 = f7;
                f6 = f8;
            }

            /* velocity BCs on north-side (lid) using bounceback for a moving wall, from
            * Krueger T, Kusumaatmaja H, Kuzmin A, Shardt O, Silva G, Viggen EM.
            * The Lattice Boltzmann Method: Principles and Practice. Springer, 2016.
            * eq (5.26)
            */
            if ((y == Ny - 1) && (x > 0) && (x < Nx-1))
            {
                float rho0 = f0 + f1 + f3 + 2.*(f2 + f5 + f6);
                float ru = rho0*u_lid;
                f4 = f2;
                f7 = f5 - 1./6.*ru;
                f8 = f6 + 1./6.*ru;
            }

            /* corners need to be treated explicitly
            * top corners are treated as part of resting wall and
            * bounced back accordingly. Inactive directions that are
            * streamed from solid are set to zero
            */

            // corner of north-west node
            if ((x == 0) && (y == Ny-1))
            {
                f4 = f2;
                f8 = f6;
                f5 = 0;
                f7 = 0;
            }

            // corner of north-east node
            if ((x == Nx-1) && (y == Ny-1))
            {
                f4 = f2;
                f7 = f5;
                f6 = 0;
                f8 = 0;
            }

            // corner of south-west node
            if ((x == 0) && (y == 0))
            {
                f6 = 0;
                f8 = 0;
            }

            // 	corner of south-east node
            if ((x == Nx-1) && (y == 0))
            {
                f5 = 0;
                f7 = 0;
            }
            
            // compute density
            float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
            if (rho < 0 )
            {
            std::cout << "Error: Negative density at ( " << x << " , " << y << " )\n";
            exit(1);
            }
            
            // collide and update fluid nodes
            if (!solid_node[arr_idx(Nx, x, y)])
            {
                // compute velocities quantities
                float ux = (f1 + f5 + f8 - (f3 + f6 + f7))/rho;
                float uy = (f2 + f5 + f6 - (f4 + f7 + f8))/rho;

                // store to memory only when needed for output
                if (save)
                {
                    ux_arr[arr_idx(Nx, x, y)] = ux;
                    uy_arr[arr_idx(Nx, x, y)] = uy;
                    //rho_arr[arr_idx(Nx, x, y)] = rho;
                }

                float uxsq = ux * ux;
                float uysq = uy * uy;
                float usq = uxsq + uysq;

                float uxuy5 =  ux + uy;
                float uxuy6 = -ux + uy;
                float uxuy7 = -ux - uy;
                float uxuy8 =  ux - uy;

                float c = 1 - 1.5*usq;

                float w_rho0_omega = w0 * rho * omega;
                float w_rho1_omega = w1 * rho * omega;
                float w_rho2_omega = w2 * rho * omega;

                f[f_idx_cpu(Nx, x, y, 0)] = one_omega*f0 + w_rho0_omega*(c                            );
                f[f_idx_cpu(Nx, x, y, 1)] = one_omega*f1 + w_rho1_omega*(c + 3.*ux    + c2*uxsq       );
                f[f_idx_cpu(Nx, x, y, 2)] = one_omega*f2 + w_rho1_omega*(c + 3.*uy    + c2*uysq       );
                f[f_idx_cpu(Nx, x, y, 3)] = one_omega*f3 + w_rho1_omega*(c - 3.*ux    + c2*uxsq       );
                f[f_idx_cpu(Nx, x, y, 4)] = one_omega*f4 + w_rho1_omega*(c - 3.*uy    + c2*uysq       );
                f[f_idx_cpu(Nx, x, y, 5)] = one_omega*f5 + w_rho2_omega*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
                f[f_idx_cpu(Nx, x, y, 6)] = one_omega*f6 + w_rho2_omega*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
                f[f_idx_cpu(Nx, x, y, 7)] = one_omega*f7 + w_rho2_omega*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
                f[f_idx_cpu(Nx, x, y, 8)] = one_omega*f8 + w_rho2_omega*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);
            }
            else
            {
                // write bounced back distributions at boundaries to memory
                f[f_idx_cpu(Nx, x, y, 0)] = f0;
                f[f_idx_cpu(Nx, x, y, 1)] = f1;
                f[f_idx_cpu(Nx, x, y, 2)] = f2;
                f[f_idx_cpu(Nx, x, y, 3)] = f3;
                f[f_idx_cpu(Nx, x, y, 4)] = f4;
                f[f_idx_cpu(Nx, x, y, 5)] = f5;
                f[f_idx_cpu(Nx, x, y, 6)] = f6;
                f[f_idx_cpu(Nx, x, y, 7)] = f7;
                f[f_idx_cpu(Nx, x, y, 8)] = f8;
            }
        }
    }
}


/**
 * Stream and collide CPU function for the SRT model with LES applied.
 * Note that this is not used in this study but added for completeness.
 *
 * @param[in] Nx, Ny domain size
 * @param[out] rho_arr array storing the density
 * @param[out] ux_arr, uy_arr arrays storing the velocities
 * @param[in] u_lid, lid velocity
 * @param[out] f array storing the distributions
 * @param[in] solid_node array storing the position of solid nodes
 * @param[in] tau relaxation time
 * @param[in] omega inverse relaxation time
 * @param[in] save bool for determinig if saving at current iteration
 * @param[in] use_LES<false> LES helper type set to true
 * @param[in] use_MRT<false> MRT helper type set to false
 */
void stream_collide_gpu(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u_lid, float* f,
                        bool* solid_node, float tau, float omega, bool save, use_LES<true>,  use_MRT<false>)
{	
    float C_smg = 0.10;  // Smagorinsky constant, sets length scale as fraction of mesh size

	for (int y = 0; y < Ny; y++)
	{
		// don't stream beyond boundary nodes
		int yn = (y>0) ? y-1 : -1;
		int yp = (y<Ny-1) ? y+1 : -1;

		for (int x = 0; x < Nx; x++)
		{
			int xn = (x>0) ? x-1 : -1;
			int xp = (x<Nx-1) ? x+1 : -1;

            float f0=-1, f1=-1, f2=-1, f3=-1, f4=-1, f5=-1, f6=-1, f7=-1, f8=-1;

                                      f0 = f[f_idx_cpu(Nx, x,  y,  0)];
            if (xn != -1            ) f1 = f[f_idx_cpu(Nx, xn, y,  1)];
            if (yn != -1            ) f2 = f[f_idx_cpu(Nx, x,  yn, 2)];
            if (xp != -1            ) f3 = f[f_idx_cpu(Nx, xp, y,  3)];
            if (yp != -1            ) f4 = f[f_idx_cpu(Nx, x,  yp, 4)];
            if (xn != -1 && yn != -1) f5 = f[f_idx_cpu(Nx, xn, yn, 5)];
            if (xp != -1 && yn != -1) f6 = f[f_idx_cpu(Nx, xp, yn, 6)];
            if (xp != -1 && yp != -1) f7 = f[f_idx_cpu(Nx, xp, yp, 7)];
            if (xn != -1 && yp != -1) f8 = f[f_idx_cpu(Nx, xn, yp, 8)];

            
            // bounceback on west wall
            if (x == 0)
            {
                f1 = f3;
                f5 = f7;
                f8 = f6;
            }

            // bounceback at east wall
            if (x == Nx-1)
            {
                f3 = f1;
                f7 = f5;
                f6 = f8;
            }

            // bounceback at south wall
            if (y == 0)
            {
                f2 = f4;
                f5 = f7;
                f6 = f8;
            }

            /* velocity BCs on north-side (lid) using bounceback for a moving wall, from
            * Krueger T, Kusumaatmaja H, Kuzmin A, Shardt O, Silva G, Viggen EM.
            * The Lattice Boltzmann Method: Principles and Practice. Springer, 2016.
            * eq (5.26)
            */
            if ((y == Ny - 1) && (x > 0) && (x < Nx-1))
            {
                float rho0 = f0 + f1 + f3 + 2.*(f2 + f5 + f6);
                float ru = rho0*u_lid;
                f4 = f2;
                f7 = f5 - 1./6.*ru;
                f8 = f6 + 1./6.*ru;
            }

            /* corners need to be treated explicitly
            * top corners are treated as part of resting wall and
            * bounced back accordingly. Inactive directions that are
            * streamed from solid are set to zero
            */

            // corner of north-west node
            if ((x == 0) && (y == Ny-1))
            {
                f4 = f2;
                f8 = f6;
                f5 = 0;
                f7 = 0;
            }

            // corner of north-east node
            if ((x == Nx-1) && (y == Ny-1))
            {
                f4 = f2;
                f7 = f5;
                f6 = 0;
                f8 = 0;
            }

            // corner of south-west node
            if ((x == 0) && (y == 0))
            {
                f6 = 0;
                f8 = 0;
            }

            // 	corner of south-east node
            if ((x == Nx-1) && (y == 0))
            {
                f5 = 0;
                f7 = 0;
            }
            
            // compute density
            float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
            if (rho < 0 )
            {
            std::cout << "Error: Negative density at ( " << x << " , " << y << " )\n";
            exit(1);
            }

            // collide and update fluid nodes
            if (!solid_node[arr_idx(Nx, x, y)])
            {

                // compute velocities
                float ux = (f1 + f5 + f8 - (f3 + f6 + f7))/rho;
                float uy = (f2 + f5 + f6 - (f4 + f7 + f8))/rho;

                // store to memory only when needed for output
                if (save)
                {
                    ux_arr[arr_idx(Nx, x, y)] = ux;
                    uy_arr[arr_idx(Nx, x, y)] = uy;
                    //rho_arr[arr_idx(Nx, x, y)] = rho;
                }

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

                // calculate equilibrium function
                float feq0 = w_rho0*(c                            );
                float feq1 = w_rho1*(c + 3.*ux    + c2*uxsq       );
                float feq2 = w_rho1*(c + 3.*uy    + c2*uysq       );
                float feq3 = w_rho1*(c - 3.*ux    + c2*uxsq       );
                float feq4 = w_rho1*(c - 3.*uy    + c2*uysq       );
                float feq5 = w_rho2*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
                float feq6 = w_rho2*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
                float feq7 = w_rho2*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
                float feq8 = w_rho2*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);
 
                /* perform large eddy simulation
                 * LES equation (12-22), here adapted to D2Q9, from
                 * Krafczyk, Manfred & Tolke, J & Luo, Li-Shi. (2003).
                 * Large eddy simulation with a multiple-relaxation-time LBE model.
                 * INTERNATIONAL JOURNAL OF MODERN PHYSICS B. 17. 33-39. 10.1142/S0217979203017059.
                 */

                /* tensor Q_ij defined from non-equilibrium distribution functions
                * Zhenhua Chai, Baochang Shi, Zhaoli Guo, Fumei Rong,
                * Multiple-relaxation-time lattice Boltzmann model for generalized Newtonian fluid flows,
                * Journal of Non-Newtonian Fluid Mechanics, Volume 166, Issues 5–6, 2011,
                * eq (18): Q_ij = sum[ex[a]*ey[a]*(f_a - feq_a)]
                */
                float Q_xx = (f1-feq1) + (f3-feq3) + (f5-feq5) + (f6-feq6) + (f7-feq7) + (f8-feq8);
                float Q_yy = (f2-feq2) + (f4-feq4) + (f5-feq5) + (f6-feq6) + (f7-feq7) + (f8-feq8);
                float Q_xy = (f5-feq5) - (f6-feq6) + (f7-feq7) - (f8-feq8);

                float Q_bar = std::sqrt(2*(Q_xx*Q_xx + Q_yy*Q_yy + 2*Q_xy*Q_xy));

                // calculate turbulence viscosity from eq (22)
                float tau_turb = 0.5 * ( std::sqrt(tau*tau + 18.*C_smg*C_smg*Q_bar) - tau );

                float tau_eff = tau + tau_turb;  // effective viscosity

                float omega_eff = 1/tau_eff;
                float one_omega_eff = 1 - omega_eff; // 1 - 1/tau

                // update distributions from LBM formula
                f[f_idx_cpu(Nx, x, y, 0)] = one_omega_eff*f0 + feq0*omega_eff;
                f[f_idx_cpu(Nx, x, y, 1)] = one_omega_eff*f1 + feq1*omega_eff;
                f[f_idx_cpu(Nx, x, y, 2)] = one_omega_eff*f2 + feq2*omega_eff;
                f[f_idx_cpu(Nx, x, y, 3)] = one_omega_eff*f3 + feq3*omega_eff;
                f[f_idx_cpu(Nx, x, y, 4)] = one_omega_eff*f4 + feq4*omega_eff;
                f[f_idx_cpu(Nx, x, y, 5)] = one_omega_eff*f5 + feq5*omega_eff;
                f[f_idx_cpu(Nx, x, y, 6)] = one_omega_eff*f6 + feq6*omega_eff;
                f[f_idx_cpu(Nx, x, y, 7)] = one_omega_eff*f7 + feq7*omega_eff;
                f[f_idx_cpu(Nx, x, y, 8)] = one_omega_eff*f8 + feq8*omega_eff;
            }
            else
            {
                // write bounced back distributions to memory
                f[f_idx_cpu(Nx, x, y, 0)] = f0;
                f[f_idx_cpu(Nx, x, y, 1)] = f1;
                f[f_idx_cpu(Nx, x, y, 2)] = f2;
                f[f_idx_cpu(Nx, x, y, 3)] = f3;
                f[f_idx_cpu(Nx, x, y, 4)] = f4;
                f[f_idx_cpu(Nx, x, y, 5)] = f5;
                f[f_idx_cpu(Nx, x, y, 6)] = f6;
                f[f_idx_cpu(Nx, x, y, 7)] = f7;
                f[f_idx_cpu(Nx, x, y, 8)] = f8;

            }
        }
    }
}

/**
 * Stream and collide CPU function for the MRT model.
 *
 * @param[in] Nx, Ny domain size
 * @param[out] rho_arr array storing the density
 * @param[out] ux_arr, uy_arr arrays storing the velocities
 * @param[in] u_lid, lid velocity
 * @param[out] f array storing the distributions
 * @param[in] solid_node array storing the position of solid nodes
 * @param[in] tau relaxation time
 * @param[in] omega inverse relaxation time
 * @param[in] save bool for determinig if saving at current iteration
 * @param[in] use_LES<false> LES helper type set to false
 * @param[in] use_MRT<false> MRT helper type set to true
 */
void stream_collide_gpu(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u_lid, float* f,
                        bool* solid_node, float tau, float omega, bool save, use_LES<false>, use_MRT<true>)
{	

    /*
     * Relaxation parameters: s0 = s3 = s5 = 0,  s1 = s2 = 1.4,  s4 = s6 = 1.2, s7 = s8 = omega = 1 / tau
     * From:
     * D’Humieres, D., Ginzburg, I., Krafczyk, M., Lallemand, P. & Luo, L.-S. (2002),
     * ‘Multiple-relaxation-time lattice Boltzmann models in three dimensions’, 
     * Philosophical transactions. Series A, Mathe-matical, physical, and engineering sciences360(1792), 437–451.
     */
    float s1_2 = 1.4, s4_6 = 1.2, s7_8 = omega;
    
	for (int y = 0; y < Ny; y++)
	{
		// don't stream beyond boundary nodes
		int yn = (y>0) ? y-1 : -1;
		int yp = (y<Ny-1) ? y+1 : -1;

		for (int x = 0; x < Nx; x++)
		{
			int xn = (x>0) ? x-1 : -1;
			int xp = (x<Nx-1) ? x+1 : -1;

            float f0=-1, f1=-1, f2=-1, f3=-1, f4=-1, f5=-1, f6=-1, f7=-1, f8=-1;

                                      f0 = f[f_idx_cpu(Nx, x,  y,  0)];
            if (xn != -1            ) f1 = f[f_idx_cpu(Nx, xn, y,  1)];
            if (yn != -1            ) f2 = f[f_idx_cpu(Nx, x,  yn, 2)];
            if (xp != -1            ) f3 = f[f_idx_cpu(Nx, xp, y,  3)];
            if (yp != -1            ) f4 = f[f_idx_cpu(Nx, x,  yp, 4)];
            if (xn != -1 && yn != -1) f5 = f[f_idx_cpu(Nx, xn, yn, 5)];
            if (xp != -1 && yn != -1) f6 = f[f_idx_cpu(Nx, xp, yn, 6)];
            if (xp != -1 && yp != -1) f7 = f[f_idx_cpu(Nx, xp, yp, 7)];
            if (xn != -1 && yp != -1) f8 = f[f_idx_cpu(Nx, xn, yp, 8)];

            
            // bounceback on west wall
            if (x == 0)
            {
                f1 = f3;
                f5 = f7;
                f8 = f6;
            }

            // bounceback at east wall
            if (x == Nx-1)
            {
                f3 = f1;
                f7 = f5;
                f6 = f8;
            }

            // bounceback at south wall
            if (y == 0)
            {
                f2 = f4;
                f5 = f7;
                f6 = f8;
            }

            /* velocity BCs on north-side (lid) using bounceback for a moving wall, from
            * Krueger T, Kusumaatmaja H, Kuzmin A, Shardt O, Silva G, Viggen EM.
            * The Lattice Boltzmann Method: Principles and Practice. Springer, 2016.
            * eq (5.26)
            */
            if ((y == Ny - 1) && (x > 0) && (x < Nx-1))
            {
                float rho0 = f0 + f1 + f3 + 2.*(f2 + f5 + f6);
                float ru = rho0*u_lid;
                f4 = f2;
                f7 = f5 - 1./6.*ru;
                f8 = f6 + 1./6.*ru;
            }

            /* corners need to be treated explicitly
            * top corners are treated as part of resting wall and
            * bounced back accordingly. Inactive directions that are
            * streamed from solid are set to zero
            */

            // corner of north-west node
            if ((x == 0) && (y == Ny-1))
            {
                f4 = f2;
                f8 = f6;
                f5 = 0;
                f7 = 0;
            }

            // corner of north-east node
            if ((x == Nx-1) && (y == Ny-1))
            {
                f4 = f2;
                f7 = f5;
                f6 = 0;
                f8 = 0;
            }

            // corner of south-west node
            if ((x == 0) && (y == 0))
            {
                f6 = 0;
                f8 = 0;
            }

            // 	corner of south-east node
            if ((x == Nx-1) && (y == 0))
            {
                f5 = 0;
                f7 = 0;
            }

            float m[Q];  // distribution in moment space

            // collide and update fluid nodes
            if (!solid_node[arr_idx(Nx, x, y)])
            {

                // transform distribution into moment space
                for (int a = 0; a<Q; a++)
                {
                    m[a] =  M[a*Q + 0]*f0 + M[a*Q + 1]*f1 + M[a*Q + 2]*f2 + M[a*Q + 3]*f3 + M[a*Q + 4]*f4
                          + M[a*Q + 5]*f5 + M[a*Q + 6]*f6 + M[a*Q + 7]*f7 + M[a*Q + 8]*f8;
                }

                // store to memory only when needed for output
                // m0 is density, m3 and m5 is x,y momentum and were calculated in previous loop
                if (m[0] < 0 )
                {
                    std::cout << "Error: Negative density at ( " << x << " , " << y << " )\n";
                    exit(1);
                }
                
                if (save)
                {
                    ux_arr[arr_idx(Nx, x, y)] = m[3]/m[0];
                    uy_arr[arr_idx(Nx, x, y)] = m[5]/m[0];
                    //rho_arr[arr_idx(Nx, x, y)] = m[0];
                }
                
                /* perform collision step in moment space
                * f_+1 - f = -Minv * S * (m - meq)
                * m_+1 = m - S*(m - meq) 
                * S is a diagonal relaxation times matrix
                * expressions for m_eq given in
                * Lallemand P, Luo L-S. Theory of the lattice Boltzmann method: dispersion,
                * dissipation, isotropy, Galilean invariance, and stability. Physics Review E 2000; 61: 6546-6562.
                */
            
                float momsq = m[3]*m[3] + m[5]*m[5];
                                                                      // meq is expression in ()
             // m[0] = m[0] - 0*(m[0] - m[0]) = m[0]                  // rho - density  
                m[1] = m[1] - s1_2*(m[1] - (-2.*m[0] + 3.*momsq  ));  // e - energy
                m[2] = m[2] - s1_2*(m[2] - (    m[0] - 3.*momsq  ));  // epsilon - energy squared
             // m[3] = m[3] - 0*(m[3] - m[3]) = m[3]                  // jx - x momentum   
                m[4] = m[4] - s4_6*(m[4] - (-m[3]                ));  // qx - energy flux
             // m[5] = m[5] - 0*(m[5] - m[5]) = m[5]                  // jy - y momentum 
                m[6] = m[6] - s4_6*(m[6] - (-m[5]                ));  // qy - energy flux
                m[7] = m[7] - s7_8*(m[7] - (m[3]*m[3] - m[5]*m[5]));  // pxx - strain rate
                m[8] = m[8] - s7_8*(m[8] - (m[3]*m[5]            ));  // pxy - strain rate

                // transform back into distribution functions
                // f = Minv*m_+1
                for (int a = 0; a<Q; a++)
                    f[f_idx_cpu(Nx, x, y, a)] =  Minv[a*Q + 0]*m[0] + Minv[a*Q + 1]*m[1] + Minv[a*Q + 2]*m[2]
                                               + Minv[a*Q + 3]*m[3] + Minv[a*Q + 4]*m[4] + Minv[a*Q + 5]*m[5]
                                               + Minv[a*Q + 6]*m[6] + Minv[a*Q + 7]*m[7] + Minv[a*Q + 8]*m[8];
            }
            else
            {
                // write bounced back distributions to memory
                f[f_idx_cpu(Nx, x, y, 0)] = f0;
                f[f_idx_cpu(Nx, x, y, 1)] = f1;
                f[f_idx_cpu(Nx, x, y, 2)] = f2;
                f[f_idx_cpu(Nx, x, y, 3)] = f3;
                f[f_idx_cpu(Nx, x, y, 4)] = f4;
                f[f_idx_cpu(Nx, x, y, 5)] = f5;
                f[f_idx_cpu(Nx, x, y, 6)] = f6;
                f[f_idx_cpu(Nx, x, y, 7)] = f7;
                f[f_idx_cpu(Nx, x, y, 8)] = f8;
            }
        }
    }
}


/**
 * Stream and collide CPU function for the MRT model with LES applied.
 *
 * @param[in] Nx, Ny domain size
 * @param[out] rho_arr array storing the density
 * @param[out] ux_arr, uy_arr arrays storing the velocities
 * @param[in] u_lid, lid velocity
 * @param[out] f array storing the distributions
 * @param[in] solid_node array storing the position of solid nodes
 * @param[in] tau relaxation time
 * @param[in] omega inverse relaxation time
 * @param[in] save bool for determinig if saving at current iteration
 * @param[in] use_LES<false> LES helper type set to true
 * @param[in] use_MRT<false> MRT helper type set to true
 */
void stream_collide_gpu(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u_lid, float* f,
                        bool* solid_node, float tau, float omega, bool save, use_LES<true>, use_MRT<true>)
{	
    float C_smg = 0.10;  // Smagorinsky constant, sets length scale as fraction of mesh size

	for (int y = 0; y < Ny; y++)
	{
		// don't stream beyond boundary nodes
		int yn = (y>0) ? y-1 : -1;
		int yp = (y<Ny-1) ? y+1 : -1;

		for (int x = 0; x < Nx; x++)
		{
			int xn = (x>0) ? x-1 : -1;
			int xp = (x<Nx-1) ? x+1 : -1;

            float f0=-1, f1=-1, f2=-1, f3=-1, f4=-1, f5=-1, f6=-1, f7=-1, f8=-1;

                                      f0 = f[f_idx_cpu(Nx, x,  y,  0)];
            if (xn != -1            ) f1 = f[f_idx_cpu(Nx, xn, y,  1)];
            if (yn != -1            ) f2 = f[f_idx_cpu(Nx, x,  yn, 2)];
            if (xp != -1            ) f3 = f[f_idx_cpu(Nx, xp, y,  3)];
            if (yp != -1            ) f4 = f[f_idx_cpu(Nx, x,  yp, 4)];
            if (xn != -1 && yn != -1) f5 = f[f_idx_cpu(Nx, xn, yn, 5)];
            if (xp != -1 && yn != -1) f6 = f[f_idx_cpu(Nx, xp, yn, 6)];
            if (xp != -1 && yp != -1) f7 = f[f_idx_cpu(Nx, xp, yp, 7)];
            if (xn != -1 && yp != -1) f8 = f[f_idx_cpu(Nx, xn, yp, 8)];

            
            // bounceback on west wall
            if (x == 0)
            {
                f1 = f3;
                f5 = f7;
                f8 = f6;
            }

            // bounceback at east wall
            if (x == Nx-1)
            {
                f3 = f1;
                f7 = f5;
                f6 = f8;
            }

            // bounceback at south wall
            if (y == 0)
            {
                f2 = f4;
                f5 = f7;
                f6 = f8;
            }

            /* velocity BCs on north-side (lid) using bounceback for a moving wall, from
            * Krueger T, Kusumaatmaja H, Kuzmin A, Shardt O, Silva G, Viggen EM.
            * The Lattice Boltzmann Method: Principles and Practice. Springer, 2016.
            * eq (5.26)
            */
            if ((y == Ny - 1) && (x > 0) && (x < Nx-1))
            {
                float rho0 = f0 + f1 + f3 + 2.*(f2 + f5 + f6);
                float ru = rho0*u_lid;
                f4 = f2;
                f7 = f5 - 1./6.*ru;
                f8 = f6 + 1./6.*ru;
            }

            /* corners need to be treated explicitly
            * top corners are treated as part of resting wall and
            * bounced back accordingly. Inactive directions that are
            * streamed from solid are set to zero
            */

            // corner of north-west node
            if ((x == 0) && (y == Ny-1))
            {
                f4 = f2;
                f8 = f6;
                f5 = 0;
                f7 = 0;
            }

            // corner of north-east node
            if ((x == Nx-1) && (y == Ny-1))
            {
                f4 = f2;
                f7 = f5;
                f6 = 0;
                f8 = 0;
            }

            // corner of south-west node
            if ((x == 0) && (y == 0))
            {
                f6 = 0;
                f8 = 0;
            }

            // 	corner of south-east node
            if ((x == Nx-1) && (y == 0))
            {
                f5 = 0;
                f7 = 0;
            }

            float m[Q];  // distribution in moment space

            // collide and update fluid nodes
            if (!solid_node[arr_idx(Nx, x, y)])
            {

                // transform distribution into moment space
                for (int a = 0; a<Q; a++)
                {
                    m[a] =  M[a*Q + 0]*f0 + M[a*Q + 1]*f1 + M[a*Q + 2]*f2 + M[a*Q + 3]*f3 + M[a*Q + 4]*f4
                          + M[a*Q + 5]*f5 + M[a*Q + 6]*f6 + M[a*Q + 7]*f7 + M[a*Q + 8]*f8;
                }

                // store to memory only when needed for output
                // m0 is density, m3 and m5 is x,y momentum and were calculated in previous loop
                if (m[0] < 0 )
                {
                    std::cout << "Error: Negative density at ( " << x << " , " << y << " )\n";
                    exit(1);
                }
                
                if (save)
                {
                    ux_arr[arr_idx(Nx, x, y)] = m[3]/m[0];
                    uy_arr[arr_idx(Nx, x, y)] = m[5]/m[0];
                    //rho_arr[arr_idx(Nx, x, y)] = m[0];
                }

                // perform large eddy simulation
                /* LES equation (12-22), here adapted to D2Q9, from
                * Krafczyk, Manfred & Tolke, J & Luo, Li-Shi. (2003).
                * Large eddy simulation with a multiple-relaxation-time LBE model.
                * INTERNATIONAL JOURNAL OF MODERN PHYSICS B. 17. 33-39. 10.1142/S0217979203017059.
                */

                float Pxx = 1./6.*(m[1] + 4*m[0] + 3.*m[7]);
                float Pyy = 1./6.*(m[1] + 4*m[0] - 3.*m[7]);
                float Pxy = m[8];

                float Q_xx = 1./3.*m[0] + m[3]*m[3] - Pxx;
                float Q_yy = 1./3.*m[0] + m[5]*m[5] - Pyy;
                float Q_xy = m[3]*m[5] - Pxy;

                float Q_bar = std::sqrt(2*(Q_xx*Q_xx + Q_yy*Q_yy + 2*Q_xy*Q_xy));

                float tau_turb = 0.5 * ( std::sqrt(tau*tau + 18.*C_smg*C_smg*Q_bar) - tau );

                float tau_eff = tau + tau_turb;  // effective viscosity

                /* Relaxation parameters: s0 = s3 = s5 = 0,  s1 = s2 = 1.4,  s4 = s6 = 1.2, s7 = s8 = omega = 1 / tau
                * From:
                * D’Humieres, D., Ginzburg, I., Krafczyk, M., Lallemand, P. & Luo, L.-S. (2002),
                * ‘Multiple-relaxation-time lattice Boltzmann models in three dimensions’, 
                * Philosophical transactions. Series A, Mathe-matical, physical, and engineering sciences360(1792), 437–451.
                */
                float s1_2 = 1.4, s4_6 = 1.2, s7_8 = 1/tau_eff;
                
                /* perform collision in moment space
                * f_+1 - f = -Minv * S * (m - meq)
                * m_+1 = m - S*(m - meq) 
                * S is a diagonal relaxation times matrix
                * expressions for m_eq given in
                * Lallemand P, Luo L-S. Theory of the lattice Boltzmann method: dispersion,
                * dissipation, isotropy, Galilean invariance, and stability. Physics Review E 2000; 61: 6546-6562.
                */
            
                float momsq = m[3]*m[3] + m[5]*m[5];
                                                                      // meq is expression in ()
             // m[0] = m[0] - 0*(m[0] - m[0]) = m[0]                  // rho - density  
                m[1] = m[1] - s1_2*(m[1] - (-2.*m[0] + 3.*momsq  ));  // e - energy
                m[2] = m[2] - s1_2*(m[2] - (    m[0] - 3.*momsq  ));  // epsilon - energy squared
             // m[3] = m[3] - 0*(m[3] - m[3]) = m[3]                  // jx - x momentum   
                m[4] = m[4] - s4_6*(m[4] - (-m[3]                ));  // qx - energy flux
             // m[5] = m[5] - 0*(m[5] - m[5]) = m[5]                  // jy - y momentum 
                m[6] = m[6] - s4_6*(m[6] - (-m[5]                ));  // qy - energy flux
                m[7] = m[7] - s7_8*(m[7] - (m[3]*m[3] - m[5]*m[5]));  // pxx - strain rate
                m[8] = m[8] - s7_8*(m[8] - (m[3]*m[5]            ));  // pxy - strain rate

                // transform back into distribution functions
                // f_+1 = Minv*m_+1
                for (int a = 0; a<Q; a++)
                    f[f_idx_cpu(Nx, x, y, a)] =  Minv[a*Q + 0]*m[0] + Minv[a*Q + 1]*m[1] + Minv[a*Q + 2]*m[2]
                                                + Minv[a*Q + 3]*m[3] + Minv[a*Q + 4]*m[4] + Minv[a*Q + 5]*m[5]
                                                + Minv[a*Q + 6]*m[6] + Minv[a*Q + 7]*m[7] + Minv[a*Q + 8]*m[8];
            
            }
            else
            {
                // write bounced back distributions to memory
                f[f_idx_cpu(Nx, x, y, 0)] = f0;
                f[f_idx_cpu(Nx, x, y, 1)] = f1;
                f[f_idx_cpu(Nx, x, y, 2)] = f2;
                f[f_idx_cpu(Nx, x, y, 3)] = f3;
                f[f_idx_cpu(Nx, x, y, 4)] = f4;
                f[f_idx_cpu(Nx, x, y, 5)] = f5;
                f[f_idx_cpu(Nx, x, y, 6)] = f6;
                f[f_idx_cpu(Nx, x, y, 7)] = f7;
                f[f_idx_cpu(Nx, x, y, 8)] = f8;
            }
        }
    }
}
