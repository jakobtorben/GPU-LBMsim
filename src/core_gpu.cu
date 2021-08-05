#include "core_gpu.hpp"
#include <iostream>  // delete later
#include <stdio.h>  // delete later
#include <cuda.h>
#include <assert.h>

// D2Q9 streaming direction scheme
// 6 2 5
// 3 0 1
// 7 4 8

__device__ __forceinline__ size_t f_index(unsigned int Nx, unsigned int Ny, unsigned int x, unsigned int y, unsigned int a)
{
    return ((Ny*a + y)*Nx + x);
    //return (x + Nx*y)*9 + a;
}

// lid driven cavity boudnary conditions - MRT
__global__ void stream_collide_gpu_lid(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u0, float* f, bool* solid_node, float tau, bool save, is_periodic<false>, use_LES<false>, use_MRT<true>)
{	
	int y = blockIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;

	// don't stream beyond boundary nodes
    int yn = (y>0) ? y-1 : -1;
	int yp = (y<Ny-1) ? y+1 : -1;
    int xn = (x>0) ? x-1 : -1;;
    int xp = (x<Nx-1) ? x+1 : -1;

    float f0=-1, f1=-1, f2=-1, f3=-1, f4=-1, f5=-1, f6=-1, f7=-1, f8=-1;

                              f0 = f[f_index(Nx, Ny, x, y, 0)];
    if (xn != -1            ) f1 = f[f_index(Nx, Ny, xn, y, 1)];
    if (yn != -1            ) f2 = f[f_index(Nx, Ny, x, yn, 2)];
    if (xp != -1            ) f3 = f[f_index(Nx, Ny, xp, y, 3)];
    if (yp != -1            ) f4 = f[f_index(Nx, Ny, x, yp, 4)];
    if (xn != -1 && yn != -1) f5 = f[f_index(Nx, Ny, xn, yn, 5)];
    if (xp != -1 && yn != -1) f6 = f[f_index(Nx, Ny, xp, yn, 6)];
    if (xp != -1 && yp != -1) f7 = f[f_index(Nx, Ny, xp, yp, 7)];
    if (xn != -1 && yp != -1) f8 = f[f_index(Nx, Ny, xn, yp, 8)];

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

    // velocity BCs on north-side (lid) using bounceback on a moving wall
    // as it has better stability than Ze & Hou
    // eq (5.26)
    // Krueger T, Kusumaatmaja H, Kuzmin A, Shardt O, Silva G, Viggen EM.
    // The Lattice Boltzmann Method: Principles and Practice. Springer, 2016. 690 p.
    if ((y == Ny - 1) && (x > 0) && (x < Nx-1))
	{
		float rho0 = f0 + f1 + f3 + 2.*(f2 + f5 + f6);
		float ru = rho0*u0;
		f4 = f2;
        f7 = f5 - 1./6.*ru;
		f8 = f6 + 1./6.*ru;
	}

    // corner of south-west inlet
    if ((x == 0) && (y == 0))
    {
        // streaming from solid nodes, zero from standard bounceback
        f6 = 0;
        f8 = 0;
    }

	// 	corner of south-east outlet
    if ((x == Nx-1) && (y == 0))
    {
        // streaming from solid nodes, zero from standard bounceback
        f5 = 0;
        f7 = 0;
    }

    // unknown distributions at singular corner points are
    //  considered part of the lid and extrapolated it

	// corner of north-west inlet
    if ((x == 0) && (y == Ny-1))
    {
        f0 = f[f_index(Nx, Ny, 1, Ny-1, 0)];
        f1 = f[f_index(Nx, Ny, 1, Ny-1, 1)];
        f2 = f[f_index(Nx, Ny, 1, Ny-1, 2)];
        f3 = f[f_index(Nx, Ny, 1, Ny-1, 3)];
        f4 = f[f_index(Nx, Ny, 1, Ny-1, 4)];
        f5 = f[f_index(Nx, Ny, 1, Ny-1, 5)];
        f6 = f[f_index(Nx, Ny, 1, Ny-1, 6)];
        f7 = f[f_index(Nx, Ny, 1, Ny-1, 7)];
        f8 = f[f_index(Nx, Ny, 1, Ny-1, 8)];
    }

	// corner of north-east outlet
    if ((x == Nx-1) && (y == Ny-1))
    {

        f0 = f[f_index(Nx, Ny, Nx-2, Ny-1, 0)];
        f1 = f[f_index(Nx, Ny, Nx-2, Ny-1, 1)];
        f2 = f[f_index(Nx, Ny, Nx-2, Ny-1, 2)];
        f3 = f[f_index(Nx, Ny, Nx-2, Ny-1, 3)];
        f4 = f[f_index(Nx, Ny, Nx-2, Ny-1, 4)];
        f5 = f[f_index(Nx, Ny, Nx-2, Ny-1, 5)];
        f6 = f[f_index(Nx, Ny, Nx-2, Ny-1, 6)];
        f7 = f[f_index(Nx, Ny, Nx-2, Ny-1, 7)];
        f8 = f[f_index(Nx, Ny, Nx-2, Ny-1, 8)];
    }

    // D. d’Humières. Multiple–relaxation–time lattice boltzmann models in three dimensions
    // s0 = s3 = s5 = 1,  s1 = s2 = 1.4,  s4 = s6 = 1.2, s7 = s8 = omega = 1 / tau
    float s1_2 = 1.4, s4_6 = 1.2, s7_8 = 1/tau;

    float m[Q];  // distribution in moment space

    if (!solid_node[x + Nx*y])
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
            printf("Fatal error: negative density  at ( %d , %d )\n", x, y);
            assert(0);
        }
        
        if (save)
        {
            ux_arr[x + Nx*y] = m[3]/m[0];
            uy_arr[x + Nx*y] = m[5]/m[0];
            //rho_arr[cord] = m[0];
        }
        
        // perform relaxation step in moment space
        //f_+1 - f = -Minv * S * (m - meq)
        // m_+1 = m - S*(m - meq) 
        // S is a diagonal relaxation times matrix
        // expressions for meq given in
        //  Lallemand P, Luo L-S. Theory of the lattice Boltzmann method: dispersion,
        // dissipation, isotropy, Galilean invariance, and stability. Physics Review E 2000; 61: 6546-6562.
    
        float momsq = m[3]*m[3] + m[5]*m[5];
                                                              // meq is expression in ()
        m[1] = m[1] - s1_2*(m[1] - (-2.*m[0] + 3.*momsq  ));  // e - energy
        m[2] = m[2] - s1_2*(m[2] - (    m[0] - 3.*momsq  ));  // epsilon - energy squared
        m[4] = m[4] - s4_6*(m[4] - (-m[3]                ));  // qx - energy flux
        m[6] = m[6] - s4_6*(m[6] - (-m[5]                ));  // qy - energy flux
        m[7] = m[7] - s7_8*(m[7] - (m[3]*m[3] - m[5]*m[5]));  // pxx - strain rate
        m[8] = m[8] - s7_8*(m[8] - (m[3]*m[5]            ));  // pxy - strain rate

        // transform back into distribution functions
        // f = Minv*m_+1
        for (int a = 0; a<Q; a++)
            f[f_index(Nx, Ny, x, y, a)] =  Minv[a*Q + 0]*m[0] + Minv[a*Q + 1]*m[1] + Minv[a*Q + 2]*m[2]
                                         + Minv[a*Q + 3]*m[3] + Minv[a*Q + 4]*m[4] + Minv[a*Q + 5]*m[5]
                                         + Minv[a*Q + 6]*m[6] + Minv[a*Q + 7]*m[7] + Minv[a*Q + 8]*m[8];
    
    }
    else
    {
        // Apply standard bounceback at boundaries
        f[f_index(Nx, Ny, x, y, 1)] = f3;
        f[f_index(Nx, Ny, x, y, 2)] = f4;
        f[f_index(Nx, Ny, x, y, 3)] = f1;
        f[f_index(Nx, Ny, x, y, 4)] = f2;
        f[f_index(Nx, Ny, x, y, 5)] = f7;
        f[f_index(Nx, Ny, x, y, 6)] = f8;
        f[f_index(Nx, Ny, x, y, 7)] = f5;
        f[f_index(Nx, Ny, x, y, 8)] = f6;
    }
}


// lid driven cavity boudnary conditions
__global__ void stream_collide_gpu_lid(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u0, float* f, bool* solid_node, float tau, bool save, is_periodic<false>, use_LES<false>, use_MRT<false>)
{	
	int y = blockIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;

	// don't stream beyond boundary nodes
    int yn = (y>0) ? y-1 : -1;
	int yp = (y<Ny-1) ? y+1 : -1;
    int xn = (x>0) ? x-1 : -1;;
    int xp = (x<Nx-1) ? x+1 : -1;

    float f0=-1, f1=-1, f2=-1, f3=-1, f4=-1, f5=-1, f6=-1, f7=-1, f8=-1;

                              f0 = f[f_index(Nx, Ny, x, y, 0)];
    if (xn != -1            ) f1 = f[f_index(Nx, Ny, xn, y, 1)];
    if (yn != -1            ) f2 = f[f_index(Nx, Ny, x, yn, 2)];
    if (xp != -1            ) f3 = f[f_index(Nx, Ny, xp, y, 3)];
    if (yp != -1            ) f4 = f[f_index(Nx, Ny, x, yp, 4)];
    if (xn != -1 && yn != -1) f5 = f[f_index(Nx, Ny, xn, yn, 5)];
    if (xp != -1 && yn != -1) f6 = f[f_index(Nx, Ny, xp, yn, 6)];
    if (xp != -1 && yp != -1) f7 = f[f_index(Nx, Ny, xp, yp, 7)];
    if (xn != -1 && yp != -1) f8 = f[f_index(Nx, Ny, xn, yp, 8)];

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

    // velocity BCs on north-side (lid) using bounceback on a moving wall
    // as it has better stability than Ze & Hou
    // eq (5.26)
    // Krueger T, Kusumaatmaja H, Kuzmin A, Shardt O, Silva G, Viggen EM.
    // The Lattice Boltzmann Method: Principles and Practice. Springer, 2016. 690 p.
    if ((y == Ny - 1) && (x > 0) && (x < Nx-1))
	{
		float rho0 = f0 + f1 + f3 + 2.*(f2 + f5 + f6);
		float ru = rho0*u0;
		f4 = f2;
        f7 = f5 - 1./6.*ru;
		f8 = f6 + 1./6.*ru;
	}

    // corner of south-west inlet
    if ((x == 0) && (y == 0))
    {
        // streaming from solid nodes, zero from standard bounceback
        f6 = 0;
        f8 = 0;
    }

	// 	corner of south-east outlet
    if ((x == Nx-1) && (y == 0))
    {
        // streaming from solid nodes, zero from standard bounceback
        f5 = 0;
        f7 = 0;
    }

    // unknown distributions at singular corner points are
    //  considered part of the lid and extrapolated it

	// corner of north-west inlet
    if ((x == 0) && (y == Ny-1))
    {
        f0 = f[f_index(Nx, Ny, 1, Ny-1, 0)];
        f1 = f[f_index(Nx, Ny, 1, Ny-1, 1)];
        f2 = f[f_index(Nx, Ny, 1, Ny-1, 2)];
        f3 = f[f_index(Nx, Ny, 1, Ny-1, 3)];
        f4 = f[f_index(Nx, Ny, 1, Ny-1, 4)];
        f5 = f[f_index(Nx, Ny, 1, Ny-1, 5)];
        f6 = f[f_index(Nx, Ny, 1, Ny-1, 6)];
        f7 = f[f_index(Nx, Ny, 1, Ny-1, 7)];
        f8 = f[f_index(Nx, Ny, 1, Ny-1, 8)];
    }

	// corner of north-east outlet
    if ((x == Nx-1) && (y == Ny-1))
    {

        f0 = f[f_index(Nx, Ny, Nx-2, Ny-1, 0)];
        f1 = f[f_index(Nx, Ny, Nx-2, Ny-1, 1)];
        f2 = f[f_index(Nx, Ny, Nx-2, Ny-1, 2)];
        f3 = f[f_index(Nx, Ny, Nx-2, Ny-1, 3)];
        f4 = f[f_index(Nx, Ny, Nx-2, Ny-1, 4)];
        f5 = f[f_index(Nx, Ny, Nx-2, Ny-1, 5)];
        f6 = f[f_index(Nx, Ny, Nx-2, Ny-1, 6)];
        f7 = f[f_index(Nx, Ny, Nx-2, Ny-1, 7)];
        f8 = f[f_index(Nx, Ny, Nx-2, Ny-1, 8)];
    }
    
    float w0 = 4./9., w1 = 1./9., w2 = 1./36.;
	float c2 = 9./2.;
	float omega = 1/tau;
	float one_omega = 1 - omega; // 1 - 1/tau

    // compute density
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    if (rho < 0 )
    {
    printf("Fatal error: Negative density at ( %d , %d )\n", x, y);
    assert(0);
    }


    int cord = x + Nx*y;
    if (!solid_node[cord])
    {
        // compute velocities quantities
        float ux = (f1 + f5 + f8 - (f3 + f6 + f7))/rho;
        float uy = (f2 + f5 + f6 - (f4 + f7 + f8))/rho;

        // store to memory only when needed for output
        if (save)
        {
            ux_arr[cord] = ux;
            uy_arr[cord] = uy;
            //rho_arr[cord] = rho;
        }

        float w_rho0_omega = w0 * rho * omega;
        float w_rho1_omega = w1 * rho * omega;
        float w_rho2_omega = w2 * rho * omega;

        float uxsq = ux * ux;
        float uysq = uy * uy;
        float usq = uxsq + uysq;

        float uxuy5 =  ux + uy;
        float uxuy6 = -ux + uy;
        float uxuy7 = -ux - uy;
        float uxuy8 =  ux - uy;

        float c = 1 - 1.5*usq;
        f[f_index(Nx, Ny, x, y, 0)] = one_omega*f0 + w_rho0_omega*(c                           );
        f[f_index(Nx, Ny, x, y, 1)] = one_omega*f1 + w_rho1_omega*(c + 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 2)] = one_omega*f2 + w_rho1_omega*(c + 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 3)] = one_omega*f3 + w_rho1_omega*(c - 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 4)] = one_omega*f4 + w_rho1_omega*(c - 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 5)] = one_omega*f5 + w_rho2_omega*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
        f[f_index(Nx, Ny, x, y, 6)] = one_omega*f6 + w_rho2_omega*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
        f[f_index(Nx, Ny, x, y, 7)] = one_omega*f7 + w_rho2_omega*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
        f[f_index(Nx, Ny, x, y, 8)] = one_omega*f8 + w_rho2_omega*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);
    }
    else
    {
        // Apply standard bounceback at boundaries
        f[f_index(Nx, Ny, x, y, 1)] = f3;
        f[f_index(Nx, Ny, x, y, 2)] = f4;
        f[f_index(Nx, Ny, x, y, 3)] = f1;
        f[f_index(Nx, Ny, x, y, 4)] = f2;
        f[f_index(Nx, Ny, x, y, 5)] = f7;
        f[f_index(Nx, Ny, x, y, 6)] = f8;
        f[f_index(Nx, Ny, x, y, 7)] = f5;
        f[f_index(Nx, Ny, x, y, 8)] = f6;
    }
}


// lid driven cavity boudnary conditions - with LES
__global__ void stream_collide_gpu_lid(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float u0, float* f, bool* solid_node, float tau, bool save, is_periodic<false>, use_LES<true>,  use_MRT<false>)
{	
	int y = blockIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;

	// don't stream beyond boundary nodes
    int yn = (y>0) ? y-1 : -1;
	int yp = (y<Ny-1) ? y+1 : -1;
    int xn = (x>0) ? x-1 : -1;;
    int xp = (x<Nx-1) ? x+1 : -1;

    float f0=-1, f1=-1, f2=-1, f3=-1, f4=-1, f5=-1, f6=-1, f7=-1, f8=-1;

                              f0 = f[f_index(Nx, Ny, x, y, 0)];
    if (xn != -1            ) f1 = f[f_index(Nx, Ny, xn, y, 1)];
    if (yn != -1            ) f2 = f[f_index(Nx, Ny, x, yn, 2)];
    if (xp != -1            ) f3 = f[f_index(Nx, Ny, xp, y, 3)];
    if (yp != -1            ) f4 = f[f_index(Nx, Ny, x, yp, 4)];
    if (xn != -1 && yn != -1) f5 = f[f_index(Nx, Ny, xn, yn, 5)];
    if (xp != -1 && yn != -1) f6 = f[f_index(Nx, Ny, xp, yn, 6)];
    if (xp != -1 && yp != -1) f7 = f[f_index(Nx, Ny, xp, yp, 7)];
    if (xn != -1 && yp != -1) f8 = f[f_index(Nx, Ny, xn, yp, 8)];

    
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

    // velocity BCs on north-side (lid) using bounceback on a moving wall
    // as it has better stability than Ze & Hou
    // eq (5.26)
    // Krueger T, Kusumaatmaja H, Kuzmin A, Shardt O, Silva G, Viggen EM.
    // The Lattice Boltzmann Method: Principles and Practice. Springer, 2016. 690 p.
    if ((y == Ny - 1) && (x > 0) && (x < Nx-1))
	{
		float rho0 = f0 + f1 + f3 + 2.*(f2 + f5 + f6);
		float ru = rho0*u0;
		f4 = f2;
        f7 = f5 - 1./6.*ru;
		f8 = f6 + 1./6.*ru;
	}

    // corner of south-west inlet
    if ((x == 0) && (y == 0))
    {
        // streaming from solid nodes, zero from standard bounceback
        f6 = 0;
        f8 = 0;
    }

	// 	corner of south-east outlet
    if ((x == Nx-1) && (y == 0))
    {
        // streaming from solid nodes, zero from standard bounceback
        f5 = 0;
        f7 = 0;
    }

    // unknown distributions at singular corner points are
    //  considered part of the lid and extrapolated it

	// corner of north-west inlet
    if ((x == 0) && (y == Ny-1))
    {
        f0 = f[f_index(Nx, Ny, 1, Ny-1, 0)];
        f1 = f[f_index(Nx, Ny, 1, Ny-1, 1)];
        f2 = f[f_index(Nx, Ny, 1, Ny-1, 2)];
        f3 = f[f_index(Nx, Ny, 1, Ny-1, 3)];
        f4 = f[f_index(Nx, Ny, 1, Ny-1, 4)];
        f5 = f[f_index(Nx, Ny, 1, Ny-1, 5)];
        f6 = f[f_index(Nx, Ny, 1, Ny-1, 6)];
        f7 = f[f_index(Nx, Ny, 1, Ny-1, 7)];
        f8 = f[f_index(Nx, Ny, 1, Ny-1, 8)];
    }

	// corner of north-east outlet
    if ((x == Nx-1) && (y == Ny-1))
    {

        f0 = f[f_index(Nx, Ny, Nx-2, Ny-1, 0)];
        f1 = f[f_index(Nx, Ny, Nx-2, Ny-1, 1)];
        f2 = f[f_index(Nx, Ny, Nx-2, Ny-1, 2)];
        f3 = f[f_index(Nx, Ny, Nx-2, Ny-1, 3)];
        f4 = f[f_index(Nx, Ny, Nx-2, Ny-1, 4)];
        f5 = f[f_index(Nx, Ny, Nx-2, Ny-1, 5)];
        f6 = f[f_index(Nx, Ny, Nx-2, Ny-1, 6)];
        f7 = f[f_index(Nx, Ny, Nx-2, Ny-1, 7)];
        f8 = f[f_index(Nx, Ny, Nx-2, Ny-1, 8)];
    }

    // compute density
    float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    if (rho < 0 )
    {
    printf("Fatal error: negative density %f at ( %d , %d )\n", rho, x, y);
    assert(0);
    }

    int cord = x + Nx*y;
       if (!solid_node[cord])
    {

        // compute velocities
        float ux = (f1 + f5 + f8 - (f3 + f6 + f7))/rho;
        float uy = (f2 + f5 + f6 - (f4 + f7 + f8))/rho;

        // store to memory only when needed for output
        if (save)
        {
            ux_arr[cord] = ux;
            uy_arr[cord] = uy;
            //rho_arr[cord] = rho;
        }

        float uxsq = ux * ux;
        float uysq = uy * uy;
        float usq = uxsq + uysq;

        float uxuy5 =  ux + uy;
        float uxuy6 = -ux + uy;
        float uxuy7 = -ux - uy;
        float uxuy8 =  ux - uy;

        float c = 1 - 1.5*usq;
        float w0 = 4./9., w1 = 1./9., w2 = 1./36.;
	    float c2 = 9./2.;
        float w_rho0 = w0 * rho;
        float w_rho1 = w1 * rho;
        float w_rho2 = w2 * rho;

        // calculate equilibrium function
        float feq0 = w_rho0*(c                           );
        float feq1 = w_rho1*(c + 3.*ux  + c2*uxsq        );
        float feq2 = w_rho1*(c + 3.*uy  + c2*uysq        );
        float feq3 = w_rho1*(c - 3.*ux  + c2*uxsq        );
        float feq4 = w_rho1*(c - 3.*uy  + c2*uysq        );
        float feq5 = w_rho2*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
        float feq6 = w_rho2*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
        float feq7 = w_rho2*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
        float feq8 = w_rho2*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);

        // perform large eddy simulation
        float C_smg = 0.14;  // Smagorinsky constant, sets length scale as fraction of mesh size

         // filtered momentum flux Q_ij defined from non-equilibrium distribution functions
        // https://onlinelibrary.wiley.com/doi/full/10.1002/zamm.201900301#zamm201900301-bib-0038
        // eq (45)
        // double check this if not using MRT to estimate Q later
        // Q_ij = sum[ex[a]*ey[a]*(f_a - feq_a)]
        float Q_xx = (f1-feq1) + (f3-feq3) + (f5-feq5) + (f6-feq6) + (f7-feq7) + (f8-feq8);
        float Q_yy = (f2-feq2) + (f4-feq4) + (f5-feq5) + (f6-feq6) + (f7-feq7) + (f8-feq8);
        float Q_xy = (f5-feq5) - (f6-feq6) + (f7-feq7) - (f8-feq8);

        float Q_bar = std::sqrt(Q_xx*Q_xx + Q_yy*Q_yy + 2*Q_xy*Q_xy);

        // calculate turbulence viscosity from eq (22)
        float tau_turb = 0.5 * ( std::sqrt(tau*tau + 18.*C_smg*C_smg*Q_bar) - tau );

        float tau_eff = tau + tau_turb;  // effective viscosity

        //printf("x %d y %d Q_xx %f, Q_yy %f, Q_xy %f tau %f, tau_turb %f, tau_eff %f \n", x, y, Q_xx*Q_xx, Q_yy*Q_yy, Q_xy*Q_xy, tau, tau_turb, tau_eff);

        float omega = 1/tau_eff;
	    float one_omega = 1 - omega; // 1 - 1/tau

        // update distributions from LBM formula
        f[f_index(Nx, Ny, x, y, 0)] = one_omega*f0 + feq0*omega;
        f[f_index(Nx, Ny, x, y, 1)] = one_omega*f1 + feq1*omega;
        f[f_index(Nx, Ny, x, y, 2)] = one_omega*f2 + feq2*omega;
        f[f_index(Nx, Ny, x, y, 3)] = one_omega*f3 + feq3*omega;
        f[f_index(Nx, Ny, x, y, 4)] = one_omega*f4 + feq4*omega;
        f[f_index(Nx, Ny, x, y, 5)] = one_omega*f5 + feq5*omega;
        f[f_index(Nx, Ny, x, y, 6)] = one_omega*f6 + feq6*omega;
        f[f_index(Nx, Ny, x, y, 7)] = one_omega*f7 + feq7*omega;
        f[f_index(Nx, Ny, x, y, 8)] = one_omega*f8 + feq8*omega;
    }
    else
    {
        // Apply standard bounceback at all inner solids
        f[f_index(Nx, Ny, x, y, 1)] = f3;
        f[f_index(Nx, Ny, x, y, 2)] = f4;
        f[f_index(Nx, Ny, x, y, 3)] = f1;
        f[f_index(Nx, Ny, x, y, 4)] = f2;
        f[f_index(Nx, Ny, x, y, 5)] = f7;
        f[f_index(Nx, Ny, x, y, 6)] = f8;
        f[f_index(Nx, Ny, x, y, 7)] = f5;
        f[f_index(Nx, Ny, x, y, 8)] = f6;

    }
}





// channel flow boundary conditions
__global__ void stream_collide_gpu(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float ux0, float* f, bool* solid_node, float tau, bool save, is_periodic<false>, use_LES<false>)
{	
	int y = blockIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;

	// don't stream beyond boundary nodes
    int yn = (y>0) ? y-1 : -1;
	int yp = (y<Ny-1) ? y+1 : -1;
    int xn = (x>0) ? x-1 : -1;;
    int xp = (x<Nx-1) ? x+1 : -1;

    float f0=-1, f1=-1, f2=-1, f3=-1, f4=-1, f5=-1, f6=-1, f7=-1, f8=-1;

                              f0 = f[f_index(Nx, Ny, x, y, 0)];
    if (xn != -1            ) f1 = f[f_index(Nx, Ny, xn, y, 1)];
    if (yn != -1            ) f2 = f[f_index(Nx, Ny, x, yn, 2)];
    if (xp != -1            ) f3 = f[f_index(Nx, Ny, xp, y, 3)];
    if (yp != -1            ) f4 = f[f_index(Nx, Ny, x, yp, 4)];
    if (xn != -1 && yn != -1) f5 = f[f_index(Nx, Ny, xn, yn, 5)];
    if (xp != -1 && yn != -1) f6 = f[f_index(Nx, Ny, xp, yn, 6)];
    if (f[f_index(Nx, Ny, xp, yn, 6)] < 0)
        printf("x %d y %d xp %d yn %d pull f6 %f\n", x, y, xp, yn, f[f_index(Nx, Ny, xp, yn, 6)]);
    if (xp != -1 && yp != -1) f7 = f[f_index(Nx, Ny, xp, yp, 7)];
    if (xn != -1 && yp != -1) f8 = f[f_index(Nx, Ny, xn, yp, 8)];

	// velocity BCs on west-side (inlet) using Zou and He.
    if ((x == 0) && (y < Ny-1))
	{
		float rho0 = (f0 + f2 + f4 + 2.*(f3 + f7 + f6)) / (1. - ux0);
		float ru = rho0*ux0;
		f1 = f3 + (2./3.)*ru;
		f5 = f7 + (1./6.)*ru - 0.5*(f2 - f4);
        if (f5 < 0)
            printf("inlet x %d y %d f5 %f f7 %f ru %f f2 %f f4 %f\n", x, y, f5, f7, ru, f2, f4);
		f8 = f6 + (1./6.)*ru - 0.5*(f4 - f2);
	}

	// BCs at east-side (outlet) using extrapolation from previous node (Nx-2) xn x-dirn
    // Since pre-collision distributions are not stored, approximate by using post-collision distributions
    if ((x == Nx-1) && (y < Ny))
	{
		f0 = f[f_index(Nx, Ny, x-1, y, 0)];
		f1 = f[f_index(Nx, Ny, x-1, y, 1)];
		f2 = f[f_index(Nx, Ny, x-1, y, 2)];
		f3 = f[f_index(Nx, Ny, x-1, y, 3)];
		f4 = f[f_index(Nx, Ny, x-1, y, 4)];
		f5 = f[f_index(Nx, Ny, x-1, y, 5)];
		f6 = f[f_index(Nx, Ny, x-1, y, 6)];
		f7 = f[f_index(Nx, Ny, x-1, y, 7)];
		f8 = f[f_index(Nx, Ny, x-1, y, 8)];
	}

	// bounceback at top wall
    if ((y == Ny - 1) && (x < Nx-1))
	{
		f4 = f2;
		f7 = f5;
		f8 = f6;
	}

	// bounceback at bottom wall
    if ((y == 0) && (x < Nx-1))
	{
		f2 = f4;
		f5 = f7;
		f6 = f8;
	}

	// corners need special treatment as we have an extra unknown.
	// Treatment based on Zou & He (1997), for further details see
	// https://palabos-forum.unige.ch/t/corner-nodes-2d-channel-boundary-condition-zou-he/577/5

	// corner of south-west inlet
    if ((x == 0) && (y == 0))
    {
        // extrapolate density from neighbour node
        // approximate by using post-collision distribution
        float rho = 0.0;
        for (int a = 0; a < Q; a++)
            rho += f[f_index(Nx, Ny, 0, 1, a)];

        f1 = f3;
        f2 = f4;
        f5 = f7;
        // f6 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f5 + f7)) eq(29)
        f6 = 0.5*(rho - f0) - (f1 + f2 + f5);
        f8 = f6;
    }

	// 	corner of south-east outlet
    if ((x == Nx-1) && (y == 0))
    {
        //extrapolate neighbour density
        float rho = 0.0;
        for (int a = 0; a < Q; a++)
            rho += f[f_index(Nx, Ny, Nx-1, 1, a)];
        f2 = f4;
        f3 = f1;
        f6 = f8;
        // f5 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f6 + f8))
        f5 = 0.5*(rho - f0) - (f2 + f3 + f6);
        f7 = f5;
    }

	// corner of north-west inlet
    if ((x == 0) && (y == Ny-1))
    {
        //extrapolate neighbour density
        float rho = 0.0;
        for (int a = 0; a < Q; a++)
            rho += f[f_index(Nx, Ny, 0, Ny-2, a)];

        f1 = f3;
        f4 = f2;
        f8 = f6;
        // f5 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f6 + f8))
        f5 = 0.5*(rho - f0) - (f2 + f3 + f6);
        f7 = f5;
    }

	// corner of north-east outlet
    if ((x == Nx-1) && (y == Ny-1))
    {
        // extrapolate density from neighbour node
        float rho = 0.0;
        for (int a = 0; a < Q; a++)
            rho += f[f_index(Nx, Ny, Nx-1, Ny-2, a)];

        f3 = f1;
        f4 = f2;
        f7 = f5;
        // f6 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f5 + f7))
        f6 = 0.5*(rho - f0) - (f3 + f4 + f7);
        f8 = f6;
    }

    
    float w0 = 4./9., w1 = 1./9., w2 = 1./36.;
	float c2 = 9./2.;
	float omega = 1/tau;
	float one_omega = 1 - omega; // 1 - 1/tau

    int cord = x + Nx*y;
    if (!solid_node[cord])
    {

        // compute macroscopic quantities
        float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
        float ux = (f1 + f5 + f8 - (f3 + f6 + f7))/rho;
        float uy = (f2 + f5 + f6 - (f4 + f7 + f8))/rho;

        // store to memory only when needed for output
        if (save)
        {
            ux_arr[cord] = ux;
            uy_arr[cord] = uy;
            //rho_arr[cord] = rho;
        }

        float w_rho0_omega = w0 * rho * omega;
        float w_rho1_omega = w1 * rho * omega;
        float w_rho2_omega = w2 * rho * omega;

        float uxsq = ux * ux;
        float uysq = uy * uy;
        float usq = uxsq + uysq;

        float uxuy5 =  ux + uy;
        float uxuy6 = -ux + uy;
        float uxuy7 = -ux - uy;
        float uxuy8 =  ux - uy;

        float c = 1 - 1.5*usq;
        f[f_index(Nx, Ny, x, y, 0)] = one_omega*f0 + w_rho0_omega*(c                           );
        f[f_index(Nx, Ny, x, y, 1)] = one_omega*f1 + w_rho1_omega*(c + 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 2)] = one_omega*f2 + w_rho1_omega*(c + 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 3)] = one_omega*f3 + w_rho1_omega*(c - 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 4)] = one_omega*f4 + w_rho1_omega*(c - 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 5)] = one_omega*f5 + w_rho2_omega*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
        f[f_index(Nx, Ny, x, y, 6)] = one_omega*f6 + w_rho2_omega*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
        f[f_index(Nx, Ny, x, y, 7)] = one_omega*f7 + w_rho2_omega*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
        f[f_index(Nx, Ny, x, y, 8)] = one_omega*f8 + w_rho2_omega*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);
    }
    else
    {
        // Apply standard bounceback at all inner solids (on-grid)
        f[f_index(Nx, Ny, x, y, 1)] = f3;
        f[f_index(Nx, Ny, x, y, 2)] = f4;
        f[f_index(Nx, Ny, x, y, 3)] = f1;
        f[f_index(Nx, Ny, x, y, 4)] = f2;
        f[f_index(Nx, Ny, x, y, 5)] = f7;
        f[f_index(Nx, Ny, x, y, 6)] = f8;
        f[f_index(Nx, Ny, x, y, 7)] = f5;
        f[f_index(Nx, Ny, x, y, 8)] = f6;
    }
}

// periodic boundary conditions
__global__ void stream_collide_gpu(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float ux0, float* f, bool* solid_node, float tau, bool save, is_periodic<true>, use_LES<false>)
{	
	int y = blockIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    
    int yn = (y>0   ) ? (y-1) : (Ny-1);
    int yp = (y<Ny-1) ? (y+1) : (0   );
    int xn = (x>0   ) ? (x-1) : (Nx-1);
    int xp = (x<Nx-1) ? (x+1) : (0   );
    
    // stream by pulling densities from neighbour nodes
    float f0 = f[f_index(Nx, Ny, x, y, 0)];
    float f1 = f[f_index(Nx, Ny, xn, y, 1)];
    float f2 = f[f_index(Nx, Ny, x, yn, 2)];
    float f3 = f[f_index(Nx, Ny, xp, y, 3)];
    float f4 = f[f_index(Nx, Ny, x, yp, 4)];
    float f5 = f[f_index(Nx, Ny, xn, yn, 5)];
    float f6 = f[f_index(Nx, Ny, xp, yn, 6)];
    float f7 = f[f_index(Nx, Ny, xp, yp, 7)];
    float f8 = f[f_index(Nx, Ny, xn, yp, 8)];
    
    
    float w0 = 4./9., w1 = 1./9., w2 = 1./36.;
	float c2 = 9./2.;
	float omega = 1/tau;
	float one_omega = 1 - omega; // 1 - 1/tau

    int cord = x + Nx*y;
    if (!solid_node[cord])
    {
        // compute macroscopic quantities
        float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
        float ux = (f1 + f5 + f8 - (f3 + f6 + f7))/rho;
        float uy = (f2 + f5 + f6 - (f4 + f7 + f8))/rho;

        // store to memory only when needed for output
        if (save)
        {
            ux_arr[cord] = ux;
            uy_arr[cord] = uy;
            //rho_arr[cord] = rho;
        }

        float w_rho0_omega = w0 * rho * omega;
        float w_rho1_omega = w1 * rho * omega;
        float w_rho2_omega = w2 * rho * omega;

        float uxsq = ux * ux;
        float uysq = uy * uy;
        float usq = uxsq + uysq;

        float uxuy5 =  ux + uy;
        float uxuy6 = -ux + uy;
        float uxuy7 = -ux - uy;
        float uxuy8 =  ux - uy;

        float c = 1 - 1.5*usq;
        f[f_index(Nx, Ny, x, y, 0)] = one_omega*f0 + w_rho0_omega*(c                           );
        f[f_index(Nx, Ny, x, y, 1)] = one_omega*f1 + w_rho1_omega*(c + 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 2)] = one_omega*f2 + w_rho1_omega*(c + 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 3)] = one_omega*f3 + w_rho1_omega*(c - 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 4)] = one_omega*f4 + w_rho1_omega*(c - 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 5)] = one_omega*f5 + w_rho2_omega*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
        f[f_index(Nx, Ny, x, y, 6)] = one_omega*f6 + w_rho2_omega*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
        f[f_index(Nx, Ny, x, y, 7)] = one_omega*f7 + w_rho2_omega*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
        f[f_index(Nx, Ny, x, y, 8)] = one_omega*f8 + w_rho2_omega*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);
    }
    else
    {
        // Apply standard bounceback at all inner solids (on-grid)
        f[f_index(Nx, Ny, x, y, 1)] = f3;
        f[f_index(Nx, Ny, x, y, 2)] = f4;
        f[f_index(Nx, Ny, x, y, 3)] = f1;
        f[f_index(Nx, Ny, x, y, 4)] = f2;
        f[f_index(Nx, Ny, x, y, 5)] = f7;
        f[f_index(Nx, Ny, x, y, 6)] = f8;
        f[f_index(Nx, Ny, x, y, 7)] = f5;
        f[f_index(Nx, Ny, x, y, 8)] = f6;
    }
}

// periodic boundary conditions with Large eddy simulation (LES)
// LES equations implemented from:
// Krafczyk, Manfred & Tolke, J & Luo, Li-Shi. (2003).
// Large eddy simulation with a multiple-relaxation-time LBE model.
// INTERNATIONAL JOURNAL OF MODERN PHYSICS B. 17. 33-39. 10.1142/S0217979203017059. 
__global__ void stream_collide_gpu(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float ux0, float* f, bool* solid_node, float tau, bool save, is_periodic<true>, use_LES<true>)
{	
	int y = blockIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;

	int yn = (y>0   ) ? (y-1) : (Ny-1);
    int yp = (y<Ny-1) ? (y+1) : (0   );
    int xn = (x>0   ) ? (x-1) : (Nx-1);
    int xp = (x<Nx-1) ? (x+1) : (0   );
    
    // stream by pulling densities from neighbour nodes
    float f0 = f[f_index(Nx, Ny, x, y, 0)];
    float f1 = f[f_index(Nx, Ny, xn, y, 1)];
    float f2 = f[f_index(Nx, Ny, x, yn, 2)];
    float f3 = f[f_index(Nx, Ny, xp, y, 3)];
    float f4 = f[f_index(Nx, Ny, x, yp, 4)];
    float f5 = f[f_index(Nx, Ny, xn, yn, 5)];
    float f6 = f[f_index(Nx, Ny, xp, yn, 6)];
    float f7 = f[f_index(Nx, Ny, xp, yp, 7)];
    float f8 = f[f_index(Nx, Ny, xn, yp, 8)];

    int cord = x + Nx*y;
    if (!solid_node[cord])
    {

        // compute macroscopic quantities
        float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
        float ux = (f1 + f5 + f8 - (f3 + f6 + f7))/rho;
        float uy = (f2 + f5 + f6 - (f4 + f7 + f8))/rho;

        // store to memory only when needed for output
        if (save)
        {
            ux_arr[cord] = ux;
            uy_arr[cord] = uy;
            //rho_arr[cord] = rho;
        }

        float uxsq = ux * ux;
        float uysq = uy * uy;
        float usq = uxsq + uysq;

        float uxuy5 =  ux + uy;
        float uxuy6 = -ux + uy;
        float uxuy7 = -ux - uy;
        float uxuy8 =  ux - uy;

        float c = 1 - 1.5*usq;
        float w0 = 4./9., w1 = 1./9., w2 = 1./36.;
	    float c2 = 9./2.;
        float w_rho0 = w0 * rho;
        float w_rho1 = w1 * rho;
        float w_rho2 = w2 * rho;

        // calculate equilibrium function
        float feq0 = w_rho0*(c                           );
        float feq1 = w_rho1*(c + 3.*ux  + c2*uxsq        );
        float feq2 = w_rho1*(c + 3.*uy  + c2*uysq        );
        float feq3 = w_rho1*(c - 3.*ux  + c2*uxsq        );
        float feq4 = w_rho1*(c - 3.*uy  + c2*uysq        );
        float feq5 = w_rho2*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
        float feq6 = w_rho2*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
        float feq7 = w_rho2*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
        float feq8 = w_rho2*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);

        // perform large eddy simulation
        float C_smg = 0.1;  // Smagorinsky constant, sets length scale as fraction of mesh size

         // filtered momentum flux Q_ij defined from non-equilibrium distribution functions
        // https://onlinelibrary.wiley.com/doi/full/10.1002/zamm.201900301#zamm201900301-bib-0038
        // eq (45)
        // double check this if not using MRT to estimate Q later
        // Q_ij = sum[ex[a]*ey[a]*(f_a - feq_a)]
        float Q_xx = (f1-feq1) + (f3-feq3) + (f5-feq5) + (f6-feq6) + (f7-feq7) + (f8-feq8);
        float Q_yy = (f2-feq2) + (f4-feq4) + (f5-feq5) + (f6-feq6) + (f7-feq7) + (f8-feq8);
        float Q_xy = (f5-feq5) - (f6-feq6) + (f7-feq7) - (f8-feq8);

        float Q_bar = std::sqrt(Q_xx*Q_xx + Q_yy*Q_yy + 2*Q_xy*Q_xy);

        // calculate turbulence viscosity from eq (22)
        float tau_turb = 0.5 * ( std::sqrt(tau*tau + 18.*C_smg*C_smg*Q_bar) - tau );

        float tau_eff = tau + tau_turb;  // effective viscosity

        //printf("x %d y %d Q_xx %f, Q_yy %f, Q_xy %f tau %f, tau_turb %f, tau_eff %f \n", x, y, Q_xx*Q_xx, Q_yy*Q_yy, Q_xy*Q_xy, tau, tau_turb, tau_eff);

        float omega = 1/tau_eff;
	    float one_omega = 1 - omega; // 1 - 1/tau

        // update distributions from LBM formula
        f[f_index(Nx, Ny, x, y, 0)] = one_omega*f0 + feq0*omega;
        f[f_index(Nx, Ny, x, y, 1)] = one_omega*f1 + feq1*omega;
        f[f_index(Nx, Ny, x, y, 2)] = one_omega*f2 + feq2*omega;
        f[f_index(Nx, Ny, x, y, 3)] = one_omega*f3 + feq3*omega;
        f[f_index(Nx, Ny, x, y, 4)] = one_omega*f4 + feq4*omega;
        f[f_index(Nx, Ny, x, y, 5)] = one_omega*f5 + feq5*omega;
        f[f_index(Nx, Ny, x, y, 6)] = one_omega*f6 + feq6*omega;
        f[f_index(Nx, Ny, x, y, 7)] = one_omega*f7 + feq7*omega;
        f[f_index(Nx, Ny, x, y, 8)] = one_omega*f8 + feq8*omega;
    }
    else
    {
        // Apply standard bounceback at all inner solids
        f[f_index(Nx, Ny, x, y, 1)] = f3;
        f[f_index(Nx, Ny, x, y, 2)] = f4;
        f[f_index(Nx, Ny, x, y, 3)] = f1;
        f[f_index(Nx, Ny, x, y, 4)] = f2;
        f[f_index(Nx, Ny, x, y, 5)] = f7;
        f[f_index(Nx, Ny, x, y, 6)] = f8;
        f[f_index(Nx, Ny, x, y, 7)] = f5;
        f[f_index(Nx, Ny, x, y, 8)] = f6;
    }
}


// Channel flow boundary conditions with Large eddy simulation (LES)
// LES equations implemented from:
// Krafczyk, Manfred & Tolke, J & Luo, Li-Shi. (2003).
// Large eddy simulation with a multiple-relaxation-time LBE model.
// INTERNATIONAL JOURNAL OF MODERN PHYSICS B. 17. 33-39. 10.1142/S0217979203017059. 
__global__ void stream_collide_gpu(int Nx, int Ny, float* rho_arr, float* ux_arr, float* uy_arr, float ux0, float* f, bool* solid_node, float tau, bool save, is_periodic<false>, use_LES<true>)
{	
	int y = blockIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;

	// don't stream beyond boundary nodes    
    int yn = (y>0) ? (y-1) : -1;
	int yp = (y<Ny-1) ? (y+1) : -1;
    int xn = (x>0) ? (x-1) : -1;
    int xp = (x<Nx-1) ? (x+1) : -1;

    float f0=-1, f1=-1, f2=-1, f3=-1, f4=-1, f5=-1, f6=-1, f7=-1, f8=-1;

                              f0 = f[f_index(Nx, Ny, x, y, 0)];
    if (xn != -1            ) f1 = f[f_index(Nx, Ny, xn, y, 1)];
    if (yn != -1            ) f2 = f[f_index(Nx, Ny, x, yn, 2)];
    if (xp != -1            ) f3 = f[f_index(Nx, Ny, xp, y, 3)];
    if (yp != -1            ) f4 = f[f_index(Nx, Ny, x, yp, 4)];
    if (xn != -1 && yn != -1) f5 = f[f_index(Nx, Ny, xn, yn, 5)];
    if (xp != -1 && yn != -1) f6 = f[f_index(Nx, Ny, xp, yn, 6)];
    if (xp != -1 && yp != -1) f7 = f[f_index(Nx, Ny, xp, yp, 7)];
    if (xn != -1 && yp != -1) f8 = f[f_index(Nx, Ny, xn, yp, 8)];

	// velocity BCs on west-side (inlet) using Zou and He.
    if ((x == 0) && (y < Ny-1))
	{
		float rho0 = (f0 + f2 + f4 + 2.*(f3 + f7 + f6)) / (1. - ux0);
		float ru = rho0*ux0;
		f1 = f3 + (2./3.)*ru;
		f5 = f7 + (1./6.)*ru - 0.5*(f2 - f4);
		f8 = f6 + (1./6.)*ru - 0.5*(f4 - f2);
	}

	// BCs at east-side (outlet) using extrapolation from previous node (Nx-2) xn x-dirn
    // Since pre-collision distributions are not stored, approximate by using post-collision distributions
    if ((x == Nx-1) && (y < Ny))
	{
		f0 = f[f_index(Nx, Ny, x-1, y, 0)];
		f1 = f[f_index(Nx, Ny, x-1, y, 1)];
		f2 = f[f_index(Nx, Ny, x-1, y, 2)];
		f3 = f[f_index(Nx, Ny, x-1, y, 3)];
		f4 = f[f_index(Nx, Ny, x-1, y, 4)];
		f5 = f[f_index(Nx, Ny, x-1, y, 5)];
		f6 = f[f_index(Nx, Ny, x-1, y, 6)];
		f7 = f[f_index(Nx, Ny, x-1, y, 7)];
		f8 = f[f_index(Nx, Ny, x-1, y, 8)];
	}

	// bounceback at top wall
    if ((y == Ny - 1) && (x < Nx-1))
	{
		f4 = f2;
		f7 = f5;
		f8 = f6;
	}

	// bounceback at bottom wall
    if ((y == 0) && (x < Nx-1))
	{
		f2 = f4;
		f5 = f7;
		f6 = f8;
	}

	// corners need special treatment as we have an extra unknown.
	// Treatment based on Zou & He (1997), for further details see
	// www.palabos-forum.unige.ch/t/corner-nodes-2d-channel-boundary-condition-zou-he/577/5

	// corner of south-west inlet
    if ((x == 0) && (y == 0))
    {
        // extrapolate density from neighbour node
        // approximate by using post-collision distribution
        float rho = 0.0;
        for (int a = 0; a < Q; a++)
            rho += f[f_index(Nx, Ny, 0, 1, a)];

        f1 = f3;
        f2 = f4;
        f5 = f7;
        // f6 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f5 + f7))
        f6 = 0.5*(rho - f0) - (f1 + f2 + f5);
        f8 = f6;
    }

	// 	corner of south-east outlet
    if ((x == Nx-1) && (y == 0))
    {
        //extrapolate neighbour density
        float rho = 0.0;
        for (int a = 0; a < Q; a++)
            rho += f[f_index(Nx, Ny, Nx-1, 1, a)];
        f2 = f4;
        f3 = f1;
        f6 = f8;
        // f5 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f6 + f8))
        f5 = 0.5*(rho - f0) - (f2 + f3 + f6);
        f7 = f5;
    }

	// corner of north-west inlet
    if ((x == 0) && (y == Ny-1))
    {
        //extrapolate neighbour density
        float rho = 0.0;
        for (int a = 0; a < Q; a++)
            rho += f[f_index(Nx, Ny, 0, Ny-2, a)];

        f1 = f3;
        f4 = f2;
        f8 = f6;
        // f5 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f6 + f8))
        f5 = 0.5*(rho - f0) - (f2 + f3 + f6);
        f7 = f5;
    }

	// corner of north-east outlet
    if ((x == Nx-1) && (y == Ny-1))
    {
        // extrapolate density from neighbour node
        float rho = 0.0;
        for (int a = 0; a < Q; a++)
            rho += f[f_index(Nx, Ny, Nx-1, Ny-2, a)];

        f3 = f1;
        f4 = f2;
        f7 = f5;
        // f6 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f5 + f7))
        f6 = 0.5*(rho - f0) - (f3 + f4 + f7);
        f8 = f6;
    }

    int cord = x + Nx*y;
    if (!solid_node[cord])
    {

        // compute macroscopic quantities
        float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
        float ux = (f1 + f5 + f8 - (f3 + f6 + f7))/rho;
        float uy = (f2 + f5 + f6 - (f4 + f7 + f8))/rho;

        // store to memory only when needed for output
        if (save)
        {
            ux_arr[cord] = ux;
            uy_arr[cord] = uy;
            //rho_arr[cord] = rho;
        }

        float uxsq = ux * ux;
        float uysq = uy * uy;
        float usq = uxsq + uysq;

        float uxuy5 =  ux + uy;
        float uxuy6 = -ux + uy;
        float uxuy7 = -ux - uy;
        float uxuy8 =  ux - uy;

        float c = 1 - 1.5*usq;
        float w0 = 4./9., w1 = 1./9., w2 = 1./36.;
	    float c2 = 9./2.;
        float w_rho0 = w0 * rho;
        float w_rho1 = w1 * rho;
        float w_rho2 = w2 * rho;

        // calculate equilibrium function
        float feq0 = w_rho0*(c                           );
        float feq1 = w_rho1*(c + 3.*ux  + c2*uxsq        );
        float feq2 = w_rho1*(c + 3.*uy  + c2*uysq        );
        float feq3 = w_rho1*(c - 3.*ux  + c2*uxsq        );
        float feq4 = w_rho1*(c - 3.*uy  + c2*uysq        );
        float feq5 = w_rho2*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
        float feq6 = w_rho2*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
        float feq7 = w_rho2*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
        float feq8 = w_rho2*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);

        // perform large eddy simulation
        float C_smg = 0.1;  // Smagorinsky constant, sets length scale as fraction of mesh size

         // filtered momentum flux Q_ij defined from non-equilibrium distribution functions
        // https://onlinelibrary.wiley.com/doi/full/10.1002/zamm.201900301#zamm201900301-bib-0038
        // eq (45)
        // double check this if not using MRT to estimate Q later
        // Q_ij = sum[ex[a]*ey[a]*(f_a - feq_a)]
        float Q_xx = (f1-feq1) + (f3-feq3) + (f5-feq5) + (f6-feq6) + (f7-feq7) + (f8-feq8);
        float Q_yy = (f2-feq2) + (f4-feq4) + (f5-feq5) + (f6-feq6) + (f7-feq7) + (f8-feq8);
        float Q_xy = (f5-feq5) - (f6-feq6) + (f7-feq7) - (f8-feq8);

        float Q_bar = std::sqrt(Q_xx*Q_xx + Q_yy*Q_yy + 2*Q_xy*Q_xy);

        // calculate turbulence viscosity from eq (22)
        float tau_turb = 0.5 * ( std::sqrt(tau*tau + 18.*C_smg*C_smg*Q_bar) - tau );

        float tau_eff = tau + tau_turb;  // effective viscosity

        //printf("x %d y %d Q_xx %f, Q_yy %f, Q_xy %f tau %f, tau_turb %f, tau_eff %f \n", x, y, Q_xx*Q_xx, Q_yy*Q_yy, Q_xy*Q_xy, tau, tau_turb, tau_eff);

        float omega = 1/tau_eff;
	    float one_omega = 1 - omega; // 1 - 1/tau

        // update distributions from LBM formula
        f[f_index(Nx, Ny, x, y, 0)] = one_omega*f0 + feq0*omega;
        f[f_index(Nx, Ny, x, y, 1)] = one_omega*f1 + feq1*omega;
        f[f_index(Nx, Ny, x, y, 2)] = one_omega*f2 + feq2*omega;
        f[f_index(Nx, Ny, x, y, 3)] = one_omega*f3 + feq3*omega;
        f[f_index(Nx, Ny, x, y, 4)] = one_omega*f4 + feq4*omega;
        f[f_index(Nx, Ny, x, y, 5)] = one_omega*f5 + feq5*omega;
        f[f_index(Nx, Ny, x, y, 6)] = one_omega*f6 + feq6*omega;
        f[f_index(Nx, Ny, x, y, 7)] = one_omega*f7 + feq7*omega;
        f[f_index(Nx, Ny, x, y, 8)] = one_omega*f8 + feq8*omega;
    }
    else
    {
        // Apply standard bounceback at all inner solids
        f[f_index(Nx, Ny, x, y, 1)] = f3;
        f[f_index(Nx, Ny, x, y, 2)] = f4;
        f[f_index(Nx, Ny, x, y, 3)] = f1;
        f[f_index(Nx, Ny, x, y, 4)] = f2;
        f[f_index(Nx, Ny, x, y, 5)] = f7;
        f[f_index(Nx, Ny, x, y, 6)] = f8;
        f[f_index(Nx, Ny, x, y, 7)] = f5;
        f[f_index(Nx, Ny, x, y, 8)] = f6;
    }
}