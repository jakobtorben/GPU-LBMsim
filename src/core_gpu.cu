#include "core_gpu.hpp"
#include <iostream>  // delete later
#include <stdio.h>
#include <cuda.h>

// D2Q9 streaming direction scheme
// 6 2 5
// 3 0 1
// 7 4 8

__device__ __forceinline__ size_t f_index(unsigned int Nx, unsigned int Ny, unsigned int x, unsigned int y, unsigned int a)
{
    return ((Ny*a + y)*Nx + x);
    //return (x + Nx*y)*9 + a;
}

__global__ void stream_collide_gpu(unsigned int Nx, unsigned int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float ux0, float* f, float* ftemp, bool* solid_node, float tau, bool save)
{	
	unsigned int y = blockIdx.y;
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

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
	// palabos-forum.unige.ch/t/corner-nodes-2d-channel-boundary-condition-zou-he/577/5

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
    
    float w0 = 4./9., w1 = 1./9., w2 = 1./36.;
	float c2 = 9./2.;
	float tauinv = 1/tau;
	float one_tauinv = 1 - tauinv; // 1 - 1/tau

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

        float w_rho0_tauinv = w0 * rho * tauinv;
        float w_rho1_tauinv = w1 * rho * tauinv;
        float w_rho2_tauinv = w2 * rho * tauinv;

        float uxsq = ux * ux;
        float uysq = uy * uy;
        float usq = uxsq + uysq;

        float uxuy5 =  ux + uy;
        float uxuy6 = -ux + uy;
        float uxuy7 = -ux - uy;
        float uxuy8 =  ux - uy;

        float c = 1 - 1.5*usq;
        f[f_index(Nx, Ny, x, y, 0)] = one_tauinv*f0 + w_rho0_tauinv*(c                           );
        f[f_index(Nx, Ny, x, y, 1)] = one_tauinv*f1 + w_rho1_tauinv*(c + 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 2)] = one_tauinv*f2 + w_rho1_tauinv*(c + 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 3)] = one_tauinv*f3 + w_rho1_tauinv*(c - 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 4)] = one_tauinv*f4 + w_rho1_tauinv*(c - 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 5)] = one_tauinv*f5 + w_rho2_tauinv*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
        f[f_index(Nx, Ny, x, y, 6)] = one_tauinv*f6 + w_rho2_tauinv*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
        f[f_index(Nx, Ny, x, y, 7)] = one_tauinv*f7 + w_rho2_tauinv*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
        f[f_index(Nx, Ny, x, y, 8)] = one_tauinv*f8 + w_rho2_tauinv*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);
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


__global__ void stream_collide_periodic_gpu(unsigned int Nx, unsigned int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float* f, float* ftemp, bool* solid_node, float tau, bool save)
{	
	unsigned int y = blockIdx.y;
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    
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
	float tauinv = 1/tau;
	float one_tauinv = 1 - tauinv; // 1 - 1/tau

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

        float w_rho0_tauinv = w0 * rho * tauinv;
        float w_rho1_tauinv = w1 * rho * tauinv;
        float w_rho2_tauinv = w2 * rho * tauinv;

        float uxsq = ux * ux;
        float uysq = uy * uy;
        float usq = uxsq + uysq;

        float uxuy5 =  ux + uy;
        float uxuy6 = -ux + uy;
        float uxuy7 = -ux - uy;
        float uxuy8 =  ux - uy;

        float c = 1 - 1.5*usq;
        f[f_index(Nx, Ny, x, y, 0)] = one_tauinv*f0 + w_rho0_tauinv*(c                           );
        f[f_index(Nx, Ny, x, y, 1)] = one_tauinv*f1 + w_rho1_tauinv*(c + 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 2)] = one_tauinv*f2 + w_rho1_tauinv*(c + 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 3)] = one_tauinv*f3 + w_rho1_tauinv*(c - 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 4)] = one_tauinv*f4 + w_rho1_tauinv*(c - 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 5)] = one_tauinv*f5 + w_rho2_tauinv*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
        f[f_index(Nx, Ny, x, y, 6)] = one_tauinv*f6 + w_rho2_tauinv*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
        f[f_index(Nx, Ny, x, y, 7)] = one_tauinv*f7 + w_rho2_tauinv*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
        f[f_index(Nx, Ny, x, y, 8)] = one_tauinv*f8 + w_rho2_tauinv*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);
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