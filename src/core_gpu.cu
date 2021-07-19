#include "core_gpu.hpp"
#include <iostream>  // delete later
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

// streaming step - without periodic boundary condittions
__global__ void stream_gpu(unsigned int Nx, unsigned int Ny, float* ftemp, float* f, bool* solid_node)
{
    unsigned int y = blockIdx.y;
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	// don't stream beyond boundary nodes
	int yn = (y>0) ? (y-1) : -1;
	int yp = (y<Ny-1) ? (y+1) : -1;
    int xn = (x>0) ? (x-1) : -1;
    int xp = (x<Nx-1) ? (x+1) : -1;

                              ftemp[f_index(Nx, Ny, x, y, 0)] = f[f_index(Nx, Ny, x, y, 0)];
    if (xp != -1            ) ftemp[f_index(Nx, Ny, xp, y, 1)] = f[f_index(Nx, Ny, x, y, 1)];
    if (yp != -1            ) ftemp[f_index(Nx, Ny, x, yp, 2)] = f[f_index(Nx, Ny, x, y, 2)];
    if (xn != -1            ) ftemp[f_index(Nx, Ny, xn, y, 3)] = f[f_index(Nx, Ny, x, y, 3)];
    if (yn != -1            ) ftemp[f_index(Nx, Ny, x, yn, 4)] = f[f_index(Nx, Ny, x, y, 4)];
    if (xp != -1 && yp != -1) ftemp[f_index(Nx, Ny, xp, yp, 5)] = f[f_index(Nx, Ny, x, y, 5)];
    if (xn != -1 && yp != -1) ftemp[f_index(Nx, Ny, xn, yp, 6)] = f[f_index(Nx, Ny, x, y, 6)];
    if (xn != -1 && yn != -1) ftemp[f_index(Nx, Ny, xn, yn, 7)] = f[f_index(Nx, Ny, x, y, 7)];
    if (xp != -1 && yn != -1) ftemp[f_index(Nx, Ny, xp, yn, 8)] = f[f_index(Nx, Ny, x, y, 8)];
}

__global__ void boundary_gpu(unsigned int Nx, unsigned int Ny, int Q, float ux0, float* ftemp, float* f, bool* solid_node)
{
    unsigned int y = blockIdx.y;
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	// velocity BCs on west-side (inlet) using Zou and He.
    if ((x == 0) && (y < Ny-1))
	{
		int cord = x + Nx*y;
		float rho0 = (ftemp[f_index(Nx, Ny, x, y, 0)] + ftemp[f_index(Nx, Ny, x, y, 2)] + ftemp[f_index(Nx, Ny, x, y, 4)]
			+ 2.*(ftemp[f_index(Nx, Ny, x, y, 3)] + ftemp[f_index(Nx, Ny, x, y, 7)] + ftemp[f_index(Nx, Ny, x, y, 6)])) / (1. - ux0);
		float ru = rho0*ux0;
		ftemp[f_index(Nx, Ny, x, y, 1)] = ftemp[f_index(Nx, Ny, x, y, 3)] + (2./3.)*ru;
		ftemp[f_index(Nx, Ny, x, y, 5)] = ftemp[f_index(Nx, Ny, x, y, 7)] + (1./6.)*ru - 0.5*(ftemp[f_index(Nx, Ny, x, y, 2)]-ftemp[f_index(Nx, Ny, x, y, 4)]);
		ftemp[f_index(Nx, Ny, x, y, 8)] = ftemp[f_index(Nx, Ny, x, y, 6)] + (1./6.)*ru - 0.5*(ftemp[f_index(Nx, Ny, x, y, 4)]-ftemp[f_index(Nx, Ny, x, y, 2)]);
	}

	// BCs at east-side (outlet) using extrapolation from previous node (Nx-2) xn x-dirn
    if ((x == Nx-1) && (y < Ny))
	{
		ftemp[f_index(Nx, Ny, x, y, 0)] = ftemp[f_index(Nx, Ny, x-1, y, 0)];
		ftemp[f_index(Nx, Ny, x, y, 1)] = ftemp[f_index(Nx, Ny, x-1, y, 1)];
		ftemp[f_index(Nx, Ny, x, y, 2)] = ftemp[f_index(Nx, Ny, x-1, y, 2)];
		ftemp[f_index(Nx, Ny, x, y, 3)] = ftemp[f_index(Nx, Ny, x-1, y, 3)];
		ftemp[f_index(Nx, Ny, x, y, 4)] = ftemp[f_index(Nx, Ny, x-1, y, 4)];
		ftemp[f_index(Nx, Ny, x, y, 5)] = ftemp[f_index(Nx, Ny, x-1, y, 5)];
		ftemp[f_index(Nx, Ny, x, y, 6)] = ftemp[f_index(Nx, Ny, x-1, y, 6)];
		ftemp[f_index(Nx, Ny, x, y, 7)] = ftemp[f_index(Nx, Ny, x-1, y, 7)];
		ftemp[f_index(Nx, Ny, x, y, 8)] = ftemp[f_index(Nx, Ny, x-1, y, 8)];
	}

	// bounceback at top wall
    if ((y == Ny - 1) && (x < Nx-1))
	{
		ftemp[f_index(Nx, Ny, x, y, 4)] = ftemp[f_index(Nx, Ny, x, y, 2)];
		ftemp[f_index(Nx, Ny, x, y, 7)] = ftemp[f_index(Nx, Ny, x, y, 5)];
		ftemp[f_index(Nx, Ny, x, y, 8)] = ftemp[f_index(Nx, Ny, x, y, 6)];
	}

	// bounceback at bottom wall
    if ((y == 0) && (x < Nx-1))
	{
		ftemp[f_index(Nx, Ny, x, y, 2)] = ftemp[f_index(Nx, Ny, x, y, 4)];
		ftemp[f_index(Nx, Ny, x, y, 5)] = ftemp[f_index(Nx, Ny, x, y, 7)];
		ftemp[f_index(Nx, Ny, x, y, 6)] = ftemp[f_index(Nx, Ny, x, y, 8)];
	}

	// corners need special treatment as we have an extra unknown.
	// Treatment based on Zou & He (1997), for further details see
	// palabos-forum.unige.ch/t/corner-nodes-2d-channel-boundary-condition-zou-he/577/5

	// corner of south-west inlet
    if ((x == 0) && (y == 0))
    {
        // extrapolate density from neighbour node
        float rho = 0.0;
        for (int a = 0; a < Q; a++)
            rho += ftemp[f_index(Nx, Ny, 0, 1, a)];

        ftemp[f_index(Nx, Ny, x, y, 1)] = ftemp[f_index(Nx, Ny, x, y, 3)];
        ftemp[f_index(Nx, Ny, x, y, 2)] = ftemp[f_index(Nx, Ny, x, y, 4)];
        ftemp[f_index(Nx, Ny, x, y, 5)] = ftemp[f_index(Nx, Ny, x, y, 7)];
        // f6 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f5 + f7))
        ftemp[f_index(Nx, Ny, x, y, 6)] =  0.5*(rho - ftemp[f_index(Nx, Ny, x, y, 0)]) 
                                         - (ftemp[f_index(Nx, Ny, x, y, 1)] 
                                         + ftemp[f_index(Nx, Ny, x, y, 2)] 
                                         + ftemp[f_index(Nx, Ny, x, y, 5)]);
        ftemp[f_index(Nx, Ny, x, y, 8)] = ftemp[f_index(Nx, Ny, x, y, 6)];
    }

	// 	corner of south-east outlet
    if ((x == Nx-1) && (y == 0))
    {
        //extrapolate neighbour density
        float rho = 0.0;
        for (int a = 0; a < Q; a++)
            rho += ftemp[f_index(Nx, Ny, Nx-1, 1, a)];
        ftemp[f_index(Nx, Ny, x, y, 2)] = ftemp[f_index(Nx, Ny, x, y, 4)];
        ftemp[f_index(Nx, Ny, x, y, 3)] = ftemp[f_index(Nx, Ny, x, y, 1)];
        ftemp[f_index(Nx, Ny, x, y, 6)] = ftemp[f_index(Nx, Ny, x, y, 8)];
        // f5 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f6 + f8))
        ftemp[f_index(Nx, Ny, x, y, 5)] =  0.5*(rho - ftemp[f_index(Nx, Ny, x, y, 0)]) 
                                          - (ftemp[f_index(Nx, Ny, x, y, 2)] 
                                          + ftemp[f_index(Nx, Ny, x, y, 3)] 
                                          + ftemp[f_index(Nx, Ny, x, y, 6)]);
        ftemp[f_index(Nx, Ny, x, y, 7)] = ftemp[f_index(Nx, Ny, x, y, 5)];
    }

	// corner of north-west inlet
    if ((x == 0) && (y == Ny-1))
    {
        //extrapolate neighbour density
        float rho = 0.0;
        for (int a = 0; a < Q; a++)
            rho += ftemp[f_index(Nx, Ny, 0, Ny-2, a)];

        ftemp[f_index(Nx, Ny, x, y, 1)] = ftemp[f_index(Nx, Ny, x, y, 3)];
        ftemp[f_index(Nx, Ny, x, y, 4)] = ftemp[f_index(Nx, Ny, x, y, 2)];
        ftemp[f_index(Nx, Ny, x, y, 8)] = ftemp[f_index(Nx, Ny, x, y, 6)];
        // f5 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f6 + f8))
        ftemp[f_index(Nx, Ny, x, y, 5)] =  0.5*(rho - ftemp[f_index(Nx, Ny, x, y, 0)]) 
                                          - (ftemp[f_index(Nx, Ny, x, y, 2)] 
                                          + ftemp[f_index(Nx, Ny, x, y, 3)] 
                                          + ftemp[f_index(Nx, Ny, x, y, 6)]);
        ftemp[f_index(Nx, Ny, x, y, 7)] = ftemp[f_index(Nx, Ny, x, y, 5)];
    }

	// corner of north-east outlet
    if ((x == Nx-1) && (y == Ny-1))
    {
        // extrapolate density from neighbour node
        float rho = 0.0;
        for (int a = 0; a < Q; a++)
            rho += ftemp[f_index(Nx, Ny, Nx-1, Ny-2, a)];

        ftemp[f_index(Nx, Ny, x, y, 3)] = ftemp[f_index(Nx, Ny, x, y, 1)];
        ftemp[f_index(Nx, Ny, x, y, 4)] = ftemp[f_index(Nx, Ny, x, y, 2)];
        ftemp[f_index(Nx, Ny, x, y, 7)] = ftemp[f_index(Nx, Ny, x, y, 5)];
        // f6 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f5 + f7))
        ftemp[f_index(Nx, Ny, x, y, 6)] =  0.5*(rho - ftemp[f_index(Nx, Ny, x, y, 0)]) 
                                         - (ftemp[f_index(Nx, Ny, x, y, 3)] 
                                         + ftemp[f_index(Nx, Ny, x, y, 4)] 
                                         + ftemp[f_index(Nx, Ny, x, y, 7)]);
        ftemp[f_index(Nx, Ny, x, y, 8)] = ftemp[f_index(Nx, Ny, x, y, 6)];
    }
}


__global__ void collide_gpu(unsigned int Nx, unsigned int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float* f, float* ftemp, bool* solid_node, float tau, bool save)
{	
	unsigned int y = blockIdx.y;
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    
    float w0 = 4./9., w1 = 1./9., w2 = 1./36.;
	float c2 = 9./2.;
	float tauinv = 1/tau;
	float one_tauinv = 1 - tauinv; // 1 - 1/tau

    int cord = x + Nx*y;
    if (!solid_node[cord])
    {

        // compute macroscopic quantities
        double rho =  ftemp[f_index(Nx, Ny, x, y, 0)] + ftemp[f_index(Nx, Ny, x, y, 1)] + ftemp[f_index(Nx, Ny, x, y, 2)]
                    + ftemp[f_index(Nx, Ny, x, y, 3)] + ftemp[f_index(Nx, Ny, x, y, 4)] + ftemp[f_index(Nx, Ny, x, y, 5)]
                    + ftemp[f_index(Nx, Ny, x, y, 6)] + ftemp[f_index(Nx, Ny, x, y, 7)] + ftemp[f_index(Nx, Ny, x, y, 8)];

        float ux =  (ftemp[f_index(Nx, Ny, x, y, 1)] + ftemp[f_index(Nx, Ny, x, y, 5)] + ftemp[f_index(Nx, Ny, x, y, 8)])
                    - (ftemp[f_index(Nx, Ny, x, y, 3)] + ftemp[f_index(Nx, Ny, x, y, 6)] + ftemp[f_index(Nx, Ny, x, y, 7)]);
        float uy =  (ftemp[f_index(Nx, Ny, x, y, 2)] + ftemp[f_index(Nx, Ny, x, y, 5)] + ftemp[f_index(Nx, Ny, x, y, 6)])
                    - (ftemp[f_index(Nx, Ny, x, y, 4)] + ftemp[f_index(Nx, Ny, x, y, 7)] + ftemp[f_index(Nx, Ny, x, y, 8)]);
        ux /= rho;
        uy /= rho;

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
        f[f_index(Nx, Ny, x, y, 0)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 0)] + w_rho0_tauinv*(c                           );
        f[f_index(Nx, Ny, x, y, 1)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 1)] + w_rho1_tauinv*(c + 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 2)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 2)] + w_rho1_tauinv*(c + 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 3)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 3)] + w_rho1_tauinv*(c - 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 4)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 4)] + w_rho1_tauinv*(c - 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 5)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 5)] + w_rho2_tauinv*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
        f[f_index(Nx, Ny, x, y, 6)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 6)] + w_rho2_tauinv*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
        f[f_index(Nx, Ny, x, y, 7)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 7)] + w_rho2_tauinv*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
        f[f_index(Nx, Ny, x, y, 8)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 8)] + w_rho2_tauinv*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);
    }
    else
    {
        // Apply standard bounceback at all inner solids (on-grid)
        f[f_index(Nx, Ny, x, y, 1)] = ftemp[f_index(Nx, Ny, x, y, 3)];
        f[f_index(Nx, Ny, x, y, 2)] = ftemp[f_index(Nx, Ny, x, y, 4)];
        f[f_index(Nx, Ny, x, y, 3)] = ftemp[f_index(Nx, Ny, x, y, 1)];
        f[f_index(Nx, Ny, x, y, 4)] = ftemp[f_index(Nx, Ny, x, y, 2)];
        f[f_index(Nx, Ny, x, y, 5)] = ftemp[f_index(Nx, Ny, x, y, 7)];
        f[f_index(Nx, Ny, x, y, 6)] = ftemp[f_index(Nx, Ny, x, y, 8)];
        f[f_index(Nx, Ny, x, y, 7)] = ftemp[f_index(Nx, Ny, x, y, 5)];
        f[f_index(Nx, Ny, x, y, 8)] = ftemp[f_index(Nx, Ny, x, y, 6)];
    }
}


__global__ void stream_collide_periodic_gpu(int Nx, int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float* f, float* ftemp, bool* solid_node, float tau, bool save)
{	
	unsigned int y = blockIdx.y;
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    
    int yn = (y>0   ) ? (y-1) : (Ny-1);
    int yp = (y<Ny-1) ? (y+1) : (0   );
    int xn = (x>0   ) ? (x-1) : (Nx-1);
    int xp = (x<Nx-1) ? (x+1) : (0   );

    ftemp[f_index(Nx, Ny, x, y, 0)] = f[f_index(Nx, Ny, x, y, 0)];
    ftemp[f_index(Nx, Ny, xp, y, 1)] = f[f_index(Nx, Ny, x, y, 1)];
    ftemp[f_index(Nx, Ny, x, yp, 2)] = f[f_index(Nx, Ny, x, y, 2)];
    ftemp[f_index(Nx, Ny, xn, y, 3)] = f[f_index(Nx, Ny, x, y, 3)];
    ftemp[f_index(Nx, Ny, x, yn, 4)] = f[f_index(Nx, Ny, x, y, 4)];
    ftemp[f_index(Nx, Ny, xp, yp, 5)] = f[f_index(Nx, Ny, x, y, 5)];
    ftemp[f_index(Nx, Ny, xn, yp, 6)] = f[f_index(Nx, Ny, x, y, 6)];
    ftemp[f_index(Nx, Ny, xn, yn, 7)] = f[f_index(Nx, Ny, x, y, 7)];
    ftemp[f_index(Nx, Ny, xp, yn, 8)] = f[f_index(Nx, Ny, x, y, 8)];
    
    
    float w0 = 4./9., w1 = 1./9., w2 = 1./36.;
	float c2 = 9./2.;
	float tauinv = 1/tau;
	float one_tauinv = 1 - tauinv; // 1 - 1/tau

    int cord = x + Nx*y;
    if (!solid_node[cord])
    {

        // compute macroscopic quantities
        double rho =  ftemp[f_index(Nx, Ny, x, y, 0)] + ftemp[f_index(Nx, Ny, x, y, 1)] + ftemp[f_index(Nx, Ny, x, y, 2)]
                    + ftemp[f_index(Nx, Ny, x, y, 3)] + ftemp[f_index(Nx, Ny, x, y, 4)] + ftemp[f_index(Nx, Ny, x, y, 5)]
                    + ftemp[f_index(Nx, Ny, x, y, 6)] + ftemp[f_index(Nx, Ny, x, y, 7)] + ftemp[f_index(Nx, Ny, x, y, 8)];

        float ux =  (ftemp[f_index(Nx, Ny, x, y, 1)] + ftemp[f_index(Nx, Ny, x, y, 5)] + ftemp[f_index(Nx, Ny, x, y, 8)])
                    - (ftemp[f_index(Nx, Ny, x, y, 3)] + ftemp[f_index(Nx, Ny, x, y, 6)] + ftemp[f_index(Nx, Ny, x, y, 7)]);
        float uy =  (ftemp[f_index(Nx, Ny, x, y, 2)] + ftemp[f_index(Nx, Ny, x, y, 5)] + ftemp[f_index(Nx, Ny, x, y, 6)])
                    - (ftemp[f_index(Nx, Ny, x, y, 4)] + ftemp[f_index(Nx, Ny, x, y, 7)] + ftemp[f_index(Nx, Ny, x, y, 8)]);
        ux /= rho;
        uy /= rho;

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
        f[f_index(Nx, Ny, x, y, 0)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 0)] + w_rho0_tauinv*(c                           );
        f[f_index(Nx, Ny, x, y, 1)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 1)] + w_rho1_tauinv*(c + 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 2)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 2)] + w_rho1_tauinv*(c + 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 3)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 3)] + w_rho1_tauinv*(c - 3.*ux  + c2*uxsq        );
        f[f_index(Nx, Ny, x, y, 4)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 4)] + w_rho1_tauinv*(c - 3.*uy  + c2*uysq        );
        f[f_index(Nx, Ny, x, y, 5)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 5)] + w_rho2_tauinv*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
        f[f_index(Nx, Ny, x, y, 6)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 6)] + w_rho2_tauinv*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
        f[f_index(Nx, Ny, x, y, 7)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 7)] + w_rho2_tauinv*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
        f[f_index(Nx, Ny, x, y, 8)] = one_tauinv*ftemp[f_index(Nx, Ny, x, y, 8)] + w_rho2_tauinv*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);
    }
    else
    {
        // Apply standard bounceback at all inner solids (on-grid)
        f[f_index(Nx, Ny, x, y, 1)] = ftemp[f_index(Nx, Ny, x, y, 3)];
        f[f_index(Nx, Ny, x, y, 2)] = ftemp[f_index(Nx, Ny, x, y, 4)];
        f[f_index(Nx, Ny, x, y, 3)] = ftemp[f_index(Nx, Ny, x, y, 1)];
        f[f_index(Nx, Ny, x, y, 4)] = ftemp[f_index(Nx, Ny, x, y, 2)];
        f[f_index(Nx, Ny, x, y, 5)] = ftemp[f_index(Nx, Ny, x, y, 7)];
        f[f_index(Nx, Ny, x, y, 6)] = ftemp[f_index(Nx, Ny, x, y, 8)];
        f[f_index(Nx, Ny, x, y, 7)] = ftemp[f_index(Nx, Ny, x, y, 5)];
        f[f_index(Nx, Ny, x, y, 8)] = ftemp[f_index(Nx, Ny, x, y, 6)];
    }
}