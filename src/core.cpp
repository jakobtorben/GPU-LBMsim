#include "core.hpp"
#include <iostream>  // delete later

// D2Q9 streaming direction scheme
// 6 2 5
// 3 0 1
// 7 4 8

// streaming step - periodic boundary conditions
void stream_periodic(int Nx, int Ny, int Q, float* ftemp, float* f, bool* solid_node)
{
	for (int y = 0; y < Ny; y++)
	{
		int yn = (y>0   ) ? (y-1) : (Ny-1);
		int yp = (y<Ny-1) ? (y+1) : (0   );

		for (int x = 0; x < Nx; x++)
		{
			int cord = x + Nx*y;
			int xn = (x>0   ) ? (x-1) : (Nx-1);
			int xp = (x<Nx-1) ? (x+1) : (0   );
			// can later skip this for interiour nodes
			ftemp[Q*(x  + Nx*y     )] = f[Q*cord  ];
			ftemp[Q*(xp + Nx*y)  + 1] = f[Q*cord + 1];
			ftemp[Q*(x  + Nx*yp) + 2] = f[Q*cord + 2];
			ftemp[Q*(xn + Nx*y)  + 3] = f[Q*cord + 3];
			ftemp[Q*(x  + Nx*yn) + 4] = f[Q*cord + 4];
			ftemp[Q*(xp + Nx*yp) + 5] = f[Q*cord + 5];
			ftemp[Q*(xn + Nx*yp) + 6] = f[Q*cord + 6];
			ftemp[Q*(xn + Nx*yn) + 7] = f[Q*cord + 7];
			ftemp[Q*(xp + Nx*yn) + 8] = f[Q*cord + 8];
		}
	}
}

// streaming step - without periodic boundary condittions
void stream(int Nx, int Ny, int Q, float* ftemp, float* f, bool* solid_node)
{
	for (int y = 0; y < Ny; y++)
	{
		// don't stream beyond boundary nodes
		int yn = (y>0) ? (y-1) : -1;
		int yp = (y<Ny-1) ? (y+1) : -1;

		for (int x = 0; x < Nx; x++)
		{
			// TODO: skip this for interiour nodes
			int cord = x + Nx*y;
			int xn = (x>0) ? (x-1) : -1;
			int xp = (x<Nx-1) ? (x+1) : -1;

				                      ftemp[Q*(x  + Nx*y)     ] = f[Q*cord    ];
			if (xp != -1            ) ftemp[Q*(xp + Nx*y)  + 1] = f[Q*cord + 1];
			if (yp != -1            ) ftemp[Q*(x  + Nx*yp) + 2] = f[Q*cord + 2];
			if (xn != -1            ) ftemp[Q*(xn + Nx*y)  + 3] = f[Q*cord + 3];
			if (yn != -1            ) ftemp[Q*(x  + Nx*yn) + 4] = f[Q*cord + 4];
			if (xp != -1 && yp != -1) ftemp[Q*(xp + Nx*yp) + 5] = f[Q*cord + 5];
			if (xn != -1 && yp != -1) ftemp[Q*(xn + Nx*yp) + 6] = f[Q*cord + 6];
			if (xn != -1 && yn != -1) ftemp[Q*(xn + Nx*yn) + 7] = f[Q*cord + 7];
			if (xp != -1 && yn != -1) ftemp[Q*(xp + Nx*yn) + 8] = f[Q*cord + 8];
		}
	}
}

void boundary(int Nx, int Ny, int Q, float ux0, float* ftemp, float* f, bool* solid_node)
{
	// velocity BCs on west-side (inlet) using Zou and He.
	int x = 0;
	for (int y = 1; y < Ny - 1; y++)
	{
		int cord = x + Nx*y;
		float rho0 = (ftemp[Q*cord + 0] + ftemp[Q*cord + 2] + ftemp[Q*cord + 4]
			+ 2.*(ftemp[Q*cord + 3] + ftemp[Q*cord + 7] + ftemp[Q*cord + 6])) / (1. - ux0);
		float ru = rho0*ux0;
		ftemp[Q*cord + 1] = ftemp[Q*cord + 3] + (2./3.)*ru;
		ftemp[Q*cord + 5] = ftemp[Q*cord + 7] + (1./6.)*ru - 0.5*(ftemp[Q*cord + 2]-ftemp[Q*cord + 4]);
		ftemp[Q*cord + 8] = ftemp[Q*cord + 6] + (1./6.)*ru - 0.5*(ftemp[Q*cord + 4]-ftemp[Q*cord + 2]);
	}

	// BCs at east-side (outlet) using extrapolation from previous node (Nx-2) xn x-dirn
	x = Nx-1;
	for (int y = 0; y < Ny; y++)
	{
		int cord = x + Nx*y;
		ftemp[Q*cord + 0] = ftemp[Q*cord - Q + 0];
		ftemp[Q*cord + 1] = ftemp[Q*cord - Q + 1];
		ftemp[Q*cord + 2] = ftemp[Q*cord - Q + 2];
		ftemp[Q*cord + 3] = ftemp[Q*cord - Q + 3];
		ftemp[Q*cord + 4] = ftemp[Q*cord - Q + 4];
		ftemp[Q*cord + 5] = ftemp[Q*cord - Q + 5];
		ftemp[Q*cord + 6] = ftemp[Q*cord - Q + 6];
		ftemp[Q*cord + 7] = ftemp[Q*cord - Q + 7];
		ftemp[Q*cord + 8] = ftemp[Q*cord - Q + 8];
	}

	// bounceback at top wall
	int y  = Ny - 1;
	for (int x = 1; x < Nx - 1; x++)
	{
		int cord = x + Nx*y;
		ftemp[Q*cord + 4] = ftemp[Q*cord + 2];
		ftemp[Q*cord + 7] = ftemp[Q*cord + 5];
		ftemp[Q*cord + 8] = ftemp[Q*cord + 6];
	}

	// bounceback at bottom wall
	y = 0;
	for (int x = 1; x < Nx - 1; x++)
	{
		int cord = x + Nx*y;
		ftemp[Q*cord + 2] = ftemp[Q*cord + 4];
		ftemp[Q*cord + 5] = ftemp[Q*cord + 7];
		ftemp[Q*cord + 6] = ftemp[Q*cord + 8];
	}

	// corners need special treatment as we have extra unknown.
	// Treatment based on Zou & He (1997), for further details see
	// palabos-forum.unige.ch/t/corner-nodes-2d-channel-boundary-condition-zou-he/577/5

	// corner of south-west inlet
	int cord = 0 + Nx*1; // extrapolate density from neighbour node
	float rho = 0.0;
	for (int a = 0; a < Q; a++)
		rho += ftemp[Q*cord + a];
	cord = 0 + Nx*0;
	ftemp[Q*cord + 1] = ftemp[Q*cord + 3];
	ftemp[Q*cord + 2] = ftemp[Q*cord + 4];
	ftemp[Q*cord + 5] = ftemp[Q*cord + 7];
	// f6 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f5 + f7))
	ftemp[Q*cord + 6] = 0.5*(rho - ftemp[Q*cord]) - (ftemp[Q*cord + 1] + ftemp[Q*cord + 2] + ftemp[Q*cord + 5]);
	ftemp[Q*cord + 8] = ftemp[Q*cord + 6];


	// 	corner of south-east outlet
	cord = (Nx - 1) + Nx*1; //extrapolate neighbour density
	rho = 0.0;
	for (int a = 0; a < Q; a++)
		rho += ftemp[Q*cord + a];
	cord = (Nx-1) + Nx*0;
	ftemp[Q*cord + 2] = ftemp[Q*cord + 4];
	ftemp[Q*cord + 3] = ftemp[Q*cord + 1];
	ftemp[Q*cord + 6] = ftemp[Q*cord + 8];
	// f5 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f6 + f8))
	ftemp[Q*cord + 5] = 0.5*(rho - ftemp[Q*cord]) - (ftemp[Q*cord + 2] + ftemp[Q*cord + 3] + ftemp[Q*cord + 6]);
	ftemp[Q*cord + 7] = ftemp[Q*cord + 5];


	// corner of north-west inlet
	cord = 0 + Nx*(Ny - 2);  // extrapolate neighbour density
	rho = 0.0;
	for (int a = 0; a < Q; a++)
		rho += ftemp[Q*cord + a];
	cord = 0 + Nx*(Ny - 1);
	ftemp[Q*cord + 1] = ftemp[Q*cord + 3];
	ftemp[Q*cord + 4] = ftemp[Q*cord + 2];
	ftemp[Q*cord + 8] = ftemp[Q*cord + 6];
	// f5 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f6 + f8))
	ftemp[Q*cord + 5] = 0.5*(rho - ftemp[Q*cord]) - (ftemp[Q*cord + 2] + ftemp[Q*cord + 3] + ftemp[Q*cord + 6]);
	ftemp[Q*cord + 7] = ftemp[Q*cord + 5];


	// corner of north-east outlet
	cord = (Nx - 1) + Nx*(Ny - 2);  // extrapolate neighbour density
	rho = 0.0;
	for (int a = 0; a < Q; a++)
		rho += ftemp[Q*cord + a];
	cord = (Nx - 1) + Nx*(Ny - 1);
	ftemp[Q*cord + 3] = ftemp[Q*cord + 1];
	ftemp[Q*cord + 4] = ftemp[Q*cord + 2];
	ftemp[Q*cord + 7] = ftemp[Q*cord + 5];
	// f6 = 1/2 * (rho - (f0 + f1 + f2 + f3 + f4 + f5 + f7))
	ftemp[Q*cord + 6] = 0.5*(rho - ftemp[Q*cord]) - (ftemp[Q*cord + 3] + ftemp[Q*cord + 4] + ftemp[Q*cord + 7]);
	ftemp[Q*cord + 8] = ftemp[Q*cord + 6];
}

void collide(int Nx, int Ny, int Q, float* rho_arr, float* ux_arr, float* uy_arr, float* f, float* ftemp, bool* solid_node, float tau, bool save)
{	
	float w0 = 4./9., w1 = 1./9., w2 = 1./36.;
	float c2 = 9./2.;
	float tauinv = 1/tau;
	float one_tauinv = 1 - tauinv; // 1 - 1/tau
	for (int y = 0; y < Ny; y++)
		for (int x = 0; x < Nx; x++)
		{
			int cord = x + Nx*y;
			if (!solid_node[cord])
			{

				// compute macroscopic quantities
				double rho =  ftemp[Q*cord] + ftemp[Q*cord + 1] + ftemp[Q*cord + 2]
							+ ftemp[Q*cord + 3] + ftemp[Q*cord + 4] + ftemp[Q*cord + 5]
							+ ftemp[Q*cord + 6] + ftemp[Q*cord + 7] + ftemp[Q*cord + 8];

				float ux =  (ftemp[Q*cord + 1] + ftemp[Q*cord + 5] + ftemp[Q*cord + 8])
						  - (ftemp[Q*cord + 3] + ftemp[Q*cord + 6] + ftemp[Q*cord + 7]);
				float uy =  (ftemp[Q*cord + 2] + ftemp[Q*cord + 5] + ftemp[Q*cord + 6])
						  - (ftemp[Q*cord + 4] + ftemp[Q*cord + 7] + ftemp[Q*cord + 8]);
				ux /= rho;
				uy /= rho;

				// store to memory only when needed for output
				if (save)
				{
					ux_arr[cord] = ux;
					uy_arr[cord] = uy;
					rho_arr[cord] = rho;
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

				f[Q*cord    ] = one_tauinv*ftemp[Q*cord    ] + w_rho0_tauinv*(c                           );
				f[Q*cord + 1] = one_tauinv*ftemp[Q*cord + 1] + w_rho1_tauinv*(c + 3.*ux  + c2*uxsq        );
				f[Q*cord + 2] = one_tauinv*ftemp[Q*cord + 2] + w_rho1_tauinv*(c + 3.*uy  + c2*uysq        );
				f[Q*cord + 3] = one_tauinv*ftemp[Q*cord + 3] + w_rho1_tauinv*(c - 3.*ux  + c2*uxsq        );
				f[Q*cord + 4] = one_tauinv*ftemp[Q*cord + 4] + w_rho1_tauinv*(c - 3.*uy  + c2*uysq        );
				f[Q*cord + 5] = one_tauinv*ftemp[Q*cord + 5] + w_rho2_tauinv*(c + 3.*uxuy5 + c2*uxuy5*uxuy5);
				f[Q*cord + 6] = one_tauinv*ftemp[Q*cord + 6] + w_rho2_tauinv*(c + 3.*uxuy6 + c2*uxuy6*uxuy6);
				f[Q*cord + 7] = one_tauinv*ftemp[Q*cord + 7] + w_rho2_tauinv*(c + 3.*uxuy7 + c2*uxuy7*uxuy7);
				f[Q*cord + 8] = one_tauinv*ftemp[Q*cord + 8] + w_rho2_tauinv*(c + 3.*uxuy8 + c2*uxuy8*uxuy8);
			}
            else
            // Apply standard bounceback at all inner solids (on-grid)
		    {
				f[Q*cord + 1] = ftemp[Q*cord + 3];
				f[Q*cord + 2] = ftemp[Q*cord + 4];
				f[Q*cord + 3] = ftemp[Q*cord + 1];
				f[Q*cord + 4] = ftemp[Q*cord + 2];
				f[Q*cord + 5] = ftemp[Q*cord + 7];
				f[Q*cord + 6] = ftemp[Q*cord + 8];
				f[Q*cord + 7] = ftemp[Q*cord + 5];
				f[Q*cord + 8] = ftemp[Q*cord + 6];
		    }
		}
}