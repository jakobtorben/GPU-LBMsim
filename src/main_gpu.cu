#include <iostream>
#include <string>
#include <math.h>
#include <chrono>

#include <stdio.h>
#include <cuda.h>

#include "core_gpu.hpp"
#include "utils.hpp"
#include "init.hpp"

using namespace std;



int main(int argc, char* argv[])
{
    // Read simulation inputs from file
    string inputfile;
	input_struct input;
    inputfile = argv[1];
	read_input(inputfile, input);

	const int Q = 9;			    // number of velocity components
	const int Nx = input.Nx;		// grid size x-direction
	const int Ny = input.Ny;		// grid size y-direction
    float cs = sqrt(1./3.);			// speed of sound**2 D2Q9
	// inital speed in x direction
	float ux0 = input.reynolds*input.kin_visc / float(Ny/4-1);  // Ny/4 is diameter of cylinder
    float mach = ux0 / cs;			// mach number
    float tau = (3. * input.kin_visc + 0.5); // collision timescale	

	// print constants
	cout << "Nx: " << Nx << " Ny: " << Ny << endl;
	cout << "Reynolds number: " << input.reynolds << endl;
	cout << "kinematic viscosity: " << input.kin_visc << endl;
	cout << "ux0: " << ux0 << endl;
	cout << "mach number: " << mach << endl;
	cout << "tau : " << tau << endl;

    // set up GPU
    cudaSetDevice(0);
    int deviceId = 0;
    cudaGetDevice(&deviceId);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    
    size_t gpu_free_mem, gpu_total_mem;
    cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);

    cout << "CUDA information\n";
    cout << "      device number: " << deviceId << "\n";
    cout << "           GPU name: " << deviceProp.name << "\n";
    cout << " compute capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
    cout << "    multiprocessors: " << deviceProp.multiProcessorCount << "\n";
    cout << "      global memory: " << deviceProp.totalGlobalMem/(1024.*1024.) << "MiB\n";
    cout << "        free memory: " << gpu_free_mem/(1024.*1024.) << "MiB\n";

    // allocate grid
    float* f          = new float[Nx * Ny * Q];
    float* ftemp      = new float[Nx * Ny * Q];

    bool*  solid_node = new bool [Nx * Ny];
    float* ux_arr        = new float[Nx * Ny];
    float* uy_arr        = new float[Nx * Ny];
    float* rho_arr        = new float[Nx * Ny];


	// define geometry
	read_geometry(Nx, Ny, solid_node);

	// apply initial conditions - flow to the rigth
	initialise(Nx, Ny, Q, ux0, f, ftemp, rho_arr, ux_arr, uy_arr, solid_node);

    // set threads to dimension of blocks
    const int num_threads = blockDim.x;

	// blocks in grid
    dim3  grid(Nx/num_threads, Ny, 1);
    // threads in block
    dim3  threads(num_threads, 1, 1);

    // simulation main loop
	cout << "Running simulation...\n";
	auto start = std::chrono::system_clock::now();
	int it = 0, out_cnt = 0;
	bool save = input.save;
	while (0)//it < input.iterations)
	{
		save = input.save && (it % input.printstep == 0);
		// streaming step
		stream_periodic_gpu(Nx, Ny, Q, ftemp, f, solid_node);

		// enforces bounadry conditions
		//boundary_gpu(Nx, Ny, Q, ux0, ftemp, f, solid_node);

		// collision step
		collide_gpu<<< grid, threads >>>(Nx, Ny, Q, rho_arr, ux_arr, uy_arr, f, ftemp, solid_node, tau, save);

        // synchronise threads
        //__syncthreads();

		// write to file
		if (save)
		{
			cout << "iteration: " << it << "\toutput: " << out_cnt << endl;
			write_to_file(out_cnt, ux_arr, uy_arr, Nx, Ny);
			out_cnt++;
		}
		it++;
	}

	timings(start, input);

	delete[] f;
	delete[] ftemp;
	delete[] solid_node;
	delete[] ux_arr;
	delete[] uy_arr;
	delete[] rho_arr;

}