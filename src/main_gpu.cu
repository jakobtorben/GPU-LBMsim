#include <iostream>
#include <string>
#include <math.h>
#include <chrono>

#include <cuda.h>

#include "utils.hpp"
#include "core_gpu.hpp"
#include "init_gpu.hpp"

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
    float mach = 0.1;               // mach number
	float ux0 =  mach * cs;         // inital speed in x direction
    float kin_visc = ux0 * float(Ny/4-1) / input.reynolds; // Ny/4 is diameter of cylinder		
    float tau = (3. * kin_visc + 0.5); // collision timescale	

	// print constants
	cout << "Nx: " << Nx << " Ny: " << Ny << endl;
	cout << "Reynolds number: " << input.reynolds << endl;
	cout << "kinematic viscosity: " << kin_visc << endl;
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
    cout << "device number: " << deviceId << "\n";
    cout << "GPU name: " << deviceProp.name << "\n";
    cout << "compute capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
    cout << "multiprocessor count: " << deviceProp.multiProcessorCount << "\n";
    cout << "global memory: " << deviceProp.totalGlobalMem/(1024.*1024.) << " MiB\n";
    cout << "free memory: " << gpu_free_mem/(1024.*1024.) << " MiB\n";

    // allocate memory
    float *f_gpu, *ftemp_gpu;
    float *ux_arr_gpu, *uy_arr_gpu, *rho_arr_gpu;
    bool *solid_node_gpu;
    const size_t arr_size = sizeof(float)*Nx*Ny;
    const size_t f_size = sizeof(float)*Nx*Ny*Q;
    cudaMalloc((void**)&f_gpu, f_size);
    cudaMalloc((void**)&ftemp_gpu, f_size);
    cudaMalloc((void**)&ux_arr_gpu, arr_size);
    cudaMalloc((void**)&uy_arr_gpu, arr_size);
    cudaMalloc((void**)&rho_arr_gpu, arr_size);
    cudaMalloc((void**)&solid_node_gpu, arr_size);
    float* ux_arr_host        = new float[Nx * Ny];
    float* uy_arr_host        = new float[Nx * Ny];
    //float* rho_arr_host        = new float[Nx * Ny];

    // set threads to nVidia's warp size to run all threads concurrently 
    const int num_threads = 32;
    if (Nx % num_threads != 0)
        throw std::invalid_argument( "Nx must be a multiple of num_threads (32)" ); 

	// blocks in grid
    dim3  grid(Nx/num_threads, Ny, 1);
    // threads in block
    dim3  threads(num_threads, 1, 1);

    // define geometry
	read_geometry<<< grid, threads >>>(Nx, Ny, solid_node_gpu);

	// apply initial conditions - flow to the rigth
	initialise<<< grid, threads >>>(Nx, Ny, Q, ux0, f_gpu, ftemp_gpu, rho_arr_gpu, ux_arr_gpu, uy_arr_gpu, solid_node_gpu);

    // simulation main loop
	cout << "Running simulation...\n";
	auto start = std::chrono::system_clock::now();
	int it = 0, out_cnt = 0;
	bool save = input.save;
	while (it < input.iterations)
	{
		save = input.save && (it > input.printstart) && (it % input.printstep == 0);
		// streaming step
        //stream_gpu<<< grid, threads >>>(Nx, Ny, Q, ftemp_gpu, f_gpu, solid_node_gpu);

		// enforces bounadry conditions
		//boundary_gpu(Nx, Ny, Q, ux0, ftemp_gpu, f_gpu, solid_node_gpu);

		// collision step
        stream_collide_periodic_gpu<<< grid, threads >>>(Nx, Ny, Q, rho_arr_gpu, ux_arr_gpu, uy_arr_gpu, f_gpu, ftemp_gpu, solid_node_gpu, tau, save);
		//collide_gpu<<< grid, threads >>>(Nx, Ny, Q, rho_arr_gpu, ux_arr_gpu, uy_arr_gpu, f_gpu, ftemp_gpu, solid_node_gpu, tau, save);

		// write to file
		if (save)
		{
            cout << "iteration: " << it << "\toutput: " << out_cnt << endl;
             // transfer memory from GPU to host
            cudaMemcpy(ux_arr_host, ux_arr_gpu, arr_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(uy_arr_host, uy_arr_gpu, arr_size, cudaMemcpyDeviceToHost);
            //cudaMemcpy(rho_arr_host, rho_arr_gpu, arr_size, cudaMemcpyDeviceToHost);
			write_to_file(out_cnt, ux_arr_host, uy_arr_host, Nx, Ny);
			out_cnt++;
		}
		it++;
	}

	timings(start, input);

	cudaFree(f_gpu);
	cudaFree(ftemp_gpu);
	cudaFree(solid_node_gpu);
	cudaFree(ux_arr_gpu);
	cudaFree(uy_arr_gpu);
	cudaFree(rho_arr_gpu);
    delete[] ux_arr_host;
	delete[] uy_arr_host;
	//delete[] rho_arr_host;

    // release GPU device resources
    cudaDeviceReset();

}