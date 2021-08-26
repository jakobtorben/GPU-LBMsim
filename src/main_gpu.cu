/* Driver code for setting up and running simulation on GPU.
 * 
 * Filename: main_gpu.cu
 * Author: Jakob Torben
 * Created: 13.07.2021
 * Last modified: 26.08.2021
 * 
 * This code is provided under the MIT license. See LICENSE.txt.
 */

#include <iostream>
#include <string>
#include <chrono>
#include <cuda.h>

#include "init_gpu.hpp"
#include "core_gpu.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char* argv[])
{
    // Read simulation inputs from file
	input_struct input;
    string inputfile = argv[1];
	read_input(inputfile, input);

    // initialise simulation parameters
    float u_lid, tau, omega;
    const int Nx = input.Nx;
    const int Ny = input.Ny;
    string fname;
    initialise_simulation(input, fname, u_lid, tau, omega);

    set_up_GPU();

    // allocate memory
    const size_t arr_size = sizeof(float)*Nx*Ny;
    const size_t f_size = sizeof(float)*Nx*Ny*Q;
    float *f_gpu;
    float *ux_arr_gpu, *uy_arr_gpu, *rho_arr_gpu;
    bool *solid_node_gpu;
    cudaMalloc((void**)&f_gpu, f_size);
    cudaMalloc((void**)&ux_arr_gpu, arr_size);
    cudaMalloc((void**)&uy_arr_gpu, arr_size);
    cudaMalloc((void**)&rho_arr_gpu, arr_size);
    cudaMalloc((void**)&solid_node_gpu, arr_size);
    float* ux_arr_host = new float[arr_size];
    float* uy_arr_host = new float[arr_size];
    //float* rho_arr_host = new float[Nx * Ny];  // uncomment if storing density

    // set up dimensions of grid 
    dim3  grid(Nx/num_threads, Ny, 1);
    // define number of threads in a block
    dim3  threads(num_threads, 1, 1);

    // define geometry of walls
    define_geometry<<< grid, threads >>>(Nx, Ny, solid_node_gpu);

	// apply initial conditions - lid moving to the right
    initialise_distributions<<< grid, threads >>>(Nx, Ny, u_lid, f_gpu, rho_arr_gpu, ux_arr_gpu, uy_arr_gpu);

    // simulation main loop
	cout << "Running simulation...\n";
	auto start = std::chrono::system_clock::now();
	int out_cnt = 0;
	bool save = input.save;
	for (int it = 0; it < input.iterations; it++)
	{
		save = input.save && (it > input.printstart) && (it % input.printstep == 0);

        // streaming and collision step combined to one kernel
        stream_collide_gpu<<< grid, threads >>>(Nx, Ny, rho_arr_gpu, ux_arr_gpu, uy_arr_gpu, u_lid,
                                                    f_gpu, solid_node_gpu, tau, omega, save, use_LES<les>(), use_MRT<mrt>());
		
        // write to file
		if (save)
		{
            cout << "iteration: " << it << "\toutput: " << out_cnt << endl;
             // transfer memory from GPU to host
            cudaMemcpy(ux_arr_host, ux_arr_gpu, arr_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(uy_arr_host, uy_arr_gpu, arr_size, cudaMemcpyDeviceToHost);
            //cudaMemcpy(rho_arr_host, rho_arr_gpu, arr_size, cudaMemcpyDeviceToHost);  // uncomment if storing density
			write_to_file(fname, out_cnt, ux_arr_host, uy_arr_host, u_lid, Nx, Ny);
			out_cnt++;
		}
	}

	timings(start, input);

	cudaFree(f_gpu);
	cudaFree(solid_node_gpu);
	cudaFree(ux_arr_gpu);
	cudaFree(uy_arr_gpu);
	cudaFree(rho_arr_gpu);
    delete[] ux_arr_host;
	delete[] uy_arr_host;
	//delete[] rho_arr_host;  // uncomment if storing density

    // release GPU device resources
    cudaDeviceReset();

}