# GPU accelerated lattice Boltzmann method for high Reynolds number flows

#### Author:
Jakob Torben

#### Supervisors:
Prof. Matthew Piggott

Dr. Steven Dargaville


## Description and aims
This codebase accompanies a study that aims to evaluate the lattice Boltzmann Method (LBM) for simulating high Reynolds number flows. The Lattice Boltzmann Method (LBM) is widespread in CFD due to its ability to model complex flows and its computational efficiency. Since each node only depends on its nearest neighbours, it is an excellent candidate for parallelisation. In the study, three different models were evaluated:  single-relaxation-time (SRT), multiple-relaxation-time (MRT) and a Smagorinsky model based large-eddy simulation(MRT-LES). Further details can be found in the report included in this repo.




## Requirements
- C++11 compatible compiler
- CMake version 3.18 or newer [1]
- CUDA enabled GPU
- CUDA toolkit [2].


## Implementation
The project was implemented using C++ for the CPU code and NVIDIA's Compute Unified Device Architecture (CUDA) for the GPU code. The CUDA implementation was developed and tested using CUDA toolkit version 11.4 and a NVIDIA GTX 1060 GPU with compute capability 6.1. Previous versions should also be compatible but have not been tested. The GPU functionality is decoupled from the CPU code in separate files. CMake is used for a cross-platform compatible build system and compiling the CUDA tagets. The general project structure is organised as follows:

- GPU:
     - `main_gpu.cu` - main funciton that sets up and runs the GPU simulation
    - `init_gpu.cu` - initialisation of GPU device, simulation and distribution functions
    - `core_gpu.cu` - GPU LBM solver that encompasses streaming, boundary conditions and collision

- CPU:
    - `main.cpp` - main funciton that sets up and runs the simulation for the CPU simulation
    - `init.cpp` - initialisaiton of simulation and distribution functions
    - `core.cpp` - CPU LBM solver that encompasses streaming, boundary conditions and collision
    - `utils.cpp` - utility functions


## Building and running

Here instructions are provided for Linux:

1. Clone the project from the GitHub repository:

        git clone https://github.com/acse-2020/acse2020-acse9-finalreport-acse-jrt3817

2. Navigate to the project directory and run:

    `mkdir build`

    `cd build`

    `cmake ..`

    `make all`

3. Adjust the simulation parameters in the inputfile located in `project-dir/input`.
4. Run the simulation using the following commands in the project directory:

    `bin/GPU_target input/inputfile.in`

    `bin/CPU_target input/inputfile.in`

The velocity fields are printed to file in the `project-dir/out` directory.


### Compiler options
 By default, targets for CPU and GPU are built. An only CPU target can be built by setting the CMake option USE_GPU to OFF: `cmake -DUSE_GPU=OFF ..`. The model used in the simulation must be set at compilation time and is by default set to the MRT-LES model. The table below summarises the compile time options and their default values.

| Option | Description | Default | 
| :---         |     :---:      |          ---: |
| CMAKE_CUDA_ARCHITECTURES     | CUDA architecture of GPU used | 61       |
| USE_GPU  | Flag to compile GPU code |  ON |
| MRT |  Use multiple-relaxation time model | ON     |
| LES     | Use LES turbulence model | ON |


### Changing simulation parameters

The simulation parameters are read in from an input file and no recompilation is needed to change the simulation. The parameters are summarised in the table below.

| Parameter | Description |
| :---         |     :---:      |
| Nx  | X-domain size | 
| Ny  | Y-domain size |
| reynolds | Reynolds number of flow  |
| iterations     | Number of iterations to run |
| printstart  | The iteration where print to file starts |
| printstep |  Steps between each print to file |
| save  | bool to control if saving to file or not |


### Performance benchmark
A script to build and run performance benchmarks are provided in the scripts directory, which prints the performance results to screen. On the laptop used in this study, the CPU  and GPU benchmark, takes approximately 6 minutes and 1 minute to complete, respectively.


### Visualisation
The simulation outputs files in the vtk format that is readable by the open-source visualisation software ParaView, which is freely available [3].


### Documentation
Documentation has automatically been generated using the Doxygen tool [4], and can be found in the docs folder.


## Validation

Since streaming, boundary conditions and collisions are combined into a single function/kernel for performance reasons, unittests are not possible. Instead, the implementation is validated against a numerical solution in literature [5]. The plot below shows a good agreement to literature for all the three models considered. The validation process is automated in the script `velocity_validation.sh`, which runs the simulation for all three models and generates a plot with quoted errors in the scripts folder. On the GPU, this takes approximately 1 minute.


### Lid-driven cavity flow
Parameters: 
Nx = Ny = 512
Re = 1000
Iterations = 300 000


#### X-velocity along vertical centerlilne 
<img src="https://github.com/acse-2020/acse2020-acse9-finalreport-acse-jrt3817/blob/main/figures/velocity_validation_xvel.png" width="600">


#### Y-velocity along horizontal centerlilne 
<img src="https://github.com/acse-2020/acse2020-acse9-finalreport-acse-jrt3817/blob/main/figures/velocity_validation_yvel.png" width="600">


## Limitations
- The implementation is limited to the lid-driven cavity flwo problem, but only a small change is required to incorporate other flow problems
- The SRT model is known to be unstable as the relaxation time approaches 1/2. For a domain size 512x512, instabilities were encountered for Re above 5000
- At Reynolds numbers greater than 20 000, the first-order accurate boundary condition limits the accuracy



## References

[1]
Cedilnik, A., Hoffman, B., King, B., Martin, K. & Neundorf, A. (2021), ‘CMake, release:  3.21.0’.URL: <https://cmake.org>

[2]
NVIDIA, Vingelmann, P. & Fitzek, F. H. P. (2021), ‘CUDA, release:  11.4.48’. URL: <https://developer.nvidia.com/cuda-toolkit>

[3]
Ayachit, U., Cedilnik, A., Hoffman, B., King, B., Martin, K. & Neundorf, A. (2021), ‘ParaView:  An End-User Tool for Large Data Visualization, release:  5.9.1’. URL <https://www.paraview.org>


[4] 
Heesch, Dimitri v. (2021), ‘Doxygen’. URL: <https://www.doxygen.nl/index.html>

[5]
Erturk,  E.,  Corke,  T.  C.  &   Gökçöl,  C.  (2005),  ‘Numerical  solutions  of  2-D  steady  incompressibledriven  cavity  flow  at  high  Reynolds  numbers’,International Journal for Numerical Methods inFluids48(7), 747–774.
