## ACSE-9 - LBM Turbulence Modelling on a GPU

#### Author:
Jakob Torben

#### Supervisors:
Prof. Matthew Piggott

Dr. Steven Dargaville


### Introduction
In this project a Lattice Boltzmann simulation is implemented for the purpose of turbulence modelling. The code will be parallised for Graphics Processing Units (GPU), using Nvidia's CUDA language.


### Requirements
- C++11 compatible compiler
- CMake version 3.14 or newer



### Build and run simulation

The code is cross-platform compatible and is built using CMake. To build the code for Linux run the following commands in root

`mkdir build`

`cd build`

`cmake ..`

`make all`

after successfully building the code, a suite of tests is run with 

`make test`

A script to run the Poiseuille flow validation is provided in the scripts directory.

### Visualisation
The simulation outputs files in the vtk format that is readable by the open-source visualisation software ParaView, which is freely available [here](https://www.paraview.org/download/) 


### Validation

#### Poiseuille flow
Parameters: 
Nx = 60
Ny = 20
Kinematic viscosity = 0.1667

<img src="https://github.com/acse-2020/acse2020-acse9-finalreport-acse-jrt3817/blob/main/figures/Poiseuille_flow.png" width="600">

<img src="https://github.com/acse-2020/acse2020-acse9-finalreport-acse-jrt3817/blob/main/figures/Poiseuille_flow_comparison.png" width="500">


#### Flow past cylinder
Parameters: 
Nx = 300
Ny = 200
Iterations = 15000

##### Reynolds number 5 and kinematic viscosity 0.1
<img src="https://github.com/acse-2020/acse2020-acse9-finalreport-acse-jrt3817/blob/main/figures/flow_past_cylinder_Re5_Nx300_Ny200_kinvisc_0_1_it15000.png" width="600">

##### Reynolds number 40 and kinematic viscosity 0.03
<img src="https://github.com/acse-2020/acse2020-acse9-finalreport-acse-jrt3817/blob/main/figures/flow_past_cylinder_Re40_Nx300_Ny200_kinvisc_0_03_it15000.png" width="600">

##### Reynolds number 100 and kinematic viscosity 0.03
<img src="https://github.com/acse-2020/acse2020-acse9-finalreport-acse-jrt3817/blob/main/figures/flow_past_cylinder_Re100_Nx300_Ny200_kinvisc_0_03_it15000.png" width="600">


### Admin
Overleaf documents:

[Project plan](https://www.overleaf.com/read/ycmmnbmxkvzx)


[Project report](https://www.overleaf.com/read/kdqvpnffdbwn)

