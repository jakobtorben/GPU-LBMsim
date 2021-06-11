## ACSE-9 - LBM Turbulence Modelling on a GPU

#### Author:
Jakob Torben

#### Supervisors:
Prof. Matthew Piggott

Dr. Steven Dargaville


### Introduction
In this project a Lattice Boltzmann simulation is implemented for the purpose of turbulence modelling. The code will be parallised for Graphics Processing Units (GPU), using Nvidia's CUDA language.


### Requirements
C++11 compatible compiler


### Build and run simulation

The code is simply built using make, running the following command in root

`make`

after successfully building the code, the simulation defined in 'main.cpp' is executed with

`make run`

### Visualisation
The simulation outputs files in the vtk format that is readable by the open-source visualisation software ParaView, which is freely available [here](https://www.paraview.org/download/) 


### Validation

Note that the code is not yet fully tested and validated, and that currently only periodic boundary conditions are implemented. The following results are more to demonstrate the output of the simulation.

#### Flow past cylinder
Parameters: 
Nx = 600
Ny = 300
Kinematic viscosity = 0.015

##### Reynolds number 5
![Reynolds number 5](https://github.com/acse-2020/acse2020-acse9-finalreport-acse-jrt3817/blob/main/figures/flow_cylinder_Re5.png?raw=true)


### Admin
Overleaf documents:

[Project plan](https://www.overleaf.com/read/ycmmnbmxkvzx)


[Project report](https://www.overleaf.com/read/kdqvpnffdbwn)

