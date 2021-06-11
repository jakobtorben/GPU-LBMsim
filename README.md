## ACSE-9 - LBM Turbulence Modelling on a GPU

#### Author:
Jakob Torben

#### Supervisors:
Prof. Matthew Piggott

Dr. Steven Dargaville


### Introduction



### Requirements
C++11 compatible compiler


### Build and run simulation

The code is simply built using make, running the following command in root

`make`

after successfully building the code, the simulation defined in 'main.cpp' is executed with

`make run`

### Visualisation
The simulation outputs files in the vtk format that is readable by the


### Validation

#### Flow past cylinder
Parameters: 
Nx = 600
Ny = 300
Kinematic viscosity = 0.015


![alt text](https://github.com/acse-2020/acse2020-acse9-finalreport-acse-jrt3817/blob/main/figures/flow_cylinder_Re5.png?raw=true "Reynolds number 5")


### Admin
