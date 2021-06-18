import numpy as np
import matplotlib.pyplot as plt

# read output file in vtk format
filename = "../build/out/output_8.vtk"
file = open(filename)
for row, line in enumerate(file):
    if row == 4:
        Nx, Ny = line.split(" ")[1:3]
        Nx, Ny = int(Nx), int(Ny)
        break

# get viscosity
file = open("../input/Poiseuille.in")
for row, line in enumerate(file):
    if "kin_visc" in line:
        kin_visc = float(line.split(" ")[2].split(";")[0])
        break

# start at middle profile
skiprows = 14 + Nx//2 * Ny

# only read velocities along centerline
max_rows = Ny

vel = np.loadtxt(filename, skiprows=14, usecols=(0,1))

# find index along centerline
mid = Nx//2
cross_ind = np.arange(mid-1, Ny*Nx, Nx)
cross_vel = vel[cross_ind, :]

# speed at each point
magn = np.sqrt(cross_vel[:,0]**2 + cross_vel[:,1]**2)

# points between plate in interval [-a, a] where a = Ny/2
a = Ny/2
x = np.linspace(-a, a, Ny)

# analytical solution
G = 5.1e-9  # replace this with (Pin - Pout)/Nx when implemented Dirichlet (pressure) boundaries
analytical = G/(2*kin_visc)*(a**2 - x**2)

plt.plot(x, magn, '.', label="LBM simulation")
plt.plot(x, analytical, label="Analytical solution")
plt.xlabel("y")
plt.ylabel("u(y)")
plt.legend()
plt.show()