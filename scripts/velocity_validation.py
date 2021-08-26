import numpy as np
import matplotlib.pyplot as plt


# Benchmark solution from Eturk et al
data_y_arr = np.array([1.00, 0.99 ,0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.50, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.00])
data_xvel_Re1000 = np.array([1.0000, 0.8486, 0.7065, 0.5917, 0.5102, 0.4582, 0.4276, 0.4101, 0.3993, 0.3913, 0.3838, -0.0620, -0.3756, -0.3869, -0.3854, -0.3690, -0.3381, -0.2960, -0.2472, -0.1951, -0.1392, -0.0757, 0.0000])

data_x_arr = np.array([1.000, 0.985 ,0.970 ,0.955 ,0.940 ,0.925 ,0.910 ,0.895 ,0.880 ,0.865 ,0.850 ,0.500 ,0.150 ,0.135 ,0.120 ,0.105 ,0.090, 0.075, 0.060, 0.045, 0.030, 0.015, 0.000])
data_yvel_Re1000 = np.array([0.0000, -0.0973, -0.2173, -0.3400, -0.4417, -0.5052, -0.5263, -0.5132, -0.4803, -0.4407, -0.4028, 0.0258, 0.3756, 0.3705, 0.3605, 0.3460, 0.3273, 0.3041, 0.2746, 0.2349, 0.1792, 0.1019, 0.0000]) 

def get_vel(filename, xvel_data, yvel_data, plot=False):
    file = open(filename)
    for row, line in enumerate(file):
        if row == 4:
            Nx, Ny = line.split(" ")[1:3]
            Nx, Ny = int(Nx), int(Ny)
            break
    # find index along vertical centerline
    vertical_ind = np.arange(Nx//2-1, Ny*Nx, Nx)

    #find index along horizontal centerline
    horizontal_ind = np.arange((Ny*Nx)//2, (Ny*Nx)//2 + Nx, 1)

    vel = np.loadtxt(filename, skiprows=14, usecols=(0,1))
    vertical_vel = vel[vertical_ind, :]
    horizontal_vel = vel[horizontal_ind, :]

    # extract velocities along centrelines
    xvel = vertical_vel[:,0]
    yvel = horizontal_vel[:,1]

    # read in x and y arrays from the file
    sim_x_arr = np.loadtxt(filename, skiprows = 6, max_rows =1)
    sim_y_arr = np.loadtxt(filename, skiprows = 8, max_rows =1)

    sim_x_values = np.interp(data_y_arr, sim_y_arr, xvel)
    sim_y_values = np.interp(data_x_arr, sim_x_arr, yvel) 
    xnorm = np.linalg.norm(sim_x_values - xvel_data)
    ynorm = np.linalg.norm(sim_y_values - yvel_data)

    return xvel, yvel, sim_x_arr, sim_y_arr, xnorm, ynorm

SRT_512_Re1000 = get_vel("../out/LBM_sim_SRT_Nx512_Ny512_Re1000_it0.vtk", data_xvel_Re1000, data_yvel_Re1000)
MRT_512_Re1000 = get_vel("../out/LBM_sim_MRT_Nx512_Ny512_Re1000_it0.vtk", data_xvel_Re1000, data_yvel_Re1000)
MRT_LES_512_Re1000 = get_vel("../out/LBM_sim_MRT_LES_Nx512_Ny512_Re1000_it0.vtk", data_xvel_Re1000, data_yvel_Re1000)


#plt.style.use('./plot.mplstyle')
plt.rc('font', family='serif')

# Re 5000
fig1 = plt.figure(figsize=(4, 3.2))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(data_xvel_Re1000, data_y_arr, 'ko', fillstyle='none', markersize=3, label="Erturk et al.")
ax1.plot(SRT_512_Re1000[0], SRT_512_Re1000[3], 'k--', label="SRT, $L^2$={:.3f}".format(SRT_512_Re1000[4]), alpha=0.8)
ax1.plot(MRT_512_Re1000[0], MRT_512_Re1000[3], 'k:', label="MRT, $L^2$={:.3f}".format(MRT_512_Re1000[4]), alpha=0.8)
ax1.plot(MRT_LES_512_Re1000[0], MRT_LES_512_Re1000[3], 'k-', label="MRT-LES, $L^2$={:.3f}".format(MRT_LES_512_Re1000[4]), alpha=0.8)
ax1.set_xlabel("$u_{x} / U_{lid}$")
ax1.set_ylabel("$y$")
ax1.legend(loc=4)
ax1.minorticks_off()
fig1.tight_layout()
fig1.savefig('velocity_validation_xvel.png', dpi=300, transparent=False, bbox_inches='tight')

fig2 = plt.figure(figsize=(4, 3.2))
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(data_x_arr, data_yvel_Re1000, 'ko', fillstyle='none', markersize=3, label="Erturk et al")
ax2.plot(SRT_512_Re1000[2], SRT_512_Re1000[1], 'k--', label="SRT, $L^2$={:.3f}".format(SRT_512_Re1000[5]), alpha=0.8)
ax2.plot(MRT_512_Re1000[2], MRT_512_Re1000[1], 'k:', label="MRT, $L^2$={:.3f}".format(MRT_512_Re1000[5]), alpha=0.8)
ax2.plot(MRT_LES_512_Re1000[2], MRT_LES_512_Re1000[1], 'k-', label="MRT-LES, $L^2$={:.3f}".format(MRT_LES_512_Re1000[5]), alpha=0.8)
ax2.set_xlabel("$x$ ")
ax2.set_ylabel("$u_{y} / U_{lid}$")
ax2.legend(loc=3)
ax2.minorticks_off()
fig2.tight_layout()
fig2.savefig('velocity_validation_yvel.png', dpi=300, transparent=False, bbox_inches='tight')