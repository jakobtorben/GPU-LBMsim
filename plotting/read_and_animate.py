
"""
2D wave equation animation

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib import cm

    
def read_file(it):
    fname = "../out/output_" + str(it) + ".dat"
    data = np.loadtxt(fname, float)
    data = data.reshape(imax, jmax)
    return data


def update_plot(frame_number):
    plot[0].remove()

    Z = read_file(frame_number)
    plot[0] = ax.contourf(X, Y, Z, cmap=cm.coolwarm, antialiased=False, linewidth=0.2)

x0 = 0.0
y0 = 0.0

imax = 100
jmax = 100

fps = 30 # frame per sec
frn = 10 # frame number of the animation

dx = 1
dy = 1

x = np.linspace(0, (imax-1)*dx, imax)
y = np.linspace(0, (jmax-1)*dy, jmax)

X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111)

#ax.set_xlim3d(0, imax)
#ax.set_ylim3d(y0, jmax)
#ax.view_init(elev=30., azim=-110)

ax.set_xlabel("x")
ax.set_ylabel("y")
#ax.set_title("Periodic")

Z = read_file(1)
#plot = [ax.contourf(X, Y, Z, cmap=cm.coolwarm, antialiased=False, linewidth=0.2)]
levels = np.linspace(0, 2, 30)
ax.contourf(X, Y, Z, 20, cmap=cm.coolwarm)#, levels=levels)
plt.show()

#ani = animation.FuncAnimation(fig, update_plot, frn, interval=1000/fps)
#fname = 'figures/plot_surface_Periodic' + str(frn)
#ani.save(fname + '.gif', writer='imagemagick', fps=fps)
