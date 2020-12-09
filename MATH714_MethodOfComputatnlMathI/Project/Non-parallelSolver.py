import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)


def f(x: float):
    return math.exp(-400.0 * (x - 0.5) * (x - 0.5))


start_time = time.time()

maxT = 2
maxX = 1
maxY = 1
deltaT = 0.0005
deltaX = 0.005
Nt = int(maxT / deltaT) + 1
Nx = int(maxX / deltaX) + 1

u = [[[0 for i in range(Nx)] for j in range(Nx)] for t in range(Nt)]

# initial condition
for i in range(Nx):
    for j in range(Nx):
        u[0][i][j] = 0
for i in range(Nx):
    for j in range(Nx):
        u[1][i][j] = u[0][i][j] + deltaT * f(i * deltaX) * f(j * deltaX)

# boundary condition
for t in range(Nt):
    for i in range(Nx):
        u[t][0][i] = 0
        u[t][i][0] = 0
        u[t][Nx - 1][i] = 0
        u[t][i][Nx - 1] = 0

for t in range(2, Nt):
    for i in range(1, Nx - 1):
        for j in range(1, Nx - 1):
            u[t][i][j] = 2 * u[t - 1][i][j] - u[t - 2][i][j] + \
                         (deltaT * deltaT / deltaX / deltaX) * (u[t - 1][i + 1][j] + u[t - 1][i - 1][j] +
                                                                u[t - 1][i][j + 1] + u[t - 1][i][j - 1] - 4 *
                                                                u[t - 1][i][j])

X = np.arange(0, 1 + deltaX, deltaX)
Y = np.arange(0, 1 + deltaX, deltaX)

X, Y = np.meshgrid(X, Y)

finalU = np.array(u[Nt - 1])

end_time = time.time()
print('time spent is '+str(end_time - start_time)+' seconds')

ax.plot_surface(X, Y, finalU, cmap='rainbow')
plt.show()
