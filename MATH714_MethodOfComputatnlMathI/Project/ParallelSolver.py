import numpy as np
import math
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from mpi4py import MPI
import copy


def f(x: float):
    return math.exp(-400.0 * (x - 0.5) * (x - 0.5))


fig = plt.figure()
ax = Axes3D(fig)

maxT = 2
maxX = 1
maxY = 1
deltaT = 0.0005
deltaX = 0.005
Nt = int(maxT / deltaT) + 1
Nx = int(maxX / deltaX) + 1

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

length = int((Nx - 1) / size)

pre_result = np.zeros((Nx, Nx))
now_result = np.zeros((Nx, Nx))

if rank == 0:
    print('length = ' + str(length))
    print('core size = ' + str(size))

start_time = time.time()

# initial condition
if rank == 0:
    for i in range(Nx):
        for j in range(Nx):
            pre_result[i][j] = 0

# n=1 and boundary condition
if rank == 0:
    for i in range(Nx):
        for j in range(Nx):
            now_result[i][j] = pre_result[i][j] + deltaT * f(i * deltaX) * f(j * deltaX)
    for i in range(Nx):
        now_result[i][0] = 0
        now_result[i][Nx - 1] = 0
        now_result[0][i] = 0
        now_result[Nx - 1][i] = 0

# start loop
for t in range(2, Nt):
    last_data = comm.bcast(now_result if rank == 0 else None, root=0)
    last2_data = comm.bcast(pre_result if rank == 0 else None, root=0)
    rank_data = []

    if rank == 4:
        rank_data = np.zeros((length - 1, Nx))
        for i in range(length - 1):
            for j in range(1, Nx - 1):
                # corresponding position is [rank * length + i + 1][j]
                rank_data[i][j] = 2 * last_data[rank * length + i + 1][j] - last2_data[rank * length + i + 1][j] + (
                            deltaT * deltaT / deltaX / deltaX) \
                                  * (last_data[rank * length + i][j] + last_data[rank * length + i + 2][j] +
                                     last_data[rank * length + i + 1][j - 1] + last_data[rank * length + i + 1][
                                         j + 1] - 4 * last_data[rank * length + i + 1][j])

    else:
        rank_data = np.zeros((length, Nx))
        for i in range(length):
            for j in range(1, Nx - 1):
                # corresponding position is [rank * length + i + 1][j]
                rank_data[i][j] = 2 * last_data[rank * length + i + 1][j] - last2_data[rank * length + i + 1][j] + (
                            deltaT * deltaT / deltaX / deltaX) \
                                  * (last_data[rank * length + i][j] + last_data[rank * length + i + 2][j] +
                                     last_data[rank * length + i + 1][j - 1] + last_data[rank * length + i + 1][
                                         j + 1] - 4 * last_data[rank * length + i + 1][j])

    gathered_data = comm.gather(rank_data, root=0)

    if rank == 0:
        pre_result = copy.deepcopy(now_result)
        for i in range(1, Nx - 1):
            now_result[i] = copy.deepcopy(gathered_data[int((i - 1) / length)][i - int((i - 1) / length) * length - 1])

if rank == 0:
    end_time = time.time()
    print('time spent is ' + str(end_time - start_time) + ' seconds')
    X = np.arange(0, 1 + deltaX, deltaX)
    Y = np.arange(0, 1 + deltaX, deltaX)
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, now_result, cmap='rainbow')
    plt.show()
