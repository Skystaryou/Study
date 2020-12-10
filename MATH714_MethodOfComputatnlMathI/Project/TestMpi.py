from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = []

start_time = time.time()

if rank == 0:
    n = 20000000
    An = [i for i in range(n)]
    size = int(n/5)
    for i in range(5):
        data.append(np.array(An[i*size: (i+1)*size]))

data = np.array(data)
data = comm.scatter(data, root=0)

max_number = -1
for i in range(len(data)):
    if data[i] > max_number:
        max_number = data[i]

receiv = comm.gather(max_number, root=0)
result = -1

if rank == 0:
    for i in range(len(receiv)):
        if receiv[i] > result:
            result = receiv[i]
    end_time = time.time()
    time_spent = (end_time - start_time)
    print('result = '+str(result))
    print('time spent = '+str(time_spent)+' seconds')


# print('now rank = ' + str(rank))
