import mpi4py
from mpi4py import MPI
import numpy as np
import time
from matplotlib import pyplot as plt
import sys

all_time = []
all_n = []

for n in range(1000000, 20000000, 1000000):

    start_time = time.time()
    all_n.append(n)

    print(n)
    An = np.array([i for i in range(n)])

    max_value = -1
    for i in range(len(An)):
        if An[i] > max_value:
            max_value = An[i]
    end_time = time.time()
    all_time.append(end_time - start_time)

# all_n = np.array(all_n)
# all_time = np.array(all_time)
# print('result is '+str(max_value))
# print('time spent is ' + str(end_time - start_time) + ' seconds')
