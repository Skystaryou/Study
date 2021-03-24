import numpy
import math
from matplotlib import pyplot as plt

def phi(x):
    if x <= 1 or x >= 3:
        return 0
    if x <= 2:
        return x - 1
    if x >= 2:
        return 3 - x


c = 2
list_t = [1/4, 3/8, 0.5, 1, 2]
list_sol = []
list_x = []
for t in list_t:
    sol = []
    x_list = []
    h = 1/16
    for xx in range(int(5/h)):
        x = xx*h
        x_list.append(x)
        if x <= c*t:
            sol.append(0.5*(phi(x+c*t)-phi(c*t-x)))
        else:
            sol.append(0.5*(phi(x+c*t)+phi(x-c*t)))
    list_x.append(x_list)
    list_sol.append(sol)

for i in range(len(list_t)):
    plt.plot(list_x[i], list_sol[i])
plt.legend(list_t)
plt.show()
plt.close()
