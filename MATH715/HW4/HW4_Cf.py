import matplotlib.pyplot as plt
import numpy as np
import math

T = 3
h = 0.001
x0 = 0
x1 = 1
N = 100


def rhs(k, j, h, N, a):
    # the first term in equation (10)
    ans = 0
    for n in range(1, N + 1):
        if (n + k) % 2 == 1:
            ans = ans + (k * n / (k * k - n * n) * a[n-1][j])
    ans = (-4) * ans / math.pi

    # the second term in equation (10)
    if k % 2 == 1:
        ans = ans + (2 / k)
    else:
        ans = ans - (2 / k)

    # the last term in equation 10
    if k % 2 == 1:
        ans = ans + (4 * j * h / (k * math.pi))

    return ans


# initialize {a_k} for k = 1, 2, ..., N
a = [[0] for i in range(N)]

for j in range(1, int(round(T / h))):
    for k in range(N):
        a[k].append(a[k][j - 1] + h * rhs(k+1, j-1, h, N, a))


def measure(a, x, j):
    ans = 0
    for k in range(N):
        ans = ans + a[k][j] * math.sin(math.pi * (k+1) * x)
    return ans

delta_x = 0.001
M = int(round((x1-x0)/delta_x))
u_x_list = []
x_list = []
for i in range(M+1):
    x_list.append(i*delta_x)
    u_x_list.append(measure(a, i*delta_x, int(round(1/h))))

plt.plot(x_list, u_x_list)
plt.title("N = "+str(N))
plt.show()
plt.close