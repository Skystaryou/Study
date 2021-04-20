import matplotlib.pyplot as plt
import numpy as np
import math


class C:
    real = 0
    imaginary = 0

    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary

    @staticmethod
    def plus(c1, c2):
        result = C(c1.real + c2.real, c1.imaginary + c2.imaginary)
        return result

    @staticmethod
    def minus(c1, c2):
        result = C(c1.real - c2.real, c1.imaginary - c2.imaginary)
        return result

    @staticmethod
    def multiple(c1, c2):
        result = C(c1.real * c2.real - c1.imaginary * c2.imaginary, c1.imaginary * c2.real + c1.real * c2.imaginary)
        return result


h = 0.001
N = 100
T = 5
a1 = [C(0, -1)]
for n in range(1, int(T / h)):
    a1.append(C.minus(a1[n - 1], C.multiple(C(0, 2 * math.pi * h), a1[n - 1])))


def evaluate(x, t):
    return C.multiple(a1[int(t / h)], C(math.cos(2 * math.pi * x), math.sin(2 * math.pi * x))).real


x0 = 0
x1 = 1
delta_x = (x1-x0)/(2*N)
t_list = [0, 0.2, 0.4, 0.6, 0.8]
legend_list = []

for t in t_list:
    legend_list.append("t = "+str(t))
    result = []
    x = []
    for i in range(int(round((x1-x0)/delta_x))):
        result.append(evaluate(delta_x*i, t))
        x.append(delta_x*i)
    plt.plot(x, result)

plt.title("the result for u_N under different t of C(d)")
plt.legend(legend_list)
plt.show()
plt.close()