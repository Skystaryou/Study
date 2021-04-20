import matplotlib.pyplot as plt
import numpy as np
import math


def p(N, x):
    if abs(x) < 1e-5 or abs(x - 2 * math.pi) < 1e-5:
        a = (2 / N) * (N / 2 - 1)
    else:
        a = (2 / N) * (math.sin((N / 4 - 0.5) * x) * math.cos((N / 4) * x)) / math.sin(x / 2)
    return 1 / N + a + (1 / N) * math.cos((N / 2) * x)


m_list = [4, 5, 6, 7, 8, 9, 10]
line_list = []
result_list = []
for m in m_list:
    N = 2 ** m
    result = []
    line = []
    h = 2 * math.pi / N
    for j in range(N + 1):
        line.append(j * h)
        result.append(p(N, j * h))
    result_list.append(result)
    line_list.append(line)

ll = []
for i in range(len(line_list)):
    plt.plot(line_list[i], result_list[i])
    ll.append("m=" + str(m_list[i]))

plt.legend(ll)
plt.title("p(x) in [0,2pi] with different m")
plt.show()
plt.close()
