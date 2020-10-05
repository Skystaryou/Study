
import numpy as np
import math
import copy
import matplotlib.pyplot as plt


def GaussSeider(mx, mr, n=100):
    if len(mx) == len(mr):
        x = []
        for i in range(len(mr)):
            x.append(0)
        count = 0
        while count < n:
            for i in range(len(x)):
                nxi = mr[i]
                for j in range(len(mx[i])):
                    if j != i:
                        nxi = nxi + (-mx[i][j]) * x[j]
                if (mx[i][i]==0):
                    print("mx[i][i]==0, mx=")
                    print(np.array(mx))
                    return False
                nxi = nxi / mx[i][i]
                x[i] = nxi
            count = count + 1
        return np.array(x)
    else:
        return False

def MatrixAdd(mx,my):
    result=copy.deepcopy(mx)
    for i in range(len(mx)):
        for j in range(len(mx[0])):
            result[i][j]=result[i][j]+my[i][j]
    return np.array(result)

def MatrixMultiply(mx,a):
    result=copy.deepcopy(mx)
    for i in range(len(mx)):
        for j in range(len(mx[0])):
            result[i][j]=result[i][j]*a
    return np.array(result)

def IdentityMatrix(n):
    result=np.array([[0.0 for j in range(n)]for i in range(n)])
    for i in range(n):
        result[i][i]=1.0
    return result

def BlankMatrix(n,m):
    result=np.array([[0.0 for j in range(m)]for i in range(n)])
    return result


def initial_fun(x):
    return math.cos(2*math.pi*(x))


def TrySolution(mtemp):

    m = mtemp

    h = 1 / (m + 1)

    # initial condition
    f = np.array([0 for i in range((m + 2) * (m + 2))])
    for i in range(m + 2):
        f[i * (m + 2)] = initial_fun(i / (m + 1))

    A = BlankMatrix(m + 2, m + 2)
    A[0][0] = h * h
    A[m + 1][m + 1] = h * h
    for i in range(1, m + 1):
        A[i][i - 1] = 1
        A[i][i] = -2
        A[i][i + 1] = 1
    A = MatrixMultiply(A, (1 / (h * h)))

    B = BlankMatrix(m + 2, m + 2)
    B[0][0] = -1 * h
    B[0][1] = h
    B[m + 1][m] = h
    B[m + 1][m + 1] = -1 * h
    for i in range(1, m + 1):
        B[i][i - 1] = 1
        B[i][i] = -2
        B[i][i + 1] = 1
    B = MatrixMultiply(B, (1 / (h * h)))

    I = IdentityMatrix(m + 2)

    D = MatrixAdd(np.kron(A, I), np.kron(I, B))

    result_temp = GaussSeider(D, f, 10)
    result = np.array([[0.0 for j in range(m + 2)] for i in range(m + 2)])
    for i in range(len(result_temp)):
        result[int((i - (i % (m + 2))) / (m + 2))][i % (m + 2)] = result_temp[i]

    return result

print(TrySolution(20))