# functions to help you
import numpy as np
import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve
import scipy.integrate as integrate
from matplotlib import pyplot as plt


class Mesh:
    def __init__(self, points):
        # self.p    array with the node points (sorted) type : np.array dim: (n_p)
        # self.n_p  number of node points               type : int
        # self.s    array with indices of points per    type : np.array dim: (n_s, 2)
        #           segment
        # self.n_s  number of segments                  type : int
        # self.bc.  array with the indices of boundary  type : np.array dim: (2)
        #           points

        self.p = points
        self.n_p = len(points)

        self.s = np.array([[i, i + 1] for i in range(len(points) - 1)])
        self.n_s = len(self.s)

        self.bc = np.array([0, len(points) - 1])


class V_h:
    def __init__(self, mesh):
        # self.mesh Mesh object containg geometric info type: Mesh
        # self.sim  dimension of the space              type: in

        self.mesh = mesh
        self.dim = mesh.n_p

    def eval(self, xi, x):
        """ evaluation of the piece wise local polynomial given by
           the coefficients xi, at the point x
        """

        # compute the index of the interval in which x is contained
        index_interval = 0
        if abs(x - self.mesh.p[-1]) < 1e-9:
            index_interval = self.mesh.n_s - 1
        else:
            for i in range(self.mesh.n_s):
                if self.mesh.p[self.mesh.s[i][0]] <= x < self.mesh.p[self.mesh.s[i][1]]:
                    index_interval = i
                    break

        # compute the size of the interval
        length_interval = self.mesh.p[self.mesh.s[index_interval][1]] - self.mesh.p[self.mesh.s[index_interval][0]]

        return xi[index_interval] * (self.mesh.p[index_interval + 1] - x) / length_interval + xi[index_interval + 1] * (
                x - self.mesh.p[index_interval]) / length_interval  # here return the value of the fucnciton


class Function:
    def __init__(self, xi, v_h):
        self.xi = xi
        self.v_h = v_h

    def __call__(self, x):
        # wrapper for calling eval in V_h

        # use the fucntion defined in v_h
        return v_h.eval(self.xi, x)


def mass_matrix(v_h):
    # sparse matrix easy to change sparsity pattern
    # this initializes an empty sparse matrix of
    # size v_h.dim x v_h.dim
    M = spsp.lil_matrix((v_h.dim, v_h.dim))

    # for loop
    for i in range(v_h.mesh.n_s):
        # extract the indices
        # compute the lengh of the segment
        h = v_h.mesh.p[i + 1] - v_h.mesh.p[i]
        # add the values to the matrix
        M[i, i] = M[i, i] + h / 3
        M[i, i + 1] = M[i, i + 1] + h / 6
        M[i + 1, i] = M[i + 1, i] + h / 6
        M[i + 1, i + 1] = M[i + 1, i + 1] + h / 3

    return M


def stiffness_matrix(v_h, sigma):
    # matrix easy to change sparsity pattern
    S = spsp.lil_matrix((v_h.dim, v_h.dim))

    # for loop
    for i in range(v_h.mesh.n_s):
        # extract the indices
        # compute the lengh of the segment
        h = v_h.mesh.p[i + 1] - v_h.mesh.p[i]
        # sample sigma
        x_mid = (v_h.mesh.p[i + 1] + v_h.mesh.p[i]) / 2
        sigma_mid = sigma(x_mid)
        # update the stiffness matrix
        S[i, i] = S[i, i] + sigma_mid / h
        S[i, i + 1] = S[i, i + 1] - sigma_mid / h
        S[i + 1, i] = S[i + 1, i] - sigma_mid / h
        S[i + 1, i + 1] = S[i + 1, i + 1] + sigma_mid / h

    return S


# show differences between Trapezoidal rule and Simpson rule
def load_vector(v_h, f):
    # allocate the vector
    b = np.zeros(v_h.dim)

    # for loop over the segments
    for i in range(v_h.mesh.n_s):
        # extracting the indices
        # computing the lenght of the interval
        h = v_h.mesh.p[i + 1] - v_h.mesh.p[i]
        # update b
        b[i] = b[i] + f(v_h.mesh.p[i]) * h / 2
        b[i + 1] = b[i + 1] + f(v_h.mesh.p[i + 1]) * h / 2

    return b


def source_assembler(v_h, f, u_dirichlet, sigma):
    # computing the load vector (use the function above)
    b = load_vector(v_h, f)

    # extract the interval index for left boundary
    left_index = v_h.mesh.bc[0]

    # compute the lenght of the interval
    left_interval_length = v_h.mesh.p[left_index + 1] - v_h.mesh.p[left_index]

    # sample sigma at the middle point
    x_mid = (v_h.mesh.p[left_index + 1] + v_h.mesh.p[left_index]) / 2
    sigma_mid = sigma(x_mid)

    # update the source_vector
    b[left_index + 1] = b[left_index + 1] - sigma_mid * u_dirichlet[0] / left_interval_length

    # extract the interval index for the right boudanry
    right_index = v_h.mesh.bc[1]

    # compute the length of the interval
    right_interval_length = v_h.mesh.p[right_index] - v_h.mesh.p[right_index - 1]

    # sample sigma at the middle point
    x_mid = (v_h.mesh.p[right_index] + v_h.mesh.p[right_index - 1]) / 2
    sigma_mid = sigma(x_mid)

    # update the source_vector
    b[right_index - 1] = b[right_index - 1] - sigma_mid * u_dirichlet[1] / right_interval_length

    # return only the interior nodes
    return b[1:-1]


def solve_poisson_dirichelet(v_h, f, sigma,
                             u_dirichlet=np.zeros((2))):
    """ function to solve the Poisson equation with
    Dirichlet boundary conditions
    input:  v_h         function space
            f           load (python function)
            sigma       conductivity
            u_dirichlet boundary conditions
    output: u           approximation (Function class)
    """

    # we compute the stiffness matrix, we only use the
    # the interior dof, and we need to convert it to
    # a csc_matrix
    S = stiffness_matrix(v_h, sigma)
    S = S[[i for i in range(1, v_h.dim - 1)], :][:, [i for i in range(1, v_h.dim - 1)]]
    S = S.tocsc()

    # we build the source
    b = source_assembler(v_h, f, u_dirichlet, sigma)

    # solve for the interior degrees of freedom
    u_interior = spsolve(S, b)

    # concatenate the solution to add the boundary
    # conditions
    xi_u = np.concatenate([u_dirichlet[:1],
                           u_interior,
                           u_dirichlet[1:]])

    # return the function
    return Function(xi_u, v_h)


def pi_h(v_h, f):
    """interpolation function
      input:  v_h   function space
              f     function to project
      output: pih_f function that is the interpolation
                    of f into v_h
    """
    xi = []
    for i in range(v_h.mesh.n_p):
        xi.append(f(v_h.mesh.p[i]))
    xi = np.array(xi)

    pi_h_f = Function(xi, v_h)

    return pi_h_f


def p_h(v_h, f):
    """projection function
      input:  v_h   function space
              f     function to project
      output: ph_f  function that is the projection
                    of f into v_h
    """
    # compute load vector
    b = load_vector(v_h, f)

    # compute Mass matrix and convert it to csc type
    M = mass_matrix(v_h)

    # solve the system
    xi = spsolve(M, b)

    # create the new function (this needs to be an instance)
    # of a Function class
    ph_f = Function(xi, v_h)

    return ph_f


if __name__ == "__main__":
    """ This is the main function, which will run 
    if you try to run this script, this code was provided 
    to you to help you debug your code. 
    """
    result_k = []
    result = []
    n = 1001
    for k in range(1, 29):
        print(str(k)+" of "+str(30))
        h = 1 / (n - 1)
        x = np.linspace(0, 1, n)
        mesh = Mesh(x)
        v_h = V_h(mesh)

        f = lambda x: np.sin(2 * k * np.pi * x)
        f_dif2 = lambda x: np.square(((2 * k * np.pi) ** 2) * np.sin(2 * k * np.pi * x))
        f_ph = p_h(v_h, f)
        left = lambda x: np.square(f(x) - f_ph(x))
        l2_left_2 = integrate.quad(left, 0.0, 1.)[0]

        right = 0
        for i in range(v_h.mesh.n_s):
            h_i = v_h.mesh.p[i + 1] - v_h.mesh.p[i]
            integrate_f_d_d = integrate.quad(f_dif2, v_h.mesh.p[i], v_h.mesh.p[i + 1])[0]
            right = right + (h_i * h_i * h_i * h_i) * integrate_f_d_d

        if not right == 0:
            result_k.append(k)
            result.append(l2_left_2 / right)

    # plt.axes(xscale="log")
    plt.plot(result_k, result)
    plt.show()
    plt.close()
