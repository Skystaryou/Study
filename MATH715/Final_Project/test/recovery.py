import context
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import numpy.linalg as npla
import scipy.sparse as spsp
# from scipy.sparse.linalg import spsolve
import time
from scikits.umfpack import spsolve

import scipy.optimize as op

import os
import sys

from eit import *

mat_fname = 'data/data_reconstruction.mat'
mat_contents = sio.loadmat(mat_fname)
# points
p = mat_contents['p']
# triangle
t = mat_contents['t'] - 1  # all the indices should be reduced by one
# volumetric indices
vol_idx = mat_contents['vol_idx'].reshape((-1,)) - 1  # all the indices should be reduced by one
# indices at the boundaries
bdy_idx = mat_contents['bdy_idx'].reshape((-1,)) - 1  # all the indices should be reduced by one
# define the mesh
mesh = Mesh(p, t, bdy_idx, vol_idx)
# define the approximation space
v_h = V_h(mesh)

dtn_data = mat_contents['DtN']

# this is the guess
sigma_vec_0 = 2 + np.zeros((t.shape[0], 1))


# simple optimization routine
def J(x):
    return misfit_sigma(v_h, dtn_data, x)


# we define a relatively high tolerance
# recall that this is the square of the misfit
opt_tol = 1.e-6

print("Begin optimization")
start_time = time.time()
# running the optimization routine
res = op.minimize(J, sigma_vec_0,  # method='L-BFGS-B',
				  jac=True,
				  options={'eps': opt_tol,
                           'maxiter': 500,
                           'disp': True})
end_time = time.time()
print("Optimization finished")
print("time spent during optimization is: " + str(end_time - start_time) + " s")

# extracting guess from the resulting optimization
sigma_guess = res.x

# we proejct sigma back to V in order to plot it
p_v_w = projection_v_w(v_h)
Mass = spsp.csr_matrix(mass_matrix(v_h))

sigma_v = spsolve(Mass, p_v_w @ sigma_guess)

# create a triangulation object
triangulation = tri.Triangulation(p[:, 0], p[:, 1], t)
# plot the triangles
plt.triplot(triangulation, '-k')
# plotting the solution
plt.tricontourf(triangulation, sigma_v)
# plotting a colorbar
plt.colorbar()
plt.title("result of optimization (sigma)")
# show
plt.show()
plt.close()
