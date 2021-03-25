"""This script solves the wave equation in scalar form

    partial_tt u =  div grad u(t, x, y) + f(t, x, y)

on a double slit geometry with source term

    f(t, x, y) = 0

and boundary conditions given by

              u(0, x, y) = exp(-(x^2 + y^2)/0.05)

    partial_t u(0, x, y) = 0
"""
from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import time

# Create mesh
camera1 = Rectangle(Point(-3, -3), Point(3, 3))
channel1 = Rectangle(Point(3, 0.5), Point(5, 1.5))
channel2 = Rectangle(Point(3, -1.5), Point(5, -0.5))
camera2 = Rectangle(Point(5, -3), Point(11, 3))

domain = camera2 + channel1 + channel2 + camera1
mesh = generate_mesh(domain, 64)

# Define function spae
V_h = FunctionSpace(mesh, 'P', 2)

# time stepping
delta_t = 0.005
#############################################
# Define boundary conditions
#############################################

# define the expression for the boundary conditions
u_D = Expression('0', degree=3)

# define the expression for the initial conditions
u_0 = Expression('exp(-(x[0]*x[0] + x[1]*x[1])/0.05)', degree=3)
du_0dt = Expression('0', degree=3)


# define where the expression will be evaluated
def boundary(x, on_boundary):
    return on_boundary


# define the BC using the space and the function above
bc = DirichletBC(V_h, u_D, boundary)

#############################################
# Define variational problem
#############################################

# define trial space
u = TrialFunction(V_h)

# define test space
v = TestFunction(V_h)

# define right-hand side: this could be a function of
# time, in which case you need to update at every time step
f = Expression('0', degree=3)

# define function where we will store the solution
u_ = Function(V_h)

# define the function that will store the values at time n and n-1
# function for u_0
u_n_1 = interpolate(u_0, V_h)

# we store the derivative here, before using it to define
# the u_1 in the next line
u_n = interpolate(du_0dt, V_h)

# define u_1 as a function of u_0 and d/dt u_0
# here you can use u_n that we used to store the derivative
u_n.assign(u_n * delta_t + u_n_1)

# define the bilinear form
var_form = u * v * dx - 2 * u_n * v * dx - (-1 * u_n_1 * v * dx) - \
           (-1) * (delta_t * delta_t) * dot(nabla_grad(u_n), nabla_grad(v)) * dx - \
           (delta_t * delta_t) * f * v * dx

# split the variotonal formulation into bilinar and linear forms
# the bilinear goes to the lhs and the linear to rhs
a = lhs(var_form)
l = rhs(var_form)

# We will assemble before hand given that this side
# doesn't change during the loop
A = assemble(a)
bc.apply(A)

# # uncomment this line to factorize the matrix before-hand
# solver = LUSolver(A)

#############################################
# Compute the solution by time-stepping
#############################################

t = 0

num_steps = 1200

# boolean to control the plotting
# turn it False when timing your solutions
plot_bool = False  # or False

start = time.time()
for n in range(num_steps):
    tt = (n + 2) * delta_t
    print(tt)

    # Update current time
    t += delta_t

    # assemble the right-hand side
    b = assemble(l)
    bc.apply(b)

    # we solve the sysmte
    # using the vanilla solver (iterative by default)
    solve(A, u_.vector(), b)
    # # using preconditioned gmres
    # solve(A, u_.vector(), b, 'gmres', 'amg')
    # # or by prefactorizing the matrix (you need to uncoment)
    # # above at the definition of solver
    # solver.solve(A, u_.vector(), b)

    # Update previous solution
    u_n_1.assign(u_n)
    u_n.assign(u_)

    if abs(tt - 5) < 1e-5:
        plot_bool = True
    else:
        plot_bool = False

    if plot_bool:
        c = plot(u_n)
        plt.colorbar(c)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
        break

end = time.time()
print('delta_t = ' + str(delta_t))
print("average runtime is %.4e" % ((end - start) / num_steps))
