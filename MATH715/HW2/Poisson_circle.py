"""This script program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit cricle with source f given by

    f(x, y) = 4

and boundary conditions given by

    u(x, y) = 0
"""

from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import math
import time

# Create mesh
from mshr.cpp import Circle, generate_mesh

domain = Circle(Point(0, 0), 1)
mesh = generate_mesh(domain, 8)

# Define function space
V_h = FunctionSpace(mesh, 'P', 2)

#############################################
# Define boundary conditions
#############################################

# define overal expression
u_D = Expression('1 - x[0]*x[0] - x[1]*x[1]', degree=3)


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

# define right-hand side
f = Constant(4.0)

# define the bilinear form
a = dot(grad(u), grad(v)) * dx

# define the linear form
L = f * v * dx

#############################################
# Compute the solution
#############################################

# define function where we will store the solution
u_h = Function(V_h)

# solve the variational problem
solve(a == L, u_h, bc)

#############################################
# Plot and compute error
#############################################

# Plot solution and mesh
plot(u_h)
plot(mesh)

# Compute error in L2 norm
error_L2 = errornorm(u_D, u_h, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u_h.compute_vertex_values(mesh)
import numpy as np

error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print("Test for h=1/8 and the first figure is the figure of the solution:")
print('error_L2  =', error_L2)
print('error_max =', error_max)

# Hold plot
plt.show()
plt.close()

print()
print("The second figure is a semi-log plot of the error w.r.t h when using linear polynomials.")

list_h = []
list_l2_err = []

for n in range(8, 100):
    print(n)
    mesh = generate_mesh(domain, n)
    # Define function space
    V_h = FunctionSpace(mesh, 'P', 1)
    #############################################
    # Define boundary conditions
    #############################################
    # define overal expression
    u_D = Expression('1 - x[0]*x[0] - x[1]*x[1]', degree=3)


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
    # define right-hand side
    f = Constant(4.0)
    # define the bilinear form
    a = dot(grad(u), grad(v)) * dx
    # define the linear form
    L = f * v * dx

    #############################################
    # Compute the solution
    #############################################
    # define function where we will store the solution
    u_h = Function(V_h)
    # solve the variational problem
    solve(a == L, u_h, bc)

    #############################################
    # Compute the error and record it
    #############################################
    # Compute error in L2 norm
    error_L2 = errornorm(u_D, u_h, 'L2')
    list_h.append(math.log(1 / n))
    list_l2_err.append(error_L2)

plt.plot(list_h, list_l2_err)
plt.xlabel("log(h)")
plt.ylabel("l2_error")
plt.show()
plt.close

list_h_2 = []
list_l2_err_2 = []

for n in range(8, 100):
    print(n)
    mesh = generate_mesh(domain, n)
    # Define function space
    V_h = FunctionSpace(mesh, 'P', 2)
    #############################################
    # Define boundary conditions
    #############################################
    # define overal expression
    u_D = Expression('1 - x[0]*x[0] - x[1]*x[1]', degree=3)


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
    # define right-hand side
    f = Constant(4.0)
    # define the bilinear form
    a = dot(grad(u), grad(v)) * dx
    # define the linear form
    L = f * v * dx

    #############################################
    # Compute the solution
    #############################################
    # define function where we will store the solution
    u_h = Function(V_h)
    # solve the variational problem
    solve(a == L, u_h, bc)

    #############################################
    # Compute the error and record it
    #############################################
    # Compute error in L2 norm
    error_L2 = errornorm(u_D, u_h, 'L2')
    list_h_2.append(math.log(1 / n))
    list_l2_err_2.append(error_L2)

plt.plot(list_h, list_l2_err)
plt.plot(list_h_2, list_l2_err_2)
plt.xlabel("log(h)")
plt.ylabel("l2_error")
plt.title("using ")
plt.legend(['linear', 'quadrature'])
plt.show()
plt.close

print("The third figure is the comparison of the error of linear and quadratic polynomials.")
