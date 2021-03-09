"""This program solves the Eigenvalue problem

    - div grad u(x, y) = labda u(x, y)

on the unit circle and boundary conditions given by

    u(x, y) = 0
"""

from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import math

# Create mesh
domain = Circle(Point(0, 0), 1)
mesh = generate_mesh(domain, 20)

# Define function space
V_h = FunctionSpace(mesh, 'P', 1)

#############################################
# Define boundary conditions
#############################################

# define overal expression
u_D = Expression('0', degree=3)


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

# define the bilinear form
a = dot(grad(u), grad(v)) * dx

#############################################
# Assemble the final matrix
#############################################

# define where to store the matrix
A = PETScMatrix()

# assemble the stiffness matrix and store in A
assemble(a, tensor=A)

# apply the boundary conditions
bc.apply(A)

#############################################
# Compute eigenvalues
#############################################

# define eigen solver
eigensolver = SLEPcEigenSolver(A)

# Compute all eigenvalues of A x = \lambda x
print("Computing eigenvalues. This can take a minute.")
eigensolver.solve()

#############################################
# Extract and plot eigenfunctions
#############################################
# Extract smallest (last) eigenpair
r, c, rx, cx = eigensolver.get_eigenpair(A.array().shape[0] - 1)

# print eigenvalue
print("Smallest eigenvalue: ", r)

list_r = []
for i in range(A.array().shape[0]):
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    list_r.append(r)

# Initialize function and assign eigenvector
u = Function(V_h)
u.vector()[:] = rx

# Plot eigenfunction
plot(u)
plt.show()
plt.close()

# Create mesh
domain = Circle(Point(0, 0), 1)
# mesh = generate_mesh(domain, 20)
n = 21
mesh = UnitSquareMesh(n, n)

# Define function space
V_h = FunctionSpace(mesh, 'P', 1)

#############################################
# Define boundary conditions
#############################################

# define overal expression
u_D = Expression('0', degree=3)


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

# define the bilinear form
a = dot(grad(u), grad(v)) * dx

#############################################
# Assemble the final matrix
#############################################

# define where to store the matrix
A = PETScMatrix()

# assemble the stiffness matrix and store in A
assemble(a, tensor=A)

# apply the boundary conditions
bc.apply(A)

#############################################
# Compute eigenvalues
#############################################

# define eigen solver
eigensolver = SLEPcEigenSolver(A)

# Compute all eigenvalues of A x = \lambda x
print("Computing eigenvalues. This can take a minute.")
eigensolver.solve()

#############################################
# Extract and plot eigenfunctions
#############################################
# Extract smallest (last) eigenpair
r, c, rx, cx = eigensolver.get_eigenpair(A.array().shape[0] - 1)

# print eigenvalue
print("Smallest eigenvalue: ", r)

# Initialize function and assign eigenvector
u = Function(V_h)
u.vector()[:] = rx

list_r_2 = []
for i in range(A.array().shape[0]):
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    list_r_2.append(r)

plt.plot(list_r)
plt.plot(list_r_2)
plt.legend(['circle', 'square'])
plt.show()
plt.close()

# Plot eigenfunction
plot(u)
plt.show()

# c

import mshr

# Create mesh
domain = mshr.Rectangle(Point(0, 0), Point(0.7, 0.2 * math.sqrt(3)))
t = mshr.Polygon([Point(0, 0), Point(0.1, 0), Point(0, 0.1 * math.sqrt(3))])
domain = domain - t
t = mshr.Polygon([Point(0.7, 0.2 * math.sqrt(3)), Point(0.5, 0.2 * math.sqrt(3)),
                  Point(0.7, 0)])
domain = domain - t
t = mshr.Polygon([Point(0.4, 0.2 * math.sqrt(3)), Point(0.4, 0.1 * math.sqrt(3)),
                  Point(0.5, 0.2 * math.sqrt(3))])
domain = domain - t
t = mshr.Rectangle(Point(0, 0.1 * math.sqrt(3)), Point(0.4, 0.2 * math.sqrt(3)))
domain = domain - t

mesh = mshr.generate_mesh(domain, 20)

# Define function space
V_h = FunctionSpace(mesh, 'P', 1)

#############################################
# Define boundary conditions
#############################################

# define overal expression
u_D = Expression('0', degree=3)


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

# define the bilinear form
a = dot(grad(u), grad(v)) * dx

#############################################
# Assemble the final matrix
#############################################

# define where to store the matrix
A = PETScMatrix()

# assemble the stiffness matrix and store in A
assemble(a, tensor=A)

# apply the boundary conditions
bc.apply(A)

#############################################
# Compute eigenvalues
#############################################

# define eigen solver
eigensolver = SLEPcEigenSolver(A)

# Compute all eigenvalues of A x = \lambda x
print("Computing eigenvalues. This can take a minute.")
eigensolver.solve()

#############################################
# Extract and plot eigenfunctions
#############################################
# Extract smallest (last) eigenpair
r, c, rx, cx = eigensolver.get_eigenpair(A.array().shape[0] - 1)

# print eigenvalue
print("Smallest eigenvalue: ", r)

list_r = []
for i in range(A.array().shape[0]):
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    list_r.append(r)

# Initialize function and assign eigenvector
u = Function(V_h)
u.vector()[:] = rx

# Plot eigenfunction
plot(u)
plt.show()
plt.close()

# Create mesh
domain = mshr.Rectangle(Point(0, 0), Point(0.5, 0.3 * math.sqrt(3)))
t = mshr.Polygon([Point(0.2, 0), Point(0.5, 0), Point(0.5, 0.3 * math.sqrt(3))])
domain = domain - t
t = mshr.Polygon([Point(0.2, 0.2 * math.sqrt(3)), Point(0, 0.2 * math.sqrt(3)),
                  Point(0, 0)])
domain = domain - t
t = mshr.Polygon([Point(0.1, 0.3 * math.sqrt(3)), Point(0.1, 0.2 * math.sqrt(3)),
                  Point(0.2, 0.2 * math.sqrt(3))])
domain = domain - t
t = mshr.Rectangle(Point(0, 0.2 * math.sqrt(3)), Point(0.1, 0.3 * math.sqrt(3)))
domain = domain - t

mesh = mshr.generate_mesh(domain, 20)

# Define function space
V_h = FunctionSpace(mesh, 'P', 1)

#############################################
# Define boundary conditions
#############################################

# define overal expression
u_D = Expression('0', degree=3)


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

# define the bilinear form
a = dot(grad(u), grad(v)) * dx

#############################################
# Assemble the final matrix
#############################################

# define where to store the matrix
A = PETScMatrix()

# assemble the stiffness matrix and store in A
assemble(a, tensor=A)

# apply the boundary conditions
bc.apply(A)

#############################################
# Compute eigenvalues
#############################################

# define eigen solver
eigensolver = SLEPcEigenSolver(A)

# Compute all eigenvalues of A x = \lambda x
print("Computing eigenvalues. This can take a minute.")
eigensolver.solve()

#############################################
# Extract and plot eigenfunctions
#############################################
# Extract smallest (last) eigenpair
r, c, rx, cx = eigensolver.get_eigenpair(A.array().shape[0] - 1)

# print eigenvalue
print("Smallest eigenvalue: ", r)

list_r2 = []
for i in range(A.array().shape[0]):
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    list_r2.append(r)

# Initialize function and assign eigenvector
u = Function(V_h)
u.vector()[:] = rx

# Plot eigenfunction
plot(u)
plt.show()
plt.close()

plt.plot(list_r)
plt.plot(list_r2)
plt.show()
plt.close()
