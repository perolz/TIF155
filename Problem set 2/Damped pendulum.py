import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot3d
from sympy.plotting import plot_parametric
from mpl_toolkits.mplot3d import Axes3D
import pickle

sy.init_printing()

(g, l, gamma, m) = sy.symbols('g l gamma m', positive=True, constant=True)
theta, omega = sy.symbols('theta omega', function=True)
t = sy.symbols('t')
# A=sy.Matrix([[0,1],[sy.Rational(g,l)*sy.sin(),sigma-3]])

eq1 = sy.Eq(sy.Derivative(theta(t), t), omega(t))
eq2 = sy.Eq(sy.Derivative(omega(t), t), -g / l * sy.sin(theta(t)) - gamma / m * omega(t))

system = [eq1, eq2]

# print(sy.solve(system))
(theta0, omega0, t0, sigma) = sy.symbols('theta0,omega0,t0,sigma', positive=True)
tprim = sy.symbols('tprim')
(x, y) = sy.symbols('x,y', function=True)

# For assignment a-d
# eq3 = sy.Eq(y(tprim), t0 / theta0 * omega0 * y(tprim))
# eq4 = sy.Eq(-sy.sin(x(tprim)) - sigma * y(tprim),
#             sy.simplify(t0 / omega0 * (-g / l * sy.sin(theta0 * x(tprim)) - gamma * omega0 / m * y(tprim))))
#
# substitutedSystem = [z.subs([(theta(t), theta0 * x(tprim)), (omega(t), omega0 * y(tprim)), (t, t0 * tprim)]) for z in
#                      system]
# sol = [eq3, eq4]
# sol = [z.subs(theta0, 1) for z in sol]
# sy.pprint(substitutedSystem)
# newsol = sy.nonlinsolve(sol, t0, omega0, sigma)

# Reduced system with x and y
#eq5 = sy.Eq(sy.Derivative(x(tprim), tprim), y)
#eq6 = sy.Eq(sy.Derivative(y(tprim), tprim), -sy.sin(x) - sigma * y(tprim))
eq5 = sy.Eq(0, y)
eq6 = sy.Eq(0, -sy.sin(x) - sigma * y)
system2 = [eq5, eq6]
print(sy.solve(system2))
#sy.pprint(system2)

sigma_value=0
w = 4
#Y, X = np.mgrid[-w:w:100j, -w:w:100j]
#U=Y
#V=-np.sin(X)-sigma_value*Y
#Z=np.array([[np.linalg.norm([U[i][j],V[i][j]]) for i in range(100)]for j in range(100)])
#speed = np.sqrt(U**2 + V**2)
#UN=U/speed
#VN=U/speed


v=np.linspace(0, 2.0, 15, endpoint=True)
fig = plt.figure()

ax0 = fig.add_subplot(1,1,1)
X=[-np.pi,0,np.pi]
Y=[0,0,0]

plt.scatter(X,Y,s=80,facecolors='none',edgecolors='r')

#ax0.set_ylim(-2,2)
#levels = [0,0.001,0.01,0.1, 0.5, 1]
#cp = plt.contourf(Y, X, Z,levels)

plt.title(r'Phase plot for every $\sigma$')
plt.xlabel('x',fontweight='bold')
plt.ylabel('x\'',fontweight='bold')

#cb = plt.colorbar(cp,ticks=v)
#plt.quiver(X, Y, UN, VN,color='Red')
fig.savefig('Images/Plot_for_damped_pendelum.png')
plt.show()



#print(sy.solve(system2,[x,y],check=False))

