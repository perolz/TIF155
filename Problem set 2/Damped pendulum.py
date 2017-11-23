import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot3d
from sympy.plotting import plot_parametric
from mpl_toolkits.mplot3d import Axes3D
import pickle

(g,l,gamma,m)=sy.symbols('g l gamma m',positive=True,constant=True)
theta,omega=sy.symbols('theta omega',function=True)
t=sy.symbols('t')
#A=sy.Matrix([[0,1],[sy.Rational(g,l)*sy.sin(),sigma-3]])

eq1=sy.Eq(sy.Derivative(theta(t),t),omega(t))
eq2=sy.Eq(sy.Derivative(omega(t),t),-g/l*sy.sin(theta(t))-gamma/m*omega(t))

system=[eq1,eq2]


#print(sy.solve(system))
(theta0,omega0,t0,sigma)=sy.symbols('theta0,omega0,t0,sigma',positive=True,function=True,constant=True)
tprim=sy.symbols('tprim')
(x,y)=sy.symbols('x,y',function=True)


eq3=sy.Eq(y(tprim),t0/theta0*omega0*y(tprim))
eq4=sy.Eq(-sy.sin(x(tprim))-sigma*y(tprim),
          sy.simplify(t0/omega0*(-g/l*sy.sin(theta0*x(tprim))-gamma*omega0/m*y(tprim))))

substitutedSystem=[z.subs([(theta(t),theta0*x(tprim)),(omega(t),omega0*y(tprim)),(t,t0*tprim)]) for z in system]
sol=[eq3,eq4]
sol=[z.subs(theta0,1) for z in sol]
#sy.pprint(substitutedSystem)
newsol=sy.nonlinsolve(sol,t0,omega0,sigma)

sy.pprint(newsol)






