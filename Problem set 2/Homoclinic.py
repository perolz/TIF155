import sympy as sy
import scipy
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
x,y=sy.symbols('x,y',function=True)
mu,t=sy.symbols('mu,t',real=True)

eq1=mu*x+y-x**2
eq2=-x+mu*y+2*x**2

system=sy.Matrix([eq1,eq2])
b=sy.Matrix([0,0])
fixedPoints=list(sy.nonlinsolve(system,[x,y]))

print(fixedPoints[0])
Y=sy.Matrix([x,y])
M=system.jacobian(Y)

test=M.subs(x,fixedPoints[1][0])
#delta=test.det()
eigen1=list(test.eigenvals())


print(eigen1)


