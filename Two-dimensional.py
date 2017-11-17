import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot3d
from sympy.plotting import plot_parametric
from mpl_toolkits.mplot3d import Axes3D
import pickle
sy.init_printing(use_unicode=False)

sigma,t = sy.symbols('sigma t',real=True)
x,y =sy.symbols('x y',function=True)
M=sy.Matrix([[sigma+1,3],[-2,sigma-1]])
eq=[sy.Eq(sy.Derivative(x(t),t),(sigma+1)*x(t)+3*y(t)),
    sy.Eq(sy.Derivative(y(t),t),-2*x(t)+(sigma-1)*y(t))]
solutions=sy.dsolve(eq)

C1,C2,omega =sy.symbols('C1 C2 omega')
initialConditions=[z.subs(t,0) for z in solutions]
solutionsForC1C2=sy.solve(initialConditions)

newSolutions=[z.subs([(C1,solutionsForC1C2[C1]),(C2,solutionsForC1C2[C2]),(sy.sqrt(5),omega)]) for z in solutions]
print(solutions)
sy.pprint(sy.simplify(newSolutions[0]))
sy.pprint(sy.simplify(newSolutions[1]))

