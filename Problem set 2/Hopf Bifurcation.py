import sympy as sy
import scipy
import numpy as np
import matplotlib.pyplot as plt

#Linearisation gives
x,y=sy.symbols('x,y',function=True)
mu,t=sy.symbols('mu,t')
omega=sy.symbols('omega',complex=True)
A=sy.Matrix([[mu,-3],[3,mu]])


yprim,xprim=sy.symbols('yprim,xprim',complex=True)
f,g=sy.symbols('f g',function=True)


f=x**3
g=2*y**3
# f=-x**2
# g=2*x**2
w=3
a=sy.symbols('a')
equation=sy.Eq(16*a,f.diff(x,x,x)+f.diff(x,y,y)+g.diff(x,x,y)+g.diff(y,y,y)+
               sy.Rational(1,w)*(f.diff(x,y)*(f.diff(x,x)+f.diff(y,y))-g.diff(x,y)*(g.diff(x,x)+g.diff(y,y))
                    -f.diff(x,x)*g.diff(x,x)+f.diff(y,y)*g.diff(y,y))
               )
#sy.pprint(sy.solve(equation))

eq1=sy.Eq(sy.Derivative(x(t),t),mu*x-3*y-x**3)
eq2=sy.Eq(sy.Derivative(y(t),t),3*x+mu*y+2*y**3)
system=[eq1,eq2]

muValue=4
newSystem=[z.subs(mu,muValue) for z in system]

sy.pprint(newSystem)

print(sy.dsolve(system))
#print(sy.solve(system,[omega,yprim,xprim]))