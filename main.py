import sympy as sy
import numpy as np
import matplotlib.pyplot as plt

sy.init_printing()


sigma=sy.symbols('sigma')
A=sy.Matrix([[(sigma+3),4],[-(9/4),sigma-3]])
eigen=A.eigenvals()



print(A.eigenvects())
print(A.inv())

(t,C1,C2) = sy.symbols('t C1 C2')
x, y = sy.symbols('x, y', function=True)

eq1=sy.Eq(sy.Derivative(x(t),t),(sigma+3)*x(t)+4*y(t))
eq2=sy.Eq(sy.Derivative(y(t),t),-(9/4)*x(t)+(sigma-3)*y(t))


general_solution=sy.dsolve([eq1,eq2],[x(t),y(t)])
#Specific Value
#value = [general_solution[0].subs([(C2,0.1),(C1,0.1)]),general_solution[1].subs([(C2,0.1),(C1,0.1)])]



#print(value)

xlam = sy.lambdify((t,sigma,C1,C2), general_solution[0].rhs, modules='numpy')
ylam = sy.lambdify((t,sigma,C1,C2), general_solution[1].rhs, modules='numpy')

tspace=np.linspace(-1.5,1.5,1000)

x_vals = xlam(tspace,1,1,1)
y_vals = ylam(tspace,1,1,1)
plt.plot(x_vals,y_vals)
plt.show()
#for i in range(-1,1):
