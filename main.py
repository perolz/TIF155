import sympy as sy
sy.init_printing(use_unicode=True)


sigma=sy.symbols('sigma')
A=sy.Matrix([[(sigma+3),4],[-(9/4),sigma-3]])
eigen=A.eigenvals()


print(eigen)

t = sy.symbols('t')
x, y = sy.symbols('x, y', function=True)

eq=(sy.Eq(sy.Derivative(x(t),t),(sigma+3)*x(t)+4*y(t)),
    sy.Eq(sy.Derivative(y(t),t),-(9/4)*x(t)+(sigma-3)*y(t)))
print(eq.subs(x,1))

solution=sy.dsolve()
print(solution)