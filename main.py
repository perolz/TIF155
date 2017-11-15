import sympy as sy
import numpy as np
import matplotlib.pyplot as plt

sy.init_printing()

def first_exercise(sigma_value):
    sigma=sy.symbols('sigma')
    A=sy.Matrix([[(sigma+3),4],[sy.Rational(-9,4),sigma-3]])
    eigen=A.eigenvals()


    print('Eigenvectors for system')
    sy.pprint(A.eigenvects())
    print('Inverse')
    #sy.pprint(A.inv()[0])
    sy.pprint(sy.simplify(A.inv()))

    inverse=sy.simplify(A.inv())
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

    x_vals = xlam(tspace,sigma_value,1,1)
    y_vals = ylam(tspace,sigma_value,1,1)
    plt.plot(x_vals,y_vals)
    #plt.show()

def second_exercise():
    sigma,c,d = sy.symbols('sigma,c,d')
    A = sy.Matrix([[(sigma - c*d), d**2], [-c**2, sigma + c*d]])
    eigen = A.eigenvals()
    print('Eigenvectors for system')
    sy.pprint(A.eigenvects())
    print('Inverse')
    # sy.pprint(A.inv()[0])
    sy.pprint(sy.simplify(A.inv()))

    values=A.subs(sigma,-1)
    print(values)
    sy.pprint(values)


def normal_form():
    #x=sy.Function('f')
    (r,x)=sy.symbols('r x')
    #eq=sy.Eq(sy.Derivative(x(t),t),x(t)**3-3*x(t)**2+(r+2)*x(t)-r)
    eq = [sy.Eq(x ** 3 - 3 * x ** 2 + (r + 2) * x - r,0),
          sy.Eq(3*x ** 2 - 6 * x + (r + 2),0)]
    solution=sy.solve(eq)
    sy.pprint(solution[0])
    y=sy.symbols('y')
    newEq=eq[0].subs(x,y+solution[0][x])
    sy.pprint(newEq)


if __name__=='__main__':
    #first_exercise(1)
    #print('\n')
    #second_exercise()
    normal_form()
