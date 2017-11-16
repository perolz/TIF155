import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot3d
from sympy.plotting import plot_parametric
from mpl_toolkits.mplot3d import Axes3D


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
    solution=sy.solve(eq)[0]

    (y,t)=sy.symbols('y t')

    newEq=[sy.expand(eq[0].subs([(x,y+solution[x]),(r,t+solution[r])]))
        ,sy.expand(eq[0].subs([(x,y+solution[x]),(r,t+solution[r])]))]
    expression=x ** 3 - 3 * x ** 2 + (r + 2) * x - r
    newExp=expression.subs([(x,y+solution[x]),(r,t+solution[r])])
    print(sy.expand(newExp))
    sy.plot(newExp.subs(t,0))

    #sy.lambdify((t, y), newExp, modules='numpy')
    #tspace = np.linspace(-1.5, 1.5, 1000)

def saddle_node():
    (x,h,r,t)=sy.symbols('x h r t')
    #x = sy.symbols('x',function=True)
    equation=[sy.Eq(h+r*x-x**2,0),sy.Eq(r-2*x,0)]
    solution=sy.solve(equation)
    print(solution)
    xvalue=sy.lambdify(x,solution[0][h])
    yvalue=sy.lambdify(x,solution[0][r])

    test=10-10*x-x**2
    print(sy.solve(sy.Eq(test,0)))
    #ax=plot_parametric(x,test)

    fig=plt.figure(1)
    ax=fig.add_subplot(1,1,1)
    ax.set_xlabel('h')
    ax.set_ylabel('r')
    ax.set_title('Excersice_a')
    tspace = np.linspace(-10, 10, 1000)
    ax.plot(xvalue(tspace),yvalue(tspace),color='r')
    # ax.fill_between(xvalue(tspace[-500:]),yvalue(tspace[-500:]),20,facecolor='blue')
    # ax.fill_between(tspace[500:700],-20,20,facecolors='blue')
    # ax.fill_between(xvalue(tspace[:500]),yvalue(tspace[:500]),-20,facecolor='blue')

    ax.text(-80,18,'one stable and one unstable fixed point outside red line')
    ax.text(-80, 7, 'one unstable fixed point on red line')
    ax.text(-80, 0, 'zero fixed points between red lines')
    # ax.annotate('1 fixed point on red line', xy=(xvalue(4), yvalue(4)), xytext=(3, 1.5),
    #             arrowprops=dict(facecolor='black', shrink=0.05),
    #             )
    ax.set_xlim([-100, 3])
    ax.set_ylim([-20, 20])


    ax.grid(True,which='both')
    #plt.show()
    fig.savefig('Excersice_a.png')
    fig2 = plt.figure(2)
    ax2 = fig2.gca(projection='3d')

    X, Y= np.meshgrid(xvalue(tspace),yvalue(tspace))
    tmeshx,tmeshy=np.meshgrid(tspace,tspace)
    ax2.set_title('Excersice_b')
    ax2.set_xlabel('h')
    ax2.set_ylabel('r')
    ax2.set_zlabel('x')
    ax2.plot_surface(xvalue(tmeshy),yvalue(tmeshx),tmeshy)
    fig2.savefig('Excersice_b')
    plt.show()

if __name__=='__main__':
    #first_exercise(1)
    #print('\n')
    #second_exercise()
    # normal_form()
    saddle_node()