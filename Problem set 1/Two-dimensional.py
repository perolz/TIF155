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

#For a)
newSolutions=[z.subs([(C1,solutionsForC1C2[C1]),(C2,solutionsForC1C2[C2])]) for z in solutions]

#b)
func=[sy.lambdify((t,sigma,x(0),y(0)),z.rhs) for z in newSolutions]

tspace=np.linspace(-2,9,1000)
x0=1
y0=1
sigmaValue=[-0.1,0,1/10]
# for i in sigmaValue:
#     fig=plt.figure()
#     plt.title(r'Plot for $\sigma$=%1.1f' %i)
#     plt.xlabel(x(t))
#     plt.ylabel(y(t))
#     plt.tight_layout()
#     test0=[y-x for x in func[0](tspace[:-1], i, x0, y0) for y in func[0](tspace[1:], i, x0, y0)]
#     test1 = [y - x for x in func[1](tspace[:-1], i, x0, y0) for y in func[1](tspace[1:], i, x0, y0)]
#     #plt.quiver(func[0](tspace[:-1], i, x0, y0),func[1](tspace[:-1], i, x0, y0),
#                #test0,test1)
#
#     plt.plot(func[0](tspace, i, x0, y0), func[1](tspace, i, x0, y0))
#
#     for j in range(9):
#         plt.quiver(func[0](tspace[100*j], i, x0, y0), func[1](tspace[100*j], i, x0, y0),
#                    func[0](tspace[100*j+10], i, x0, y0) - func[0](tspace[100*j], i, x0, y0),
#                    func[1](tspace[100*j+10], i, x0, y0) - func[1](tspace[100*j], i, x0, y0))
#     # plt.plot(func[0](tspace[-1], i, x0, y0), func[1](tspace[-1], i, x0, y0), 'r>')
#     fig.savefig('Images/plot15sigma%1.1f.png'%i)

#c)


sigma0Eq=[z.subs([(sigma,0),(x(0),1),(y(0),1)]) for z in newSolutions]

angle=sy.sympify(sigma0Eq[1].rhs/sigma0Eq[0].rhs)
test=sy.Eq(angle)
sigma0Solutions=sy.simplify(sy.solve(test))
tmp=[sy.N(z) for z in sigma0Solutions]
print(tmp)

fig=plt.figure()


# sigma0Solutions=sy.dsolve(sigma0Eq)
# sigma0Solutions=[z.subs(sy.sqrt(5),omega) for z in sigma0Solutions]


# omegaSolution=sy.simplify(sy.solve(sigma0Solutions,omega))
# print(sy.solve(sigma0Solutions,omega))
# sigma0InitialConditions=[z.subs([(t,0),(sy.sqrt(5),omega)]) for z in sigma0Solutions]
# sigma0SolutionsForC1C2=sy.solve(sigma0InitialConditions)
#
# print(sigma0SolutionsForC1C2)

plt.show()