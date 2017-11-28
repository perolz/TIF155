import sympy as sy
import scipy
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# def model1(X,t):
#     dxdt=-X[0]**3+4*X[0]-3*X[1]
#     dydt=3*X[0]+2*X[1]**3+4*X[1]
#     return [dxdt,dydt]
#
# ts = np.linspace(0, 12, 100)
# P0 = [1, 1]
# Ps = odeint(model1, P0, ts)
# plt.plot(Ps[:,0],Ps[:,1])
# plt.show()


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
dxdt=sy.lambdify((mu,x,y),eq1.rhs)
dydt=sy.lambdify((mu,x,y),eq2.rhs)

newSystem=[z.subs(mu,muValue) for z in system]
newSystem=[z.subs('**','^') for z in newSystem]
print(newSystem)
ts = np.linspace(-1, 1, 100)
Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U=muValue*Y-X**2
V=-X+muValue*Y+2*X**2

# U = -X**3 +muValue * X-3 *Y
# V = 3*X + 2*Y**3 + muValue*Y
speed = np.sqrt(U*U + V*V)
fig = plt.figure(figsize=(7, 9))


#  Varying density along a streamline
ax0 = fig.add_subplot(1,1,1)
strm = ax0.streamplot(X, Y, U, V, linewidth=2)
ax0.set_title(r'$\dot{x}=4*x+y-x^2 \quad \dot{y}=-x +4*y+2*x^2$')
plt.xlabel('x',fontweight='bold')
plt.ylabel('y',fontweight='bold')
plt.suptitle(r'Hopf bifucation with $\mu$=%d' %muValue)


plt.savefig('Images/Hopfeq2%d.png' %muValue)


plt.show()
