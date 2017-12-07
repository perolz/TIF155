import sympy as sy
import scipy
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers.inequalities import reduce_rational_inequalities

def solutionAandB():
    x,y,r,theta,dx,dy=sy.symbols('x,y,r,theta,dx,dy',function=True)
    mu,omega,nu=sy.symbols('mu,omega,nu',real=True,positive=True)

    dr=mu*r-r**3
    dtheta=omega+nu*r**2

    DX=sy.Eq(dx,1/x*(r*dr-y*dy))
    DY=sy.Eq(dy,((x**2+y**2)*dtheta+y*dx)/x)
    print(DX)
    system=[DX,DY]
    system=[z.subs(r,sy.sqrt(x**2+y**2)) for z in system]
    solutiondxdy=sy.solve(system,[dx,dy])

    print(sy.simplify(solutiondxdy))
    #print(solutiondxdy[dy].subs([(mu,1),(nu,1),(omega,1)]))

    muValue=6
    nuValue=1
    omegaValue=1
    ts = np.linspace(-1, 1, 100)
    Y, X = np.mgrid[-3:3:100j, -3:3:100j]
    U=sy.lambdify((x,y,mu,nu,omega),solutiondxdy[dx], modules='numpy')
    W=sy.lambdify((x,y,mu,nu,omega),solutiondxdy[dy], modules='numpy')
    Uval=U(X,Y,muValue,nuValue,omegaValue)
    Wval=W(X,Y,muValue,nuValue,omegaValue)
    #.subs([(mu,1),(nu,1),(omega,1)])
    limitCycle=sy.solve(sy.Eq(dr.subs(mu,muValue),0))
    tspan=np.linspace(0,2*np.pi,100)
    #tspan=np.linspace(-np.sqrt(muValue),np.sqrt(muValue),1000)
    #yval=np.sqrt(muValue-tspan**2)
    xval=limitCycle[-1]*np.cos(tspan)
    yval=limitCycle[-1]*np.sin(tspan)

    V=sy.lambdify((x),limitCycle, modules='numpy')

    #speed = np.sqrt(U*U + V*V)
    fig = plt.figure(figsize=(9, 9))


    ax0 = fig.add_subplot(1,1,1)
    strm = ax0.streamplot(X, Y, Uval, Wval, linewidth=2)
    ax0.set_title(r'$\dot{r}=\mu r-r^3 \quad \dot{\theta}=\omega +\nu r^2$')
    plt.xlabel('x',fontweight='bold')
    plt.ylabel('y',fontweight='bold')
    plt.suptitle(r'Streamplot of system $\mu$=%d, $\omega$=%d and $\nu$=%d' %(muValue,omegaValue,nuValue))
    plt.plot(xval,yval,'r')
    fig.savefig('Images/Exercise1b.eps')

    plt.show()

def solutionC():
    X1, X2= sy.symbols('X1,X2', nonzero=True)
    f1=X1/10-X2**3-X1*X2**2-X1**2*X2-X2-X1**3
    f2=X1+X2/10+X1*X2**2+X1**3-X2**3-X1**2*X2

    F=sy.Matrix([f1,f2])
    J=F.jacobian([X1,X2])

    M11, M12,M21,M22 = sy.symbols('M11,M12,M21,M22', positive=True,real=True)
    M=sy.Matrix([[M11,M12],[M21,M22]])
    Mdot=J*M

    f3=Mdot[0]
    f4=Mdot[1]
    f5=Mdot[2]
    f6=Mdot[3]
    print(J)
    print(sy.nonlinsolve((f1,f2,f3,f4,f5,f6),(X1,X2,M11,M12,M21,M22)))

    tmp=f5.subs([(X1,0.1),(X2,-0.1)])
    print(tmp)
    #reduce_rational_inequalities([[
    #    ((sy.Poly(tmp),0), '<')]],(M11,M21))
    solution=sy.solve(tmp<0,M11)

    print(solution)


if __name__ == '__main__':
    solutionC()