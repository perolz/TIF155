import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot3d
from sympy.plotting import plot_parametric
from mpl_toolkits.mplot3d import Axes3D
import pickle


if __name__ == '__main__':
    fig = plt.figure(figsize=(9, 9))
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.set_title(r'$\dot{r}=\mu r-r^3 \quad \dot{\theta}=\omega +\nu r^2$')
    plt.xlabel('x', fontweight='bold')
    plt.ylabel('y', fontweight='bold')

    for i in np.linspace(-0.6,1.5):
        #for j in np.linspace(-0.6,1,10):
        x=[i]
        y=[i]

        for k in range(100):
            y.append(0.3*x[-1])
            x.append(y[-1]+1-1.4*x[-1]**2)
        if(np.max(x)>1000):
            print(y)
        plt.plot(x, y)
        del x
        del y

    plt.show()