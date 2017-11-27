import sympy as sy
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

numberOfArrows=20
tspace=np.linspace(0,2*np.pi,numberOfArrows)
xval=np.cos(tspace)
yval=np.sin(tspace)

#X=[first**2 for first in xval]
#Y=yval

#X=[second-first for first,second in zip(xval,yval)]
#Y=[first**2 for first in xval]

#X=[first**3 for first in yval]
#Y=xval

#X=[second*first for first,second in zip(xval,yval)]
#Y=[second+first for first,second in zip(xval,yval)]

#X=1-np.cos(3*np.arccos(xval))
#Y=np.sin(3*np.arcsin(yval))
x,y =sy.symbols('x y',real=True)

z=x+sy.I*y
index=-3
test=z**2
print(sy.expand(test).as_real_imag())

# X=[]
# X.append( [x**2 - y for x,y in zip(xval,yval)])
# X.append([2*x*y for x,y in zip(xval,yval)])

# X.append( [x**3+x for x in xval])
# X.append([y for y in yval])
#
# X.append([x**3+x for x in xval])
# X.append([y**3+y for y in yval])
#
# X4 = [x**2 for x in xval]
# Y4=[y**2+y for y in yval]

print(sy.series(sy.asin(y),y,0,11,'+'))



#phi=[np.arctan(first/second) for first,second in zip(X,Y)]

#print(phi)
#
# fig =plt.figure()
#
#
# ax=fig.add_subplot(1,1,1)
# axes = plt.gca()
# axes.set_xlim([np.min(X[0])-1,np.max(X[0])+1])
# axes.set_ylim([np.min(X[1])-1,np.max(X[1])+1])
# for i in range(numberOfArrows):
#      ax.arrow(0,0,X[0][i],X[1][i],head_width=0.05)
#      plt.draw()
#      plt.pause(0.2)

w = 3
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U = X**3 - 3*X*Y**2
V = -3*X**2*Y + Y**3

speed = np.sqrt(U*U + V*V)

fig = plt.figure(figsize=(7, 9))
gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])


#  Varying density along a streamline
ax0 = fig.add_subplot(1,1,1)
ax0.streamplot(X, Y, U, V, density=[1, 1])
ax0.set_title(r'$\dot{x}=x^3 - 3*x*y^2 \quad \dot{y}=-3*x^2*y + y^3$')
plt.xlabel('x',fontweight='bold')
plt.ylabel('x\'',fontweight='bold')
plt.suptitle(r'Index for plot $I_C$=%d' %index)


plt.savefig('Images/Indexing%d.png' %index)


plt.show()