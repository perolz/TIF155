import matplotlib.pyplot as plt
import pickle


with open('myplot14.pkl','rb') as fid:
    fig = pickle.load(fid)
plt.title('Subcritical pitchfork')
plt.xlabel('r')
plt.ylabel('x')
plt.savefig('Subcritical.png')
plt.show()
