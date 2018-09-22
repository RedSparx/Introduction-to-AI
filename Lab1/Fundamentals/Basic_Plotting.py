import numpy as np
import matplotlib.pyplot as plt

N=100
x=np.linspace(0, 10, N)
n=0.5*np.random.randn(N)
y=np.sin(x)+n

plt.plot(x, y, '.-')
plt.show()
