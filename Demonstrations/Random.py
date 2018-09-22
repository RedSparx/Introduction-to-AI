import numpy as np
import matplotlib.pyplot as plt

N=1000
t=np.linspace(0, 3, N)
x=np.random.randn(N)
plt.plot(x+20*np.sin(t),'-')
plt.show()
