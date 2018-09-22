""" Generate data sets for 2D model fitting (regression).
"""

import numpy as np

def save_file(fname, x,y):
    Data=np.vstack((x,y)).transpose()
    np.savetxt(fname,Data,delimiter=',')

N=500
x=np.linspace(3, 9, N)

# Data set 1.
y=0.1*np.exp(-0.02*x)
save_file(r'Data\Ideal_Exponential.csv', x, y)

# Data set 2.
y = x**3 + x**2 +x + 1
save_file(r'Data\Ideal_Cubic.csv', x, y)

# Random data sets (varying in noise).
y = (2*np.sin(x)+1)+(1.375*np.random.randn(N))+(2*x+1)
save_file(r'Data\Data_Set_1.csv', x, y)
y = (2*np.sin(x)+1)+(0.75*np.random.randn(N))+(2*x+1)
save_file(r'Data\Data_Set_2.csv', x, y)
y = (2*np.sin(x)+1)+(5.5*np.random.randn(N))+(2*x+1)
save_file(r'Data\Data_Set_3.csv', x, y)


# y=100*np.random.randn(N)+0.1

# endregion