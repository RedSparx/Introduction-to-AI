"""LABORATORY 3
EXERCISE #1: Create an error surface for a regression problem.
    (a) Create a simple linear model using fixed coefficients.
    (b) Create a new data set with additive Gaussian noise.
    (c) Compute the error, absolute error as well as the squared error. Plot them.
    (d) Estimate the linear (polynomial) coefficients for the noisy model.
    (e) Plot the error surface in a,b space identifying the minima.
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import *

# region Create the ideal linear model with parameters a and b.
N = 150
a = 3
b = 2
x = np.linspace(-4, 4, N)
y = a*x+b
# endregion
# region Create noisy data from the model.
mu = 0
sigma = 3.5
noise = sigma*np.random.randn(N)+mu
yn = y+noise
# endregion
# region Using the noisy data, compute the error.
Error = y-yn
Abs_Error = np.abs(y-yn)
Sq_Error = (y-yn)**2
# endregion
# region Plot the models and errors.
plt.subplot(2, 2, 2)
plt.plot(x, y, label='Linear Model')
plt.plot(x, yn, '.', label='Noisy Model')
plt.title('Linear Model: $y=%4.2fx+%4.2f$'%(a,b))
plt.legend(loc=2)
plt.subplot(2, 2, 4)
plt.plot(x, Error, '--', label='Error')
plt.plot(x, Abs_Error, '-.', label='Absolute Error')
plt.plot(x, Sq_Error, 'r', label='Squared Error', linewidth=1.5)
plt.legend(loc=2)
# plt.show()
# endregion
# region Estimate the linear (polynomial) coefficients for the noisy model.
a_est, b_est = polyfit(x,yn, 1)
# endregion
# region Plot the error surface in a,b space identifying the minima.
a_val = np.linspace(-5, 5, N)
b_val = np.linspace(-20, 20, N)

A, B = np.meshgrid(a_val, b_val)
Mean_Square_Error = np.zeros_like(A)

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        Mean_Square_Error[i,j] = np.sum((A[i,j]*x + B[i,j] - yn)**2)

plt.subplot(2,2,(1,3))
plt.plot(a,b,'g*', label='Linear Model')
plt.plot(a_est, b_est,'r*', label='Regression Model')
plt.text(a_est, b_est, '(%3.2f,%3.2f)   '%(a_est, b_est),
         horizontalalignment='right',
         verticalalignment='center',
         fontsize=6)
plt.contourf(A, B, Mean_Square_Error, contours=100, cmap='binary')
plt.xlabel('a')
plt.ylabel('b')
plt.legend(loc=2)
plt.show()
# endregion
