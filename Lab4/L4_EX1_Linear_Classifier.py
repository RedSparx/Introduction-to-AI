"""LABORATORY 4
EXERCISE #1: Train a perceptron to separate data into two classes.  Use the learned decision line to classify new data.
    (a) Load the raw data with class assignments for each point.
    (b) Plot the data sets.  Use two different colors or symbols to distinguish between class data points.
    (c) Train the perceptron using the delta learning rule.
    (d) Plot the class separation line.
    (e) Create a function that classifies a data point using the decision line.
    (f) Load the unknown data set and classify all points into one of the two classes.
"""

import numpy as np
import matplotlib.pyplot as plt
# region Set up dataset: (x,y,1) data separated into two target classes {+1, -1}
N = 2500
x = np.hstack((-4 + 2.5*np.random.randn(int(N/2)), 2 + 3.5*np.random.randn(int(N/2))))
y = np.hstack((-1 + 4.5*np.random.randn(int(N/2)), 5 + 2.5*np.random.randn(int(N/2))))
b = np.ones(N)
X = np.vstack((b, x, y)).transpose()
t = np.hstack((-1*np.zeros(int(N/2)), +1*np.ones(int(N/2))))
print(X.shape)
# endregion
# region Plot the data points for each class.
plt.scatter(x[np.where(t==1)], y[np.where(t==1)], color='b', alpha=0.5, marker='o')
plt.scatter(x[np.where(t==0)], y[np.where(t==0)], color='g', alpha=0.5, marker='*')
# endregion
# region Train the perceptron.
mu = 1E-6
w = np.random.randn(3)
Epochs = 1000
Error=np.zeros(Epochs)
for i in range(Epochs):
    for n in range(N):
        e = (np.dot(w, X[n, :])) - t[n]
        Error[i] = e
        w += mu*e*X[n, :]
w /=w.max() # OPTIONAL: normalizes the separation equation.
# endregion
# region Display the class separation line.
x_line = np.linspace(x.min(), x.max(), 100)
a = -(w[2]/w[1])
b = -w[0]/w[1]
y_line =  a* x_line + b
plt.plot(x_line,y_line, 'r-', linewidth=1)
plt.show()
print('y = %4.2fx%+4.2f'%(a,b))
# endregion
