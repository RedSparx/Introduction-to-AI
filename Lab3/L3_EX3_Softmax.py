"""LABORATORY 3
EXERCISE #3: Given a data set, find the best fit model and make a prediction based on it.
    (a) Perform regression modeling of data against several models: (i)y=a*sin(x)+b,(ii)ln(y)=m*ln(x)+b,(iii)y=ax^3+b*x^2+c
    (b) Plot all three models with the data.
    (c) Compute residuals for each model.
    (d) Determine which of the models is the best fit for the given data.
    (e) Given a set of input values and the best fit model, predict the output.
"""

import numpy as np

def softmax(vals):
    y = np.exp(vals)
    y /= np.sum(y)
    return y


x=np.random.randn(5)
print(softmax(x))
