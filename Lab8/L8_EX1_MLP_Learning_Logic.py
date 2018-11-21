"""LABORATORY 8
EXERCISE #1: Train a MLP on non-linearly separable XOR logic as well as separable AND and OR logic.
    (a) Create arrays to hold input vectors.  Inputs are binary {0,1}.
    (b) For the input space, add 1000 random values to each point (gaussian, zero-mean, 0.15 variance).
    (c) Create the binary target output vectors for XOR, AND and OR logic.
    (d) Create the neural network by defining it sequentially: Three dense layers with 3, 5 and  1 units respectively.
    (e) Add activation functions: 'tanh' to all layers.
    (f) Train the network over 100 epochs with a verbose output so that progress can be observed.
    (g) Plot the training accuracy with a title and axis labels.
    (h) Sample the entire input space and predict the output and plot the decision surface. The "region of interest"
        (ROI) consists of points on a grid for which a prediction is made using the network.  To use the neural network,
         we first need to convert the grid array of points to a list of points.  We can then reshape the predicted
         values back into a grid for plotting.
    (i) Plot the input data and superimpose a contour plot of the decision surface.  Note that because we wish for our
        output to be {0,1}, we will assume that the decision boundary for the output is 0.5.  This boundary will be
        plotted.
    (j) Save the resulting plot.
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# region Arrays to hold the logic truth tables.
Input_A = np.array([0, 1, 0, 1])
Input_B = np.array([0, 0, 1, 1])
AND = np.array([0, 0, 0, 1])
OR = np.array([0, 1, 1, 1])
XOR = np.array([0, 1, 1, 0])
# endregion
# region Add 1000 random values to each of the inputs.  Combine these arrays into a single array.  Plot these points.
N=1000
var = 0.15
Input_A_Random = np.array([i+var*np.random.randn(N) for i in Input_A]).flatten()
Input_B_Random = np.array([i+var*np.random.randn(N) for i in Input_B]).flatten()
Input_AB = np.vstack((Input_A_Random, Input_B_Random)).transpose()
# endregion
# region Create an output target vector for the logic gates.
AND_Output = np.array([i*np.ones(N) for i in AND]).flatten()
OR_Output = np.array([i*np.ones(N) for i in OR]).flatten()
XOR_Output = np.array([i*np.ones(N) for i in XOR]).flatten()
# endregion
# region Define a 3 layer neural network with 2 inputs, one output and 1 hidden layer. The input layer should contain:
#   Layer 1: 3 units
#   Layer 2: 5 units
#   Layer 3: 1 unit
Inputs=2
Outputs=1

model = Sequential()
model.add(Dense(3, input_shape=(Inputs,)))
model.add(Activation('tanh'))
model.add(Dense(5))
model.add(Activation('tanh'))
model.add(Dense(Outputs))
model.add(Activation('tanh'))
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
# endregion
# region Train the neural network and plot the history of its training accuracy in a subplot.
Hist = model.fit(Input_AB, XOR_Output, epochs=150, verbose=True)
plt.subplot(121)
plt.plot(Hist.history['acc'])
plt.xlabel('Training Epochs')
plt.ylabel('Training Accuracy')
plt.title('Accuracy History')
# endregion
# region Sample the entire input space and predict the output and plot the decision surface. The
# "region of interest" (ROI) consists of points on a grid for which a prediction is made using the network.  To use the
# neural network, we first need to convert the grid array of points to a list of points.  We can then reshape the
# predicted values back into a grid for plotting.
Ngrid = 100
x = np.linspace(-1, +2, Ngrid)
y = np.linspace(-1, +2, Ngrid)
X, Y = np.meshgrid(x, y)
ROI = np.array([X.ravel(), Y.ravel()]).transpose()
Z = model.predict(ROI).reshape(X.shape)
# endregion
# region Plot the input data and superimpose a contour plot of the decision surface.  Note that because we wish for our
# output to be {0,1}, we will assume that the decision boundary for the output is 0.5.  This boundary will be plotted.
plt.subplot(1,2,2)
ax1 = plt.gca()
plt.scatter(Input_A_Random, Input_B_Random, marker='*', alpha=0.1)
plt.text(0, 0, '0 0', color = 'y', horizontalalignment='center', verticalalignment='center')
plt.text(0, 1, '0 1', color = 'w', horizontalalignment='center', verticalalignment='center')
plt.text(1, 0, '1 0', color = 'w', horizontalalignment='center', verticalalignment='center')
plt.text(1, 1, '1 1', color = 'y', horizontalalignment='center', verticalalignment='center')

cs = ax1.contour(X, Y, Z, levels=[0.5], colors=['r'], linestyles=['dashed'], linewidths=[3], alpha=1)
plt.axis('tight')
plt.title('Decision Boundary: XOR Logic')
plt.xlabel('Input A')
plt.ylabel('Input B')
plt.savefig(r'Data\MLP_XOR_Logic.png')
plt.show()
# endregion
