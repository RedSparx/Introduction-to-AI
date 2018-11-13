"""LABORATORY 6
EXERCISE #1: Create a scalable layer of parallel Adaline units.  Test the custom class using 2 inputs and 2 outputs
    (a) Generate between 200 to 400 data points around each of these points: (0,0), (0,5), (5,0), (5,5). Points should
        be Gaussian distributed about each point. Labels are (-1,-1), (-1,+1), (+1,-1), (+1,+1) respectively.
    (b) Define a scalable adaline layer class. Create a constructor that will take on default values for the learning
        rate(1E-6) and maximum number of training iterations (100).
    (c) Create a training member function "fit" that will take in an input X and target labels T.  The function should
        infer the number of inputs from X and the number of parallel adaline units from T.  When training is complete,
        the function should return the weight matrix W.
    (d) Create a member function "predict" that will use the (post-training) weight matrix to estimate output labels for
        each input row.
    (e) After instantiating an instance of the single perceptron layer and using the previously generated data, train
        the system.
    (f) Using the same data set, test the system.  Determine the training accuracy.
    (g) Print the weight matrix.
    (h) Print each data point using a different color for each label (red, yellow, green, and blue respectively).
"""

import numpy as np
import matplotlib.pyplot as plt

# region Custom Adaline class.
class adaline:
    def __init__(self, mu=1E-6, max_iter=100):
        # Constructor with default values for critical parameters.
        self.mu = mu
        self.max_iter = max_iter
        self.depth = 0

    def fit(self, X, T):
        # Pad the data array with ones so that the bias is accounted for in the weights (last column).
        X = np.pad(X, ((0, 0), (0, 1)), mode='constant', constant_values=1)

        # Number of rows (data points) and columns (dimensions) in X.
        N, D = X.shape
        self.depth = T.shape[1]
        print('depth = %d'%self.depth)
        print('N=%4.2f\tD=%4.2f'%(N, D))
        self.W = np.random.randn(self.depth, D)
        for t in range(self.max_iter):
            for i in range(N):
                y = np.dot(self.W, X[i, :])
                e = T[i] - y
                delta = 2*self.mu*np.outer(e, X[i, :])
                self.W = self.W + delta
        return self.W

    def predict(self, X):
        X = np.pad(X, ((0, 0), (0, 1)), mode='constant', constant_values=1)
        N, D = X.shape
        y = np.zeros((N, D-1))
        for i in range(N):
            y[i, :] = np.sign(np.dot(W, X[i, :]))
        return y
# endregion

# region Generate a dataset for testing.
def Gen_Data_Set(set=1):
    if set == 1:
        T1 = -np.ones((100, 1))
        X1 = np.hstack([np.random.randn(100, 1), np.random.randn(100, 1)])
        T2 = np.ones((100,1))
        X2 = np.hstack([np.random.randn(100, 1) + 5, np.random.randn(100, 1) + 5])

        T = np.vstack([T1, T2])
        X = np.vstack([X1, X2])
        return X, T

    elif set == 2:
        T1 = np.hstack([+np.ones((100, 1)), +np.ones((100, 1))])
        T2 = np.hstack([+np.ones((100, 1)), -np.ones((100, 1))])
        T3 = np.hstack([-np.ones((100, 1)), +np.ones((100, 1))])
        T4 = np.hstack([-np.ones((100, 1)), -np.ones((100, 1))])
        T = np.vstack([T1, T2, T3, T4])

        X1 = np.hstack([np.random.randn(100, 1) + 0, np.random.randn(100, 1) + 0])
        X2 = np.hstack([np.random.randn(100, 1) + 0, np.random.randn(100, 1) + 5])
        X3 = np.hstack([np.random.randn(100, 1) + 5, np.random.randn(100, 1) + 0])
        X4 = np.hstack([np.random.randn(100, 1) + 5, np.random.randn(100, 1) + 5])
        X = np.vstack([X1, X2, X3, X4])
        return X, T
# endregion

if __name__ == '__main__':
    # region Generate data.
    Dataset_Number = 2
    X, T = Gen_Data_Set(set=Dataset_Number)
    Data = np.hstack((X, T))
    np.savetxt('Data_4_Classes_Set%1d.dat'%Dataset_Number, Data, delimiter=',')
    # endregion
    # region Define and train an adaline layer.  The number of inputs and outputs are defined just prior to training.
    Ada = adaline(max_iter=1500, mu=10E-6)
    W = Ada.fit(X, T)
    # endregion
    # region Use the adaline layer to predict the output based on the input. Determine the accuracy against the known labels.
    P = Ada.predict(X)
    Training_Accuracy = (1-np.count_nonzero(T-P)/len(X)) * 100
    print('Training Accuracy Achieved: %4.2f%%'%Training_Accuracy)
    # endregion
    # region Display the weight matrix.
    np.set_printoptions(precision=4, suppress=True)
    print('Exported Weight Matrix:')
    print(W)
    # endregion
    # region Iterate through each of the data points and plot them.  Change the color depending on the predicted output labels.
    plt.figure()
    for i in range(len(P)):
        if (P[i]==[-1, -1]).all():
            plt.plot(X[i, 0], X[i, 1], 'r.')
        if (P[i] == [+1, -1]).all():
            plt.plot(X[i, 0], X[i, 1], 'y.')
        if (P[i] == [-1, +1]).all():
            plt.plot(X[i, 0], X[i, 1], 'g.')
        if (P[i] == [+1, +1]).all():
            plt.plot(X[i, 0], X[i, 1], 'b.')
    plt.title('Predicted Labels from Data (%6.2f%% Accuracy)'%Training_Accuracy)
    # endregion
    # region Draw the decision boundaries.
    y = np.linspace(-5, 10, 10)
    w1, w2, b = W[1]
    plt.plot(y, -(w2/w1)*y - (b/w1), 'k-.', label='Boundary 1')

    x = np.linspace(-5, 10, 10)
    w1, w2, b = W[1]
    plt.plot(x, -(w1/w2)*x - (b/w2), 'k--', label='Boundary 2')

    plt.axis([-4, 9, -4, 9])
    plt.legend(loc='lower right')
    plt.savefig('Perceptron_Layer.png')
    plt.show()
    # endregion


