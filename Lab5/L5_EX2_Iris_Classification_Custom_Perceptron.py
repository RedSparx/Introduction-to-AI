"""LABORATORY 5
EXERCISE #2: Create a custom perceptron class and train it to distinguish between setosa and versicolor petals using
    length and width (Iris dataset).
    (a) Load the Iris Dataset from scikit.  Plot the sepal length vs sepal width for the setosa and versicolor flowers.
    (b) Assemble training data into an array with data point in rows (columns contain features).
    (c) Plot the data.
    (d) Create a perceptron class (Percept) with a constructor that will: initialize random weights (W), a maximum
        number of iterations (iterations=2000), learning rate (mu=1E-3), and a training score (score).  The constructor
        should take on optional parameters lrate and iter for learning rate and number of iterations respectively.
        Weights should be randomly initialized. Be sure to shuffle the data so that labeled data are in random positions.
    (e) Add a member function train(X,T) that takes data (X) and target labels (T) and implements the learning rule.
        Training should iterate over the entire data set using the set number of iterations.  It should return the
        training score.
    (f) Add a member function predict(X) that will return the output predictions given an input data vector.
    (g) Instantiate the class using the default parameters.  Train the perceptron using the data and target
        classification.
    (h) Predict flower classifications for the training data.
    (i) Using the perceptron weights, plot the decision line.
    (j) Save the plot.
"""
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

class Percept:
    def __init__(self, lrate = 1E-3, iter=2000):
        # Initialize class properties.
        self.W = np.random.randn(3)  # (X0, X1, B)
        self.mu = lrate
        self.iterations = iter
        self.score = 0

    def train(self, X, T):
        # Shuffle the training data
        N = X.shape[0]
        Shuffle = list(range(N))
        np.random.shuffle(Shuffle)
        X = X[Shuffle, :]
        T = T[Shuffle]

        # Train the perceptron using the learning rule.
        Output = np.zeros(N)
        Ones = np.ones((N, 1))
        # Pad the input vector with ones.  This is to facilitate handling of the bias.  (x0, x1, b)
        X1 = np.hstack((X, Ones))
        for j in range(self.iterations):
            for i in range(N):
                # Compute the output of the perceptron.
                Output[i] = 1 if np.dot(self.W, X1[i, :])> 0 else 0
                # Compare the output against the target.  Only apply the learning rule if there is a mistake.
                if Output[i] != T[i]:
                    self.W = self.W + self.mu*(T[i] - Output[i]) * X1[i, :]
        # Compute the training score from the number of mistakes encountered from the last training cycle.
        self.score = 1 - np.count_nonzero(T-Output)/N
        return self.score

    def predict(self, X):
        # Given a data set, predict the output of the perceptron in each case.
        N = X.shape[0]
        Output = np.zeros(N)
        Ones = np.ones((N, 1))
        X1 = np.hstack((X, Ones))
        for i in range(N):
            Output[i] = 1 if np.dot(self.W, X1[i, :]) > 0 else 0
        return Output


if __name__ == '__main__':
    # region Load the iris dataset from scikit.  Separate data for the setosa and versicolor irises.
    Dataset = load_iris()
    Filter = np.hstack((np.where(Dataset.target==0), np.where(Dataset.target==1)))[0]
    Training_Data = Dataset.data[Filter, :2]
    Target = Dataset.target[Filter]
    # endregion
    # region Plot the data for each flower using different colors.
    plt.figure(figsize=[7,6])
    plt.style.use('seaborn-dark')
    plt.scatter(Training_Data[np.where(Target==0), 0], Training_Data[np.where(Target==0), 1],
                color='b', marker='*', label='Setosa Iris')
    plt.scatter(Training_Data[np.where(Target==1), 0], Training_Data[np.where(Target==1), 1],
                color='g', marker='v', label='Versicolor Iris')
    plt.xlabel(Dataset.feature_names[0])
    plt.ylabel(Dataset.feature_names[1])
    # endregion
    # region Train the perceptron using the data and the target classification.
    P = Percept(lrate=50E-1, iter=3000)
    Score = P.train(Training_Data, Target)
    # endregion
    # region Predict flower classifications for the training data.
    Result = P.predict(Training_Data)
    # endregion
    # region Plot the decision line and save the figure.
    # Recall that W*X + b = 0.  Therefore the decision line can be found from:
    #
    #                  w0*x + w1*y + b = 0
    #                  w1*y = -w0*x + b
    #                  y = -w0/w1*x - b/w1
    #
    x = np.linspace(np.min(Training_Data[:, 0]), np.max(Training_Data[:, 0]), 5)
    y = -P.W[0]/P.W[1] * x - P.W[2]/P.W[1]
    plt.plot(x, y, 'r-.', linewidth=2)
    P.W = P.W/np.max(P.W)
    Classifier_String = '%.2fx%+.2fy%+.2f > 0\nAccuracy: %6.2f%%'%(P.W[0], P.W[1], P.W[2], Score*100)
    plt.text(6.3, 2.0, Classifier_String)
    plt.title('Iris Sepal Data')
    plt.legend(loc=2)
    plt.axis('tight')
    plt.savefig('Data/Iris_Sepal_Data.png')
    plt.show()
    # endregion
