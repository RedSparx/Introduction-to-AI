import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# TODO: Complete the animation function.

class Percept:
    def __init__(self, lrate = 1E-3, iter=1000):
        # Initialize class properties.
        self.W = 10*np.random.randn(3)  # (X0, X1, B)
        # self.W = np.zeros(3)
        self.mu = lrate
        self.iterations = iter
        self.score = 0

        self.fig = plt.figure()
        self.ax = plt.axes()
        self.line, = self.ax.plot([],[])

    def train(self, X, T, showtraining=False):
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

        # plt.plot(x_line, y_line, 'r')

        # Compute the training score from the number of mistakes encountered from the last training cycle.
        self.score = 1 - np.count_nonzero(T-Output)/N
        return self.score

    def animate_decision_line(self):
        # Plot the decision line (x, y, b)
        NPoints = 2
        # x_line = np.linspace(np.min(X[:,0]), np.max(X[:,0]), NPoints)
        x_line = np.linspace(-3, +3, NPoints)
        a = -(self.W[0] / self.W[1])
        b = -(self.W[2] / self.W[1])
        y_line = a * x_line + b
        eq_str = '%.6fx%+.6fy%+.6f > 0' % (self.W[0], self.W[1], self.W[2])
        plt.plot(x_line, y_line, 'k', alpha=0.05)
        # print(eq_str)

    def predict(self, X):
        # Given a data set, predict the output of the perceptron in each case.
        N = X.shape[0]
        Output = np.zeros(N)
        Ones = np.ones((N, 1))
        X1 = np.hstack((X, Ones))
        for i in range(N):
            Output[i] = 1 if np.dot(self.W, X1[i, :]) > 0 else 0
        return Output

    def init_plot(self):
        self.line.set_data([],[])


    def plot_decision_line(self, i):
        x_line = np.linspace(x.min(), x.max(), 100)
        a = -(self.W[1] / self.W[2])
        b = -self.W[0] / self.W[2]
        y_line = a * x_line + b
        # print('y = %.6fx%+.6f' % (a, b))
        dline, =  plt.plot(x_line, y_line, 'r--', linewidth=1.5)
        return dline,

if __name__ =='__main__':
    pcpt = Percept(lrate=1E-3, iter=5000)
    fig = plt.figure(figsize=(5, 5))

    # region Synthesize up a labelled 2D dataset : (x,y,c) data separated into two target classes c = {+1, -1}
    N = 250
    v = 0.5
    x = np.hstack((-np.random.randn() + v * np.random.randn(int(N / 2)), +np.random.randn() + v * np.random.randn(int(N / 2))))
    y = np.hstack((-np.random.randn() + v * np.random.randn(int(N / 2)), +np.random.randn() + v * np.random.randn(int(N / 2))))
    # Combine the above data into a single convenient data vector and target vector.
    X = np.vstack((x, y)).transpose()  # DATA VECTOR: X = (b, x, y)
    t = np.hstack((0 * np.ones(int(N / 2)), +1 * np.ones(int(N / 2))))
    # endregion
    # region Plot the data points for each class.

    plt.scatter(x[np.where(t == 1)], y[np.where(t == 1)], color='b', alpha=0.5, marker='o')
    plt.scatter(x[np.where(t == 0)], y[np.where(t == 0)], color='g', alpha=0.5, marker='*')
    axis_limits = [-3, 3, -3, 3]
    plt.title('Decision Boundary Movement: Perceptron Training')

    # endregion
    # region Train a perceptron.
    anim = animation.FuncAnimation(fig, pcpt.plot_decision_line, init_func=pcpt.init_plot, 100, repeat=False, blit=True)
    pcpt.train(X, t)
    # print(pcpt.predict(X))

    # endregion
    # region Print the equation and display the class separation line in a plot.
    # x_line = np.linspace(x.min(), x.max(), 100)
    # a = -(w[1] / w[2])
    # b = -w[0] / w[2]
    # y_line = a * x_line + b
    # print('y = %4.2fx%+4.2f' % (a, b))
    # plt.subplot(1, 2, 2)
    # plt.plot(x_line, y_line, 'g--', linewidth=1.5)
    plt.axis(axis_limits)
    # endregion
    # region Display the classification of each point.
    # Classify = lambda x, w: +1 if np.dot(x, w) >= 0 else -1
    # perceptron_output = np.ones_like(t) * np.nan
    # for i in range(N):
    #     perceptron_output[i] = Classify(X[i], w)
    #     Class_Assignment = 'A' if perceptron_output[i] == +1 else 'B'
    #     Target_Assignment = 'A' if t[i] == +1 else 'B'
    #     # Display the classification in RED if incorrect, BLACK otherwise.
    #     if perceptron_output[i] != t[i]:
    #         plt.text(x[i], y[i], Target_Assignment, color='r', horizontalalignment='center', verticalalignment='center')
    #     else:
    #         plt.text(x[i], y[i], Class_Assignment, color='k', horizontalalignment='center', verticalalignment='center')
    # plt.axis(axis_limits)
    # endregion
    # region Determine the classifier accuracy: The difference between the output label and target label.
    # missclassifications = np.count_nonzero(np.abs(perceptron_output - t))
    # accuracy = (N - missclassifications) / N * 100
    # plt.title('ADALINE Output (%4.2f%% accuracy)' % accuracy)
    # endregion
    # region Display the figure and save it to file as well.
    # plt.axis('equal')
    plt.axis(axis_limits)
    plt.tight_layout()
    # plt.savefig(r'Data\ADALINE_Output.png')
    plt.show()
    # endregion