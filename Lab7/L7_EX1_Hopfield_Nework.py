import pyfiglet
import numpy as np
import matplotlib.pyplot as plt
import pickle


class hopfield:
    def __init__(self):
        self.W = 0

    def fit(self, X):
        N, D = X.shape
        self.W = np.zeros((D,D))
        for i in range(N):
            if N==1:
                self.W += np.outer(X, X)
            else:
                self.W += np.outer(X[i, :], X[i, :])
            np.fill_diagonal(self.W, 0)
        self.W /=N
        return self.W

    def predict(self, X):
        Y = np.array([
            +1 if s >= 0 else -1
            for s in np.dot(W, X)])
        return Y


if __name__ == '__main__':
    f = open(r'Data\Hopfield_Test_Data.pkl', 'rb')
    Data = pickle.load(f)
    f.close()
    Pr = lambda s: print(pyfiglet.figlet_format(s, font='big'))
    Pr('Synchronous Update')
    H = hopfield()
    Bipolar = lambda vect: 2*vect-1

    # Try patterns: (1)ABCXYZ, (2)ABCDEF, (3)JKLXYZ (4)JKLSTU
    Char1 = Bipolar(Data['P']).flatten()
    Char2 = Bipolar(Data['Y']).flatten()
    Char3 = Bipolar(Data['T']).flatten()
    Char4 = Bipolar(Data['H']).flatten()
    Char5 = Bipolar(Data['O']).flatten()
    Char6 = Bipolar(Data['N']).flatten()

    X = np.vstack((Char1, Char2, Char3, Char4, Char5, Char6))
    W = H.fit(X)

    plt.figure(figsize=[6, 10])
    plt.subplot(4, 6, (1, 12))
    plt.spy(W)

    for i in range(6):
        Test = X[i]
        Y = (H.predict(Test).reshape((7, 7))+1)/2

        plt.subplot(4, 6, 13+i)
        plt.imshow((Test.reshape((7, 7))+1)/2, 'Reds')
        plt.axis('equal')
        plt.axis('off')
        plt.title('Test %d'%(i+1))

        plt.subplot(4, 6, 19+i)
        plt.imshow(Y, cmap='binary')
        plt.axis('equal')
        plt.axis('off')
        plt.title('Output %d'%(i+1))

    plt.show()
