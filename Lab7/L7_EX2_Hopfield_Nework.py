import pyfiglet
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import pickle


class hopfield:
    def __init__(self, biased=False):
        self.W = np.array([])
        self.B = np.array([])
        self.Energy = np.array([])
        self.History = np.array([])
        self.Capacity = np.nan
        self.Biased = biased

    def fit(self, X):
        if X.ndim > 1:
            N, D = X.shape

        else:
            D = X.shape[0]
            N = 1

        if self.Biased:
            X = np.hstack((X, np.ones((N, 1))))
            self.W = np.zeros((D+1, D+1))
        else:
            self.W = np.zeros((D, D))

        # self.Capacity = D/2*np.log(N)
        for i in range(N):
            if N==1:
                self.W += np.outer(X, X)
            else:
                self.W += np.outer(X[i, :], X[i, :])
            np.fill_diagonal(self.W, 0)
        self.W /= N
        if self.Biased:
            self.B = self.W[:-1, -1]
            self.W = self.W[:-1, :-1]
            return self.W, self.B
        else:
            return self.W

    def predict(self, X, asynchronous=False, verbose=False):
        if X.ndim > 1:
            N, D = X.shape
            raise ValueError('Only one input vector can be predicted at a time.  You provided %d.'%N)
        else:
            D = X.shape[0]
            N = 1

        Y = X
        if asynchronous:
            # Asynchronous updating of all neurons.
            self.History = np.empty((0, 0))
            self.Energy = np.empty((0, 0))
            Bin2Str = lambda A: ''.join([str(int((a + 1) / 2)) for a in A])

            Y_old = np.nan*np.ones_like(Y)
            while np.all(Y != Y_old):
                # Activate each neuron one at a time, asynchronously.
                idx=np.array(list(range(D)))
                # Randomize neuron update.
                np.random.shuffle(idx)
                for i in idx:
                    if self.Biased:
                        Y[i] = +1 if (np.dot(W[i], Y) + self.B[i]) >= 0 else -1
                    else:
                        Y[i] = +1 if np.dot(W[i], Y) >= 0 else -1
                    E = -0.5 * np.dot(np.dot(W, Y), Y)
                    if verbose:
                        print('Neuron %-2d: %s       E=%6.2f' % (i, Bin2Str(Y), E))
                    if self.History.size == 0:
                        self.History = Y
                        self.Energy = E
                    else:
                        self.History = np.vstack((Y, self.History))
                        self.Energy = np.append(self.Energy, E)
                Y_old = Y
                print('x')
            return Y
        else:
            self.History = np.nan
            if N == 1:
                # Synchronous updating of all neurons (single input vector).
                Y = np.array([
                    +1 if s >= 0 else -1
                    for s in np.dot(self.W, X)+self.B])
                self.Energy = -0.5 * np.dot(np.dot(W, Y), Y)
            else:
                # Synchronous updating of all neurons (multiple input vectors).
                Y = np.zeros_like(X)
                self.Energy = np.zeros_like(X)
                for i in range(N):
                    if self.Biased:
                        Y[i] = np.array([
                            +1 if s >= 0 else -1
                            for s in np.dot(self.W, X[i])+self.B])
                    else:
                        Y[i] = np.array([
                            +1 if s >= 0 else -1
                            for s in np.dot(self.W, X[i])])
                    self.Energy[i] = -0.5 * np.dot(np.dot(W, Y[i]), Y[i])
            return Y

def ShowChar(i):
    plt.imshow(Vec2Img(YTest), cmap='gray')


if __name__ == '__main__':
    f = open(r'Data\Hopfield_Test_Data.pkl', 'rb')
    Data = pickle.load(f)
    f.close()
    Pr = lambda s: print(pyfiglet.figlet_format(s, font='big'))
    Pr('Hopfield Network')

    Bin2Str = lambda A: ''.join([str(int((a+1)/2)) for a in A])

    # region Setup and train the Hopfield network.
    H = hopfield()
    H2 = hopfield()
    H3 = hopfield(biased=True)
    Bipolar = lambda vect: 2*vect-1

    # Try patterns: (1)ABCXYZ, (2)ABCDEF, (3)JKLXYZ (4)JKLSTU
    Char1 = Bipolar(Data['X']).flatten()
    Char2 = Bipolar(Data['T']).flatten()
    Char3 = Bipolar(Data['Z']).flatten()
    Char4 = Bipolar(Data['X']).flatten()
    Char5 = Bipolar(Data['Y']).flatten()
    Char6 = Bipolar(Data['Z']).flatten()

    Chars = 'PYTHON'
    X = np.array([Bipolar(Data[c]).flatten() for c in Chars])

    # X = np.vstack((Char1, Char2, Char3, Char4, Char5, Char6))
    # X = np.vstack((Char1, Char2))
    # X = Char1
    # W = H.fit(X)
    # W = H2.fit(X)

    W, B = H3.fit(X)
    Test = H3.predict(X, asynchronous=True, verbose=True)
    Vec2Img = lambda v: (v.reshape((7, 7)) + 1) / 2
    plt.imshow(Vec2Img(Test))
    plt.show()



    # # TestVect = np.ones_like(X[0, :])
    # TestVect = np.sign(np.random.randn(len(X[0, :])))
    # print('Initial:   %s'%Bin2Str(TestVect))
    #
    # YTest = H.predict(TestVect, asynchronous=True)
    # YTest2 = H2.predict(TestVect, asynchronous=False)
    #
    # print('Asynchronous Energy: %6.2f'%H.Energy[-1])
    # print('Synchronous Energy:  %6.2f'%H2.Energy)
    #
    # Vec2Img = lambda v: (v.reshape((7, 7)) + 1) / 2
    #
    # print(Vec2Img(YTest))
    #
    # plt.imshow(Vec2Img(YTest))
    # plt.show()

    # for i in range(48, 0, -1):
    #     plt.subplot(7, 7, i+1)
    #     plt.imshow(Vec2Img(YTest[i, :]), cmap='gray')
    #     plt.axis('off')
    # plt.figure()
    # plt.plot(np.flipud(ETest))
    # plt.title('Hopfield Network Energy')
    # plt.show()
    # endregion

