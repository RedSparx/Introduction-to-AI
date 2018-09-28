"""LABORATORY 5
EXERCISE #1: Demonstrate limitations of perceptrons: Learning 2-bit logic (AND, OR, XOR).
    (a) Create arrays that will hold inputs and corresponding outputs for each gate: AND, OR, XOR (truth tables).
    (b) To each input, add 500 gaussian random variables (0.2 variance).  There should now be 2000 points.
    (c) Create a target vector containing the target value representing the output for each logic gate.
    (d) Set up one perceptron for each logic gate.  Train it for a maximum number of 5000 iterations and a learning rate
        of 1E-6.
    (e) Print the learned truth table.
    (f) Plot the decision lines for the perceptrons trained on AND, OR, and XOR.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# region Arrays to hold the logic truth tables.
Input_A = np.array([0, 1, 0, 1])
Input_B = np.array([0, 0, 1, 1])
AND = np.array([0, 0, 0, 1])
OR = np.array([0, 1, 1, 1])
XOR = np.array([0, 1, 1, 0])
# endregion
# region Add 500 random values to each of the inputs.  Combine these arrays into a single array.  Plot these points.
N=500
var = 0.15
Input_A_Random = np.array([i+var*np.random.randn(N) for i in Input_A]).flatten()
Input_B_Random = np.array([i+var*np.random.randn(N) for i in Input_B]).flatten()
Input_AB = np.vstack((Input_A_Random, Input_B_Random)).transpose()
plt.scatter(Input_A_Random, Input_B_Random, marker='*', alpha=0.1)
# endregion
# region Create an output target vector for the logic gates.
AND_Output = np.array([i*np.ones(N) for i in AND]).flatten()
OR_Output = np.array([i*np.ones(N) for i in OR]).flatten()
XOR_Output = np.array([i*np.ones(N) for i in XOR]).flatten()
# endregion D
# region Train perceptrons to learn the logic.  Display the training accuracy score.
lrate = 1E-6
imax = 5000
P1 = Perceptron(max_iter=imax, eta0=lrate, random_state=None)
P2 = Perceptron(max_iter=imax, eta0=lrate, random_state=None)
P3 = Perceptron(max_iter=imax, eta0=lrate, random_state=None)

P1.fit(Input_AB, AND_Output)
P2.fit(Input_AB, OR_Output)
P3.fit(Input_AB, XOR_Output)

print('Accuracy for AND Logic: %4.2f%%'%(P1.score(Input_AB, AND_Output)*100))
print('Accuracy for OR Logic: %4.2f%%'%(P2.score(Input_AB, OR_Output)*100))
print('Accuracy for XOR Logic: %4.2f%%'%(P3.score(Input_AB, XOR_Output)*100))
# endregion
# region Print the learned truth table.
Test_Input = [[0, 0], [0, 1], [1, 0], [1, 1]]

print('\n\n A     B    AND   OR    XOR')
print('----------------------------')

for i in range(4):
    print(' %d     %d     %d     %d     %d'%(Test_Input[i][0], Test_Input[i][1],
                                        P1.predict([Test_Input[i]]),
                                        P2.predict([Test_Input[i]]),
                                        P3.predict([Test_Input[i]])))
# endregion
# region Plot decision line: Logical AND
x = np.linspace(-5, 5, 100)
w0 = P1.coef_[0,0]
w1 = P1.coef_[0,1]
b = P1.intercept_[0]
AND_Sep = lambda x: -w0/w1*x - b/w1
plt.plot(x, AND_Sep(x), 'r', label = 'AND')
# endregion
# region Plot decision line: Logical OR
w0 = P2.coef_[0,0]
w1 = P2.coef_[0,1]
b = P2.intercept_[0]
OR_Sep = lambda x: -w0/w1*x - b/w1
plt.plot(x, OR_Sep(x), 'g', label = 'OR')
# endregion
# region Plot decision line: Logical XOR
w0 = P3.coef_[0,0]
w1 = P3.coef_[0,1]
b = P3.intercept_[0]
XOR_Sep = lambda x: -w0/w1*x - b/w1
plt.plot(x, XOR_Sep(x), 'b--', label = 'XOR')

plt.axis([-0.75, 1.75, -0.75, 1.75])
plt.text(0, 0, '0 0', color = 'r', horizontalalignment='center', verticalalignment='center')
plt.text(0, 1, '0 1', color = 'r', horizontalalignment='center', verticalalignment='center')
plt.text(1, 0, '1 0', color = 'r', horizontalalignment='center', verticalalignment='center')
plt.text(1, 1, '1 1', color = 'r', horizontalalignment='center', verticalalignment='center')
plt.legend()
plt.show()
# endregion