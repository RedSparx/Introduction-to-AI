"""LABORATORY 4
EXERCISE #1: Train an ADALINE unit to separate data into two classes.  Use the learned decision line to classify new data.
    (a) Synthesize up a labelled 2D dataset with 250 points: (x,y,c) data separated into two target classes c = {+1, -1}
    (b) Plot data points for each class using the labels to distinguish points with color (in a subplot).
    (c) Set up 500 training cycles for an ADALINE with a learning rate of 1E-6.
    (d) Print the equation of the class separation line.  Optional: Normalize the equation.
    (e) Plot the class separation line using the weights determined from training (in a subplot).
    (f) Use the ADALINE to classify each point.  Superimpose the classification labels over the class separation line.
    (g) Compute the accuracy of the classifier.
"""

import numpy as np
import matplotlib.pyplot as plt
# region Synthesize up a labelled 2D dataset : (x,y,c) data separated into two target classes c = {+1, -1}
N = 250
x = np.hstack((-2 + 3.5*np.random.randn(int(N/2)), 3 + 2.5*np.random.randn(int(N/2))))
y = np.hstack((-2 + 2.5*np.random.randn(int(N/2)), 5 + 2.5*np.random.randn(int(N/2))))
b = np.ones(N)
# Combine the above data into a single convenient data vector and target vector.
X = np.vstack((b, x, y)).transpose()  # DATA VECTOR: X = (b, x, y)
t = np.hstack((-1*np.ones(int(N/2)), +1*np.ones(int(N/2))))
# endregion
# region Plot the data points for each class.
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.scatter(x[np.where(t==+1)], y[np.where(t==+1)], color='b', alpha=0.5, marker='o')
plt.scatter(x[np.where(t==-1)], y[np.where(t==-1)], color='g', alpha=0.5, marker='*')
axis_limits = [-15,15,-15,15]
plt.title('Original Training Data')
plt.axis(axis_limits)
# endregion
# region Train an ADALINE unit.
mu = 1E-6
w = np.random.randn(3)
Training_Epochs = 2000
Error = np.zeros(Training_Epochs)
for i in range(Training_Epochs):
    for n in range(N):
        e = (np.dot(w, X[n, :])) - t[n]
        Error[i] = e
        w += mu*e*X[n, :]
w /=w.max() # OPTIONAL: normalizes the separation equation.
# endregion
# region Print the equation and display the class separation line in a plot.
x_line = np.linspace(x.min(), x.max(), 100)
a = -(w[1]/w[2])
b = -w[0]/w[2]
y_line = a* x_line + b
print('y = %4.2fx%+4.2f'%(a,b))
plt.subplot(1,2,2)
plt.plot(x_line,y_line, 'g--', linewidth=1.5)
plt.axis(axis_limits)
# endregion
# region Display the classification of each point.
Classify = lambda x,w: +1 if np.dot(x,w) >= 0 else -1
perceptron_output = np.ones_like(t)*np.nan
for i in range(N):
    perceptron_output[i] = Classify(X[i],w)
    Class_Assignment = 'A' if perceptron_output[i]==+1 else 'B'
    Target_Assignment = 'A' if t[i]==+1 else 'B'
    # Display the classification in RED if incorrect, BLACK otherwise.
    if perceptron_output[i]!=t[i]:
        plt.text(x[i], y[i], Target_Assignment, color='r', horizontalalignment='center', verticalalignment='center')
    else:
        plt.text(x[i], y[i], Class_Assignment, color='k', horizontalalignment='center', verticalalignment='center')
plt.axis(axis_limits)
# endregion
# region Determine the classifier accuracy: The difference between the output label and target label.
missclassifications = np.count_nonzero(np.abs(perceptron_output-t))
accuracy = (N-missclassifications) / N * 100
plt.title('ADALINE Output (%4.2f%% accuracy)'%accuracy)
# endregion
# region Display the figure and save it to file as well.
plt.tight_layout()
plt.savefig(r'Data\ADALINE_Output.png')
plt.show()
# endregion
