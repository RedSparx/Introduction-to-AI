"""LABORATORY 5
EXERCISE #2: Character recognition using a single layer of many perceptrons.
    (a) From sklearn, load the NIST handwritten dataset and display its associated description for the user.
    (b) Using 10 perceptrons from scikit and a "one-hot coding" scheme (1-of-10), train each to recognize one digit. Be
        sure to use the _entire_ dataset to train, not just data for a single digit.
    (c) Randomly select a image (of a digit) from the database.  Predict what it is.
    (d) Display the randomly selected image and the prediction of what digit it is.
    (e) Display a 10x10 image plot of some training data with the labeled randomly selected image.
    (f) Extract all of the perceptron weights and bias and store them in a text file with the bias in the last column.
    (g) Save the weights and bias in an Excel file with identifying columns (W0, W1, W2, ... W64, B).
"""

from sklearn.linear_model import Perceptron
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# region Load the NIST handwritten digit dataset and display its description
NIST_Database = load_digits()
print(NIST_Database['DESCR'])
print('The database has %d samples. Starting training sequence...'%len(NIST_Database.images))
# endregion
# region Using a one-hot coding scheme and the 10 perceptrons, train each unit to recognize a single digit.
lrate = 1E-5
imax = 1000
P = [Perceptron(max_iter=imax, eta0=lrate, random_state=None, n_jobs=-1) for p in range(10)]

# Flatten all images.  Store the indices identifying each image.
Database_Size = len(NIST_Database.images)
Flattened_Images = NIST_Database.images.reshape((Database_Size, -1))

# Train each perceptron individually.
for d in range(10):
    Target_Images = np.where(NIST_Database.target==d)
    Target_Output = np.zeros(Database_Size)
    Target_Output[Target_Images] = 1
    P[d].fit(Flattened_Images, Target_Output)
    Accuracy = P[d].score(Flattened_Images, Target_Output)
    print('Trained Perceptron %d... Accuracy = %4.2f%%' % (d, Accuracy*100))
# endregion
# region Randomly select a digit from the database.  Predict what it is.
Random_Image_Index = np.random.randint(Database_Size)
Random_Image = NIST_Database.images[Random_Image_Index]
Random_Image_Flat = Random_Image.reshape((1,64))
One_Hot_Output = np.array([P[i].predict(Random_Image_Flat)[0] for i in range(10)])
Predicted_Digits = list(np.nonzero(One_Hot_Output)[0])
# endregion
# region Display the training data in a 10x10 subplot for the first 100 training samples of the predicted image.  If
# there are more predictions, take the first one only for display purposes.
Sample_Digit = np.array(np.where(NIST_Database.target == Predicted_Digits[0]))[0]
splot = 1
plt.figure(figsize=(5,3))
for d in Sample_Digit[:100]:
    plt.subplot(10, 10, splot)
    splot += 1
    plt.imshow(NIST_Database.images[d], cmap='gray_r')
    plt.axis('off')
# endregion
# region Display the randomly selected image and the prediction for its label.
plt.subplot(1,2,2)
plt.imshow(NIST_Database.images[Random_Image_Index], cmap='gray_r')
plt.title('Digit Prediction: %s'%str(Predicted_Digits))
plt.axis('off')
plt.savefig('Data\Digit_%1d.png'%Predicted_Digits[0])
plt.show()
# endregion
# region Extract the perceptron weights and bias for each perceptron and store them in a text file. The last column
# should contain the bias.
Weights = np.vstack((P[d].coef_.flatten() for d in range(10)))
Bias = np.vstack(P[d].intercept_.flatten() for d in range(10))
Perceptron_Memory = np.hstack((Weights, Bias))
np.savetxt('Data\Digit_Classifier.dat', Perceptron_Memory, fmt='%6.4f')
# endregion
# region Save the weights to an Excel sheet.
df = pd.DataFrame()
df = df.append(Perceptron_Memory.tolist())
col_titles = ['W%1d'%i for i in range(64)]
col_titles.append('B')
df.columns = col_titles

writer = pd.ExcelWriter('Data\Digit_Classifier.xlsx')
df.to_excel(writer, 'Sheet1', index=False)
writer.save()
# endregion
