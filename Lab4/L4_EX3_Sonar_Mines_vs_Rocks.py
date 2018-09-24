"""LABORATORY 4
EXERCISE #3: Using a sklearn's perceptron model, process sonar data for rock and mine data to distinguish between them.
    (a) Use pandas to read all data into an array and randomly shuffle the data.
    (b) Use the 80/20 rule. Put 80% of the raw data into two data vectors: Sonar and Classification (input and label).
        Use the remaining 20% of the data for testing the classifier.
    (c) Train sklearn's Perceptron() unit that will adjust over a maximum of 1000 iterations with learning rate 1E-5.
    (d) Determine what the training accuracy is using sklearn's accuracy_score() function.
    (d) Assume the 20% test data has just been read by the sonar.  Predict the classification using the perceptron and
        determine its classification accuracy.  Use sklearn's accuracy_score() function to do this.
    (e) From the perceptron, extract the weight vector as well as the perceptron bias.
    (f) Implement a function that will perform a classification based on an input and the extracted weights.
    (g) Construct a pandas table (dataframe) that will hold the following columns: test data, classification from the
        function-implemented perceptron and classification from sklean's perceptron classifier.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

def Perceptron_Classifier(X, W, B):
    Weighted_Sum = np.dot(W,X)+B
    return 'R' if Weighted_Sum>0 else 'M'

# region Load data.  Shuffle the data and split it into training and test data using the 80/20 rule.
Data = np.array(pd.read_csv(r'Data/sonar.all-data').values)
np.random.shuffle(Data)
NData80 = int(np.round(Data.shape[0] * 0.80))
Training_Data = Data[:NData80, :]
Testing_Data = Data[:-NData80, :]
# endregion
# region Set up a single perceptron for training on the appropriate sonar data. Determine the training accuracy.
Sonar = np.array(Training_Data[:, :-1])
Classification = np.array(Training_Data[:, -1])
# endregion
# region Set up perceptron and train it.
detector = Perceptron(max_iter=1000, eta0=0.00001, random_state=None, n_jobs=-1)
detector.fit(Sonar, Classification)
# endregion
# region Test the classifier and determine the training accuracy based on the training labels.
Sensor_Reading = detector.predict(Sonar)
Training_Accuracy = accuracy_score(Classification, Sensor_Reading)
print('Training Accuracy: %4.1f%%'%(Training_Accuracy*100))
# endregion
# region Determine the performance of the perceptron on the test data.
Test_Sonar = Testing_Data[:, :-1]
Test_Classification = Testing_Data[:, -1]
Test_Sensor_Reading = detector.predict(Test_Sonar)
Testing_Accuracy = accuracy_score(Test_Classification, Test_Sensor_Reading)
print('Testing Accuracy: %4.1f%%'%(Testing_Accuracy*100))
# endregion
# region Extract the perceptron weights and display them.
Weights = detector.coef_
Bias = detector.intercept_
# endregion
# region Store results in an Excel sheet comparing the built in perceptron against the perceptron we made for the
# entire dataset (100% of the data).
Column1 = 'Custom Perceptron'
Column2 = 'Scikit Perceptron'
df = pd.DataFrame(columns=['Ground Truth', Column1, Column2])

for i in range(Data.shape[0]):
    Sonar_Data = np.matrix(Data[i, :-1]).transpose()
    # Prepare the data rows.
    Custom_Perceptron_Result = Perceptron_Classifier(Sonar_Data, Weights, Bias)
    Scikit_Perceptron_Result = detector.predict(Sonar_Data.transpose())[0]
    Ground_Truth = Data[i,-1]

    # Check the output of the perceptrons against the ground truth.  Store a result in the database row.
    Custom_Comparison = 'Correct' if Custom_Perceptron_Result==Ground_Truth else 'Mistake'
    Scikit_Comparison = 'Correct' if Scikit_Perceptron_Result==Ground_Truth else 'Mistake'

    # Store the rows in the appropriate column.
    df = df.append({'Ground Truth': Ground_Truth,
                    Column1: Custom_Comparison,
                    Column2: Scikit_Comparison
                    }, ignore_index=True)
print(df)
writer = pd.ExcelWriter('Perceptron_Comparison.xlsx')
df.to_excel(writer, 'Sheet1', index=False)
writer.save()
# endregion
