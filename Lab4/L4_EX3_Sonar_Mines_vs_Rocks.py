import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

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

detector = Perceptron(max_iter=1000, eta0=0.00001, random_state=None, n_jobs=-1)
detector.fit(Sonar, Classification)

Sensor_Reading = detector.predict(Sonar)
Training_Accuracy = accuracy_score(Classification, Sensor_Reading)
print('Training Accuracy: %4.1f%%'%(Training_Accuracy*100))
# endregion
# region Determine the performance of the perceptron on test data.
Test_Sonar = Testing_Data[:, :-1]
Test_Classification = Testing_Data[:, -1]

Test_Sensor_Reading = detector.predict(Test_Sonar)
Testing_Accuracy = accuracy_score(Test_Classification, Test_Sensor_Reading)
print('Testing Accuracy: %4.1f%%'%(Testing_Accuracy*100))
# endregion
print(detector.coef_)
