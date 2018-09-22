"""LABORATORY 3
EXERCISE #2: Given a data set, find the best fit model and make a prediction based on it.
    (a) Perform regression modeling of data against several models: (i)y=a*sin(x)+b,(ii)ln(y)=m*ln(x)+b,(iii)y=ax^3+b*x^2+c
    (b) Plot all three models with the data.
    (c) Compute residuals for each model.
    (d) Determine which of the models is the best fit for the given data.
    (e) Given a set of input values and the best fit model, predict the output.
"""


# import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import pandas as pd
import os

def regression_analysis(x,y, Picture_File):
    # region Perform regression fitting for each model.
    Model1 = polyfit(np.sin(x), y, 1)
    Model2 = polyfit(x, np.log(y), 1)
    Model3 = polyfit(x, y, 3)
    Model4 = polyfit(x, y, 1)
    # endregion
    # region Find the residuals from each model.
    Residual1 = y - (Model1[0] * np.sin(x) + Model1[1])
    Residual2 = y - (np.exp(Model2[1]) * np.exp(Model2[0] * x))
    Residual3 = y - (Model3[0] * x ** 3 + Model3[1] * x ** 2 + Model3[2] * x + Model3[3])
    Residual4 = y - (Model4[0] * x + Model4[1])
    # endregion
    # region For each model, find the sum of the squared-residuals.
    MSE1 = np.sum(Residual1 ** 2)
    MSE2 = np.sum(Residual2 ** 2)
    MSE3 = np.sum(Residual3 ** 2)
    MSE4 = np.sum(Residual4 ** 2)
    MSE = np.array([MSE1, MSE2, MSE3, MSE4])
    # endregion
    # region Select the best fit model.
    Model_Name = ['Sinusoidal', 'Exponential', 'Cubic', 'Linear']
    Best_Fit = argmin(MSE)
    print('Best Fit: %s' % Model_Name[Best_Fit])

    # endregion
    # region Plot the Data and the fitted models.
    fig=plt.figure()
    fig.set_size_inches(6,8)
    plt.subplot(2, 1, 1)
    plt.plot(x, y, '.k', label='Data', markersize=1)
    plt.plot(x, Model1[0] * np.sin(x) + Model1[1], 'b', label='Sinusoidal')
    plt.plot(x, np.exp(Model2[1]) * np.exp(Model2[0] * x), 'c', label='Exponential')
    plt.plot(x, Model3[0] * x ** 3 + Model3[1] * x ** 2 + Model3[2] * x + Model3[3], 'g', label='Cubic')
    plt.plot(x, Model4[0] * x + Model4[1], 'y', label='Linear')
    plt.title('Best Fit Model (Minimum $R^2$):%s '%Model_Name[Best_Fit])
    plt.legend(loc=2, fontsize=8)
    # endregion
    # region Plot the model residuals.
    plt.subplot(2, 1, 2)
    plt.plot(x, Residual1 ** 2, 'b', label='Sinusoidal(%3.2e)'%MSE1, linewidth=0.75)
    plt.plot(x, Residual2 ** 2, 'c', label='Exponential (%3.2e)'%MSE2, linewidth=0.75)
    plt.plot(x, Residual3 ** 2, 'g', label='Cubic (%3.2e)'%MSE3, linewidth=0.75)
    plt.plot(x, Residual4 ** 2, 'y', label='Linear (%3.2e)'%MSE4, linewidth=0.75)
    plt.title('Residuals ($R^2$)')
    plt.legend(loc=2, fontsize=8)
    # plt.ylim([-20, 20])
    # plt.show()
    print(Picture_File)

    fig.savefig(Picture_File)
    # endregion
    return Model_Name[Best_Fit]

def acquire_data(FName):
    # region Import data from file.
    try:
        # Read a CSV data file.  Transpose the data before unpacking them into the two required data vectors.
        x,y = pd.read_csv(FName).values.transpose()
    except IOError as e:
        print('The file %s does not exist.  Please check the name again.'%FName)
        exit()
    return x, y
    # endregion


if __name__=='__main__':
    Directory = 'Data'
    CSV_Files = os.listdir(Directory)
    Full_Path_CSV_Files = [Directory + "\\" + f  for f in CSV_Files]
    for File in Full_Path_CSV_Files:
        if File.endswith('.csv'):
            x, y = acquire_data(File)
            regression_analysis(x, y, File[:-4]+'.png')
