"""LABORATORY 2
EXERCISE #2: Synthesize Random 2D Test Data.
    (a) Write a function that will generate a set of Gaussian random vectors with a a dictionary as input with
        parameters mean, variance, and size.
    (b) Generate several different sets of random vectors.
    (c) Make a scatter plot of the vectors.
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def rvect(parameters):
    # Return a data set based on 'parameters'.  This dictionary contains 'variance', 'mean', 'size', and 'dimension'.
    return parameters['variance']*np.random.randn(parameters['size'],parameters['dimension'])+parameters['mean']


def plot_data(dat):
    plt.scatter(dat[:,0],dat[:,1])


# region Define the synthesized data set parameters.
D = 2       # Dimension
N = 1000    # Number of data points in a data set.

Set1 = {'mean': [0,0], 'variance': [0.75, 0.5], 'dimension':D, 'size': N}
Set2 = {'mean': [-3,5], 'variance': [0.5, 0.75], 'dimension':D, 'size': N}
Set3 = {'mean': [5,2], 'variance': [0.5, 0.5], 'dimension':D, 'size': N}
Set4 = {'mean': [-4,-1], 'variance': [0.75, 0.5], 'dimension':D, 'size': N}
Set5 = {'mean': [2,4], 'variance': [0.75, 0.75], 'dimension':D, 'size': N}
Set6 = {'mean': [-2,2], 'variance': [0.5, 0.5], 'dimension':D, 'size': N}

# PCA Data Set
Set7 = {'mean': [0,0], 'variance': [0.5, 0.5], 'dimension':D, 'size': N}
Set8 = {'mean': [2,2], 'variance': [0.5, 0.75], 'dimension':D, 'size': N}
Set9 = {'mean': [3,4], 'variance': [0.5, 0.5], 'dimension':D, 'size': N}
Set10 = {'mean': [-3,-1], 'variance': [0.5, 0.85], 'dimension':D, 'size': N}
Set11 = {'mean': [-3,-4], 'variance': [0.5, 0.35], 'dimension':D, 'size': N}
Set12 = {'mean': [5,5], 'variance': [0.5, 0.25], 'dimension':D, 'size': N}
# endregion
# region If the dimension is two, make a scatter plot.
if D==2:
    plt.subplot(1,2,1)
    plot_data(rvect(Set1))
    plot_data(rvect(Set2))
    plot_data(rvect(Set3))
    plot_data(rvect(Set4))
    plot_data(rvect(Set5))
    plot_data(rvect(Set6))
    plt.title('2D Data Set')
    plt.subplot(1,2,2)
    plot_data(rvect(Set7))
    plot_data(rvect(Set8))
    plot_data(rvect(Set9))
    plot_data(rvect(Set10))
    plot_data(rvect(Set11))
    plot_data(rvect(Set12))
    plt.title('2D Data Set for PCA Analysis')
    plt.show()
# endregion
# region Concatenate all independent data sets so that they are mixed.  Save this data set.
Data=np.vstack([rvect(Set1),
               rvect(Set2),
               rvect(Set3),
               rvect(Set4),
               rvect(Set5),
               rvect(Set6)])
Data_PCA=np.vstack([rvect(Set7),
               rvect(Set8),
               rvect(Set9),
               rvect(Set10),
               rvect(Set11),
               rvect(Set12)])
np.savetxt(r'Data\DAT%05d_%1dD_%1dCLASSES.csv'%(Data.shape[0],D,6), Data, delimiter=',')
np.savetxt(r'Data\DAT%05d_%1dD_%1dCLASSES_PCA.csv'%(Data_PCA.shape[0],D,6), Data_PCA, delimiter=',')
# endregion

# region Kernel density plot with histograms. Optional.
sns.set(style='darkgrid')
D=pd.DataFrame(Data_PCA, columns=['x', 'y'])
sns.jointplot(x='x', y='y', data=D, kind='kde')
# plt.title('No Clear Decision Boundaries in Projection Axes')
plt.show()
# endregion
