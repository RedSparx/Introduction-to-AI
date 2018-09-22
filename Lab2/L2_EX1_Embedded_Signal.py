"""LABORATORY 2
EXERCISE #1: Find a signal embedded in noise.
    (a) Read data from a file and process the stream in frames.
    (b) Using a sliding frame, compute a closeness score between the frame and a stored signal.
    (c) Determine the index within the data set that contains the start of the signal.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# region Read in the CSV file.  Pandas will store the dataframe as a multidimensional array. It must be converted to 1D.
Signal = pd.read_csv(r'Data\Signal_02000.csv').values
Data = pd.read_csv(r'Data\Signal_Z_EMBED_20000_S02000.csv').values
Signal = Signal.ravel()
Data = Data.ravel()
# endregion
# region Compute the correlation between the signal and the data.  The resulting vector must be flipped.
Correlation = np.correlate(Signal, Data, mode='same')[::-1]
Max_Index = np.argmax(Correlation)
# endregion
# region Plot the raw data with the target signal highlighted.  Plot the correlation on another graph.
plt.subplot(2,2,2)
plt.plot(Data)
plt.plot(range(int(Max_Index-int(len(Signal)/2)), int(Max_Index+int(len(Signal)/2))),
    Data[(Max_Index-int(len(Signal)/2)):(Max_Index+int(len(Signal)/2))],'g')
plt.xlim([0, len(Data)])
plt.title('Data with Highlighted Signal')

plt.subplot(2,2,4)
plt.plot(Correlation)
plt.xlim([0, len(Data)])
plt.title('Correlation function.')

plt.subplot(2,2,(1,3))
plt.plot(Signal)
plt.title('Embedded Signal')
plt.show()
# endregion
