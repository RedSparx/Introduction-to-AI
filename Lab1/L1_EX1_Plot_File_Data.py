"""LABORATORY 1
EXERCISE #1: Working with Tabular Data in a File.
    (a) Read data from a file and store it in memory.
    (b) Determine the dimensions of the data.
    (c) Perform a computation on the data.
    (d) Plot data based on computation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# region (a) Read data from a file and store it in memory.
Filename=r'Data\DNormal_2D_0100W.dat'
try:
    # Attempt to read the given file and store the information in a variable.
    Data = pd.read_csv(Filename)
except IOError as e:
    # Report an error and exit the program.
    print('Error reading file %s'%Filename)
    print(e)
    exit()
# endregion
# region (b) Determine the dimensions of the data.
print('The file has %d rows and %d columns.' % Data.shape)
# endregion
# region (c) Perform a computation on the data.
# Find the distance of the point from the origin (Euclidean distance).
Dist=np.empty(Data.shape[0])
i=0
for V in Data.values:
    Dist[i] = np.sqrt(V[0]**2 + V[1**2])
    i+=1
# endregion
# region (d) Plot data based on computation.
plt.plot(Dist)
plt.title('Vector Distances from Origin: %s'%Filename)
plt.show()
# endregion
