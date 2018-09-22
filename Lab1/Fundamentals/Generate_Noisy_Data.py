import numpy as np
import string

N_Data=[10, 100, 500, 1500, 5000, 10000]
N_Dimensions=[1,2,3]
Datasets=list(string.ascii_uppercase)

for D in N_Dimensions:
    for N in N_Data:
        for Data in Datasets:
            Uniform_Data = np.random.rand(N,3)
            Normal_Data = np.random.randn(N,3)
            np.savetxt(r'..\Data\DUniform_%1dD_%04d%1s.dat' % (D, N, Data), Uniform_Data, delimiter=',')
            np.savetxt(r'..\Data\DNormal_%1dD_%04d%1s.dat' % (D, N, Data), Uniform_Data, delimiter=',')


