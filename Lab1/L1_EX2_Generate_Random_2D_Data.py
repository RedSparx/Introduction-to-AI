'''LABORATORY 1
EXERCISE #2: Synthesizing Random 1D Test Data.
    (a) Write a function to generate 1D uniformly distributed data and plot a histogram with labeled axes.
    (b) Write a function to generate 1D normally distributed data and plot a histogram with labeled axes.
    (c) Simulate a noisy sine wave signal(uniform and Gaussian noise).
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Number of data points to generate.
N = 2000
Histogram_Bins = 500
# region Generate 1D random uniform data.  Let N be the number of data points to generate.
Uniform_Noise = np.random.rand(N)
plt.subplot(2,2,1)
plt.hist(Uniform_Noise, bins=Histogram_Bins)
plt.title('Uniform Distribution: %d Bins.'%Histogram_Bins)
# endregion
# region Generate 1D normal uniform data.  Let N be the number of data points to generate.
Normal_Noise = np.random.randn(N)
plt.subplot(2,2,2)
plt.hist(Normal_Noise, bins=Histogram_Bins)
plt.title('Normal Distribution: %d Bins.'%Histogram_Bins)
# endregion
# region Simulate a sine wave with first uniform then normal noise data added.
x=np.linspace(0, 8*np.pi, N)
y1 = np.sin(x) + Uniform_Noise
y2 = np.sin(x) + Normal_Noise

plt.subplot(2,2,3)
plt.plot(x,y1)
plt.title('Sine Wave with Uniform Noise')

plt.subplot(2,2,4)
plt.plot(x,y2)
plt.title('Sine Wave with Gaussian Noise')
plt.show()
# endregion



