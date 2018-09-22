""" Generate simulated sensor data with a randomly place signal embedded. The signal is non-trivial and embedded in a
random location.  Several files are generated so that there is some variance in the location for testing.  The noise is
of additive Gaussian type.
"""

import numpy as np
import matplotlib.pyplot as plt
import string

N=20000
Noise_Parameters={'Mean': 0.0, 'Variance': 0.8}
Noise = Noise_Parameters['Variance'] * np.random.randn(N) + Noise_Parameters['Mean']

# Compute the signal with a specific signal length.
Signal_Length=int(0.10 * N)
x = np.linspace(0, 8 * np.pi, Signal_Length)
Signal = np.sin(x)+0.5*np.sin(4*x)
plt.subplot(2,2,(1,3))
plt.plot(x,Signal)
plt.title('Embedded Signal (%d Samples)'%len(Signal))


Datasets=list(string.ascii_uppercase)
Signal_Filename='Signal_%05d.csv'%Signal_Length
np.savetxt(Signal_Filename, Signal)
for D in Datasets:
    Punctuated_Signal_Filename = 'Signal_%1s_PUNCT_%05d_S%05d.csv'%(D,N,Signal_Length)
    Embedded_Signal_Filename = 'Signal_%1s_EMBED_%05d_S%05d.csv'%(D,N,Signal_Length)

    # In a random location, embed the signal.
    Location = np.random.randint(0, N - Signal_Length)
    Noise_Punctured_Signal = np.copy(Noise)
    Noise_Punctured_Signal[Location:(Location + Signal_Length)] = Signal
    np.savetxt(Punctuated_Signal_Filename, Noise_Punctured_Signal, delimiter=',')

    # In the same random location, embed the signal directly within the noise.
    Embedded_Signal=np.copy(Noise)
    Embedded_Signal[Location:(Location + Signal_Length)] += Signal
    np.savetxt(Embedded_Signal_Filename, Embedded_Signal, delimiter=',')
