"""LABORATORY 4
EXERCISE #2: NLMS Noise Cancellation. Recover a signal that has been corrupted by noise using an adaptive filter.
Apply the filter to the input x and adjust its weights according to the error signal e (difference between the filter
output and the desired output d).

           ------------
  x-----> |     h      |---------+------> y
           ------------          |
                ^                |
                |  dw            |
                |                |
           ------------          V
          | LMS Update | <------(+)<----- d
           ------------    e=d-y

   x = Microphone input.
   d = Desired output (original signal).
   y = Filter output.

The difference between the filter output and the desired output is the error e=d-y.
    (a) Synthesize a pure sinusoid signal and corrupt it with noise (t={0..2pi}, f(x)= sin(5x).
    (b) Initialize the weight vector (w) for the filter (length k=500).
    (c) Initialize output and error arrays to hold these values for each shift of the filter through the data.
    (d) Recover the pure signal from the noisy data by adjusting filter weights adaptively using the NLMS learning rule.
        Store the output and error signals.
    (e) Plot the original signal, the learned noise signal, as well as the filter output.
"""
import numpy as np
import matplotlib.pyplot as plt
# region Synthesize a signal in the time domain (d) and then corrupt it with noise (n).
N = 10000
t = np.linspace(0, 2*np.pi, N)
d = np.sin(5*t)
n = np.random.normal(0, 0.5, N)
# n = 0.1*np.cos(3*t)**2
x = d + n
# endregion
# region Initialize the filter (of length k). Randomize the weights.
k = 500
mu = 1E-1
w = np.zeros(k)
# endregion
# region Initialize output and error arrays to hold these values for each shift of the filter through the data.
y = np.ones(N)*np.nan
Err = np.zeros(N)
# endregion
# region Recover the pure signal from the noisy data by adjusting filter weights adaptively using the NLMS learning rule.
# Store the output and error signals.
for n in range(N-k):
    y[n] = np.dot(w, x[n:(n + k)])
    # e = d[n:(n + k)] - y[n:(n + k)]
    e = d[n] - y[n]
    Err[n] = e
    eta = mu/(np.linalg.norm(x[n:(n+k)])**2)
    w = w + 2 * eta * e * x[n:(n+k)]
# endregion
# region Produce plots.
plt.subplot(1, 2, 1)
plt.plot(t, x, color='b', alpha=0.25, label='System Input')

plt.plot(t, d, color='g', linewidth=2, label='Desired Signal')
plt.title('Original Signal with Impairment')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, Err, 'k')
plt.title('Cancellation Signal (Learned Impairment)')

plt.subplot(2, 2, 4)
plt.plot(t, y, 'g')
plt.title('Impairment-Cancelled Signal')
plt.show()
# endregion