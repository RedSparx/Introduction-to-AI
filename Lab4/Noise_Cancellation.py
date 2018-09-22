import numpy as np
import matplotlib.pyplot as plt

# region Initialize the filter (of length k). Apply the filter to the input x and adjust its weights according to the
# error signal e (difference between the filter output and the desired output d).
#
#
#            ------------
#   x-----> |     h      |---------+------> y
#            ------------          |
#                 ^                |
#                 |  dw            |
#                 |                |
#            ------------          V
#           | LMS Update | <------(+)<----- d
#            ------------    e=d-y
#
#    x = Microphone input.
#    d = Desired output (original signal).
#    y = Filter output.
#
# The difference between the filter output and the desired output is the error e=d-y.

N=50000
time = np.linspace(0, 50, N)  # 0----->5*pi divides it up into N intervals.
music = np.sin(time)
noise = 0.3*np.random.randn(N)
noisy_music = music + noise

plt.subplot(1,2,1)
plt.plot(time, music, 'r', label='Music Signal')
plt.plot(time, noisy_music, label='Combined: Music + Noise', alpha = 0.5)
plt.legend()
plt.ylim([-5, 5])

# region Set up the Perceptron to work across a sliding window in the music.
window_size = 100
perceptron_output = np.zeros(N)
anti_noise = np.zeros(N)
w = np.random.randn(window_size)
mu = 5E-2
for n in range(N - window_size):
    n_window_end = n + window_size
    perceptron_output[n_window_end] = np.dot(w, noisy_music[n:n_window_end])
    Error = music[n_window_end] - perceptron_output[n_window_end]
    anti_noise[n_window_end] = Error
    eta = mu/(np.linalg.norm(noisy_music[n:n_window_end]) ** 2)
    w = w + eta * Error * noisy_music[n:n_window_end]

# Add the anti-noise to the original music.  This, combined signal will reduce
# the effect of new noise.
modified_music = anti_noise + music
combined_music = modified_music + perceptron_output
plt.subplot(1, 2, 2)
plt.plot(time, modified_music,'b', alpha = 0.2, label = 'Modified Music')
plt.plot(time, combined_music,'g', label = 'Combined: Music + Filter Output')
plt.ylim([-5, 5])
plt.legend()
plt.show()




# endregion


