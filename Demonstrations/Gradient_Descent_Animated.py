"""Gradient Descent Animation on the Linear Regression Problem"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from pylab import polyfit, polyval

global anim_coeffs

def sgdfit(x, y, mu = 1E-4, max_iter=5000):
    """Fit_SGD: Stochasic Gradient Descent"""
    N = len(x)
    order = 1
    coeffs = np.random.randn(order + 1)
    training_epoch = 0
    while training_epoch<=max_iter:
        for i in range(N):
            Err = y[i]-polyval(coeffs, x[i])
            coeffs = coeffs + mu*Err*np.array([x[i], 1])
        training_epoch+=1
    return coeffs

def animate_line(i, x, y, order=1, mu = 1E-4):
    global anim_coeffs
    N = len(x)
    plt.title('Iteration %d\n%s' % (i, Eqn_Str3(anim_coeffs)))
    for i in range(N):
        Err = y[i]-polyval(anim_coeffs, x[i])
        # anim_coeffs = anim_coeffs + mu*Err*np.array([x[i]**2, x[i], 1])
        X_Vect = np.fliplr(np.polynomial.polynomial.polyvander(x[i], order))[0]
        anim_coeffs = anim_coeffs + mu * Err * X_Vect\
                      #/(np.linalg.norm(X_Vect)**2)
        # print(anim_coeffs, np.fliplr(np.polynomial.polynomial.polyvander(x[i], 2)))
    line.set_xdata(x)
    line.set_ydata(polyval(anim_coeffs, x))
    return line,

def init():
    global anim_coeffs
    anim_coeffs = 0.5 * np.random.randn(Poly_Order + 1)
    return line,


if __name__ == '__main__':

    Poly_Order=3
    anim_coeffs = 5*np.random.randn(Poly_Order+1)
    # anim_coeffs = np.zeros(Poly_Order+1)

    # region Generate noisy linear data.
    N = 500
    a = 1
    b = -2
    n = 0.25*np.random.randn(N)
    x = np.linspace(-5, 5, N)
    y = (a*x +b) + n
    # endregion

    # region For reference, use built-in function to perform linear fit (using direct least-squares solution).  Make plots.
    fit_coeffs = polyfit(x, y, 1)
    # sgd_coeffs = sgdfit(x, y)
    Eqn_Str1 = lambda coefficients: '$y = %.2f x%+.2f$' % (coefficients[0], coefficients[1])
    Eqn_Str2 = lambda coefficients: '$y = %.2f x^2%+.2f x%+.2f$' % (coefficients[0], coefficients[1], coefficients[2])
    Eqn_Str3 = lambda coefficients: '$y = %.2f x^3%+.2f x^2%+.2f x%+.2f$' % (coefficients[0], coefficients[1], coefficients[2], coefficients[3])

    fig = plt.figure(figsize=[9, 7])
    # plt.axis([-5,5,-5,5])
    ax = plt.gca()
    ax.scatter(x, y, alpha=0.5, color='g', marker='*')
    ax.plot(x, polyval(fit_coeffs, x), color='k', linewidth=3, label='Linear Fit (MSE)\n%s' % Eqn_Str1(fit_coeffs), alpha=0.6)
    # line, = ax.plot(x, polyval(sgd_coeffs, x), color='b', linewidth=2, label ='Stochasic Gradient Descent\n%s'%Eqn_Str(sgd_coeffs))
    line, = ax.plot([], [], color='b', linewidth=2, label='Cubic Fit (SGD)')
    ax.legend(loc=4)

    Animation_Frames=1001
    ani = animation.FuncAnimation(fig, animate_line, range(Animation_Frames),
                                  fargs=( x, y, Poly_Order, 50E-6),
                                  interval=20, repeat=False, blit=False)
    plt.show()
    # TODO: Write code to save the animation to a file.
    print('*** END ***')
    # endregion

    # region Plot the dat

