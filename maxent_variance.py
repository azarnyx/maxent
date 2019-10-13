import numpy as np
from scipy.optimize import minimize
import pylab as plt


def fun(p):
    s = 0
    for pi in p:
        if pi<1e-10:
            continue
        # as we need to minimize, no minus
        s += np.log(pi)*pi
    return s


if __name__=='__main__':
    num_sides = 10
    mean = 4.
    sigma = 1.

    ini_guess = np.array([1/num_sides]*num_sides)

    # probability sum is one
    cons0 = {'type':'eq', 'fun': lambda x: np.ones(num_sides).dot(x) - 1. }

    # all probabilities are positive
    bnds = tuple([(0, None)]*(num_sides))

    # known variance
    A = (np.arange(1, 1+num_sides)-mean)**2
    cons1 = {'type':'eq', 'fun': lambda x: A.dot(x) - sigma**2 }

    sol = minimize(fun, ini_guess, bounds=bnds,
                   constraints=[cons0, cons1], options={'maxiter':201})

    # plot
    plt.bar(np.arange(1, 1+num_sides), sol.x)
    x = np.linspace(1,num_sides, num_sides*10)
    gauss = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mean)**2/sigma**2/2.)
    plt.plot(x, gauss, c='C1')
    plt.show()
