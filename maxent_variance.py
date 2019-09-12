#!/usr/bin/python

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import pylab as plt

def fun(p):
    s = 0
    for pi in p:
        # as we need to minimize, no minus
        s += np.log(pi)*pi
    return s

num_sides = 10
mean = 4.
sigma = 1.

ini_guess = np.array([1/num_sides]*num_sides)

# probability sum is one
cons0 = LinearConstraint(np.ones(num_sides), 1., 1.)

# all probabilities are positive (not necessary for fun(p), but might be helpful for other functions)
cons1 = LinearConstraint(np.identity(num_sides), np.zeros(num_sides), np.array([np.inf]*num_sides))

# known variance
cons3 = LinearConstraint((np.arange(1, 1+num_sides)-mean)**2, sigma**2, sigma**2)

sol = minimize(fun, ini_guess, constraints=[cons0, cons3], options={'maxiter':201})
plt.bar(np.arange(1, 1+num_sides), sol.x)
x = np.linspace(1,num_sides, num_sides*10)
plt.plot(x, 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mean)**2/sigma**2/2.), c='C1')
plt.show()
