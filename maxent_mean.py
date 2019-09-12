#!/usr/bin/python

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import pylab as plt

def fun(p):
    s = 0
    for pi in p:
        if pi==0:
            continue
        # as we need to minimize, no minus
        s += np.log(pi)*pi
    return s

num_sides = 6 #100
mean = 4.5 #40.

ini_guess = np.array([1/num_sides]*num_sides)

# probability sum is one
cons0 = LinearConstraint(np.ones(num_sides), 1., 1.)

# all probabilities are positive (not necessary for fun, but might be helpful for other functions)
cons1 = LinearConstraint(np.identity(num_sides), np.zeros(num_sides), np.array([np.inf]*num_sides))

# known mean
cons2 = LinearConstraint(np.arange(1, 1+num_sides), mean, mean)

# equal to exponential in continuous case only
# in discrete somewhat in between uniform and exponential
plt.figure(figsize=(6.4,2.4))
sol = minimize(fun, ini_guess, constraints=[cons0, cons2], options={'maxiter':1001})
plt.bar(np.arange(1, 1+num_sides), sol.x)
# x = np.linspace(1, num_sides, num_sides*10)
# plt.plot(x, 1/(mean) * np.exp(-x/mean),c='C1')
plt.xlabel("Side of dice")
plt.ylabel("Probability")
plt.savefig("mean.png", format='png', dpi=250)
plt.show()
