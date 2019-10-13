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


if __name__=="__main__":
    NUM_SIDES = 6 
    MEAN = 4.5

    ini_guess = np.array([1/NUM_SIDES]*NUM_SIDES)

    # probability sum is one
    cons0 = {'type':'eq', 'fun': lambda x: np.ones(NUM_SIDES).dot(x) - 1. }
    
    # all probabilities are positive
    # (not necessary for entropy, but might be helpful for other functions)
    bnds = tuple([(0, None)]*(NUM_SIDES))

    # known mean
    cons1 = {'type':'ineq', 'fun': lambda x: np.arange(1, 1+NUM_SIDES).dot(x)-MEAN}

    # equal to exponential in continuous case only
    # in discrete somewhat in between uniform and exponential
    plt.figure(figsize=(6.4, 2.4))
    sol = minimize(fun, ini_guess, bounds=bnds,
                   constraints=[cons0, cons1], options={'maxiter':1001})

    # plot
    plt.bar(np.arange(1, 1+NUM_SIDES), sol.x)
    plt.xlabel("Side of dice")
    plt.ylabel("Probability")
    plt.savefig("mean.png", format='png', dpi=250)
    plt.show()
