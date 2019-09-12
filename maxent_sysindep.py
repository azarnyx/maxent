#!/usr/bin/python
import numpy as np
from scipy.optimize import minimize
import pylab as plt
import itertools


num_sides1 = 6
num_sides2 = 10
mean1 = 4.5
mean2 = mean1


def f(x):
    if x==0:
        return 0.0
    return  - x*np.log(x)


def sum_form(p):
    s = 0
    for pi in p:
        # as we will to minimize, add minus
        s += - f(pi)
    return s


# let us consider that we have two dices. there are two possibilities:
# consider them as separate systems or consider them as one system

########### SEPARATE SYSTEMS#############
# probabilities have to be one
cons01 = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1.})
cons02 = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1.})

# introduce constraints on mean values
sides1 = np.arange(0, num_sides1)+1.
sides2 = np.arange(0, num_sides2)+1.
cons21 = ({'type': 'eq', 'fun': lambda p: sides1.dot(p) - mean1})
cons22 = ({'type': 'eq', 'fun': lambda q: sides2.dot(q) - mean2})

# bounds. do not need for entropy, however might need for other
# functions
bnds1 = tuple([(0, None)]*(num_sides1))
bnds2 = tuple([(0, None)]*(num_sides2))

# introduce initial guesses
ini_guess1 = np.array([1/num_sides1]*num_sides1)
ini_guess2 = np.array([1/num_sides2]*num_sides2)


sol1 = minimize(sum_form, ini_guess1,
                bounds=bnds1, constraints=[cons01, cons21], options={'maxiter':1001})
sol2 = minimize(sum_form, ini_guess2,
                bounds=bnds2, constraints=[cons02, cons22], options={'maxiter':1001})


########### JOINT SYSTEMS #############
# we have [p11, p12, ..., pnm] as joint pdfs
# probability>0
bnds = tuple([(0, None)]*(num_sides2*num_sides1))

# remember number of probabilities
lookup_map = {}
for ni, piqj in enumerate(itertools.product(sides1, sides2)):
    lookup_map[ni] = piqj

# common guesses
guess_common = np.ones(len(lookup_map))/len(lookup_map)

# common constrain on sum to 1
ccons0 = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1.})

# constrains on mean
def cons20_joint(params):
    s = 0
    for ni, (pij) in enumerate(params):
        s += pij*lookup_map[ni][0] # conditions for p
    return s

def cons21_joint(params):
    s = 0
    for ni, (pij) in enumerate(params):
        s += pij*lookup_map[ni][1] # conditions for q
    return s

ccons20 = ({'type': 'eq', 'fun': lambda x: cons20_joint(x)-mean1})
ccons21 = ({'type': 'eq', 'fun': lambda x: cons21_joint(x)-mean2})


sol_common = minimize(sum_form,
                      guess_common,
                      bounds=bnds,
                      constraints=[ccons0,
                                   ccons20,
                                   ccons21],
                      options={'maxiter':10001})

########### EVALUATE #############
# check p1*p2=p12
sep_sol = [i*j for i,j in  itertools.product(sol1.x, sol2.x)]
mse = np.sqrt(np.mean((sol_common.x-sep_sol)**2))
print("Difference:", mse)


# integrate on x1 and x2 to derive marginal pdf and plot them
p1_c = np.zeros(num_sides1)
p2_c = np.zeros(num_sides2)
for j, i in zip(lookup_map.items(), sol_common.x):
    p1_c[int(j[1][0])-1]+=i
    p2_c[int(j[1][1])-1]+=i

fig, ax = plt.subplots(2,1)
ax[1].bar(sides2, sol2.x, label='separate, second dice',color='C3')
ax[1].bar(sides2, p2_c, alpha=0.7, label='joint, second dice',color='C0')
ax[1].legend()
ax[0].bar(sides1, sol1.x, label='separate, first dice',color='C3')
ax[0].bar(sides1, p1_c, alpha=0.7, label='joint, first dice',color='C0')
ax[1].set_xlabel("Sides of dices")
ax[1].set_ylabel("Probaability")
ax[0].set_ylabel("Probaability")
ax[0].legend()
plt.savefig("Subsystems.png", format='png', dpi=250)
plt.show()
