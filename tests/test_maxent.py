import pytest
import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.pardir,'')))
import maxent_sysindep, maxent_mean, maxent_variance

LI = [1e-6, 10., 1, -1, 2, -2, 0]

def test_f_sysindep(regtest):
    for i in LI:
        val = round(maxent_sysindep.f(i), 6)
        regtest.write(str(val)+' ')


def test_f_mean(regtest):
    arr = np.array(LI)
    val = maxent_mean.fun(arr)
    regtest.write(str(val)+' ')


def test_f_variance(regtest):
    arr = np.array(LI)
    val = maxent_variance.fun(arr)
    regtest.write(str(val)+' ')


def test_sum_form_sysindep(regtest):
    arr = np.array(LI)
    val = maxent_sysindep.sum_form(arr)
    regtest.write(str(val)+' ')


def test_cons_joint_sysindep(regtest):
    lookup_map = {0: (1.0, 1.0),
                  1: (1.0, 2.0),
                  2: (2.0, 1.0),
                  3: (2.0, 2.0),
                  4: (3.0, 1.0),
                  5: (3.0, 2.0)}

    p1 = [0.5, 0.5]
    p2 = [0.33333333, 0.33333333, 0.33333333]
    regtest.write(str(maxent_sysindep.cons20_joint(p1, lookup_map)))
    regtest.write(' ')
    regtest.write(str(maxent_sysindep.cons21_joint(p2, lookup_map)))
