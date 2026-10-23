#
# shared-spaces/model/numerical/defs.py
#
# @todo: 
#

import numpy as np

def advance_Euler(dt, x, v, a, a_new = None):
    x_new = x + v * dt
    v_new = v + a * dt
    return x_new, v_new

#
# Midpoint (Modified Euler)
#
def advance_Midpoint(dt, x, v, a):
    assert(a_new != None)

    half_dt = 0.5 * dt

    v_mid = v + a * half_dt
    a_mid = 0.5 * (a + a_new)

    x_new = x + v_mid * dt
    v_new = v + a_mid * dt
    return x_new, v_new

#
# Semi-Implicit Euler
#
def advance_SIE(dt, x, v, a, a_new = None):
    v_new = v + a * dt
    x_new = x + v_new * dt
    return x_new, v_new

def advance_Leapfrog(dt, x, v, a, a_new):
    assert(a_new != None)
    x_new = x + v * dt + 0.5 * a * dt**2
    v_new = v + 0.5 * (a + a_new) * dt
    return x_new, v_new

    
