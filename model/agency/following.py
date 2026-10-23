#
# Agent and two subclasses, Pedestrian and Vehicle
#
# @todo: Move some of the Vehicle functionality into a Car subclass
# @todo: Add other vehicle subclasses, e.g. Bicycle 
#

import numpy as np
# import scipy as sp
# import scipy.stats as ss

# from collections import deque

# import model.defs as smd

import model.numerical.defs as smnd
import model.configuration as smc


#
# Generalised Force Model (GFM) [Helbing+Tilch 1998]
#
def follow_GFM(s, v, v_des, dv, a_lead):
    s_des = smc.s_safe + smc.T_headway * max(v, dv)
    E_acc = np.exp((s - s_des) / smc.GFM_D_acc)

    v_tmp = (1 - E_acc) * v_des
    a = (v_tmp - v_des) / smc.GFM_T_acc

    if dv > 0:
        E_brk = np.exp((s - s_des) / smc.GFM_D_brk)
        a -= E_brk * dv / smc.GFM_T_brk

    return a
    
#
# Intelligent Driver Model [Treiber, Hennecke, Helbing 2000]
#
def follow_IDM(s, v, v_des, dv, a_lead):
    a_lead = None
    
    s_des = smc.s_safe + v * smc.T_headway + v * dv / 2 * np.sqrt(smc.a * smc.b)

    factor = 1 - (v / v_des)**4 - (s_des / s)**2

    return factor * smc.a

#
# Improved IDM
#
def follow_IIDM(s, v, v_des, dv, a_lead):
    a_lead = None
    
    s_des = smc.CFM_s_safe + v * smc.CFM_T_headway
    if s**2 <= smnd.EPS_m_sqr:
        print("Vehicle::calc_intervehicular_influence_IIDM - s_star {}, s {}".format(s_des, s))
        # assert(False)
    z = s_des / s

    if v <= v_des and v_des**2 < smnd.EPS_m_sqr:
        a_IIDM = 0
    elif v <= v_des:
        w = v / v_des
        if z >= 1.:
            a_IIDM = (1 - z**2) * smc.IDM_a
        else:
            a_free = (1 - w**4) * smc.IDM_a
            if a_free**2 > smnd.EPS_m_sqr:
                a_IIDM = a_free * (1 - (z**2)**(smc.IDM_a / a_free))
            else:
                a_IIDM = 0
    else:
        w = v_des / v
        b_free = (1 - w**(4 * smc.IDM_a / smc.IDM_b)) * smc.b
        if z >= 1.:
            a_IIDM = (1 - z**2) * smc.IDM_a - b_free
        else:
            a_IIDM = -b_free

    return a_IIDM

#
# Constant Acceleration Heuristic
#
def follow_CAH(s, v, v_des, dv, a_lead):
    a_tilde = min(a_lead, smc.IDM_a)
    v_lead = v - dv
    
    if v_lead * dv < -2 * s * a_tilde:
        a_CAH = v**2 * a_tilde / (v_lead**2 - 2 * s * a_tilde)
    elif dv > 0:
        a_CAH = a_tilde - dv**2 / 2 * s
    else:
        a_CAH = a_tilde
        
    return a_CAH

#
# Enhanced IDM for Adaptive Cruise Control [Kesting, Treiber, Helbing 2010]
#
def follow_EIDM(s, v, v_des, dv, a_lead):
    # assert(False)
    a_CAH  = follow_CAH(s, v, v_des, dv, a_lead)
    a_IIDM = follow_IIDM(s, v, v_des, dv, a_lead)
    coolness_factor = 0.99
    if a_IIDM >= a_CAH:
        return a_IIDM
    else:
        a_tmp    = a_CAH + smc.IDM_b * np.tanh((a_IIDM - a_CAH) / smc.IDM_b)
        return (1 - coolness_factor) * a_IIDM + coolness_factor * a_tmp

#
# Full Velocity Difference Model (FVDM) [Jiang+ 2014, PLoS One], with
# modifications [Treiber+Kesting 2017, ISTT]
#
def follow_FVDM(s, v, v_des, dv, a_lead):
    beta  = 0.1
    gamma = 0.52

    v_lead = v - dv
    
    v_safe = (s - smc.CFM_s_safe) / smc.CFM_T_headway
    v_opt  = max(0, min(v_des, v_safe))

    return beta * (v_opt - v) + gamma * (v_lead - v)
    



