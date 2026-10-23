import copy as cp
import numpy as np
# import scipy as sp
# import scipy.stats as ss
import scipy.optimize as so

import model.agency.defs as smad
import model.agency.state as smas
import model.logger as sml
import model.numerical.defs as smnd
import model.geometry.shape as smgs
# import model.defs as smd
import model.configuration as smc
# import view.defs as svd
# import view.view as svv
# import model.topology.graph as smtg
# import model.topology.approachnode as smtApp
# import model.topology.activitynode as smtAct
# import model.topology.topology as smtt
# import model.geometry.grid as smgg
# import model.agency.agent as smaa
# import model.zone.zone as smzz
# import model.interaction.codec as smic
# import model.interaction.interaction as smii
# import model.force as smf

#
# In the absence of better criteria, yield to a vehicle coming from the right
# side. Used in LHT roundabouts, its asymmetry could be useful to break ties
# more symmetric criteria cannot resolve.
#
def prioritise_by_yielding_to_right(agent, ctragent):
    e_alpha_perp = smnd.rotate_90ACW(agent.dirn)
    if np.dot(e_alpha_perp, ctragent.posn - agent.posn) <= 0:
        return 0
    
    return 1

#
# In the absence of better criteria, yield to a vehicle coming from the right
# side. Used in RHT roundabouts, its asymmetry could be useful to break ties
# more symmetric criteria cannot resolve.
#
def prioritise_by_yielding_to_left(agent, ctragent):
    e_alpha_perp = smnd.rotate_90ACW(agent.dirn)
    if np.dot(e_alpha_perp, ctragent.posn - agent.posn) >= 0:
        return 0
    
    return 1

#
# Since the software is to be tested with Australian data, yield to right.
# The software is expected to also work with British, Japanese, and Thai
# traffic data, but would need to be changed over to test with European or
# North American data
#
def prioritise_by_lateral_position(agent, ctragent):
    return prioritise_by_yielding_to_right(agent, ctragent)

def prioritise_by_longitudinal_position(agent, ctragent):
    relposn = ctragent.posn - agent.posn
    if np.dot(agent.dirn, relposn) >= 0:
        return 1

    return 0
    
def prioritise_by_precedence(t0, t1):
    if t0 <= t1:
        return 0

    return 1

def compute_crossing_interval(y, v, L_cross):
    t_cross  = -y / v
    dt_cross = np.abs(L_cross / v)
    
    return (t_cross - dt_cross, t_cross + dt_cross)

def temporal_overlap_exists(T_cross_alpha, T_cross_beta):
    if T_cross_alpha[0] > T_cross_beta[1] or T_cross_alpha[1] < T_cross_beta[0]:
        return False

    if T_cross_alpha[0] > smc.T_master_horizon and T_cross_beta[0] > smc.T_master_horizon:
        return False

    if T_cross_alpha[1] <= 0 and T_cross_beta[1] <= 0:
        return False

    return True

def compute_displacement_time(agent, L):
    spd_pred = 0.5 * (np.dot(agent.v,    agent.dirn)    + agent.get_target_speed())

    return L / spd_pred

def compute_arrival_times(agent, ctragent, L_alpha, L_beta):
    t_arrival_alpha = \
        compute_displacement_time(agent,    L_alpha)
    t_arrival_beta  = \
        compute_displacement_time(ctragent, L_beta)

    return t_arrival_alpha, t_arrival_beta

def compute_passing_interval(agent, ctragent):
    l_sum   = agent.get_length_axis() + ctragent.get_length_axis()

    relposn = ctragent.posn - agent.posn
    relv    = ctragent.v    - agent.v

    x = np.dot(relposn, agent.dirn)
    u = np.dot(relv,    agent.dirn)

    if u**2 < smnd.EPS_m_sqr:
        if x**2 > l_sum**2:
            return None
        else:
            return (0, smc.T_master_horizon)

    dt_pass = np.abs(l_sum / u)
    
    t_pass = -x / u
    t_pass_min = t_pass - dt_pass
    t_pass_max = t_pass + dt_pass
    
    if t_pass_min > smc.T_master_horizon:
        return None

    if t_pass_max <= 0:
        return None

    return (t_pass_min, t_pass_max)
