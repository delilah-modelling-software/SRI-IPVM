#
#
#

import numpy as np
# import model.agency.defs as SMAD

#
# Switches
#
VIEW_ENABLED                 = True
OMNISCIENCE_ENABLED          = False
VEERING_ENABLED              = True
YIELDING_ENABLED             = True
PRUDENT_ACCELERATION         = True
STABILISED_YIELDING          = True
REFLEXIVE_REPULSION_ENABLED  = True

#
# Iterative timestep
#

# T_step        = 0.125
T_step        = 0.0625
# T_step        = 0.03125

#
# Temporal horizons:
#   SRCA
#   MRCA
#   LRCA
#
T_SRCA_horizon   = 0.5
T_MRCA_horizon   = 2.
T_LRCA_horizon   = 6.
T_master_horizon = T_LRCA_horizon

#
# Estimates of perceived safe distances for contrazone and interagent interactions.
#
d_safe_CZ         = 0.5
d_safe_IA_before  = 2.
# d_safe_IA_beside  = 0.5
d_safe_IA_beside  = 1.
d_safe_IA_behind  = 1.
d_safe_IA_maximum = d_safe_IA_before

#
# @todo: Replace this with a meaningful value.
#

# d_reflex_IA = 0.5
T_reflex_IA = 0.5

#
# Important angles and their cosines and sines:
#   theta_IAI_concern   Angular extent of driver's concern for pedestrians to
#                       their front.
#
theta_IAI_concern = 0.25 * np.pi
cos_IAI_concern   = np.cos(theta_IAI_concern)
theta_QP          = (1/18) * np.pi
cos_IAI_QP        = np.cos(theta_QP)

#
# Maximum rate of change of heading.
#
omega_max_PTN = 0.25 * np.pi
omega_max_CAR = 0.25 * np.pi

#
# The wheelbase is an approximation of the length of a Toyota Camry, and the
# maximum comfortable centrifugal acceleration is taken from Anvari+ [ABSO15].
#
L_wheelbase = 2.8
a_max_cfgl  = 3.4


#
# Car-Following Model parameters
#

# CFM_v_max          = 1.39 # 5 km/h = 25/18 m/s, probably safest.
# CFM_v_max          = 2.78 # 10 km/h = 25/9 m/s, probably even safer.
# CFM_v_max          = 4.17 # 15 km/h = 75/18 m/s, probably even safer.
# CFM_v_max          = 5.56 # 20 km/h = 50/9 m/s, probably safer.
# CFM_v_max          = 6.94 # 25 km/h = 125/18 m/s, probably marginally safer.
# CFM_v_max          = 8.33 # 30 km/h = 25/3 m/s, slightly slower than Anvari (2015), good enough for now.
CFM_v_max     = 4.17

CFM_T_headway = 1.5 # 1.7 for a truck!
CFM_s_safe    = 2.
CFM_noise_Q   = 0.2
CFM_noise_SD  = np.sqrt(CFM_noise_Q) # Assuming noise ~ N(0, SD)

#
# IDM-specific CFM parameters
#
IDM_a         = 1.
IDM_b         = 1.5 # 2.
IDM_braking_factor = 0.5 / np.sqrt(IDM_a * IDM_b)

#
# GFM-specific CFM parameters
#
GFM_D_brk = 98.78
GFM_D_acc =  5.59
GFM_T_brk =  0.74
GFM_T_acc =  0.77

