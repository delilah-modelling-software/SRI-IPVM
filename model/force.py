#
# Force
#
# @todo: 
#

import numpy as np

import model.agency.defs as smad
import model.numerical.defs as smnd

# T_step = 0.125 # 0.5s

force_defn         = "AnvariEtAl_2015"
enable_variability = False
# mollification      = True

#
# F[a, b](d) = A[a, b] exp(-d / D[a, b])
#
A_mu    = np.zeros((6, 6))
A_sigma = np.zeros((6, 6))

D_mu    = np.zeros((6, 6))
D_sigma = np.zeros((6, 6))

lambda_mu    = np.zeros((6, 6))
lambda_sigma = np.zeros((6, 6))

if force_defn == "HelbingMolnar_1995":
    A_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 2.1
    D_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 0.3

    lambda_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 0.5
elif force_defn == "HelbingJohansson_2013_Circular":
    A_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 0.42
    A_sigma[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 0.26

    D_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 1.65
    D_sigma[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 1.01

    lambda_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 0.12
    lambda_sigma[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 0.07
elif force_defn == "HelbingJohansson_2013_Elliptical":
    A_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 0.04
    A_sigma[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 0.01

    D_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 3.22
    D_sigma[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 0.67

    lambda_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN]    = 0.06
    lambda_sigma[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 0.04
elif force_defn == "AnvariEtAl_2015":
    A_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 0.7
    D_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 2.25

    A_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.CAR] = 5.
    D_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.CAR] = 3.

    A_mu[smad.AgentType.CAR, smad.AgentType.PEDESTRIAN] = 6.
    D_mu[smad.AgentType.CAR, smad.AgentType.PEDESTRIAN] = 5.

    A_mu[smad.AgentType.CAR, smad.AgentType.CAR] =  8.
    D_mu[smad.AgentType.CAR, smad.AgentType.CAR] = 12.

    if enable_variability:
        A_sigma[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 0.25
        D_sigma[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 1.
    
        A_sigma[smad.AgentType.PEDESTRIAN, smad.AgentType.CAR] = 1.
        D_sigma[smad.AgentType.PEDESTRIAN, smad.AgentType.CAR] = 1.

        A_sigma[smad.AgentType.CAR, smad.AgentType.PEDESTRIAN] = 1.
        D_sigma[smad.AgentType.CAR, smad.AgentType.PEDESTRIAN] = 1.

        A_sigma[smad.AgentType.CAR, smad.AgentType.CAR] = 1.
        D_sigma[smad.AgentType.CAR, smad.AgentType.CAR] = 1.

    #
    # @todo: lambda_03 is a personal hypothesis and lambda_30 is fairly callous, so it would be good to find better estimates (preferably empirical).
    #   Perhaps Schoenauer et al., Anvari et al., or Johora & Mueller provide better answers? 
    #
    # lambda_mu = np.zeros((6, 6))
    lambda_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.PEDESTRIAN] = 0.5
    lambda_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.CAR] = 0.5
    lambda_mu[smad.AgentType.CAR, smad.AgentType.PEDESTRIAN] = 0.2
    lambda_mu[smad.AgentType.CAR, smad.AgentType.CAR] = 0.2

#
# d = min distance between centre[-line]s - sum(radii | halfstrokewidths)
# A, D are properties of the repulsor
# Note: Passing 1/D instead may allow reduction of #divisions
#
def calc_repulsion(d, A, D): # distance, amplitude, characteristic distance
    return A * np.exp(-d / D)

#
# d = min distance between centre[-line]s - sum(radii | halfstrokewidths)
# t = time since pedestrian (or driver) perceived attractor
# A, D, T are properties of the attractor
# Note: Passing 1/D, 1/T instead may allow reduction of #divisions
#
def calc_attraction(d, t, A, D, T): # distance, time, amplitude, characteristic distance, characteristic duration
    #
    # @todo: Example: callers should consider making T a random function of the pedestrians or pedestrian+attractor concerned.
    #
    # @todo: Is attraction exponentially decaying in both space and time? Or should the below formula be more complicated?
    # @todo: Consider more complicated formulae, e.g. a generic separable formula f(d) * g(t)
    #       hence, def calc_attraction(d, t, f, g): return f(d) * g(t)
    # @todo: Consider truly generic formulae, e.g f(d, t)
    #
    return A * np.exp(-(d / D + t / T))

def calc_interagent_repulsion(d, type1, type2):
    return calc_repulsion(d, A_mu[type1, type2], D_mu[type1, type2])

#
# The following two functions were defined in F. Johansson et al. (2015):
#   PV, Preferred Velocity; and
#   PP, Preferred Position.
# I am not currently using that paper's third waiting method:
#   APP (Adaptive Preferred Position).
# Because APP can change the agent's preferred position and velocity, a
# significantly more complex design would be required to take advantage of it.
#

def calc_PV_waiting(x_focal, x_alpha, v_alpha, spd_pref, d_focal, T_rlx):
    I_wait = -v_alpha / T_rlx
    return I_wait

def calc_PP_waiting(x_focal, x_alpha, v_alpha, spd_pref, d_focal, T_rlx):
    s = x_focal - x_alpha
    if smnd.calc_norm_sqr(s) <= d_focal**2:
        v_pref = (spd_pref / d_focal) * s
    else:
        v_pref = spd_pref * smnd.calc_unit_vector(s)

    I_wait = (v_pref - v_alpha) / T_rlx
    return I_wait

