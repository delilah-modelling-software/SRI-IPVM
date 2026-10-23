import sys
import numpy as np
import scipy as sp
# import scipy.stats as ss

import view.defs as SVD
import view.view as SVV
import model.numerical.defs as SMND
# import model.force as SMF
import model.distributor as SMDis
import model.topology.node as SMTN
import model.topology.centroid as SMTC
import model.topology.topology as SMTT
import model.activity as SMA
import model.topology.activitynode as SMTAct
import model.topology.approachnode as SMTApp
import model.agency.defs as SMAD
import model.model as SMM
import model.geometry.shape as SMGS
# import model.zone.defs as SMZD
import model.zone.zone as SMZZ
import model.logger as SML
# import model.agency.agent as smaa

width  = 80
height = 80

scale_factor   = 10
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width,screen_height)

# layer_flags = SVV.ViewLayerFlagset.GEOMETRY | SVV.ViewLayerFlagset.TOPOLOGY | SVV.ViewLayerFlagset.MACROSCOPIC | SVV.ViewLayerFlagset.MICROSCOPIC
# layer_flags = SVV.ViewLayerFlagset.GEOMETRY | SVV.ViewLayerFlagset.TOPOLOGY | SVV.ViewLayerFlagset.MICROSCOPIC | SVV.ViewLayerFlagset.ATTRACTION
layer_flags = SVD.LayerFlagset.GEOMETRY | SVD.LayerFlagset.TOPOLOGY | SVD.LayerFlagset.MICROSCOPIC | SVD.LayerFlagset.ATTRACTION | SVD.LayerFlagset.INTERACTION | SVD.LayerFlagset.CONES | SVD.LayerFlagset.TRAJECTORY
view = SVV.View(width, height, screen_size, scale_factor, layer_flags)
# view = SVV.View(width, height, screen_size, scale_factor, pedestrian_upscale_factor = 1.)

centre_true = np.array([width / 2, height / 2])

corner_NE = np.array([width - 5, height - 5])
corner_NW = np.array([  5, height - 5])
corner_SW = np.array([  5,   5])
corner_SE = np.array([width - 5,   5])

centre_E = np.array([width - 5, centre_true[1]])
centre_N = np.array([centre_true[0], height - 3])
centre_W = np.array([  5, centre_true[1]])
centre_S = np.array([centre_true[0],   3])

# T_step       = 0.5 # 0.125    # Duration of each simulation time-step
# T_reaction   = T_step
T_simulation = 15 * 60  # 15 minutes
L_simulation = int(T_simulation / SMND.T_step)

spawner_uniform = SMDis.DistributorUniform()

unit_E = np.array([1, 0])
unit_N = np.array([0, 1])

ptngate_width     = 3.
ptngate_halfwidth = 0.5 * ptngate_width
ptngate_halfdepth = 0.125

cargate_width     = 4.
cargate_halfwidth = 0.5 * cargate_width
cargate_halfdepth = 0.25

# carspawner = SMDis.DistributorUniform()

safety_margin = 0.5 # 1.
cargate_NS_displacement = (safety_margin + cargate_halfwidth) * unit_N
ptngate_NS_extradisplacement = (cargate_halfwidth + safety_margin + ptngate_width) * unit_N # s/ptngate_width/ptngate_halfwidth ?!
ptngate_NS_displacement = cargate_NS_displacement + ptngate_NS_extradisplacement

cargate_EW_displacement = (safety_margin + cargate_halfwidth) * unit_E
ptngate_EW_extradisplacement = (cargate_halfwidth + safety_margin + ptngate_width) * unit_E # s/ptngate_width/ptngate_halfwidth ?!
ptngate_EW_displacement = cargate_EW_displacement + ptngate_EW_extradisplacement


#
# Pedestrian gates:
# ESE? centre_E - ptngate_NS_displacement +- ptngate_halfwidth * unit_N
# ENE? centre_E + ptngate_NS_displacement +- ptngate_halfwidth * unit_N
# NNE? centre_N + ptngate_EW_displacement +- ptngate_halfwidth * unit_E
# NNW? centre_N - ptngate_EW_displacement +- ptngate_halfwidth * unit_E
# WNW? centre_W + ptngate_NS_displacement +- ptngate_halfwidth * unit_N
# WSW? centre_W - ptngate_NS_displacement +- ptngate_halfwidth * unit_N
#
# S? centre_S +- ptngate_halfwidth * unit_E
#

ptngate_NS_hwvec = ptngate_halfwidth * unit_N
ptngate_EW_hwvec = ptngate_halfwidth * unit_E

ptngate_ESE = SMTC.create_oblong_gate(0, centre_E - ptngate_NS_displacement, ptngate_NS_hwvec, ptngate_halfdepth, SVD.colour(  0, 128, 0), SVD.colour(  0,   0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)
ptngate_ENE = SMTC.create_oblong_gate(1, centre_E + ptngate_NS_displacement, ptngate_NS_hwvec, ptngate_halfdepth, SVD.colour(  0, 128, 0), SVD.colour(  0,   0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)
ptngate_NNE = SMTC.create_oblong_gate(2, centre_N + ptngate_EW_displacement, ptngate_EW_hwvec, ptngate_halfdepth, SVD.colour(  0, 128, 0), SVD.colour(  0,   0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)
ptngate_NNW = SMTC.create_oblong_gate(3, centre_N - ptngate_EW_displacement, ptngate_EW_hwvec, ptngate_halfdepth, SVD.colour(  0, 128, 0), SVD.colour(  0,   0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)
ptngate_WNW = SMTC.create_oblong_gate(4, centre_W + ptngate_NS_displacement, ptngate_NS_hwvec, ptngate_halfdepth, SVD.colour(  0, 128, 0), SVD.colour(  0,   0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)
ptngate_WSW = SMTC.create_oblong_gate(5, centre_W - ptngate_NS_displacement, ptngate_NS_hwvec, ptngate_halfdepth, SVD.colour(  0, 128, 0), SVD.colour(  0,   0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)

ptngate_S = SMTC.create_oblong_gate(6, centre_S, ptngate_EW_hwvec, ptngate_halfdepth, SVD.colour(  0, 128, 0), SVD.colour(  0,   0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)

#
# Vehicle gates:
# ESE? centre_E - cargate_NS_displacement +- cargate_halfwidth * unit_N
# ENE? centre_E + cargate_NS_displacement +- cargate_halfwidth * unit_N
# NNE? centre_N + cargate_EW_displacement +- cargate_halfwidth * unit_E
# NNW? centre_N - cargate_EW_displacement +- cargate_halfwidth * unit_E
# WNW? centre_W + cargate_NS_displacement +- cargate_halfwidth * unit_N
# WSW? centre_W - cargate_NS_displacement +- cargate_halfwidth * unit_N
#

cargate_NS_hwvec = cargate_halfwidth * unit_N
cargate_EW_hwvec = cargate_halfwidth * unit_E

cargate_ESE = SMTC.create_oblong_gate(7, centre_E - cargate_NS_displacement, cargate_NS_hwvec, cargate_halfdepth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
cargate_ENE = SMTC.create_oblong_gate(8, centre_E + cargate_NS_displacement, cargate_NS_hwvec, cargate_halfdepth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
cargate_NNE = SMTC.create_oblong_gate(9, centre_N + cargate_EW_displacement, cargate_EW_hwvec, cargate_halfdepth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
cargate_NNW = SMTC.create_oblong_gate(10, centre_N - cargate_EW_displacement, cargate_EW_hwvec, cargate_halfdepth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
cargate_WNW = SMTC.create_oblong_gate(11, centre_W + cargate_NS_displacement, cargate_NS_hwvec, cargate_halfdepth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
cargate_WSW = SMTC.create_oblong_gate(12, centre_W - cargate_NS_displacement, cargate_NS_hwvec, cargate_halfdepth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)

#
# Origin-destination pairs:
#   Note: Parking, U-turns, and /3-/5-/7-point turns are neglected
#
ptn_centroids        = [ptngate_ESE, ptngate_ENE, ptngate_NNE, ptngate_NNW, ptngate_WNW, ptngate_WSW, ptngate_S]
car_centroids        = [cargate_ESE, cargate_ENE, cargate_NNE, cargate_NNW, cargate_WNW, cargate_WSW]
pedestrian_OD_matrix = \
    18 * np.array(
    [[2, 5, 5, 5, 5, 5, 5],
     [5, 2, 5, 5, 5, 5, 5],
     [5, 5, 2, 5, 5, 5, 5],
     [5, 5, 5, 2, 5, 5, 5],
     [5, 5, 5, 5, 2, 5, 5],
     [5, 5, 5, 5, 5, 2, 5],
     [5, 5, 5, 5, 5, 5, 2]])
six_zeroes = [0, 0, 0, 0, 0, 0]
car_OD_matrix = \
    15 * np.array(
    [[0, 0, 0, 2, 0, 2],
     six_zeroes,
     [0, 2, 0, 0, 0, 2],
     six_zeroes,
     [0, 2, 0, 2, 0, 0],
     six_zeroes])

#
# Pedestrian waypoints:
# SE? Square(centre_true - ptngate_NS_displacement + ptngate_EW_displacement, waypoint_halfwidth)
# NE? Square(centre_true + ptngate_NS_displacement + ptngate_EW_displacement, waypoint_halfwidth)
# NW? Square(centre_true + ptngate_NS_displacement - ptngate_EW_displacement, waypoint_halfwidth)
# SW? Square(centre_true - ptngate_NS_displacement - ptngate_EW_displacement, waypoint_halfwidth)
#
# SC? Square(centre_true - ptngate_NS_displacement, waypoint_halfwidth)
#
ptnwypnt_SE_centre = centre_true - ptngate_NS_displacement + ptngate_EW_displacement
ptnwypnt_NE_centre = centre_true + ptngate_NS_displacement + ptngate_EW_displacement
ptnwypnt_NW_centre = centre_true + ptngate_NS_displacement - ptngate_EW_displacement
ptnwypnt_SW_centre = centre_true - ptngate_NS_displacement - ptngate_EW_displacement
ptnwypnt_S_centre  = centre_true - ptngate_NS_displacement

ptnwypnt_NE = SMTApp.create_oblong_waypoint(13, ptnwypnt_NE_centre, unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
ptnwypnt_NW = SMTApp.create_oblong_waypoint(14, ptnwypnt_NW_centre, unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
ptnwypnt_SW = SMTApp.create_oblong_waypoint(15, ptnwypnt_SW_centre, unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
ptnwypnt_SE = SMTApp.create_oblong_waypoint(16, ptnwypnt_SE_centre, unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
ptnwypnt_S  = SMTApp.create_oblong_waypoint(17, ptnwypnt_S_centre,  unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)

#
# Vehicle waypoints:
# ESE? centre_true + cargate_EW_displacement - cargate_NS_displacement +- cargate_halfwidth * unit_N
# ENE? centre_true + cargate_EW_displacement + cargate_NS_displacement +- cargate_halfwidth * unit_N
# NNE? centre_true + cargate_NS_displacement + cargate_EW_displacement +- cargate_halfwidth * unit_E
# NNW? centre_true + cargate_NS_displacement - cargate_EW_displacement +- cargate_halfwidth * unit_E
# WNW? centre_true - cargate_EW_displacement + cargate_NS_displacement +- cargate_halfwidth * unit_N
# WSW? centre_true + cargate_EW_displacement - cargate_NS_displacement +- cargate_halfwidth * unit_N
#
# N2W? centre_true + cargate_SE_displacement +- cargate_halfwidth * unit_NE
#   where:
#       unit_SE = rsqrt(2) * (unit_E - unit_N) 
#       cargate_SE_displacement = (cargate_halfwidth + safety_margin) * unit_SE
#       unit_NE = rsqrt(2) * (unit_N + unit_E)
#
mean_car_halflength  = 2.25
carwypnt_safe_offset = safety_margin + mean_car_halflength

carwypnt_ESE_B_centre = centre_true - cargate_NS_displacement + ptngate_EW_displacement + ptngate_EW_hwvec
carwypnt_ESE_A_centre = carwypnt_ESE_B_centre + carwypnt_safe_offset * unit_E

carwypnt_ENE_A_centre = centre_true + cargate_NS_displacement + ptngate_EW_displacement + ptngate_EW_hwvec
carwypnt_ENE_B_centre = carwypnt_ENE_A_centre + carwypnt_safe_offset * unit_E

carwypnt_NNE_B_centre = centre_true + ptngate_NS_displacement + cargate_EW_displacement + ptngate_NS_hwvec
carwypnt_NNE_A_centre = carwypnt_NNE_B_centre + carwypnt_safe_offset * unit_N

carwypnt_NNW_A_centre = centre_true + ptngate_NS_displacement - cargate_EW_displacement + ptngate_NS_hwvec
carwypnt_NNW_B_centre = carwypnt_NNW_A_centre + carwypnt_safe_offset * unit_N

carwypnt_WNW_B_centre = centre_true + cargate_NS_displacement - ptngate_EW_displacement - ptngate_EW_hwvec
carwypnt_WNW_A_centre = carwypnt_WNW_B_centre - carwypnt_safe_offset * unit_E

carwypnt_WSW_A_centre = centre_true - cargate_NS_displacement - ptngate_EW_displacement - ptngate_EW_hwvec
carwypnt_WSW_B_centre = carwypnt_WSW_A_centre - carwypnt_safe_offset * unit_E

carwypnt_ESE_A = SMTApp.create_flat_waypoint(18, carwypnt_ESE_A_centre, unit_N, cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)
carwypnt_ESE_B = SMTApp.create_round_waypoint(19, carwypnt_ESE_B_centre, unit_N, cargate_halfwidth, SMAD.AgentType.CAR)

carwypnt_ENE_A = SMTApp.create_round_waypoint(20, carwypnt_ENE_A_centre, unit_N, cargate_halfwidth, SMAD.AgentType.CAR)
carwypnt_ENE_B = SMTApp.create_flat_waypoint(21, carwypnt_ENE_B_centre, unit_N, cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)

carwypnt_NNE_A = SMTApp.create_flat_waypoint(22, carwypnt_NNE_A_centre, unit_E, cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)
carwypnt_NNE_B = SMTApp.create_round_waypoint(23, carwypnt_NNE_B_centre, unit_E, cargate_halfwidth, SMAD.AgentType.CAR)

carwypnt_NNW_A = SMTApp.create_round_waypoint(24, carwypnt_NNW_A_centre, unit_E, cargate_halfwidth, SMAD.AgentType.CAR)
carwypnt_NNW_B = SMTApp.create_flat_waypoint(25, carwypnt_NNW_B_centre, unit_E, cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)

carwypnt_WNW_A = SMTApp.create_flat_waypoint(26, carwypnt_WNW_A_centre, unit_N, cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)
carwypnt_WNW_B = SMTApp.create_round_waypoint(27, carwypnt_WNW_B_centre, unit_N, cargate_halfwidth, SMAD.AgentType.CAR)

carwypnt_WSW_A = SMTApp.create_round_waypoint(28, carwypnt_WSW_A_centre, unit_N, cargate_halfwidth, SMAD.AgentType.CAR)
carwypnt_WSW_B = SMTApp.create_flat_waypoint(29, carwypnt_WSW_B_centre, unit_N, cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)


#
# Diagonal unit vectors
#
rt2     = np.sqrt(2)
inv_rt2 = 0.5 * rt2
unit_SE = inv_rt2 * (unit_E - unit_N)
unit_NE = inv_rt2 * (unit_N + unit_E)
unit_NW = -unit_SE
unit_SW = -unit_NE

#
# Experiment to find realistic values!
#
if 0:
    Lc_E2N           = SMND.calc_norm(carwypnt_NNW_A_centre - carwypnt_ESE_B_centre)
    Lc_E2N_third     = (1. / 3) * Lc_E2N
    carwypnt_E2N_pts = [carwypnt_ESE_B_centre, carwypnt_ESE_B_centre - Lc_E2N_third * unit_E, carwypnt_NNW_A_centre - Lc_E2N_third * unit_N, carwypnt_NNW_A_centre]
    
    Lc_N2E = SMND.calc_norm(carwypnt_ENE_A_centre - carwypnt_NNE_B_centre)
    Lc_N2E_third     = (1. / 3) * Lc_N2E
    carwypnt_N2E_pts = [carwypnt_NNE_B_centre, carwypnt_NNE_B_centre - Lc_E2N_third * unit_N, carwypnt_ENE_A_centre - Lc_E2N_third * unit_E, carwypnt_ENE_A_centre]

    Lc_N2W = SMND.calc_norm(carwypnt_WSW_A_centre - carwypnt_NNE_B_centre)
    Lc_N2W_third     = (1. / 3) * Lc_N2W
    carwypnt_N2W_pts = [carwypnt_NNE_B_centre, carwypnt_NNE_B_centre - Lc_E2N_third * unit_N, carwypnt_WSW_A_centre + Lc_E2N_third * unit_E, carwypnt_WSW_A_centre]
    
    Lc_W2N = SMND.calc_norm(carwypnt_NNW_A_centre - carwypnt_WNW_B_centre)
    Lc_W2N_third     = (1. / 3) * Lc_W2N
    carwypnt_W2N_pts = [carwypnt_WNW_B_centre, carwypnt_WNW_B_centre + Lc_E2N_third * unit_E, carwypnt_NNW_A_centre - Lc_E2N_third * unit_N, carwypnt_NNW_A_centre]

    carwypnt_E2N_A_centre = SMND.evaluate_cubic_spline(carwypnt_E2N_pts, 0.25)
    carwypnt_E2N_B_centre = SMND.evaluate_cubic_spline(carwypnt_E2N_pts, 0.5)
    carwypnt_E2N_C_centre = SMND.evaluate_cubic_spline(carwypnt_E2N_pts, 0.75)
    carwypnt_N2E_A_centre = SMND.evaluate_cubic_spline(carwypnt_N2E_pts, 0.25)
    carwypnt_N2E_B_centre = SMND.evaluate_cubic_spline(carwypnt_N2E_pts, 0.5)
    carwypnt_N2E_C_centre = SMND.evaluate_cubic_spline(carwypnt_N2E_pts, 0.75)
    carwypnt_N2W_A_centre = SMND.evaluate_cubic_spline(carwypnt_N2W_pts, 0.25)
    carwypnt_N2W_B_centre = SMND.evaluate_cubic_spline(carwypnt_N2W_pts, 0.5)
    carwypnt_N2W_C_centre = SMND.evaluate_cubic_spline(carwypnt_N2W_pts, 0.75)
    carwypnt_W2N_A_centre = SMND.evaluate_cubic_spline(carwypnt_W2N_pts, 0.25)
    carwypnt_W2N_B_centre = SMND.evaluate_cubic_spline(carwypnt_W2N_pts, 0.5)
    carwypnt_W2N_C_centre = SMND.evaluate_cubic_spline(carwypnt_W2N_pts, 0.75)

    carwypnt_E2N = SMTApp.create_round_waypoint(30, carwypnt_E2N_centre, unit_NE, ptngate_halfwidth, SMAD.AgentType.CAR)
    carwypnt_N2E = SMTApp.create_round_waypoint(31, carwypnt_N2E_centre, unit_NE, ptngate_halfwidth, SMAD.AgentType.CAR)
    carwypnt_N2W = SMTApp.create_round_waypoint(32, carwypnt_N2W_centre, unit_NW, ptngate_halfwidth, SMAD.AgentType.CAR)
    carwypnt_W2N = SMTApp.create_round_waypoint(33, carwypnt_W2N_centre, unit_NW, ptngate_halfwidth, SMAD.AgentType.CAR)
elif 1:
    Lc_E2N           = SMND.calc_norm(carwypnt_NNW_A_centre - carwypnt_ESE_B_centre)
    Lc_E2N_third     = (1. / 3) * Lc_E2N
    carwypnt_E2N_pts = [carwypnt_ESE_B_centre, carwypnt_ESE_B_centre - Lc_E2N_third * unit_E, carwypnt_NNW_A_centre - Lc_E2N_third * unit_N, carwypnt_NNW_A_centre]
    
    Lc_N2E = SMND.calc_norm(carwypnt_ENE_A_centre - carwypnt_NNE_B_centre)
    Lc_N2E_third     = (1. / 3) * Lc_N2E
    carwypnt_N2E_pts = [carwypnt_NNE_B_centre, carwypnt_NNE_B_centre - Lc_E2N_third * unit_N, carwypnt_ENE_A_centre - Lc_E2N_third * unit_E, carwypnt_ENE_A_centre]

    Lc_N2W = SMND.calc_norm(carwypnt_WSW_A_centre - carwypnt_NNE_B_centre)
    Lc_N2W_third     = (1. / 3) * Lc_N2W
    carwypnt_N2W_pts = [carwypnt_NNE_B_centre, carwypnt_NNE_B_centre - Lc_E2N_third * unit_N, carwypnt_WSW_A_centre + Lc_E2N_third * unit_E, carwypnt_WSW_A_centre]
    
    Lc_W2N = SMND.calc_norm(carwypnt_NNW_A_centre - carwypnt_WNW_B_centre)
    Lc_W2N_third     = (1. / 3) * Lc_W2N
    carwypnt_W2N_pts = [carwypnt_WNW_B_centre, carwypnt_WNW_B_centre + Lc_E2N_third * unit_E, carwypnt_NNW_A_centre - Lc_E2N_third * unit_N, carwypnt_NNW_A_centre]

    carwypnt_E2N_centre = SMND.evaluate_cubic_spline(carwypnt_E2N_pts, 0.5)
    carwypnt_N2E_centre = SMND.evaluate_cubic_spline(carwypnt_N2E_pts, 0.5)
    carwypnt_N2W_centre = SMND.evaluate_cubic_spline(carwypnt_N2W_pts, 0.5)
    carwypnt_W2N_centre = SMND.evaluate_cubic_spline(carwypnt_W2N_pts, 0.5)

    carwypnt_E2N = SMTApp.create_round_waypoint(30, carwypnt_E2N_centre, unit_NE, ptngate_halfwidth, SMAD.AgentType.CAR)
    carwypnt_N2E = SMTApp.create_round_waypoint(31, carwypnt_N2E_centre, unit_NE, ptngate_halfwidth, SMAD.AgentType.CAR)
    carwypnt_N2W = SMTApp.create_round_waypoint(32, carwypnt_N2W_centre, unit_NW, ptngate_halfwidth, SMAD.AgentType.CAR)
    carwypnt_W2N = SMTApp.create_round_waypoint(33, carwypnt_W2N_centre, unit_NW, ptngate_halfwidth, SMAD.AgentType.CAR)
else:   
    r_inner = ptngate_halfwidth + safety_margin + cargate_halfwidth
    r_outer = ptngate_halfwidth + safety_margin + cargate_width + safety_margin + cargate_halfwidth

    carwypnt_E2N_centre = ptnwypnt_NE_centre + r_outer * unit_SW
    carwypnt_N2E_centre = ptnwypnt_NE_centre + r_inner * unit_SW
    carwypnt_N2W_centre = ptnwypnt_NW_centre + r_outer * unit_SE
    carwypnt_W2N_centre = ptnwypnt_NW_centre + r_inner * unit_SE

    carwypnt_E2N = SMTApp.create_round_waypoint(30, carwypnt_E2N_centre, unit_NE, ptngate_halfwidth, SMAD.AgentType.CAR)
    carwypnt_N2E = SMTApp.create_round_waypoint(31, carwypnt_N2E_centre, unit_NE, ptngate_halfwidth, SMAD.AgentType.CAR)
    carwypnt_N2W = SMTApp.create_round_waypoint(32, carwypnt_N2W_centre, unit_NW, ptngate_halfwidth, SMAD.AgentType.CAR)
    carwypnt_W2N = SMTApp.create_round_waypoint(33, carwypnt_W2N_centre, unit_NW, ptngate_halfwidth, SMAD.AgentType.CAR)


#
# Probabilities reflecting the probability of through-routed pedestrians stopping to perform an activity and the
# probability of each activity being the one chosen by any pedestrian.
#
activity_probability = 0.1
activity_popularities = [("TakeawayCoffee", 0.5), ("OnsiteCoffee", 0.25), ("Swill", 0.05), ("Drink", 0.2)]
activity_demand_info = SMA.DemandInfo(activity_probability, activity_popularities)

#
# Pedestrian activity-nodes:
# SE? centre_true - area_NS_displacement + area_EW_displacement
# NE? centre_true + area_NS_displacement + area_EW_displacement
# NW? centre_true + area_NS_displacement - area_EW_displacement
# SW? centre_true - area_NS_displacement - area_EW_displacement
#

# vestibule_NS_displacement = ptngate_NS_displacement + rt2 * ptngate_halfwidth * unit_N
# vestibule_EW_displacement = ptngate_EW_displacement + rt2 * ptngate_halfwidth * unit_E
vestibule_NS_displacement = ptngate_NS_displacement + rt2 * (ptngate_width + safety_margin) * unit_N
vestibule_EW_displacement = ptngate_EW_displacement + rt2 * (ptngate_width + safety_margin) * unit_E

vestibule_NE_centre = centre_true + vestibule_NS_displacement + vestibule_EW_displacement
seating_NE_centre   = vestibule_NE_centre + inv_rt2 * vestibule_NS_displacement
ordering_NE_centre  = vestibule_NE_centre + inv_rt2 * vestibule_EW_displacement

seating_NE_halfwidth  = ptngate_halfwidth
ordering_NE_halfwidth = ptngate_halfwidth
ordering_NE_halfdepth = ptngate_halfdepth

cafe_NE_vestibule_shape = SMGS.create_square(vestibule_NE_centre, unit_NE, ptngate_halfwidth, ptngate_halfdepth)
cafe_seating_shape      = SMGS.create_square(seating_NE_centre, unit_E, seating_NE_halfwidth, ptngate_halfdepth)
cafe_ordering_shape     = SMGS.Segment(ordering_NE_centre + ordering_NE_halfwidth * unit_E, ordering_NE_centre - ordering_NE_halfwidth * unit_E, ordering_NE_halfdepth)

cafe_NE_foci     = [(cafe_ordering_shape, SVD.black), (cafe_ordering_shape, SVD.black)]
cafe_NE_barriers = []
cafe_NE          = SMTAct.create_cafe(34, cafe_NE_vestibule_shape, SVD.black, cafe_NE_foci, cafe_NE_barriers)

vestibule_NW_centre = centre_true + vestibule_NS_displacement - vestibule_EW_displacement
seating_NW_centre   = vestibule_NW_centre + rt2 * vestibule_NS_displacement
ordering_NW_centre  = vestibule_NW_centre - rt2 * vestibule_EW_displacement

seating_NW_halfwidth  = ptngate_halfwidth
ordering_NW_halfwidth = ptngate_halfwidth
ordering_NW_halfdepth = ptngate_halfdepth

bar_NW_vestibule_shape = SMGS.create_square(vestibule_NW_centre, unit_NW, ptngate_halfwidth, ptngate_halfdepth)
bar_seating_shape      = SMGS.create_square(seating_NW_centre, unit_E, seating_NW_halfwidth, ptngate_halfdepth)
bar_ordering_shape     = SMGS.Segment(ordering_NW_centre + ordering_NW_halfwidth * unit_E, ordering_NW_centre - ordering_NW_halfwidth * unit_E, ordering_NW_halfdepth)

bar_NW_foci     = [(cafe_ordering_shape, SVD.black), (cafe_ordering_shape, SVD.black)]
bar_NW_barriers = []
bar_NW          = SMTAct.create_bar(35, bar_NW_vestibule_shape, SVD.black, bar_NW_foci, bar_NW_barriers)


activity_nodes = [cafe_NE, bar_NW]









# wall_depth  = 4.  # in metres
# wall_radius = 0.5 * wall_depth


#
# Footpath placement
#
footpath_halfwidth = ptngate_halfwidth

footpath_S_shape   = SMGS.Segment(ptngate_WSW.approach_shape.posn, ptngate_ESE.approach_shape.posn, footpath_halfwidth)
footpath_ENE_shape = SMGS.Segment(ptngate_ENE.approach_shape.posn, ptnwypnt_NE.approach_shape.posn, footpath_halfwidth)
footpath_NNE_shape = SMGS.Segment(ptngate_NNE.approach_shape.posn, ptnwypnt_NE.approach_shape.posn, footpath_halfwidth)
footpath_NNW_shape = SMGS.Segment(ptngate_NNW.approach_shape.posn, ptnwypnt_NW.approach_shape.posn, footpath_halfwidth)
footpath_WNW_shape = SMGS.Segment(ptngate_WNW.approach_shape.posn, ptnwypnt_NW.approach_shape.posn, footpath_halfwidth)

footpath_S   = SMZZ.create_footpath(51, footpath_S_shape)
footpath_ENE = SMZZ.create_footpath(52, footpath_ENE_shape)
footpath_NNE = SMZZ.create_footpath(53, footpath_NNE_shape)
footpath_NNW = SMZZ.create_footpath(54, footpath_NNW_shape)
footpath_WNW = SMZZ.create_footpath(55, footpath_WNW_shape)

#
# Roadway placement
#
roadway_halfwidth = cargate_halfwidth

roadway_ESE_shape = SMGS.Segment(cargate_ESE.approach_shape.posn, carwypnt_ESE_A.approach_shape.posn, roadway_halfwidth)
roadway_ENE_shape = SMGS.Segment(cargate_ENE.approach_shape.posn, carwypnt_ENE_B.approach_shape.posn, roadway_halfwidth)
roadway_NNE_shape = SMGS.Segment(cargate_NNE.approach_shape.posn, carwypnt_NNE_A.approach_shape.posn, roadway_halfwidth)
roadway_NNW_shape = SMGS.Segment(cargate_NNW.approach_shape.posn, carwypnt_NNW_B.approach_shape.posn, roadway_halfwidth)
roadway_WNW_shape = SMGS.Segment(cargate_WNW.approach_shape.posn, carwypnt_WNW_A.approach_shape.posn, roadway_halfwidth)
roadway_WSW_shape = SMGS.Segment(cargate_WSW.approach_shape.posn, carwypnt_WSW_B.approach_shape.posn, roadway_halfwidth)

roadway_ESE = SMZZ.create_roadway(61, roadway_ESE_shape)
roadway_ENE = SMZZ.create_roadway(62, roadway_ENE_shape)
roadway_NNE = SMZZ.create_roadway(63, roadway_NNE_shape)
roadway_NNW = SMZZ.create_roadway(64, roadway_NNW_shape)
roadway_WNW = SMZZ.create_roadway(65, roadway_WNW_shape)
roadway_WSW = SMZZ.create_roadway(66, roadway_WSW_shape)

#
# @todo: Consider declaring other zones for marked or signalised non-shared crossing.
#


#
# Pedestrian symmetric centroid <-> waypoint connections
#
SMTN.symconnect(ptngate_NNE,  ptnwypnt_NE)
SMTN.symconnect(ptngate_ENE,  ptnwypnt_NE)
SMTN.symconnect(ptngate_NNE,  ptnwypnt_NE)
SMTN.symconnect(ptngate_NNW,  ptnwypnt_NW)
SMTN.symconnect(ptngate_WNW,  ptnwypnt_NW)

SMTN.symconnect(ptngate_ESE,  ptnwypnt_SE)
SMTN.symconnect(ptngate_WSW,  ptnwypnt_SW)

SMTN.symconnect(ptngate_S,   ptnwypnt_S)
# SMTN.symconnect(ptnwypnt_S,  ptnwypnt_SE)
# SMTN.symconnect(ptnwypnt_S,  ptnwypnt_SW)
SMTN.omniconnect([ptnwypnt_NE, ptnwypnt_NW, ptnwypnt_SW, ptnwypnt_S, ptnwypnt_SE])

#
# Pedestrian activity-node connections
#
SMTN.symconnect(cafe_NE,  ptnwypnt_NE)
SMTN.symconnect(bar_NW,  ptnwypnt_NW)

#
# One-way car connections
#
SMTN.seqconnect([cargate_ESE, carwypnt_ESE_A, carwypnt_ESE_B])
SMTN.seqconnect([carwypnt_ENE_A, carwypnt_ENE_B, cargate_ENE])

SMTN.seqconnect([cargate_NNE, carwypnt_NNE_A, carwypnt_NNE_B])
SMTN.seqconnect([carwypnt_NNW_A, carwypnt_NNW_B, cargate_NNW])

SMTN.seqconnect([cargate_WNW, carwypnt_WNW_A, carwypnt_WNW_B])
SMTN.seqconnect([carwypnt_WSW_A, carwypnt_WSW_B, cargate_WSW])

carwypnt_ESE_B.connect(carwypnt_WSW_A)
carwypnt_WNW_B.connect(carwypnt_ENE_A)

SMTN.seqconnect([carwypnt_ESE_B, carwypnt_E2N, carwypnt_NNW_A])
SMTN.seqconnect([carwypnt_NNE_B, carwypnt_N2E, carwypnt_ENE_A])
SMTN.seqconnect([carwypnt_NNE_B, carwypnt_N2W, carwypnt_WSW_A])
SMTN.seqconnect([carwypnt_WNW_B, carwypnt_W2N, carwypnt_NNW_A])


# cargate_ESE.connect(carwypnt_ESE)
# cargate_NNE.connect(carwypnt_NNE)
# cargate_WNW.connect(carwypnt_WNW)

# carwypnt_ESE.connect(carwypnt_WSW)
# carwypnt_ESE.connect(carwypnt_NNW) # SMTN.seqconnect([carwypnt_ESE, carwypnt_E2N, carwypnt_NNW])

# carwypnt_NNE.connect(carwypnt_ENE)
# carwypnt_NNE.connect(carwypnt_WSW) # SMTN.seqconnect([carwypnt_NNE, carwypnt_N2W, carwypnt_WSW])

# carwypnt_WNW.connect(carwypnt_NNW)
# carwypnt_WNW.connect(carwypnt_ENE)

# carwypnt_ENE.connect(cargate_ENE)
# carwypnt_NNW.connect(cargate_NNW)
# carwypnt_WSW.connect(cargate_WSW)

ptn_waypoints = [ptnwypnt_NE, ptnwypnt_NW, ptnwypnt_SW, ptnwypnt_S, ptnwypnt_SE]
car_waypoints = \
    [carwypnt_ESE_A, carwypnt_ESE_B,
     carwypnt_ENE_A, carwypnt_ENE_B,
     carwypnt_NNE_A, carwypnt_NNE_B,
     carwypnt_NNW_A, carwypnt_NNW_B,
     carwypnt_WNW_A, carwypnt_WNW_B,
     carwypnt_WSW_A, carwypnt_WSW_B,
     carwypnt_E2N, carwypnt_N2E,
     carwypnt_N2W, carwypnt_W2N]

ptn_topo_spec = SMTT.AgentNodeConfig(ptn_centroids, ptn_waypoints, activity_nodes, SMAD.AgentType.PEDESTRIAN)
car_topo_spec = SMTT.AgentNodeConfig(car_centroids, car_waypoints, [], SMAD.AgentType.CAR)
topo_specs    = [ptn_topo_spec, car_topo_spec]
# ptn_topology = SMM.PartialTopology(ptn_centroids, ptn_waypoints, [], SMAD.AgentType.PEDESTRIAN)
# car_topology = SMM.PartialTopology(car_centroids, car_waypoints, [], SMAD.AgentType.CAR)
# partial_topologies = [ptn_topology, car_topology]
# partial_topologies = [SMM.PartialTopology(ptn_centroids, ptn_waypoints, 0), SMM.PartialTopology(car_centroids, car_waypoints, 3)]


# topology = mnn.Graph()
# topology.add_nodelist(centroids + waypoints)






OD_matrices = {SMAD.AgentType.PEDESTRIAN: pedestrian_OD_matrix, SMAD.AgentType.CAR: car_OD_matrix}

footpaths        = [footpath_S, footpath_ENE, footpath_NNE, footpath_NNW, footpath_WNW]
roadways         = [roadway_ESE, roadway_ENE, roadway_NNE, roadway_NNW, roadway_WNW, roadway_WSW]
# crossings        = [] # [crossing_C]
# countercrossings = [ctrcrossing_NC, ctrcrossing_SC]
# vbarriers        = [vbarrier_N, vbarrier_S]

import controller.controller as SCC

if __name__ == "__main__":
    zones = footpaths + roadways # + crossings # + countercrossings # + vbarriers

    plaza           = SMM.Model(width, height, screen_size, scale_factor, T_simulation, 4, topo_specs, OD_matrices, zones, [], [], [])
    plaza.register_activity_demand_info(activity_demand_info)

    SML.log_static_components(plaza.topology.centroids, plaza.topology.waypoints, plaza.topology.activity_nodes, plaza.zone_config)
    logger = SML.Logger()
    plaza.register_logger(logger)

    for i in range(L_simulation):
        print("Simulation step ", i, ": start")
        plaza.spawn_agents()
        plaza.advance_timestep()
        assert(isinstance(plaza, SMM.Model))
        assert(isinstance(view, SVV.View))
        plaza.draw(view)
        plaza.despawn_agents()
        view.flip()
        # SCC.wait(10)
        # print("Simulation step ", i, ": finish")

    SML.log_stream_statistics(plaza)
