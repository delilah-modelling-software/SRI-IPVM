import sys
import numpy as np
import scipy as sp
# import scipy.stats as ss

np.seterr(invalid='raise')

import controller.controller as SCC
import view.defs as SVD
import view.view as SVV
import model.configuration as SMC
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


next_node_id = 0
sup_node_id  = 32 # 28 # 128
def generate_node_id():
    global next_node_id
    if next_node_id == sup_node_id:
        assert(False)
        return -1

    node_id = next_node_id
    next_node_id += 1
    return node_id

width  = 80
height = 80

scale_factor   = 15
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width,screen_height)

# layer_flags = SVV.ViewLayerFlagset.GEOMETRY | SVV.ViewLayerFlagset.TOPOLOGY | SVV.ViewLayerFlagset.MACROSCOPIC | SVV.ViewLayerFlagset.MICROSCOPIC
# layer_flags = SVV.ViewLayerFlagset.GEOMETRY | SVV.ViewLayerFlagset.TOPOLOGY | SVV.ViewLayerFlagset.MICROSCOPIC | SVV.ViewLayerFlagset.ATTRACTION
# layer_flags = SVD.LayerFlagset.GEOMETRY | SVD.LayerFlagset.TOPOLOGY | SVD.LayerFlagset.MICROSCOPIC | SVD.LayerFlagset.ATTRACTION | SVD.LayerFlagset.INTERACTION | SVD.LayerFlagset.CONES | SVD.LayerFlagset.TRAJECTORY
layer_flags = SVD.LayerFlagset.GEOMETRY | SVD.LayerFlagset.TOPOLOGY | SVD.LayerFlagset.MICROSCOPIC | SVD.LayerFlagset.INTERACTION | SVD.LayerFlagset.INTRAMODAL | SVD.LayerFlagset.INTERMODAL | SVD.LayerFlagset.CONES | SVD.LayerFlagset.TRAJECTORY
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
L_simulation = int(T_simulation / SMC.T_step)

spawner_uniform = SMDis.DistributorUniform()

unit_E = np.array([1, 0])
unit_N = np.array([0, 1])

ptngate_width     = 3.
ptngate_halfwidth = 0.5 * ptngate_width
ptngate_halfdepth = 0.125

#
# The void lane represents:
#   3m for an empty lane where cars are not allowed to park; and
#   0.5m of unusable footpath near the kerb.
#
voidlane_width    =  3.5

cargate_width     = 4.
cargate_halfwidth = 0.5 * cargate_width
cargate_halfdepth = 0.25

# carspawner = SMDis.DistributorUniform()

safety_margin = 0.5 # 1.
cargate_NS_displacement = (safety_margin + cargate_halfwidth) * unit_N
ptngate_NS_extradisplacement = (cargate_halfwidth + voidlane_width + ptngate_halfwidth) * unit_N
ptngate_NS_displacement = cargate_NS_displacement + ptngate_NS_extradisplacement

cargate_EW_displacement = (safety_margin + cargate_halfwidth) * unit_E
ptngate_EW_extradisplacement = (cargate_halfwidth + voidlane_width + ptngate_halfwidth) * unit_E
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

# node_id = generate_node_id()
# print("Node {}".format(node_id))

ptngate_ESE = SMTC.create_oblong_gate(generate_node_id(), centre_E - ptngate_NS_displacement, ptngate_NS_hwvec, ptngate_halfdepth, SVD.colour(0, 0, 128), SVD.colour(0, 0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)
ptngate_ENE = SMTC.create_oblong_gate(generate_node_id(), centre_E + ptngate_NS_displacement, ptngate_NS_hwvec, ptngate_halfdepth, SVD.colour(0, 0, 128), SVD.colour(0, 0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)
ptngate_NNE = SMTC.create_oblong_gate(generate_node_id(), centre_N + ptngate_EW_displacement, ptngate_EW_hwvec, ptngate_halfdepth, SVD.colour(0, 0, 128), SVD.colour(0, 0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)
ptngate_NNW = SMTC.create_oblong_gate(generate_node_id(), centre_N - ptngate_EW_displacement, ptngate_EW_hwvec, ptngate_halfdepth, SVD.colour(0, 0, 128), SVD.colour(0, 0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)
ptngate_WNW = SMTC.create_oblong_gate(generate_node_id(), centre_W + ptngate_NS_displacement, ptngate_NS_hwvec, ptngate_halfdepth, SVD.colour(0, 0, 128), SVD.colour(0, 0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)
ptngate_WSW = SMTC.create_oblong_gate(generate_node_id(), centre_W - ptngate_NS_displacement, ptngate_NS_hwvec, ptngate_halfdepth, SVD.colour(0, 0, 128), SVD.colour(0, 0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)

# assert(0)

ptngate_S = SMTC.create_oblong_gate(generate_node_id(), centre_S, ptngate_EW_hwvec, ptngate_halfdepth, SVD.colour(0, 128, 0), SVD.colour(0, 0, 255), spawner_uniform, SMAD.AgentType.PEDESTRIAN)

#
# Vehicle gates:
# ESE? centre_E - cargate_NS_displacement +- cargate_halfwidth * unit_N
# ENE? centre_E + cargate_NS_displacement +- cargate_halfwidth * unit_N
# NNE? centre_N + cargate_EW_displacement +- cargate_halfwidth * unit_E
# NNW? centre_N - cargate_EW_displacement +- cargate_halfwidth * unit_E
# WNW? centre_W + cargate_NS_displacement +- cargate_halfwidth * unit_N
# WSW? centre_W - cargate_NS_displacement +- cargate_halfwidth * unit_N
#

# cargate_NS_hwvec = cargate_halfwidth * unit_N
# cargate_EW_hwvec = cargate_halfwidth * unit_E

# cargate_ESE = SMTC.create_oblong_gate(generate_node_id(), centre_E - cargate_NS_displacement, cargate_NS_hwvec, cargate_halfdepth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
# cargate_ENE = SMTC.create_oblong_gate(generate_node_id(), centre_E + cargate_NS_displacement, cargate_NS_hwvec, cargate_halfdepth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
# cargate_NNE = SMTC.create_oblong_gate(generate_node_id(), centre_N + cargate_EW_displacement, cargate_EW_hwvec, cargate_halfdepth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
# cargate_NNW = SMTC.create_oblong_gate(generate_node_id(), centre_N - cargate_EW_displacement, cargate_EW_hwvec, cargate_halfdepth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
# cargate_WNW = SMTC.create_oblong_gate(generate_node_id(), centre_W + cargate_NS_displacement, cargate_NS_hwvec, cargate_halfdepth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
# cargate_WSW = SMTC.create_oblong_gate(generate_node_id(), centre_W - cargate_NS_displacement, cargate_NS_hwvec, cargate_halfdepth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)

cargate_ESE = SMTC.create_round_gate(generate_node_id(), centre_E - cargate_NS_displacement, cargate_halfwidth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
cargate_ENE = SMTC.create_round_gate(generate_node_id(), centre_E + cargate_NS_displacement, cargate_halfwidth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
cargate_NNE = SMTC.create_round_gate(generate_node_id(), centre_N + cargate_EW_displacement, cargate_halfwidth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
cargate_NNW = SMTC.create_round_gate(generate_node_id(), centre_N - cargate_EW_displacement, cargate_halfwidth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
cargate_WNW = SMTC.create_round_gate(generate_node_id(), centre_W + cargate_NS_displacement, cargate_halfwidth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)
cargate_WSW = SMTC.create_round_gate(generate_node_id(), centre_W - cargate_NS_displacement, cargate_halfwidth, SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), spawner_uniform, SMAD.AgentType.CAR)

#
# Origin-destination pairs:
#   Note: Parking, U-turns, and /3-/5-/7-point turns are neglected
#
ptn_centroids        = [ptngate_ESE, ptngate_ENE, ptngate_NNE, ptngate_NNW, ptngate_WNW, ptngate_WSW, ptngate_S]
car_centroids        = [cargate_ESE, cargate_ENE, cargate_NNE, cargate_NNW, cargate_WNW, cargate_WSW]
pedestrian_OD_matrix = \
    3 * np.array(
    [[2, 5, 5, 5, 5, 5, 5],
     [5, 2, 5, 5, 5, 5, 5],
     [5, 5, 2, 5, 5, 5, 5],
     [5, 5, 5, 2, 5, 5, 5],
     [5, 5, 5, 5, 2, 5, 5],
     [5, 5, 5, 5, 5, 2, 5],
     [5, 5, 5, 5, 5, 5, 2]])
six_zeroes = [0, 0, 0, 0, 0, 0]
car_OD_matrix = \
    10 * np.array(
    [[0, 0, 0, 2, 0, 2],
     six_zeroes,
     [0, 2, 0, 0, 0, 2],
     six_zeroes,
     [0, 2, 0, 2, 0, 0],
     six_zeroes])


#
# Testing Override
#
if False:
    ptn_src_idx_set = {5}
    ptn_dst_idx_set = {1, 3}
    for i in range(7):
        for j in range(7):
            if i not in ptn_src_idx_set or j not in ptn_dst_idx_set:
                pedestrian_OD_matrix[i, j] = 0

    car_src_idx_set = {2}
    car_dst_idx_set = {1, 5}
    for i in range(6):
        for j in range(6):
            if i not in car_src_idx_set or j not in car_dst_idx_set:
                car_OD_matrix[i, j] = 0

mean_car_halflength  = 2.4
mean_car_length      = 4.8

#
# Pedestrian waypoints:
# SE? Square(centre_true - ptngate_NS_displacement + ptngate_EW_displacement, waypoint_halfwidth)
# NE? Square(centre_true + ptngate_NS_displacement + ptngate_EW_displacement, waypoint_halfwidth)
# NW? Square(centre_true + ptngate_NS_displacement - ptngate_EW_displacement, waypoint_halfwidth)
# SW? Square(centre_true - ptngate_NS_displacement - ptngate_EW_displacement, waypoint_halfwidth)
#
# SC? Square(centre_true - ptngate_NS_displacement, waypoint_halfwidth)
#

rt2                            = np.sqrt(2)
waypoint_NS_extradisplacement = (ptngate_width + safety_margin) * unit_N
waypoint_EW_extradisplacement = (ptngate_width + safety_margin) * unit_E
waypoint_NS_displacement = ptngate_NS_displacement + waypoint_NS_extradisplacement
waypoint_EW_displacement = ptngate_EW_displacement + waypoint_EW_extradisplacement

ptnwypnt_ENE_centre = centre_true + ptngate_NS_displacement + waypoint_EW_displacement
ptnwypnt_NE_centre  = centre_true + ptngate_NS_displacement + ptngate_EW_displacement - mean_car_halflength * unit_E - mean_car_halflength * unit_N
ptnwypnt_NNE_centre = centre_true + waypoint_NS_displacement + ptngate_EW_displacement


ptnwypnt_NNW_centre = centre_true + waypoint_NS_displacement - ptngate_EW_displacement
ptnwypnt_NW_centre  = centre_true + ptngate_NS_displacement - ptngate_EW_displacement - mean_car_halflength * unit_N + mean_car_halflength * unit_E
ptnwypnt_WNW_centre = centre_true + ptngate_NS_displacement - waypoint_EW_displacement

ptnwypnt_SE_centre = centre_true - ptngate_NS_displacement + ptngate_EW_displacement - mean_car_halflength * unit_E
ptnwypnt_SW_centre = centre_true - ptngate_NS_displacement - ptngate_EW_displacement + mean_car_halflength * unit_E
ptnwypnt_S_centre  = centre_true - ptngate_NS_displacement

ptnwypnt_ENE = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_ENE_centre, unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
# ptnwypnt_NE  = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_NE_centre,  unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
ptnwypnt_NE  = SMTApp.create_round_waypoint(generate_node_id(), ptnwypnt_NE_centre,  unit_E, ptngate_halfwidth, SMAD.AgentType.PEDESTRIAN)
ptnwypnt_NNE = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_NNE_centre, unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)

ptnwypnt_NNW = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_NNW_centre, unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
# ptnwypnt_NW  = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_NW_centre,  unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
ptnwypnt_NW  = SMTApp.create_round_waypoint(generate_node_id(), ptnwypnt_NW_centre,  unit_E, ptngate_halfwidth, SMAD.AgentType.PEDESTRIAN)
ptnwypnt_WNW = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_WNW_centre, unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)

ptnwypnt_SW = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_SW_centre, unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
ptnwypnt_SE = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_SE_centre, unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
ptnwypnt_S  = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_S_centre,  unit_E, ptngate_halfwidth, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)

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
carwypnt_safe_offset = safety_margin + mean_car_halflength

# carwypnt_ESE_exit_centre = centre_true - cargate_NS_displacement + ptngate_EW_displacement + ptngate_EW_hwvec
# carwypnt_ESE_A_centre = carwypnt_ESE_B_centre + carwypnt_safe_offset * unit_E
carwypnt_ESE_exit_centre = centre_true - cargate_NS_displacement + ptngate_EW_displacement

# carwypnt_ENE_entry_centre = centre_true + cargate_NS_displacement + ptngate_EW_displacement # + ptngate_EW_hwvec
# carwypnt_ENE_entry_centre = centre_true + cargate_NS_displacement + ptngate_EW_displacement + ptngate_EW_hwvec
# carwypnt_ENE_B_centre = carwypnt_ENE_A_centre + carwypnt_safe_offset * unit_E
carwypnt_ENE_entry_centre = centre_true + cargate_NS_displacement + ptngate_EW_displacement

# carwypnt_NNE_exit_centre = centre_true + ptngate_NS_displacement + cargate_EW_displacement + ptngate_NS_hwvec
# carwypnt_NNE_A_centre = carwypnt_NNE_B_centre + carwypnt_safe_offset * unit_N
carwypnt_NNE_exit_centre = centre_true + ptngate_NS_displacement + cargate_EW_displacement

# carwypnt_NNW_entry_centre = centre_true + ptngate_NS_displacement - cargate_EW_displacement # + ptngate_NS_hwvec
# carwypnt_NNW_entry_centre = centre_true + ptngate_NS_displacement - cargate_EW_displacement + ptngate_NS_hwvec
# carwypnt_NNW_B_centre = carwypnt_NNW_A_centre + carwypnt_safe_offset * unit_N
carwypnt_NNW_entry_centre = centre_true + ptngate_NS_displacement - cargate_EW_displacement

# carwypnt_WNW_exit_centre = centre_true + cargate_NS_displacement - ptngate_EW_displacement - ptngate_EW_hwvec
# carwypnt_WNW_A_centre = carwypnt_WNW_B_centre - carwypnt_safe_offset * unit_E
carwypnt_WNW_exit_centre = centre_true + cargate_NS_displacement - ptngate_EW_displacement

# carwypnt_WSW_entry_centre = centre_true - cargate_NS_displacement - ptngate_EW_displacement # - ptngate_EW_hwvec
# carwypnt_WSW_entry_centre = centre_true - cargate_NS_displacement - ptngate_EW_displacement - ptngate_EW_hwvec
# carwypnt_WSW_B_centre = carwypnt_WSW_A_centre - carwypnt_safe_offset * unit_E
carwypnt_WSW_entry_centre = centre_true - cargate_NS_displacement - ptngate_EW_displacement

# carwypnt_ESE_exit = SMTApp.create_round_waypoint(generate_node_id(), carwypnt_ESE_exit_centre, unit_N, cargate_halfwidth, SMAD.AgentType.CAR)
# carwypnt_ESE_B = SMTApp.create_round_waypoint(generate_node_id(), carwypnt_ESE_B_centre, unit_N, cargate_halfwidth, SMAD.AgentType.CAR)
carwypnt_ESE_exit = SMTApp.create_oblong_waypoint(generate_node_id(), carwypnt_ESE_exit_centre, unit_E, mean_car_halflength, cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)

# carwypnt_ENE_entry = SMTApp.create_round_waypoint(generate_node_id(), carwypnt_ENE_entry_centre, unit_N, cargate_halfwidth, SMAD.AgentType.CAR)
# carwypnt_ENE_B = SMTApp.create_round_waypoint(generate_node_id(), carwypnt_ENE_B_centre, unit_N, cargate_halfwidth, SMAD.AgentType.CAR)
carwypnt_ENE_entry = SMTApp.create_oblong_waypoint(generate_node_id(), carwypnt_ENE_entry_centre, unit_E, mean_car_halflength, cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)

# carwypnt_NNE_exit = SMTApp.create_round_waypoint(generate_node_id(), carwypnt_NNE_exit_centre, unit_E, cargate_halfwidth, SMAD.AgentType.CAR)
# carwypnt_NNE_B = SMTApp.create_round_waypoint(generate_node_id(), carwypnt_NNE_B_centre, unit_E, cargate_halfwidth, SMAD.AgentType.CAR)
carwypnt_NNE_exit = SMTApp.create_oblong_waypoint(generate_node_id(), carwypnt_NNE_exit_centre, unit_N, mean_car_halflength, cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)

# carwypnt_NNW_entry = SMTApp.create_round_waypoint(generate_node_id(), carwypnt_NNW_entry_centre, unit_E, cargate_halfwidth, SMAD.AgentType.CAR)
# carwypnt_NNW_B = SMTApp.create_round_waypoint(generate_node_id(), carwypnt_NNW_B_centre, unit_E, cargate_halfwidth, SMAD.AgentType.CAR)
carwypnt_NNW_entry = SMTApp.create_oblong_waypoint(generate_node_id(), carwypnt_NNW_entry_centre, unit_N, mean_car_halflength, cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)

# carwypnt_WNW_exit = SMTApp.create_round_waypoint(generate_node_id(), carwypnt_WNW_exit_centre, unit_N, cargate_halfwidth, SMAD.AgentType.CAR)
# carwypnt_WNW_B = SMTApp.create_round_waypoint(generate_node_id(), carwypnt_WNW_B_centre, unit_N, cargate_halfwidth, SMAD.AgentType.CAR)
carwypnt_WNW_exit = SMTApp.create_oblong_waypoint(generate_node_id(), carwypnt_WNW_exit_centre, unit_E, mean_car_halflength, cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)

# carwypnt_WSW_entry = SMTApp.create_round_waypoint(generate_node_id(), carwypnt_WSW_entry_centre, unit_N, cargate_halfwidth, SMAD.AgentType.CAR)
# carwypnt_WSW_B = SMTApp.create_round_waypoint(generate_node_id(), carwypnt_WSW_B_centre, unit_N, cargate_halfwidth, SMAD.AgentType.CAR)
carwypnt_WSW_entry = SMTApp.create_oblong_waypoint(generate_node_id(), carwypnt_WSW_entry_centre, unit_E, mean_car_halflength, cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)

#
# Diagonal unit vectors
#
inv_rt2 = 0.5 * rt2
unit_SE = inv_rt2 * (unit_E - unit_N)
unit_NE = inv_rt2 * (unit_N + unit_E)
unit_NW = -unit_SE
unit_SW = -unit_NE

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

vestibule_NS_extradisplacement = rt2 * (ptngate_width + safety_margin) * unit_N
vestibule_EW_extradisplacement = rt2 * (ptngate_width + safety_margin) * unit_E
vestibule_NS_displacement = ptngate_NS_displacement + vestibule_NS_extradisplacement
vestibule_EW_displacement = ptngate_EW_displacement + vestibule_EW_extradisplacement

vestibule_NE_centre = centre_true + vestibule_NS_displacement + vestibule_EW_displacement
seating_NE_centre   = vestibule_NE_centre + ((1 + inv_rt2) * ptngate_halfwidth + safety_margin) * unit_N
ordering_NE_centre  = vestibule_NE_centre + ((1 + inv_rt2) * ptngate_halfwidth + safety_margin) * unit_E

seating_NE_halfwidth  = ptngate_halfwidth
ordering_NE_halfwidth = ptngate_halfwidth
ordering_NE_halfdepth = ptngate_halfdepth

doorway_NE = SMTApp.create_flat_waypoint(generate_node_id(), vestibule_NE_centre - ptngate_halfwidth * unit_NE, unit_SE, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)

cafe_NE_vestibule_shape = SMGS.create_segment(vestibule_NE_centre, unit_SE, ptngate_halfwidth, ptngate_halfdepth)
# cafe_NE_vestibule_shape = SMGS.create_square(vestibule_NE_centre, unit_NE, ptngate_halfwidth, ptngate_halfdepth)
cafe_seating_shape      = SMGS.create_square(seating_NE_centre, unit_E, seating_NE_halfwidth, ptngate_halfdepth)
cafe_ordering_shape     = SMGS.Segment(ordering_NE_centre + ordering_NE_halfwidth * unit_N, ordering_NE_centre - ordering_NE_halfwidth * unit_N, ordering_NE_halfdepth)

cafe_NE_foci     = [(cafe_ordering_shape, SVD.black), (cafe_seating_shape, SVD.black)]
cafe_NE_barriers = []
cafe_NE          = SMTAct.create_cafe(generate_node_id(), cafe_NE_vestibule_shape, SVD.black, cafe_NE_foci, cafe_NE_barriers)

vestibule_NW_centre = centre_true + vestibule_NS_displacement - vestibule_EW_displacement
seating_NW_centre   = vestibule_NW_centre + ((1 + inv_rt2) * ptngate_halfwidth + safety_margin) * unit_N
ordering_NW_centre  = vestibule_NW_centre - ((1 + inv_rt2) * ptngate_halfwidth + safety_margin) * unit_E

seating_NW_halfwidth  = ptngate_halfwidth
ordering_NW_halfwidth = ptngate_halfwidth
ordering_NW_halfdepth = ptngate_halfdepth

doorway_NW = SMTApp.create_flat_waypoint(generate_node_id(), vestibule_NW_centre - ptngate_halfwidth * unit_NW, unit_NE, ptngate_halfwidth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)

bar_NW_vestibule_shape = SMGS.create_segment(vestibule_NW_centre, unit_NE, ptngate_halfwidth, ptngate_halfdepth)
# bar_NW_vestibule_shape = SMGS.create_square(vestibule_NW_centre, unit_NW, ptngate_halfwidth, ptngate_halfdepth)
bar_seating_shape      = SMGS.create_square(seating_NW_centre, unit_E, seating_NW_halfwidth, ptngate_halfdepth)
bar_ordering_shape     = SMGS.Segment(ordering_NW_centre + ordering_NW_halfwidth * unit_N, ordering_NW_centre - ordering_NW_halfwidth * unit_N, ordering_NW_halfdepth)

bar_NW_foci     = [(bar_ordering_shape, SVD.black), (bar_seating_shape, SVD.black)]
bar_NW_barriers = []
bar_NW          = SMTAct.create_bar(generate_node_id(), bar_NW_vestibule_shape, SVD.black, bar_NW_foci, bar_NW_barriers)


activity_nodes = [cafe_NE, bar_NW]


# wall_depth  = 4.  # in metres
# wall_radius = 0.5 * wall_depth


#
# Footpath placement
#
footpath_halfwidth = ptngate_halfwidth

footpath_S_shape   = SMGS.Segment(ptngate_WSW.approach_shape.posn, ptngate_ESE.approach_shape.posn, footpath_halfwidth)
footpath_ENE_shape = SMGS.Segment(ptngate_ENE.approach_shape.posn, ptnwypnt_ENE.approach_shape.posn, footpath_halfwidth)
footpath_NNE_shape = SMGS.Segment(ptngate_NNE.approach_shape.posn, ptnwypnt_NNE.approach_shape.posn, footpath_halfwidth)
footpath_NNW_shape = SMGS.Segment(ptngate_NNW.approach_shape.posn, ptnwypnt_NNW.approach_shape.posn, footpath_halfwidth)
footpath_WNW_shape = SMGS.Segment(ptngate_WNW.approach_shape.posn, ptnwypnt_WNW.approach_shape.posn, footpath_halfwidth)

footpath_S   = SMZZ.create_footpath(51, footpath_S_shape)
footpath_ENE = SMZZ.create_footpath(52, footpath_ENE_shape)
footpath_NNE = SMZZ.create_footpath(53, footpath_NNE_shape)
footpath_NNW = SMZZ.create_footpath(54, footpath_NNW_shape)
footpath_WNW = SMZZ.create_footpath(55, footpath_WNW_shape)

#
# Roadway placement
#
roadway_halfwidth = cargate_halfwidth

roadway_ESE_shape = SMGS.Segment(cargate_ESE.approach_shape.posn, carwypnt_ESE_exit.approach_shape.posn  + mean_car_halflength * unit_E, roadway_halfwidth)
roadway_ENE_shape = SMGS.Segment(cargate_ENE.approach_shape.posn, carwypnt_ENE_entry.approach_shape.posn + mean_car_halflength * unit_E, roadway_halfwidth)
roadway_NNE_shape = SMGS.Segment(cargate_NNE.approach_shape.posn, carwypnt_NNE_exit.approach_shape.posn  + mean_car_halflength * unit_N, roadway_halfwidth)
roadway_NNW_shape = SMGS.Segment(cargate_NNW.approach_shape.posn, carwypnt_NNW_entry.approach_shape.posn + mean_car_halflength * unit_N, roadway_halfwidth)
roadway_WNW_shape = SMGS.Segment(cargate_WNW.approach_shape.posn, carwypnt_WNW_exit.approach_shape.posn  - mean_car_halflength * unit_E, roadway_halfwidth)
roadway_WSW_shape = SMGS.Segment(cargate_WSW.approach_shape.posn, carwypnt_WSW_entry.approach_shape.posn - mean_car_halflength * unit_E, roadway_halfwidth)

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
SMTN.symconnect(ptngate_ENE,  ptnwypnt_ENE, 5)
SMTN.symconnect(ptngate_NNE,  ptnwypnt_NNE, 5)

SMTN.symconnect(ptngate_NNW,  ptnwypnt_NNW, 5)
SMTN.symconnect(ptngate_WNW,  ptnwypnt_WNW, 5)

SMTN.symconnect(ptngate_ESE,  ptnwypnt_SE, 5)
SMTN.symconnect(ptngate_WSW,  ptnwypnt_SW, 5)

SMTN.symconnect(ptngate_S,   ptnwypnt_S, 5)
# SMTN.symconnect(ptnwypnt_S,  ptnwypnt_SE, 5)
# SMTN.symconnect(ptnwypnt_S,  ptnwypnt_SW, 5)
SMTN.omniconnect([ptnwypnt_ENE,  ptnwypnt_NE, ptnwypnt_NNE, doorway_NE], 5)
SMTN.omniconnect([ptnwypnt_NNW,  ptnwypnt_NW, ptnwypnt_WNW, doorway_NW], 5)
SMTN.omniconnect([ptnwypnt_NE, ptnwypnt_NW, ptnwypnt_SW, ptnwypnt_S, ptnwypnt_SE], 5)

#
# Pedestrian activity-node connections
#
SMTN.symconnect(cafe_NE, doorway_NE,  5)
# SMTN.symconnect(doorway_NE, ptnwypnt_NE, 5)
SMTN.symconnect(bar_NW,  doorway_NW,  5)
# SMTN.symconnect(doorway_NW, ptnwypnt_NW, 5)

#
# One-way car connections
#
cargate_ESE.connect(carwypnt_ESE_exit, 4)
carwypnt_ENE_entry.connect(cargate_ENE, 4)

cargate_NNE.connect(carwypnt_NNE_exit, 4)
carwypnt_NNW_entry.connect(cargate_NNW, 4)

cargate_WNW.connect(carwypnt_WNW_exit, 4)
carwypnt_WSW_entry.connect(cargate_WSW, 4)

carwypnt_ESE_exit.connect(carwypnt_WSW_entry, 3)
carwypnt_WNW_exit.connect(carwypnt_ENE_entry, 3)

carwypnt_ESE_exit.connect(carwypnt_NNW_entry, 1)
carwypnt_NNE_exit.connect(carwypnt_ENE_entry, 2)

carwypnt_NNE_exit.connect(carwypnt_WSW_entry, 0)
carwypnt_WNW_exit.connect(carwypnt_NNW_entry, 2)


ptn_waypoints = [ptnwypnt_ENE, ptnwypnt_NE, ptnwypnt_NNE, ptnwypnt_NNW, ptnwypnt_NW, ptnwypnt_WNW, ptnwypnt_SW, ptnwypnt_S, ptnwypnt_SE, doorway_NE, doorway_NW]
car_waypoints = \
    [carwypnt_ESE_exit,
     carwypnt_ENE_entry,
     carwypnt_NNE_exit,
     carwypnt_NNW_entry,
     carwypnt_WNW_exit,
     carwypnt_WSW_entry]

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

np.random.seed(seed = 1)

if __name__ == "__main__":
    zones = footpaths + roadways # + crossings # + countercrossings # + vbarriers

    model = SMM.Model(width, height, T_simulation, 4, topo_specs, OD_matrices, zones)
    model.register_activity_demand_info(activity_demand_info)
    if SMC.VIEW_ENABLED:
        view  = SVV.View(width, height, screen_size, scale_factor, layer_flags)
    # plaza           = SMM.Model(width, height, screen_size, scale_factor, T_simulation, 4, topo_specs, OD_matrices, zones, [], [], [])
    # plaza.register_activity_demand_info(activity_demand_info)

    SML.log_static_components(model.topology.centroids, model.topology.waypoints, model.topology.activity_nodes, model.zone_config)
    logger = SML.Logger()
    model.register_logger(logger)
    model.register_timer_posn(corner_NW)

    for i in range(L_simulation):
        if SMC.VIEW_ENABLED:
            SCC.ignore_events()

        print("Simulation step ", i, ": start")
        model.spawn_agents()
        model.advance_timestep()
        # assert(isinstance(model, SMM.Model))
        # assert(isinstance(view, SVV.View))
        if SMC.VIEW_ENABLED:
            model.draw(view)
            view.flip()

        model.despawn_agents()
        # SCC.wait(10)
        # print("Simulation step ", i, ": finish")

    SML.log_stream_statistics(plaza)
