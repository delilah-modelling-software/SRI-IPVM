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
import time

#
# Constant across multiple runs.
#
width  = 80
height = 80

scale_factor   = 15
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width,screen_height)

# layer_flags = SVD.LayerFlagset.MACROSCOPIC |SVD.LayerFlagset.GEOMETRY | SVD.LayerFlagset.TOPOLOGY | SVD.LayerFlagset.MICROSCOPIC | SVD.LayerFlagset.INTERACTION | SVD.LayerFlagset.INTRAMODAL | SVD.LayerFlagset.INTERMODAL | SVD.LayerFlagset.CONES | SVD.LayerFlagset.TRAJECTORY
layer_flags = SVD.LayerFlagset.GEOMETRY | SVD.LayerFlagset.TOPOLOGY | SVD.LayerFlagset.MICROSCOPIC | SVD.LayerFlagset.INTERACTION | SVD.LayerFlagset.INTRAMODAL | SVD.LayerFlagset.INTERMODAL | SVD.LayerFlagset.CONES | SVD.LayerFlagset.TRAJECTORY

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
T_simulation = 60 * 60  # 60 minutes
# T_simulation = 15 * 60  # 15 minutes
L_simulation = int(T_simulation / SMC.T_step)

unit_E = np.array([1, 0])
unit_N = np.array([0, 1])

ptnlane_width  = 3.
voidlane_width = 3.
carlane_width  = 3.

ptngate_width  = ptnlane_width
cargate_width  = carlane_width

ptngate_halfwidth = 0.5 * ptngate_width
ptngate_halfdepth = 0.125

#
# The void lane represents:
#   3m for an empty lane where cars are not allowed to park; and
#   0.5m of unusable footpath near the kerb.
#

cargate_halfwidth = 0.5 * cargate_width
cargate_halfdepth = 0.25

safety_margin = 0.5 # 1.
cargate_NS_displacement = (safety_margin + cargate_halfwidth) * unit_N
ptngate_NS_extradisplacement = (cargate_halfwidth + voidlane_width + ptngate_halfwidth) * unit_N
ptngate_NS_displacement = cargate_NS_displacement + ptngate_NS_extradisplacement

cargate_EW_displacement = (safety_margin + cargate_halfwidth) * unit_E
ptngate_EW_extradisplacement = (cargate_halfwidth + voidlane_width + ptngate_halfwidth) * unit_E
ptngate_EW_displacement = cargate_EW_displacement + ptngate_EW_extradisplacement

ptngate_NS_hwvec = ptngate_halfwidth * unit_N
ptngate_EW_hwvec = ptngate_halfwidth * unit_E

mean_car_halflength  = 2.4
mean_car_length      = 4.8

rt2                            = np.sqrt(2)
waypoint_NS_extradisplacement = (ptngate_width + safety_margin) * unit_N
waypoint_EW_extradisplacement = (ptngate_width + safety_margin) * unit_E
# waypoint_NS_displacement = ptngate_NS_displacement + waypoint_NS_extradisplacement
# waypoint_EW_displacement = ptngate_EW_displacement + waypoint_EW_extradisplacement
waypoint_NS_displacement = ptngate_NS_displacement - waypoint_NS_extradisplacement
waypoint_EW_displacement = ptngate_EW_displacement - waypoint_EW_extradisplacement

#
# Diagonal unit vectors
#
inv_rt2 = 0.5 * rt2
unit_SE = inv_rt2 * (unit_E - unit_N)
unit_NE = inv_rt2 * (unit_N + unit_E)
unit_NW = -unit_SE
unit_SW = -unit_NE


#
# Demand Scaling Factor: 0.8, 1., 1.2, 1.4 are the values to be tested
#
dsf = 4. * 1.0
# dsf = 1.
scaled_2 = int(dsf * 2.)
scaled_3 = int(dsf * 3.)
scaled_5 = int(dsf * 5.)
scaled_6 = int(dsf * 6.)
scaled_7 = int(dsf * 7.)
scaled_10 = int(dsf * 10.)


#
# Pedestrian waypoint positions:
#
ptnwypnt_NE_centre  = centre_true + ptngate_NS_displacement + ptngate_EW_displacement
ptnwypnt_NW_centre = centre_true + ptngate_NS_displacement - ptngate_EW_displacement
ptnwypnt_SW_centre = centre_true - ptngate_NS_displacement - ptngate_EW_displacement
ptnwypnt_S_centre = centre_true - ptngate_NS_displacement
ptnwypnt_SE_centre = centre_true - ptngate_NS_displacement + ptngate_EW_displacement

ptnwypnt_NE_E_centre  = ptnwypnt_NE_centre + ptngate_halfwidth * unit_E
ptnwypnt_NE_N_centre  = ptnwypnt_NE_centre + ptngate_halfwidth * unit_N
ptnwypnt_NE_W_centre  = ptnwypnt_NE_centre - ptngate_halfwidth * unit_E
ptnwypnt_NE_S_centre  = ptnwypnt_NE_centre - ptngate_halfwidth * unit_N

ptnwypnt_NW_E_centre  = ptnwypnt_NW_centre + ptngate_halfwidth * unit_E
ptnwypnt_NW_N_centre  = ptnwypnt_NW_centre + ptngate_halfwidth * unit_N
ptnwypnt_NW_W_centre  = ptnwypnt_NW_centre - ptngate_halfwidth * unit_E
ptnwypnt_NW_S_centre  = ptnwypnt_NW_centre - ptngate_halfwidth * unit_N

ptnwypnt_SW_E_centre  = ptnwypnt_SW_centre + ptngate_halfwidth * unit_E
ptnwypnt_SW_N_centre  = ptnwypnt_SW_centre + ptngate_halfwidth * unit_N
ptnwypnt_SW_W_centre  = ptnwypnt_SW_centre - ptngate_halfwidth * unit_E
ptnwypnt_SW_S_centre  = ptnwypnt_SW_centre - ptngate_halfwidth * unit_N

ptnwypnt_S_E_centre  = ptnwypnt_S_centre + cargate_halfwidth * unit_E
ptnwypnt_S_N_centre  = ptnwypnt_S_centre + ptngate_halfwidth * unit_N
ptnwypnt_S_W_centre  = ptnwypnt_S_centre - cargate_halfwidth * unit_E
ptnwypnt_S_S_centre  = ptnwypnt_S_centre - ptngate_halfwidth * unit_N

ptnwypnt_SE_E_centre  = ptnwypnt_SE_centre + ptngate_halfwidth * unit_E
ptnwypnt_SE_N_centre  = ptnwypnt_SE_centre + ptngate_halfwidth * unit_N
ptnwypnt_SE_W_centre  = ptnwypnt_SE_centre - ptngate_halfwidth * unit_E
ptnwypnt_SE_S_centre  = ptnwypnt_SE_centre - ptngate_halfwidth * unit_N


#
# Vehicle waypoint positions:
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

carwypnt_ESE_exit_centre = centre_true - cargate_NS_displacement + ptngate_EW_displacement + carwypnt_safe_offset * unit_E
carwypnt_ENE_entry_centre = centre_true + cargate_NS_displacement + ptngate_EW_displacement + carwypnt_safe_offset * unit_E

carwypnt_NNE_exit_centre = centre_true + ptngate_NS_displacement + cargate_EW_displacement + + carwypnt_safe_offset * unit_N
carwypnt_NNW_entry_centre = centre_true + ptngate_NS_displacement - cargate_EW_displacement + carwypnt_safe_offset * unit_N

carwypnt_WNW_exit_centre = centre_true + cargate_NS_displacement - ptngate_EW_displacement - carwypnt_safe_offset * unit_E
carwypnt_WSW_entry_centre = centre_true - cargate_NS_displacement - ptngate_EW_displacement - carwypnt_safe_offset * unit_E

#
# Set up nodes
#
next_node_id = 0
sup_node_id = 45  # 32 # 28 # 128
def generate_node_id():
    global next_node_id
    if next_node_id == sup_node_id:
        assert (False)
        return -1

    node_id = next_node_id
    next_node_id += 1
    return node_id

def reset_node_ids():
    global next_node_id
    next_node_id = 0

def construct_model():
    reset_node_ids()

    spawner_uniform = SMDis.DistributorUniform()

    #
    # Pedestrian centroids
    #
    ptngate_ESE = SMTC.create_oblong_gate(generate_node_id(), centre_E - ptngate_NS_displacement, ptngate_NS_hwvec,
                                          ptngate_halfdepth, SVD.colour(0, 0, 128), SVD.colour(0, 0, 255),
                                          spawner_uniform, SMAD.AgentType.PEDESTRIAN)
    ptngate_ENE = SMTC.create_oblong_gate(generate_node_id(), centre_E + ptngate_NS_displacement, ptngate_NS_hwvec,
                                          ptngate_halfdepth, SVD.colour(0, 0, 128), SVD.colour(0, 0, 255),
                                          spawner_uniform, SMAD.AgentType.PEDESTRIAN)
    ptngate_NNE = SMTC.create_oblong_gate(generate_node_id(), centre_N + ptngate_EW_displacement, ptngate_EW_hwvec,
                                          ptngate_halfdepth, SVD.colour(0, 0, 128), SVD.colour(0, 0, 255),
                                          spawner_uniform, SMAD.AgentType.PEDESTRIAN)
    ptngate_NNW = SMTC.create_oblong_gate(generate_node_id(), centre_N - ptngate_EW_displacement, ptngate_EW_hwvec,
                                          ptngate_halfdepth, SVD.colour(0, 0, 128), SVD.colour(0, 0, 255),
                                          spawner_uniform, SMAD.AgentType.PEDESTRIAN)
    ptngate_WNW = SMTC.create_oblong_gate(generate_node_id(), centre_W + ptngate_NS_displacement, ptngate_NS_hwvec,
                                          ptngate_halfdepth, SVD.colour(0, 0, 128), SVD.colour(0, 0, 255),
                                          spawner_uniform, SMAD.AgentType.PEDESTRIAN)
    ptngate_WSW = SMTC.create_oblong_gate(generate_node_id(), centre_W - ptngate_NS_displacement, ptngate_NS_hwvec,
                                          ptngate_halfdepth, SVD.colour(0, 0, 128), SVD.colour(0, 0, 255),
                                          spawner_uniform, SMAD.AgentType.PEDESTRIAN)

    ptngate_S = SMTC.create_oblong_gate(generate_node_id(), centre_S, ptngate_EW_hwvec, ptngate_halfdepth,
                                        SVD.colour(0, 128, 0), SVD.colour(0, 0, 255), spawner_uniform,
                                        SMAD.AgentType.PEDESTRIAN)

    ptngates = [ptngate_ESE, ptngate_ENE, ptngate_NNE, ptngate_NNW, ptngate_WNW, ptngate_WSW, ptngate_S]

    #
    # Vehicle centroids
    #
    cargate_ESE = SMTC.create_round_gate(generate_node_id(), centre_E - cargate_NS_displacement, cargate_halfwidth,
                                         SVD.colour(170, 0, 170), SVD.colour(255, 0, 255), spawner_uniform,
                                         SMAD.AgentType.CAR)
    cargate_ENE = SMTC.create_round_gate(generate_node_id(), centre_E + cargate_NS_displacement, cargate_halfwidth,
                                         SVD.colour(170, 0, 170), SVD.colour(255, 0, 255), spawner_uniform,
                                         SMAD.AgentType.CAR)
    cargate_NNE = SMTC.create_round_gate(generate_node_id(), centre_N + cargate_EW_displacement, cargate_halfwidth,
                                         SVD.colour(170, 0, 170), SVD.colour(255, 0, 255), spawner_uniform,
                                         SMAD.AgentType.CAR)
    cargate_NNW = SMTC.create_round_gate(generate_node_id(), centre_N - cargate_EW_displacement, cargate_halfwidth,
                                         SVD.colour(170, 0, 170), SVD.colour(255, 0, 255), spawner_uniform,
                                         SMAD.AgentType.CAR)
    cargate_WNW = SMTC.create_round_gate(generate_node_id(), centre_W + cargate_NS_displacement, cargate_halfwidth,
                                         SVD.colour(170, 0, 170), SVD.colour(255, 0, 255), spawner_uniform,
                                         SMAD.AgentType.CAR)
    cargate_WSW = SMTC.create_round_gate(generate_node_id(), centre_W - cargate_NS_displacement, cargate_halfwidth,
                                         SVD.colour(170, 0, 170), SVD.colour(255, 0, 255), spawner_uniform,
                                         SMAD.AgentType.CAR)

    cargates = [cargate_ESE, cargate_ENE, cargate_NNE, cargate_NNW, cargate_WNW, cargate_WSW]

    #
    # Origin-destination pairs:
    #   Note: Parking, U-turns, and 3-/5-/7-point turns are neglected
    #
    ptn_centroids = ptngates
    car_centroids = cargates
    pedestrian_OD_matrix = \
        np.array(
            [[0, scaled_3, scaled_3, scaled_3, scaled_3, scaled_3, scaled_3],
             [scaled_3, 0, scaled_3, scaled_3, scaled_3, scaled_3, scaled_3],
             [scaled_3, scaled_3, 0, scaled_3, scaled_3, scaled_3, scaled_3],
             [scaled_3, scaled_3, scaled_3, 0, scaled_3, scaled_3, scaled_3],
             [scaled_3, scaled_3, scaled_3, scaled_3, 0, scaled_3, scaled_3],
             [scaled_3, scaled_3, scaled_3, scaled_3, scaled_3, 0, scaled_3],
             [scaled_3, scaled_3, scaled_3, scaled_3, scaled_3, scaled_3, 0]])
    six_zeroes = [0, 0, 0, 0, 0, 0]
    car_OD_matrix = \
        np.array(
            [[0, 0, 0, scaled_5, 0, scaled_3],
             six_zeroes,
             [0, scaled_6, 0, 0, 0, scaled_2],
             six_zeroes,
             [0, scaled_7, 0, scaled_10, 0, 0],
             six_zeroes])

    STRESSTEST = False
    if STRESSTEST:
        stress_factor = 3
        pedestrian_OD_matrix *= stress_factor
        car_OD_matrix *= stress_factor

    #
    # Pedestrian waypoints
    #
    ptnwypnt_NE_E = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_NE_E_centre, unit_N, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnt_NE_N = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_NE_N_centre, unit_E, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnt_NE_W = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_NE_W_centre, unit_N, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnt_NE_S = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_NE_S_centre, unit_E, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnts_NE = [ptnwypnt_NE_E, ptnwypnt_NE_N, ptnwypnt_NE_W, ptnwypnt_NE_S]

    ptnwypnt_NW_E = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_NW_E_centre, unit_N, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnt_NW_N = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_NW_N_centre, unit_E, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnt_NW_W = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_NW_W_centre, unit_N, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnt_NW_S = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_NW_S_centre, unit_E, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnts_NW = [ptnwypnt_NW_E, ptnwypnt_NW_N, ptnwypnt_NW_W, ptnwypnt_NW_S]

    ptnwypnt_SW_E = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_SW_E_centre, unit_N, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnt_SW_N = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_SW_N_centre, unit_E, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnt_SW_W = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_SW_W_centre, unit_N, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    # ptnwypnt_SW_S = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_SW_S_centre, unit_E, ptngate_halfwidth, ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnts_SW = [ptnwypnt_SW_E, ptnwypnt_SW_N, ptnwypnt_SW_W]

    ptnwypnt_S_E = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_S_E_centre, unit_N, ptngate_halfwidth,
                                                 ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnt_S_N = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_S_N_centre, unit_E, ptngate_halfwidth,
                                                 ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnt_S_W = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_S_W_centre, unit_N, ptngate_halfwidth,
                                                 ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnt_S_S = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_S_S_centre, unit_E, ptngate_halfwidth,
                                                 ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnts_S = [ptnwypnt_S_E, ptnwypnt_S_N, ptnwypnt_S_W, ptnwypnt_S_S]

    ptnwypnt_SE_E = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_SE_E_centre, unit_N, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnt_SE_N = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_SE_N_centre, unit_E, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnt_SE_W = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_SE_W_centre, unit_N, ptngate_halfwidth,
                                                  ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    # ptnwypnt_SE_S = SMTApp.create_oblong_waypoint(generate_node_id(), ptnwypnt_SE_S_centre, unit_E, ptngate_halfwidth, ptngate_halfdepth, ptngate_halfdepth, SMAD.AgentType.PEDESTRIAN)
    ptnwypnts_SE = [ptnwypnt_SE_E, ptnwypnt_SE_N, ptnwypnt_SE_W]

    #
    # Vehicle waypoints
    #
    carwypnt_ESE_exit = \
        SMTApp.create_oblong_waypoint(generate_node_id(), carwypnt_ESE_exit_centre, unit_E, mean_car_halflength,
                                      cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)
    carwypnt_ENE_entry = \
        SMTApp.create_oblong_waypoint(generate_node_id(), carwypnt_ENE_entry_centre, unit_E, mean_car_halflength,
                                      cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)
    carwypnt_NNE_exit = \
        SMTApp.create_oblong_waypoint(generate_node_id(), carwypnt_NNE_exit_centre, unit_N, mean_car_halflength,
                                      cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)
    carwypnt_NNW_entry = \
        SMTApp.create_oblong_waypoint(generate_node_id(), carwypnt_NNW_entry_centre, unit_N, mean_car_halflength,
                                      cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)
    carwypnt_WNW_exit = \
        SMTApp.create_oblong_waypoint(generate_node_id(), carwypnt_WNW_exit_centre, unit_E, mean_car_halflength,
                                      cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)
    carwypnt_WSW_entry = \
        SMTApp.create_oblong_waypoint(generate_node_id(), carwypnt_WSW_entry_centre, unit_E, mean_car_halflength,
                                      cargate_halfwidth, cargate_halfdepth, SMAD.AgentType.CAR)

    #
    # Set up activity nodes.
    #

    activity_probability = 0.1
    activity_popularities = [("Sightseeing", 1.)]
    activity_demand_info = SMA.DemandInfo(activity_probability, activity_popularities)

    vestibule_Swards_displacement = 0.5 * (centre_S - centre_true)
    vestibule_EW_displacement = ptngate_EW_displacement

    vestibule_SW_centre = centre_true + vestibule_Swards_displacement - vestibule_EW_displacement
    sightseeing_SW_centre = vestibule_SW_centre - ptngate_width * unit_E

    garden_SW_vestibule_shape = SMGS.create_segment(vestibule_SW_centre, unit_N, 2 * ptngate_width, ptngate_halfdepth)
    garden_SW_sightseeing_shape = SMGS.create_segment(sightseeing_SW_centre, unit_N, 2 * ptngate_width,
                                                      ptngate_halfdepth)
    garden_SW_foci = [(garden_SW_sightseeing_shape, SVD.green)]
    garden_SW = SMTAct.create_garden(generate_node_id(), garden_SW_vestibule_shape, SVD.black, garden_SW_foci, [])

    vestibule_SE_centre = centre_true + vestibule_Swards_displacement + vestibule_EW_displacement
    sightseeing_SE_centre = vestibule_SE_centre + ptngate_width * unit_E

    garden_SE_vestibule_shape = SMGS.create_segment(vestibule_SE_centre, unit_N, 2 * ptngate_width, ptngate_halfdepth)
    garden_SE_sightseeing_shape = SMGS.create_segment(sightseeing_SE_centre, unit_N, 2 * ptngate_width,
                                                      ptngate_halfdepth)
    garden_SE_foci = [(garden_SE_sightseeing_shape, SVD.green)]
    garden_SE = SMTAct.create_garden(generate_node_id(), garden_SE_vestibule_shape, SVD.black, garden_SE_foci, [])

    activity_nodes = [garden_SW, garden_SE]  # [cafe_NE, bar_NW]

    #
    # Footpath placement
    #
    footpath_halfwidth = ptngate_halfwidth

    footpath_S_shape = SMGS.Segment(ptngate_WSW.approach_shape.posn, ptngate_ESE.approach_shape.posn,
                                    footpath_halfwidth)
    footpath_ENE_shape = SMGS.Segment(ptngate_ENE.approach_shape.posn, ptnwypnt_NE_centre, footpath_halfwidth)
    footpath_NNE_shape = SMGS.Segment(ptngate_NNE.approach_shape.posn, ptnwypnt_NE_centre, footpath_halfwidth)
    footpath_NNW_shape = SMGS.Segment(ptngate_NNW.approach_shape.posn, ptnwypnt_NW_centre, footpath_halfwidth)
    footpath_WNW_shape = SMGS.Segment(ptngate_WNW.approach_shape.posn, ptnwypnt_NW_centre, footpath_halfwidth)

    footpath_S = SMZZ.create_footpath(51, footpath_S_shape)
    footpath_ENE = SMZZ.create_footpath(52, footpath_ENE_shape)
    footpath_NNE = SMZZ.create_footpath(53, footpath_NNE_shape)
    footpath_NNW = SMZZ.create_footpath(54, footpath_NNW_shape)
    footpath_WNW = SMZZ.create_footpath(55, footpath_WNW_shape)

    #
    # Roadway placement
    #
    roadway_halfwidth = cargate_halfwidth

    roadway_ESE_shape = SMGS.Segment(cargate_ESE.approach_shape.posn,
                                     carwypnt_ESE_exit.approach_shape.posn + mean_car_halflength * unit_E,
                                     roadway_halfwidth)
    roadway_ENE_shape = SMGS.Segment(cargate_ENE.approach_shape.posn,
                                     carwypnt_ENE_entry.approach_shape.posn + mean_car_halflength * unit_E,
                                     roadway_halfwidth)
    roadway_NNE_shape = SMGS.Segment(cargate_NNE.approach_shape.posn,
                                     carwypnt_NNE_exit.approach_shape.posn + mean_car_halflength * unit_N,
                                     roadway_halfwidth)
    roadway_NNW_shape = SMGS.Segment(cargate_NNW.approach_shape.posn,
                                     carwypnt_NNW_entry.approach_shape.posn + mean_car_halflength * unit_N,
                                     roadway_halfwidth)
    roadway_WNW_shape = SMGS.Segment(cargate_WNW.approach_shape.posn,
                                     carwypnt_WNW_exit.approach_shape.posn - mean_car_halflength * unit_E,
                                     roadway_halfwidth)
    roadway_WSW_shape = SMGS.Segment(cargate_WSW.approach_shape.posn,
                                     carwypnt_WSW_entry.approach_shape.posn - mean_car_halflength * unit_E,
                                     roadway_halfwidth)

    roadway_ESE = SMZZ.create_roadway(61, roadway_ESE_shape)
    roadway_ENE = SMZZ.create_roadway(62, roadway_ENE_shape)
    roadway_NNE = SMZZ.create_roadway(63, roadway_NNE_shape)
    roadway_NNW = SMZZ.create_roadway(64, roadway_NNW_shape)
    roadway_WNW = SMZZ.create_roadway(65, roadway_WNW_shape)
    roadway_WSW = SMZZ.create_roadway(66, roadway_WSW_shape)

    #
    # Pedestrian symmetric centroid <-> waypoint connections
    #
    SMTN.symconnect(ptngate_ENE, ptnwypnt_NE_E, 5)
    SMTN.symconnect(ptngate_NNE, ptnwypnt_NE_N, 5)

    SMTN.symconnect(ptngate_NNW, ptnwypnt_NW_N, 5)
    SMTN.symconnect(ptngate_WNW, ptnwypnt_NW_W, 5)

    SMTN.symconnect(ptngate_ESE, ptnwypnt_SE_E, 5)
    SMTN.symconnect(ptngate_WSW, ptnwypnt_SW_W, 5)

    SMTN.symconnect(ptngate_S, ptnwypnt_S_S, 5)

    SMTN.symconnect(ptnwypnt_SW_E, ptnwypnt_S_W, 5)
    SMTN.symconnect(ptnwypnt_SE_W, ptnwypnt_S_E, 5)

    SMTN.symconnect(ptnwypnt_NE_W, ptnwypnt_NW_E, 5)

    SMTN.symconnect(ptnwypnt_NE_S, ptnwypnt_SE_N, 5)
    SMTN.symconnect(ptnwypnt_NE_S, ptnwypnt_S_N, 5)
    SMTN.symconnect(ptnwypnt_NW_S, ptnwypnt_SW_N, 5)
    SMTN.symconnect(ptnwypnt_NW_S, ptnwypnt_S_N, 5)

    SMTN.omniconnect([ptnwypnt_NE_E, ptnwypnt_NE_N, ptnwypnt_NE_W, ptnwypnt_NE_S], 5)
    SMTN.omniconnect([ptnwypnt_NW_E, ptnwypnt_NW_N, ptnwypnt_NW_W, ptnwypnt_NW_S], 5)
    SMTN.omniconnect([ptnwypnt_SW_E, ptnwypnt_SW_N, ptnwypnt_SW_W], 5)
    SMTN.omniconnect([ptnwypnt_S_E, ptnwypnt_S_N, ptnwypnt_S_W, ptnwypnt_S_S], 5)
    SMTN.omniconnect([ptnwypnt_SE_E, ptnwypnt_SE_N, ptnwypnt_SE_W], 5)

    SMTN.symconnect(garden_SW, ptnwypnt_S_S, 5)
    SMTN.symconnect(garden_SW, ptngate_S, 5)
    SMTN.symconnect(garden_SE, ptnwypnt_S_S, 5)
    SMTN.symconnect(garden_SE, ptngate_S, 5)
    SMTN.symconnect(garden_SW, garden_SE, 5)

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

    ptn_waypoints = ptnwypnts_NE + ptnwypnts_NW + ptnwypnts_SW + ptnwypnts_S + ptnwypnts_SE
    car_waypoints = \
        [carwypnt_ESE_exit,
         carwypnt_ENE_entry,
         carwypnt_NNE_exit,
         carwypnt_NNW_entry,
         carwypnt_WNW_exit,
         carwypnt_WSW_entry]

    ptn_topo_spec = SMTT.AgentNodeConfig(ptn_centroids, ptn_waypoints, activity_nodes, SMAD.AgentType.PEDESTRIAN)
    car_topo_spec = SMTT.AgentNodeConfig(car_centroids, car_waypoints, [], SMAD.AgentType.CAR)
    topo_specs = [ptn_topo_spec, car_topo_spec]

    OD_matrices = {SMAD.AgentType.PEDESTRIAN: pedestrian_OD_matrix, SMAD.AgentType.CAR: car_OD_matrix}

    footpaths = [footpath_S, footpath_ENE, footpath_NNE, footpath_NNW, footpath_WNW]
    roadways = [roadway_ESE, roadway_ENE, roadway_NNE, roadway_NNW, roadway_WNW, roadway_WSW]
    zones = footpaths + roadways  # + crossings # + countercrossings # + vbarriers

    model = SMM.Model(width, height, T_simulation, 4, topo_specs, OD_matrices, zones)
    model.register_activity_demand_info(activity_demand_info)

    return model

if __name__ == "__main__":
    # np.random.seed(42)
    # seeds = []
    # for i in range(8):
        # seed = np.random.randint(low = 0, high = 0xffffffff, dtype = "uint32")
        # seeds += [seed]
    # print(seeds)

    if SMC.VIEW_ENABLED:
        view = SVV.View(width, height, screen_size, scale_factor, layer_flags)

    assert len(sys.argv) == 2
    if len(sys.argv) == 2:
        seed = int(sys.argv[1])

        np.random.seed(seed)

        model = construct_model()

        SML.log_static_components(model.topology.centroids, model.topology.waypoints, model.topology.activity_nodes, model.zone_config, seed)
        logger = SML.Logger(seed)
        model.register_logger(logger)
        model.register_timer_posn(corner_NW)

        start_time = time.perf_counter()

        for i in range(L_simulation):
            if SMC.VIEW_ENABLED:
                SCC.ignore_events()

            print("Simulation step ", i, ": start")
            model.spawn_agents()
            model.advance_timestep()
            # assert(isinstance(model, SMM.Model))
            if SMC.VIEW_ENABLED:
                # assert (isinstance(view, SVV.View))
                model.draw(view)
                view.flip()

            model.despawn_agents()
            # SCC.wait(10)
            # print("Simulation step ", i, ": finish")

        end_time = time.perf_counter()
        elapsed  = end_time - start_time
        print("Elapsed time {}s".format(elapsed))

        SML.log_stream_statistics(model, seed)
        SML.log_safety_measures(model, seed)
