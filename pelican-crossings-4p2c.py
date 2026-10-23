import sys
import numpy as np
import scipy as sp
# import scipy.stats as ss

import view.view as SVV
import model.numerical.defs as SMND
# import model.force as SMF
import model.distributor as SMDis
import model.topology.node as SMTN
import model.topology.centroid as SMTC
import model.topology.topology as SMTT
import model.topology.approachnode as SMTApp
import model.agency.defs as SMAD
import model.model as SMM
import model.geometry.shape as SMGS
# import model.zone.defs as SMZD
import model.zone.zone as SMZZ
import model.logger as SML
# import model.agency.agent as smaa

width  = 100 # 75
height = 30

scale_factor   = 15
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width,screen_height)

# layer_flags = SVV.ViewLayerFlagset.GEOMETRY | SVV.ViewLayerFlagset.TOPOLOGY | SVV.ViewLayerFlagset.MACROSCOPIC | SVV.ViewLayerFlagset.MICROSCOPIC
# layer_flags = SVV.ViewLayerFlagset.GEOMETRY | SVV.ViewLayerFlagset.TOPOLOGY | SVV.ViewLayerFlagset.MICROSCOPIC | SVV.ViewLayerFlagset.ATTRACTION
layer_flags = SVV.ViewLayerFlagset.GEOMETRY | SVV.ViewLayerFlagset.TOPOLOGY | SVV.ViewLayerFlagset.MICROSCOPIC | SVV.ViewLayerFlagset.ATTRACTION | SVV.ViewLayerFlagset.REPULSION
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

spawner_NE  = SMDis.DistributorUniform()
spawner_NW  = SMDis.DistributorUniform()
spawner_SW  = SMDis.DistributorUniform()
spawner_SE  = SMDis.DistributorUniform()

unit_E = np.array([1, 0])
unit_N = np.array([0, 1])

ptngate_EW_offset = 10 * unit_E
centre_NNE = centre_N + ptngate_EW_offset
centre_NNW = centre_N - ptngate_EW_offset
centre_SSW = centre_S - ptngate_EW_offset
centre_SSE = centre_S + ptngate_EW_offset

ptncentre_E = centre_true + ptngate_EW_offset
ptncentre_W = centre_true - ptngate_EW_offset

ptngate_width     = 3.
ptngate_halfwidth = 0.5 * ptngate_width
ptngate_halfdepth = 0.125

cargate_width     = 4.
cargate_halfwidth = 0.5 * cargate_width
cargate_halfdepth = 0.25

carspawner_E = SMDis.DistributorUniform()
carspawner_W = SMDis.DistributorUniform()

safety_margin = 0.5 # 1.
cargate_NS_displacement = (cargate_halfwidth + safety_margin) * unit_N
ptngate_NS_extradisplacement = (ptngate_width + cargate_halfwidth + safety_margin) * unit_N
ptngate_NS_displacement = cargate_NS_displacement + ptngate_NS_extradisplacement

ptngate_NE_shape = SMGS.Segment(centre_NNE - ptngate_halfwidth * unit_E, centre_NNE + ptngate_halfwidth * unit_E, ptngate_halfdepth)
ptngate_NW_shape = SMGS.Segment(centre_NNW - ptngate_halfwidth * unit_E, centre_NNW + ptngate_halfwidth * unit_E, ptngate_halfdepth)
ptngate_SW_shape = SMGS.Segment(centre_SSW - ptngate_halfwidth * unit_E, centre_SSW + ptngate_halfwidth * unit_E, ptngate_halfdepth)
ptngate_SE_shape = SMGS.Segment(centre_SSE - ptngate_halfwidth * unit_E, centre_SSE + ptngate_halfwidth * unit_E, ptngate_halfdepth)
ptngate_NE  = SMTC.Centroid(1, ptngate_NE_shape, SVV.colour(114,  56,   0), SVV.colour(170,  85,   0), spawner_NE, agent_type = SMAD.AgentType.PEDESTRIAN)
ptngate_NW  = SMTC.Centroid(2, ptngate_NW_shape, SVV.colour(114,  56,   0), SVV.colour(170,  85,   0), spawner_NW, agent_type = SMAD.AgentType.PEDESTRIAN)
ptngate_SW  = SMTC.Centroid(3, ptngate_SW_shape, SVV.colour(114,   0,  56), SVV.colour(170,   0,  85), spawner_SW, agent_type = SMAD.AgentType.PEDESTRIAN)
ptngate_SE  = SMTC.Centroid(4, ptngate_SE_shape, SVV.colour(114,   0,  56), SVV.colour(170,   0,  85), spawner_SE, agent_type = SMAD.AgentType.PEDESTRIAN)


cargate_NE_shape = SMGS.Segment(centre_E + cargate_NS_displacement + cargate_halfwidth * unit_N, centre_E + cargate_NS_displacement - cargate_halfwidth * unit_N, cargate_halfdepth)
cargate_NW_shape = SMGS.Segment(centre_W + cargate_NS_displacement + cargate_halfwidth * unit_N, centre_W + cargate_NS_displacement - cargate_halfwidth * unit_N, cargate_halfdepth)
cargate_SW_shape = SMGS.Segment(centre_W - cargate_NS_displacement + cargate_halfwidth * unit_N, centre_W - cargate_NS_displacement - cargate_halfwidth * unit_N, cargate_halfdepth)
cargate_SE_shape = SMGS.Segment(centre_E - cargate_NS_displacement + cargate_halfwidth * unit_N, centre_E - cargate_NS_displacement - cargate_halfwidth * unit_N, cargate_halfdepth)
cargate_NE = SMTC.Centroid(21, cargate_NE_shape, SVV.colour(170,   0, 170), SVV.colour(255,   0, 255), carspawner_E, agent_type = SMAD.AgentType.CAR)
cargate_NW = SMTC.Centroid(22, cargate_NW_shape, SVV.colour(170,   0, 170), SVV.colour(255,   0, 255), carspawner_E, agent_type = SMAD.AgentType.CAR)
cargate_SW = SMTC.Centroid(23, cargate_SW_shape, SVV.colour(  0, 170, 170), SVV.colour(  0,   0, 255), carspawner_W, agent_type = SMAD.AgentType.CAR)
cargate_SE = SMTC.Centroid(24, cargate_SE_shape, SVV.colour(  0, 170, 170), SVV.colour(  0,   0, 255), carspawner_W, agent_type = SMAD.AgentType.CAR)

ptn_centroids        = [ptngate_NE, ptngate_NW, ptngate_SW, ptngate_SE]
car_centroids        = [cargate_NE, cargate_NW, cargate_SW, cargate_SE]
pedestrian_OD_matrix = 9 * np.array([[0, 5, 10, 10], [5, 0, 10, 10], [10, 10, 0, 5], [10, 10, 5, 0]])
car_OD_matrix        = 15 * np.array([[0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0]])



wall_depth  = 4.  # in metres
wall_radius = 0.5 * wall_depth

# building_NS_extradisplacement = (gate_radius + safety_margin + wall_radius) * unit_N
# building_NS_displacement = ptngate_NS_displacement + building_NS_extradisplacement
# building_EW_displacement = (gate_radius + safety_margin + wall_radius) * unit_E

ctrcrossing_EW_displacement = (ptngate_halfwidth + cargate_halfwidth) * unit_E

#
# Declare a few other zones:
#   two (or possibly four) footpaths;
#   four roadways; and
#   one crossing.
#
footpath_halfwidth = ptngate_halfwidth
footpath_NE_shape = SMGS.Segment(centre_NNE, ptncentre_E + ptngate_NS_displacement, footpath_halfwidth)
footpath_NW_shape = SMGS.Segment(centre_NNW, ptncentre_W + ptngate_NS_displacement, footpath_halfwidth)

roadway_halfwidth = cargate_halfwidth
roadway_NW_shape  = SMGS.Segment(cargate_NW.approach_shape.posn, ptncentre_W + cargate_NS_displacement - ctrcrossing_EW_displacement, roadway_halfwidth)
roadway_NE_shape  = SMGS.Segment(ptncentre_E + cargate_NS_displacement + ctrcrossing_EW_displacement, cargate_NE.approach_shape.posn, roadway_halfwidth)

roadway_SW_shape  = SMGS.Segment(cargate_SW.approach_shape.posn, ptncentre_W - cargate_NS_displacement - ctrcrossing_EW_displacement, roadway_halfwidth)
roadway_SE_shape  = SMGS.Segment(ptncentre_E - cargate_NS_displacement + ctrcrossing_EW_displacement, cargate_SE.approach_shape.posn, roadway_halfwidth)

footpath_SW_shape = SMGS.Segment(centre_SSW, ptncentre_W - ptngate_NS_displacement, footpath_halfwidth)
footpath_SE_shape = SMGS.Segment(centre_SSE, ptncentre_E - ptngate_NS_displacement, footpath_halfwidth)

crossing_E_shape = SMGS.Segment(ptncentre_E + ptngate_NS_displacement - footpath_halfwidth * unit_N, ptncentre_E - ptngate_NS_displacement + footpath_halfwidth * unit_N, footpath_halfwidth)
crossing_W_shape = SMGS.Segment(ptncentre_W + ptngate_NS_displacement - footpath_halfwidth * unit_N, ptncentre_W - ptngate_NS_displacement + footpath_halfwidth * unit_N, footpath_halfwidth)

# ctrcrossing_NC_centre = centre_true + cargate_NS_displacement
# ctrcrossing_SC_centre = centre_true - cargate_NS_displacement

epsilon           = 0 # 0.125
# ctrcrossing_halfW = roadway_halfwidth - epsilon

# ctrcrossing_NC_shape = SMGS.Segment(ctrcrossing_NC_centre - ctrcrossing_EW_displacement, ctrcrossing_NC_centre + ctrcrossing_EW_displacement, ctrcrossing_halfW)
# ctrcrossing_SC_shape  = SMGS.Segment(ctrcrossing_SC_centre - ctrcrossing_EW_displacement, ctrcrossing_SC_centre + ctrcrossing_EW_displacement, ctrcrossing_halfW)


footpath_NE = SMZZ.create_footpath(7, footpath_NE_shape)
footpath_NW = SMZZ.create_footpath(8, footpath_NW_shape)
footpath_SW = SMZZ.create_footpath(9, footpath_SW_shape)
footpath_SE = SMZZ.create_footpath(10, footpath_SE_shape)

roadway_NW  = SMZZ.create_roadway(30, roadway_NW_shape)
roadway_NE  = SMZZ.create_roadway(31, roadway_NE_shape)

roadway_SW  = SMZZ.create_roadway(32, roadway_SW_shape)
roadway_SE  = SMZZ.create_roadway(33, roadway_SE_shape)

crossing_E = SMZZ.create_pelican_crossing(41, crossing_E_shape)
crossing_W = SMZZ.create_pelican_crossing(42, crossing_W_shape)
# ctrcrossing_NC = SMZZ.create_counterpelican_turning(51, ctrcrossing_NC_shape, unit_N, unit_E)
# ctrcrossing_SC = SMZZ.create_counterpelican_turning(52, ctrcrossing_SC_shape, -unit_N, -unit_E)

# ptn_stopline_NS_displacement = (ptngate_NS_displacement - 2 * safety_margin) * unit_N # 0.75 * ptngate_NS_displacement
# ptn_stopline_NS_displacement = cargate_NS_displacement + (cargate_halfwidth + safety_margin + ptngate_halfdepth) * unit_N # 0.75 * ptngate_NS_displacement

waypoint_halfdepth = ptngate_halfdepth
waypoint_halfwidth = ptngate_halfwidth - waypoint_halfdepth
waypoint_NE_shape   = SMGS.Segment(ptncentre_E + ptngate_NS_displacement + waypoint_halfwidth * unit_E, ptncentre_E + ptngate_NS_displacement - waypoint_halfwidth * unit_E, waypoint_halfdepth)
waypoint_NW_shape   = SMGS.Segment(ptncentre_W + ptngate_NS_displacement + waypoint_halfwidth * unit_E, ptncentre_W + ptngate_NS_displacement - waypoint_halfwidth * unit_E, waypoint_halfdepth)
waypoint_SW_shape   = SMGS.Segment(ptncentre_W - ptngate_NS_displacement + waypoint_halfwidth * unit_E, ptncentre_W - ptngate_NS_displacement - waypoint_halfwidth * unit_E, waypoint_halfdepth)
waypoint_SE_shape   = SMGS.Segment(ptncentre_E - ptngate_NS_displacement + waypoint_halfwidth * unit_E, ptncentre_E - ptngate_NS_displacement - waypoint_halfwidth * unit_E, waypoint_halfdepth)
waypoint_NE  = SMTApp.create_waypoint(5, waypoint_NE_shape, SMAD.AgentType.PEDESTRIAN)
waypoint_NW  = SMTApp.create_waypoint(6, waypoint_NW_shape, SMAD.AgentType.PEDESTRIAN)
waypoint_SW  = SMTApp.create_waypoint(7, waypoint_SW_shape, SMAD.AgentType.PEDESTRIAN)
waypoint_SE  = SMTApp.create_waypoint(8, waypoint_SE_shape, SMAD.AgentType.PEDESTRIAN)
# waypoint_N  = SMTApp.create_waypoint(5, SMGS.Circle(centre_true + ptngate_NS_displacement, footpath_halfwidth), SMAD.AgentType.PEDESTRIAN)
# waypoint_S  = SMTApp.create_waypoint(6, SMGS.Circle(centre_true - ptngate_NS_displacement, footpath_halfwidth), SMAD.AgentType.PEDESTRIAN)

ptn_stopline_NS_displacement = cargate_NS_displacement + (cargate_halfwidth + ptngate_halfdepth) * unit_N # 0.75 * ptngate_NS_displacement
stopline_NE = SMGS.Segment(ptncentre_E + ptn_stopline_NS_displacement - footpath_halfwidth * unit_E, ptncentre_E + ptn_stopline_NS_displacement + footpath_halfwidth * unit_E, ptngate_halfdepth)
stopline_NW = SMGS.Segment(ptncentre_W + ptn_stopline_NS_displacement - footpath_halfwidth * unit_E, ptncentre_W + ptn_stopline_NS_displacement + footpath_halfwidth * unit_E, ptngate_halfdepth)
stopline_SW = SMGS.Segment(ptncentre_W - ptn_stopline_NS_displacement - footpath_halfwidth * unit_E, ptncentre_W - ptn_stopline_NS_displacement + footpath_halfwidth * unit_E, ptngate_halfdepth)
stopline_SE = SMGS.Segment(ptncentre_E - ptn_stopline_NS_displacement - footpath_halfwidth * unit_E, ptncentre_E - ptn_stopline_NS_displacement + footpath_halfwidth * unit_E, ptngate_halfdepth)
d_focal_ptn = 4 * 1.34 * 0.5 # 4 v T_rlx
approach_node_NE_shape = SMGS.create_segment(ptncentre_E + ptn_stopline_NS_displacement + d_focal_ptn * unit_N, unit_E, footpath_halfwidth, ptngate_halfdepth)
approach_node_NW_shape = SMGS.create_segment(ptncentre_W + ptn_stopline_NS_displacement + d_focal_ptn * unit_N, unit_E, footpath_halfwidth, ptngate_halfdepth)
approach_node_SW_shape = SMGS.create_segment(ptncentre_W - ptn_stopline_NS_displacement - d_focal_ptn * unit_N, unit_E, footpath_halfwidth, ptngate_halfdepth)
approach_node_SE_shape = SMGS.create_segment(ptncentre_E - ptn_stopline_NS_displacement - d_focal_ptn * unit_N, unit_E, footpath_halfwidth, ptngate_halfdepth)
approach_node_NE  = SMTApp.create_signalised_approach_node(9, approach_node_NE_shape, stopline_NE, crossing_E, SMAD.AgentType.PEDESTRIAN)
approach_node_NW  = SMTApp.create_signalised_approach_node(10, approach_node_NW_shape, stopline_NW, crossing_W, SMAD.AgentType.PEDESTRIAN)
approach_node_SW  = SMTApp.create_signalised_approach_node(11, approach_node_SW_shape, stopline_SW, crossing_W, SMAD.AgentType.PEDESTRIAN)
approach_node_SE  = SMTApp.create_signalised_approach_node(12, approach_node_SE_shape, stopline_SE, crossing_E, SMAD.AgentType.PEDESTRIAN)

# As per F. Johansson et al. (2015), within a certain distance of the focal point
# (hmmm, make that focal shape), the desired velocity is reduced so as to bring the agent to a stop at that point.
# The following formula was given in that paper and I have confirmed the mathematical working:
#   d_focal = 4 * v_preferred * T_rlx is ideal for this
# All entries to the approach node's outer shape must be >= d_focal, and d_focal + T_react * v_preferred would be better.
#
# d_focal = 10
d_focal_car = 4 * (25 / 9) * 0.77 # 10km/h = (25/9)m/s, T_rlx = 0.77
d_stopping  = d_focal_car + 0.5 * (25/9)
car_stopline_halfdepth = 0.25 * roadway_halfwidth
car_stopline_halfwidth  = roadway_halfwidth - car_stopline_halfdepth
approach_section_NW_shape = SMGS.Segment(ptncentre_W + cargate_NS_displacement - ctrcrossing_EW_displacement - d_stopping * unit_E, ptncentre_W + cargate_NS_displacement - ctrcrossing_EW_displacement, roadway_halfwidth)
approach_section_NC_shape = SMGS.Segment(ptncentre_E + cargate_NS_displacement - ctrcrossing_EW_displacement - d_stopping * unit_E, ptncentre_E + cargate_NS_displacement - ctrcrossing_EW_displacement, roadway_halfwidth)
approach_section_SC_shape = SMGS.Segment(ptncentre_W - cargate_NS_displacement + ctrcrossing_EW_displacement, ptncentre_W - cargate_NS_displacement + ctrcrossing_EW_displacement + d_stopping * unit_E, roadway_halfwidth)
approach_section_SE_shape = SMGS.Segment(ptncentre_E - cargate_NS_displacement + ctrcrossing_EW_displacement, ptncentre_E - cargate_NS_displacement + ctrcrossing_EW_displacement + d_stopping * unit_E, roadway_halfwidth)
# approach_section_NW_shape = SMGS.Segment(centre_true + cargate_NS_displacement - ctrcrossing_EW_displacement - d_focal * unit_E, centre_true + cargate_NS_displacement - ctrcrossing_EW_displacement, roadway_halfwidth)
# approach_section_SE_shape = SMGS.Segment(centre_true - cargate_NS_displacement + ctrcrossing_EW_displacement, centre_true - cargate_NS_displacement + ctrcrossing_EW_displacement + d_focal * unit_E, roadway_halfwidth)
car_stopline_NW_shape = SMGS.Segment(ptncentre_W + cargate_NS_displacement - ctrcrossing_EW_displacement + car_stopline_halfwidth * unit_N, ptncentre_W + cargate_NS_displacement - ctrcrossing_EW_displacement - car_stopline_halfwidth * unit_N, car_stopline_halfdepth)
car_stopline_NC_shape = SMGS.Segment(ptncentre_E + cargate_NS_displacement - ctrcrossing_EW_displacement + car_stopline_halfwidth * unit_N, ptncentre_E + cargate_NS_displacement - ctrcrossing_EW_displacement - car_stopline_halfwidth * unit_N, car_stopline_halfdepth)
car_stopline_SC_shape = SMGS.Segment(ptncentre_W - cargate_NS_displacement + ctrcrossing_EW_displacement - car_stopline_halfwidth * unit_N, ptncentre_W - cargate_NS_displacement + ctrcrossing_EW_displacement + car_stopline_halfwidth * unit_N, car_stopline_halfdepth)
car_stopline_SE_shape = SMGS.Segment(ptncentre_E - cargate_NS_displacement + ctrcrossing_EW_displacement - car_stopline_halfwidth * unit_N, ptncentre_E - cargate_NS_displacement + ctrcrossing_EW_displacement + car_stopline_halfwidth * unit_N, car_stopline_halfdepth)

approach_section_NW = SMTApp.create_signalised_approach_node(37, approach_section_NW_shape, car_stopline_NW_shape, crossing_W, SMAD.AgentType.CAR)
approach_section_NC = SMTApp.create_signalised_approach_node(38, approach_section_NC_shape, car_stopline_NC_shape, crossing_E, SMAD.AgentType.CAR)
approach_section_SC = SMTApp.create_signalised_approach_node(39, approach_section_SC_shape, car_stopline_SC_shape, crossing_W, SMAD.AgentType.CAR)
approach_section_SE = SMTApp.create_signalised_approach_node(40, approach_section_SE_shape, car_stopline_SE_shape, crossing_E, SMAD.AgentType.CAR)

ptn_waypoints   = [waypoint_NE, waypoint_NW, waypoint_SW, waypoint_SE, approach_node_NE, approach_node_NW, approach_node_SW, approach_node_SE]
car_waypoints   = [approach_section_NW, approach_section_NC, approach_section_SC, approach_section_SE]

#
# Pedestrian symmetric centroid <-> waypoint connections
#
# SMTN.symconnect(ptngate_N,  waypoint_N)
# SMTN.symconnect(ptngate_S,  waypoint_S)

ptngate_NE.connect(approach_node_NE)
# waypoint_NE.connect(ptngate_NE)
SMTN.symconnect(ptngate_NE,  waypoint_NE)

ptngate_NW.connect(approach_node_NW)
# waypoint_NW.connect(ptngate_NW)
SMTN.symconnect(ptngate_NW,  waypoint_NW)

ptngate_SW.connect(approach_node_SW)
# waypoint_SW.connect(ptngate_SW)
SMTN.symconnect(ptngate_SW,  waypoint_SW)

ptngate_SE.connect(approach_node_SE)
# waypoint_SE.connect(ptngate_SE)
SMTN.symconnect(ptngate_SE,  waypoint_SE)

#
# N-S symmetric pedestrian connections
#
# SMTN.symconnect(waypoint_N,   waypoint_S)
approach_node_NE.connect(waypoint_SE)
approach_node_NW.connect(waypoint_SW)
approach_node_SW.connect(waypoint_NW)
approach_node_SE.connect(waypoint_NE)

#
# E-W symmetric pedestrian connections
#
SMTN.symconnect(waypoint_NE,  waypoint_NW)
SMTN.symconnect(waypoint_SW,  waypoint_SE)

#
# E-W one-way car connections
#
# cargate_NW.connect(cargate_NE)
# cargate_SE.connect(cargate_SW)
SMTN.seqconnect([cargate_NW, approach_section_NW, approach_section_NC, cargate_NE])
SMTN.seqconnect([cargate_SE, approach_section_SE, approach_section_SC, cargate_SW])

ptn_topo_spec = SMTT.AgentNodeConfig(ptn_centroids, ptn_waypoints, [], SMAD.AgentType.PEDESTRIAN)
car_topo_spec = SMTT.AgentNodeConfig(car_centroids, car_waypoints, [], SMAD.AgentType.CAR)
topo_specs    = [ptn_topo_spec, car_topo_spec]
# ptn_topology = SMM.PartialTopology(ptn_centroids, ptn_waypoints, [], SMAD.AgentType.PEDESTRIAN)
# car_topology = SMM.PartialTopology(car_centroids, car_waypoints, [], SMAD.AgentType.CAR)
# partial_topologies = [ptn_topology, car_topology]
# partial_topologies = [SMM.PartialTopology(ptn_centroids, ptn_waypoints, 0), SMM.PartialTopology(car_centroids, car_waypoints, 3)]


# topology = mnn.Graph()
# topology.add_nodelist(centroids + waypoints)






OD_matrices = {SMAD.AgentType.PEDESTRIAN: pedestrian_OD_matrix, SMAD.AgentType.CAR: car_OD_matrix}

footpaths        = [footpath_NE, footpath_NW, footpath_SW, footpath_SE]
roadways         = [roadway_NW, roadway_NE, roadway_SW, roadway_SE]
crossings        = [crossing_E, crossing_W]

import controller.controller as SCC

if __name__ == "__main__":
    zones = footpaths + roadways + crossings # + countercrossings # + vbarriers

    plaza           = SMM.Model(width, height, screen_size, scale_factor, T_simulation, 4, topo_specs, OD_matrices, zones, [], [], [])

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
        SCC.wait(20)
        # pg.time.wait(1)
        # print("Simulation step ", i, ": finish")

    SML.log_stream_statistics(plaza)
