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
import model.topology.approachnode as SMTApp
import model.topology.topology as SMTT
import model.agency.defs as SMAD
import model.model as SMM
import model.geometry.shape as SMGS
import model.zone.zone as SMZZ
import model.logger as SML
# import model.agency.agent as SMAA

width  = 110
height =  48

scale_factor   = 14
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width,screen_height)

# layer_flags = SVD.LayerFlagset.GEOMETRY | SVD.LayerFlagset.TOPOLOGY | SVD.LayerFlagset.MACROSCOPIC | SVD.LayerFlagset.MICROSCOPIC
# layer_flags = SVD.LayerFlagset.GEOMETRY | SVD.LayerFlagset.TOPOLOGY | SVD.LayerFlagset.MICROSCOPIC | SVD.LayerFlagset.ATTRACTION
# layer_flags = SVD.LayerFlagset.GEOMETRY | SVD.LayerFlagset.TOPOLOGY | SVD.LayerFlagset.MICROSCOPIC | SVD.LayerFlagset.INTERACTION
# layer_flags = SVD.LayerFlagset.GEOMETRY | SVD.LayerFlagset.TOPOLOGY | SVD.LayerFlagset.MICROSCOPIC | SVD.LayerFlagset.INTERACTION | SVD.LayerFlagset.CONFLICT
layer_flags = SVD.LayerFlagset.GEOMETRY | SVD.LayerFlagset.TOPOLOGY | SVD.LayerFlagset.MICROSCOPIC | SVD.LayerFlagset.INTERACTION | SVD.LayerFlagset.INTERMODAL | SVD.LayerFlagset.TRAJECTORY
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
T_simulation = 6 * 60  # 15 minutes
L_simulation = int(T_simulation / SMND.T_step)

ptngate_width     = 3.
ptngate_halfwidth = 0.5 * ptngate_width
ptngate_halfdepth = 0.125
# gate_diameter = 3.
# gate_radius   = 0.5 * gate_diameter

# spawner_NE = mnsp.Betaspawner(0., 10, 10)
# spawner_N  = mnsp.Uniformspawner()
# spawner_NW = mnsp.Betaspawner(0., 5, 5)
# spawner_SW = mnsp.Betaspawner(0., 10, 10)
# spawner_S  = mnsp.Uniformspawner()
# spawner_SE = mnsp.Betaspawner(0., 5, 5)
# spawner_SE = sst.ClusterSpawner()

spawner_NE = SMDis.DistributorBeta(0., 10, 10)
spawner_N  = SMDis.DistributorUniform()
spawner_NW = SMDis.DistributorBeta(0., 5, 5)
spawner_SW = SMDis.DistributorBeta(0., 10, 10)
spawner_S  = SMDis.DistributorUniform()
spawner_SE = SMDis.DistributorBeta(0., 5, 5)

unit_E = np.array([1, 0])
unit_N = np.array([0, 1])

cargate_diameter = 3.
cargate_radius   = 0.5 * cargate_diameter

# carspawner_E = mnsp.Betaspawner(0., 15, 15)
# carspawner_W = mnsp.Betaspawner(0., 10, 10)
# carspawner_E = mnsp.Uniformspawner()
# carspawner_W = mnsp.Uniformspawner()

carspawner_E = SMDis.DistributorUniform()
carspawner_W = SMDis.DistributorUniform()

safety_margin = 1.
cargate_NS_displacement = (cargate_radius + safety_margin) * unit_N
ptngate_NS_extradisplacement = (ptngate_halfwidth + cargate_radius + safety_margin) * unit_N
ptngate_NS_displacement = cargate_NS_displacement + ptngate_NS_extradisplacement

cargate_NE = SMTC.Centroid(6, SMGS.Circle(centre_E + cargate_NS_displacement, cargate_radius), SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), carspawner_E, agent_type = SMAD.AgentType.CAR)
cargate_NW = SMTC.Centroid(7, SMGS.Circle(centre_W + cargate_NS_displacement, cargate_radius), SVD.colour(170,   0, 170), SVD.colour(255,   0, 255), carspawner_E, agent_type = SMAD.AgentType.CAR)
cargate_SW = SMTC.Centroid(8, SMGS.Circle(centre_W - cargate_NS_displacement, cargate_radius), SVD.colour(  0, 170, 170), SVD.colour(  0,   0, 255), carspawner_W, agent_type = SMAD.AgentType.CAR)
cargate_SE = SMTC.Centroid(9, SMGS.Circle(centre_E - cargate_NS_displacement, cargate_radius), SVD.colour(  0, 170, 170), SVD.colour(  0,   0, 255), carspawner_W, agent_type = SMAD.AgentType.CAR)

ptngate_NE_shape = SMGS.Segment(centre_E + ptngate_NS_displacement + ptngate_halfwidth * unit_N, centre_E + ptngate_NS_displacement - ptngate_halfwidth * unit_N, ptngate_halfdepth)
ptngate_NW_shape = SMGS.Segment(centre_W + ptngate_NS_displacement + ptngate_halfwidth * unit_N, centre_W + ptngate_NS_displacement - ptngate_halfwidth * unit_N, ptngate_halfdepth)
ptngate_SW_shape = SMGS.Segment(centre_W - ptngate_NS_displacement + ptngate_halfwidth * unit_N, centre_W - ptngate_NS_displacement - ptngate_halfwidth * unit_N, ptngate_halfdepth)
ptngate_SE_shape = SMGS.Segment(centre_E - ptngate_NS_displacement + ptngate_halfwidth * unit_N, centre_E - ptngate_NS_displacement - ptngate_halfwidth * unit_N, ptngate_halfdepth)
ptngate_N_shape = SMGS.Segment(centre_N - ptngate_halfwidth * unit_E, centre_N + ptngate_halfwidth * unit_E, ptngate_halfdepth)
ptngate_S_shape = SMGS.Segment(centre_S - ptngate_halfwidth * unit_E, centre_S + ptngate_halfwidth * unit_E, ptngate_halfdepth)

ptngate_NE = SMTC.Centroid(0, ptngate_NE_shape, SVD.colour(  0,   0, 170), SVD.colour(  0,   0, 255), spawner_NE, agent_type = SMAD.AgentType.PEDESTRIAN)
ptngate_N  = SMTC.Centroid(1, ptngate_N_shape, SVD.colour(114,  56,   0), SVD.colour(170,  85,   0), spawner_NE, agent_type = SMAD.AgentType.PEDESTRIAN)
ptngate_NW = SMTC.Centroid(2, ptngate_NW_shape, SVD.colour( 56, 114,   0), SVD.colour( 85, 170,   0), spawner_NW, agent_type = SMAD.AgentType.PEDESTRIAN)
ptngate_SW = SMTC.Centroid(3, ptngate_SW_shape, SVD.colour(170,   0,   0), SVD.colour(255,   0,   0), spawner_SW, agent_type = SMAD.AgentType.PEDESTRIAN)
ptngate_S  = SMTC.Centroid(4, ptngate_S_shape, SVD.colour(114,   0,  56), SVD.colour(170,   0,  85), spawner_SE, agent_type = SMAD.AgentType.PEDESTRIAN)
ptngate_SE = SMTC.Centroid(5, ptngate_SE_shape, SVD.colour( 56,   0, 114), SVD.colour( 85,   0, 170), spawner_SE, agent_type = SMAD.AgentType.PEDESTRIAN)

# ptngate_NE = SMTC.Centroid(0, SMGS.Circle(centre_E + ptngate_NS_displacement, gate_radius), SVD.colour(  0,   0, 170), SVD.colour(  0,   0, 255), spawner_NE, agent_type = SMAD.AgentType.PEDESTRIAN)
# ptngate_N  = SMTC.Centroid(1, SMGS.Circle(centre_N,  gate_radius), SVD.colour(114,  56,   0), SVD.colour(170,  85,   0), spawner_NE, agent_type = SMAD.AgentType.PEDESTRIAN)
# ptngate_NW = SMTC.Centroid(2, SMGS.Circle(centre_W + ptngate_NS_displacement, gate_radius), SVD.colour( 56, 114,   0), SVD.colour( 85, 170,   0), spawner_NW, agent_type = SMAD.AgentType.PEDESTRIAN)
# ptngate_SW = SMTC.Centroid(3, SMGS.Circle(centre_W - ptngate_NS_displacement, gate_radius), SVD.colour(170,   0,   0), SVD.colour(255,   0,   0), spawner_SW, agent_type = SMAD.AgentType.PEDESTRIAN)
# ptngate_S  = SMTC.Centroid(4, SMGS.Circle(centre_S,  gate_radius), SVD.colour(114,   0,  56), SVD.colour(170,   0,  85), spawner_SE, agent_type = SMAD.AgentType.PEDESTRIAN)
# ptngate_SE = SMTC.Centroid(5, SMGS.Circle(centre_E - ptngate_NS_displacement, gate_radius), SVD.colour( 56,   0, 114), SVD.colour( 85,   0, 170), spawner_SE, agent_type = SMAD.AgentType.PEDESTRIAN)


ptn_centroids        = [ptngate_NE, ptngate_N, ptngate_NW, ptngate_SW, ptngate_S, ptngate_SE]
car_centroids        = [cargate_NE, cargate_NW, cargate_SW, cargate_SE]
pedestrian_OD_matrix = 3 * np.array([[0, 5, 5, 10, 5, 5], [5, 0, 5, 5, 10, 5], [5, 5, 0, 5, 5, 10], [10, 5, 5, 0, 5, 5], [5, 10, 5, 5, 0, 5], [5, 5, 10, 5, 5, 0]])
car_OD_matrix        = np.array([[0, 0, 0, 0], [10, 0, 0, 0], [0, 0, 0, 0], [0, 0, 10, 0]])

crossing_EW_displacement = 6 * unit_E

wypt_corner_r  = ptngate_halfdepth # 0.125
wypt_hw        = ptngate_halfwidth # - wypt_corner_r

waypoint_NE_centre = centre_E + (ptngate_NS_displacement - crossing_EW_displacement)
waypoint_NW_centre = centre_W + (ptngate_NS_displacement + crossing_EW_displacement)
waypoint_SW_centre = centre_W - (ptngate_NS_displacement - crossing_EW_displacement)
waypoint_SE_centre = centre_E - (ptngate_NS_displacement + crossing_EW_displacement)
waypoint_NE_shape = SMGS.create_square(waypoint_NE_centre, unit_E, wypt_hw, wypt_corner_r)
waypoint_NW_shape = SMGS.create_square(waypoint_NW_centre, unit_E, wypt_hw, wypt_corner_r)
waypoint_SW_shape = SMGS.create_square(waypoint_SW_centre, unit_E, wypt_hw, wypt_corner_r)
waypoint_SE_shape = SMGS.create_square(waypoint_SE_centre, unit_E, wypt_hw, wypt_corner_r)

waypoint_N_centre = centre_true + ptngate_NS_displacement
waypoint_S_centre = centre_true - ptngate_NS_displacement

waypoint_N_shape = SMGS.create_rect(waypoint_N_centre, unit_E, 2 * wypt_hw, 1.5 * wypt_hw, wypt_corner_r)
waypoint_S_shape = SMGS.create_rect(waypoint_S_centre, unit_E, 2 * wypt_hw, 1.5 * wypt_hw, wypt_corner_r)

waypoint_NE = SMTApp.create_waypoint(10, waypoint_NE_shape, SMAD.AgentType.PEDESTRIAN)
waypoint_NW = SMTApp.create_waypoint(11, waypoint_NW_shape, SMAD.AgentType.PEDESTRIAN)
waypoint_SW = SMTApp.create_waypoint(12, waypoint_SW_shape, SMAD.AgentType.PEDESTRIAN)
waypoint_SE = SMTApp.create_waypoint(13, waypoint_SE_shape, SMAD.AgentType.PEDESTRIAN)
waypoint_N  = SMTApp.create_waypoint(14, waypoint_N_shape, SMAD.AgentType.PEDESTRIAN)
waypoint_S  = SMTApp.create_waypoint(15, waypoint_S_shape, SMAD.AgentType.PEDESTRIAN)
ptn_waypoints   = [waypoint_NE, waypoint_N, waypoint_NW, waypoint_SW, waypoint_S, waypoint_SE]

#
# Pedestrian symmetric centroid <-> waypoint connections
#
SMTN.symconnect(ptngate_NE,  waypoint_NE)
SMTN.symconnect(ptngate_NW,  waypoint_NW)
SMTN.symconnect(ptngate_SW,  waypoint_SW)
SMTN.symconnect(ptngate_SE,  waypoint_SE)
SMTN.symconnect(ptngate_N,  waypoint_N)
SMTN.symconnect(ptngate_S,  waypoint_S)

#
# E-W pedestrian connections
#
SMTN.symconnect(waypoint_NE, waypoint_N)
SMTN.symconnect(waypoint_N,  waypoint_NW)
SMTN.symconnect(waypoint_SW, waypoint_S)
SMTN.symconnect(waypoint_S,  waypoint_SE)

#
# N-S pedestrian connections
#
SMTN.symconnect(waypoint_NE, waypoint_SE)
SMTN.symconnect(waypoint_N,  waypoint_S)
SMTN.symconnect(waypoint_NW, waypoint_SW)

SMTN.interconnect([waypoint_NE, waypoint_N, waypoint_NW], [waypoint_SW, waypoint_S, waypoint_SE])

cargate_NW.connect(cargate_NE)
cargate_SE.connect(cargate_SW)

ptn_topo_spec = SMTT.AgentNodeConfig(ptn_centroids, ptn_waypoints, [], SMAD.AgentType.PEDESTRIAN)
car_topo_spec = SMTT.AgentNodeConfig(car_centroids, [], [], SMAD.AgentType.CAR)
topo_specs    = [ptn_topo_spec, car_topo_spec]
# ptn_topology = SMM.PartialTopology(ptn_centroids, waypoints, [], SMAD.AgentType.PEDESTRIAN)
# car_topology = SMM.PartialTopology(car_centroids, [], [], SMAD.AgentType.CAR)
# partial_topologies = [ptn_topology, car_topology]


# topology = mnn.Graph()
# topology.add_nodelist(centroids + waypoints)

building_offset = ptngate_halfwidth + 3.

wall_depth  = 4.  # 2m
wall_radius = 0.5 * wall_depth

building_NS_extradisplacement = (ptngate_halfwidth + safety_margin + wall_radius) * unit_N
# building_NS_displacement = ptngate_NS_displacement + building_NS_extradisplacement
building_EW_displacement = (ptngate_halfwidth + safety_margin + wall_radius) * unit_E

bldng_NE_pt_0  = ptngate_NE.approach_shape.posn    + building_NS_extradisplacement # building_offset * unit_N
bldng_NE_pt_1  = np.array([ptngate_NE.approach_shape.posn[0], ptngate_N.approach_shape.posn[1]])
bldng_NE_pt_2  = ptngate_N.approach_shape.posn     + building_EW_displacement
bldng_NE_pt_3  = waypoint_N_centre    + building_NS_extradisplacement + building_EW_displacement # building_offset * (unit_E + unit_N)
bldng_NE_pts   = [bldng_NE_pt_0, bldng_NE_pt_1, bldng_NE_pt_2, bldng_NE_pt_3]
bldng_NE_shape = SMGS.Polygon(bldng_NE_pts, wall_radius)

bldng_NW_pt_0  = waypoint_N_centre + building_NS_extradisplacement - building_EW_displacement # building_offset * (unit_E - unit_N)
bldng_NW_pt_1  = ptngate_N.approach_shape.posn     - building_EW_displacement # building_offset * unit_E
bldng_NW_pt_2  = np.array([ptngate_NW.approach_shape.posn[0], ptngate_N.approach_shape.posn[1]])
bldng_NW_pt_3  = ptngate_NW.approach_shape.posn    + building_NS_extradisplacement # building_offset * unit_N
bldng_NW_pts   = [bldng_NW_pt_0, bldng_NW_pt_1, bldng_NW_pt_2, bldng_NW_pt_3]
bldng_NW_shape = SMGS.Polygon(bldng_NW_pts, wall_radius)

bldng_SW_pt_0  = ptngate_S.approach_shape.posn     - building_EW_displacement # building_offset * unit_E
bldng_SW_pt_1  = waypoint_S_centre - (building_NS_extradisplacement + building_EW_displacement) # building_offset * (unit_E + unit_N)
bldng_SW_pt_2  = ptngate_SW.approach_shape.posn    - building_NS_extradisplacement # building_offset * unit_N
bldng_SW_pt_3  = np.array([ptngate_SW.approach_shape.posn[0], ptngate_S.approach_shape.posn[1]])
bldng_SW_pts   = [bldng_SW_pt_0, bldng_SW_pt_1, bldng_SW_pt_2, bldng_SW_pt_3]
bldng_SW_shape = SMGS.Polygon(bldng_SW_pts, wall_radius)

bldng_SE_pt_0  = np.array([ptngate_SE.approach_shape.posn[0], ptngate_S.approach_shape.posn[1]])
bldng_SE_pt_1  = ptngate_SE.approach_shape.posn - building_NS_extradisplacement # building_offset * unit_N
bldng_SE_pt_2  = waypoint_S_centre - building_NS_extradisplacement + building_EW_displacement # + building_offset * (unit_E - unit_N)
bldng_SE_pt_3  = ptngate_S.approach_shape.posn  + building_EW_displacement # building_offset * unit_E
bldng_SE_pts   = [bldng_SE_pt_0, bldng_SE_pt_1, bldng_SE_pt_2, bldng_SE_pt_3]
bldng_SE_shape = SMGS.Polygon(bldng_SE_pts, wall_radius)

bldng_NE       = SMZZ.create_obstacle( 8, bldng_NE_shape)
bldng_NW       = SMZZ.create_obstacle( 9, bldng_NW_shape)
bldng_SW       = SMZZ.create_obstacle(10, bldng_SW_shape)
bldng_SE       = SMZZ.create_obstacle(11, bldng_SE_shape)


#
# Declare a few other zones:
#   two (or possibly four) footpaths;
#   two streetscapes; and
#   three crossings. 
#
footpath_NW_NE_shape = SMGS.Segment(ptngate_NW.approach_shape.posn - ptngate_halfwidth * unit_E, ptngate_NE.approach_shape.posn + ptngate_halfwidth * unit_E, ptngate_halfwidth)
footpath_SW_SE_shape = SMGS.Segment(ptngate_SW.approach_shape.posn - ptngate_halfwidth * unit_E, ptngate_SE.approach_shape.posn + ptngate_halfwidth * unit_E, ptngate_halfwidth)

roadway_NW_shape  = SMGS.Segment(cargate_NW.approach_shape.posn - (ptngate_halfwidth + cargate_radius) * unit_E, waypoint_NW.approach_shape.posn - ptngate_NS_extradisplacement - (ptngate_halfwidth + cargate_radius) * unit_E, cargate_radius)
roadway_NNW_shape = SMGS.Segment(waypoint_NW.approach_shape.posn - ptngate_NS_extradisplacement + (ptngate_halfwidth + cargate_radius) * unit_E, waypoint_N_centre - ptngate_NS_extradisplacement - (ptngate_halfwidth + cargate_radius) * unit_E, cargate_radius)
roadway_NNE_shape = SMGS.Segment(waypoint_N_centre - ptngate_NS_extradisplacement + (ptngate_halfwidth + cargate_radius) * unit_E, waypoint_NE.approach_shape.posn - ptngate_NS_extradisplacement - (ptngate_halfwidth + cargate_radius) * unit_E, cargate_radius)
roadway_NE_shape  = SMGS.Segment(waypoint_NE.approach_shape.posn - ptngate_NS_extradisplacement + (ptngate_halfwidth + cargate_radius) * unit_E, cargate_NE.approach_shape.posn + (ptngate_halfwidth + cargate_radius) * unit_E, cargate_radius)

roadway_SW_shape  = SMGS.Segment(cargate_SW.approach_shape.posn - (ptngate_halfwidth + cargate_radius) * unit_E, waypoint_SW.approach_shape.posn + ptngate_NS_extradisplacement - (ptngate_halfwidth + cargate_radius) * unit_E, cargate_radius)
roadway_SSW_shape = SMGS.Segment(waypoint_SW.approach_shape.posn + ptngate_NS_extradisplacement + (ptngate_halfwidth + cargate_radius) * unit_E, waypoint_S_centre + ptngate_NS_extradisplacement - (ptngate_halfwidth + cargate_radius) * unit_E, cargate_radius)
roadway_SSE_shape = SMGS.Segment(waypoint_S_centre + ptngate_NS_extradisplacement + (ptngate_halfwidth + cargate_radius) * unit_E, waypoint_SE.approach_shape.posn + ptngate_NS_extradisplacement - (ptngate_halfwidth + cargate_radius) * unit_E, cargate_radius)
roadway_SE_shape  = SMGS.Segment(waypoint_SE.approach_shape.posn + ptngate_NS_extradisplacement + (ptngate_halfwidth + cargate_radius) * unit_E, cargate_SE.approach_shape.posn + (ptngate_halfwidth + cargate_radius) * unit_E, cargate_radius)

crossing_W_shape = SMGS.Segment(waypoint_NW.approach_shape.posn - ptngate_halfwidth * unit_N, waypoint_SW.approach_shape.posn + ptngate_halfwidth * unit_N, ptngate_halfwidth)
crossing_C_shape = SMGS.Segment(waypoint_N_centre - ptngate_halfwidth * unit_N, waypoint_S_centre + ptngate_halfwidth * unit_N, ptngate_halfwidth)
crossing_E_shape = SMGS.Segment(waypoint_NE.approach_shape.posn - ptngate_halfwidth * unit_N, waypoint_SE.approach_shape.posn + ptngate_halfwidth * unit_N, ptngate_halfwidth)

footpath_N = SMZZ.create_footpath(28, footpath_NW_NE_shape)
footpath_S = SMZZ.create_footpath(29, footpath_SW_SE_shape)

# roadway_NW  = SMZZ.create_roadway(30, roadway_NW_shape)
# roadway_NNW = SMZZ.create_roadway(31, roadway_NNW_shape)
# roadway_NNE = SMZZ.create_roadway(32, roadway_NNE_shape)
# roadway_NE  = SMZZ.create_roadway(33, roadway_NE_shape)

# roadway_SW  = SMZZ.create_roadway(34, roadway_SW_shape)
# roadway_SSW = SMZZ.create_roadway(35, roadway_SSW_shape)
# roadway_SSE = SMZZ.create_roadway(36, roadway_SSE_shape)
# roadway_SE  = SMZZ.create_roadway(37, roadway_SE_shape)

crossing_W = SMZZ.create_zebra_crossing(38, crossing_W_shape)
crossing_C = SMZZ.create_zebra_crossing(39, crossing_C_shape)
crossing_E = SMZZ.create_zebra_crossing(40, crossing_E_shape)

OD_matrices = {SMAD.AgentType.PEDESTRIAN: pedestrian_OD_matrix, SMAD.AgentType.CAR: car_OD_matrix}

buildings = [bldng_NE, bldng_NW, bldng_SW, bldng_SE]
footpaths = [footpath_N, footpath_S]
# roadways  = [roadway_NW, roadway_NNW, roadway_NNE, roadway_NE, roadway_SW, roadway_SSW, roadway_SSE, roadway_SE]
crossings = [crossing_W, crossing_C, crossing_E]

#
# @todo: Add pedestrian spawning. Each centroid should spawn uniformly over T_simulation
#

if __name__ == "__main__":
    zones = buildings + footpaths + crossings 
    
    plaza           = SMM.Model(width, height, screen_size, scale_factor, T_simulation, 4, topo_specs, OD_matrices, zones, [], [], [])

    SML.log_static_components(plaza.centroids, plaza.waypoints, plaza.activity_nodes, plaza.zone_config)
    logger = SML.Logger()
    plaza.register_logger(logger)

    for i in range(L_simulation):
        # print("Simulation step ", i, ": start")
        plaza.spawn_agents()
        plaza.advance_timestep()
        # assert(isinstance(plaza, mnm.Model))
        # assert(isinstance(view, mnv.View))
        plaza.draw(view)
        plaza.despawn_agents()
        view.flip()
        # print("Simulation step ", i, ": finish")
        # plaza.log()

    SML.log_stream_statistics(plaza)
