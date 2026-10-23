import sys
import numpy as np
import scipy as sp
# import scipy.stats as ss

import view.view as SVV
import model.force as SMF
import model.distributor as SMDis
import model.topology.node as SMTN
import model.topology.centroid as SMTC
import model.topology.approachnode as SMTApp
import model.agency.defs as SMAD
import model.model as SMM
import model.geometry.shape as SMGS
# import model.zone.defs as SMZD
import model.zone.zone as SMZZ
import model.logger as SML
# import model.agency.agent as smaa

width  = 110
height =  48

scale_factor   = 14
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width,screen_height)

view = SVV.View(width, height, screen_size, scale_factor, pedestrian_upscale_factor = 1.)

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
T_simulation = 3 * 60 # 15 * 60  # 15 minutes
L_simulation = int(T_simulation / SMF.T_step)

spawner_NE = SMDis.DistributorBeta(0., 10, 10)
spawner_N  = SMDis.DistributorUniform()
spawner_NW = SMDis.DistributorBeta(0., 5, 5)
spawner_SW = SMDis.DistributorBeta(0., 10, 10)
spawner_S  = SMDis.DistributorUniform()
spawner_SE = SMDis.DistributorBeta(0., 5, 5)

unit_E = np.array([1, 0])
unit_N = np.array([0, 1])

gate_diameter = 3.
gate_radius   = 0.5 * gate_diameter

cargate_diameter = 4.
cargate_radius   = 0.5 * gate_diameter

carspawner_E = SMDis.DistributorUniform()
carspawner_W = SMDis.DistributorUniform()

safety_margin = 1.
cargate_NS_displacement = (cargate_radius + safety_margin) * unit_N
ptngate_NS_extradisplacement = (gate_radius + cargate_radius + safety_margin) * unit_N
ptngate_NS_displacement = cargate_NS_displacement + ptngate_NS_extradisplacement

cargate_NE_posn = centre_E + cargate_NS_displacement
cargate_NW_posn = centre_W + cargate_NS_displacement
cargate_SW_posn = centre_W - cargate_NS_displacement
cargate_SE_posn = centre_E - cargate_NS_displacement

cargate_NE = SMTC.Centroid(6, SMGS.Circle(cargate_NE_posn, cargate_radius), SVV.colour(170,   0, 170), SVV.colour(255,   0, 255), carspawner_E, agent_type = SMAD.AgentType.CAR)
cargate_NW = SMTC.Centroid(7, SMGS.Circle(cargate_NW_posn, cargate_radius), SVV.colour(170,   0, 170), SVV.colour(255,   0, 255), carspawner_E, agent_type = SMAD.AgentType.CAR)
cargate_SW = SMTC.Centroid(8, SMGS.Circle(cargate_SW_posn, cargate_radius), SVV.colour(  0, 170, 170), SVV.colour(  0,   0, 255), carspawner_W, agent_type = SMAD.AgentType.CAR)
cargate_SE = SMTC.Centroid(9, SMGS.Circle(cargate_SE_posn, cargate_radius), SVV.colour(  0, 170, 170), SVV.colour(  0,   0, 255), carspawner_W, agent_type = SMAD.AgentType.CAR)

ptngate_NE_centre_posn = centre_E + ptngate_NS_displacement
ptngate_NW_centre_posn = centre_W + ptngate_NS_displacement
ptngate_SW_centre_posn = centre_W - ptngate_NS_displacement
ptngate_SE_centre_posn = centre_E - ptngate_NS_displacement

ptngate_NE = SMTC.Centroid(0, SMGS.Circle(ptngate_NE_centre_posn, gate_radius), SVV.colour(  0,   0, 170), SVV.colour(  0,   0, 255), spawner_NE, agent_type = SMAD.AgentType.PEDESTRIAN)
ptngate_N  = SMTC.Centroid(1, SMGS.Circle(centre_N,  gate_radius), SVV.colour(114,  56,   0), SVV.colour(170,  85,   0), spawner_NE, agent_type = SMAD.AgentType.PEDESTRIAN)
ptngate_NW = SMTC.Centroid(2, SMGS.Circle(ptngate_NW_centre_posn, gate_radius), SVV.colour( 56, 114,   0), SVV.colour( 85, 170,   0), spawner_NW, agent_type = SMAD.AgentType.PEDESTRIAN)
ptngate_SW = SMTC.Centroid(3, SMGS.Circle(ptngate_SW_centre_posn, gate_radius), SVV.colour(170,   0,   0), SVV.colour(255,   0,   0), spawner_SW, agent_type = SMAD.AgentType.PEDESTRIAN)
ptngate_S  = SMTC.Centroid(4, SMGS.Circle(centre_S,  gate_radius), SVV.colour(114,   0,  56), SVV.colour(170,   0,  85), spawner_SE, agent_type = SMAD.AgentType.PEDESTRIAN)
ptngate_SE = SMTC.Centroid(5, SMGS.Circle(ptngate_SE_centre_posn, gate_radius), SVV.colour( 56,   0, 114), SVV.colour( 85,   0, 170), spawner_SE, agent_type = SMAD.AgentType.PEDESTRIAN)


ptn_centroids        = [ptngate_NE, ptngate_N, ptngate_NW, ptngate_SW, ptngate_S, ptngate_SE]
car_centroids        = [cargate_NE, cargate_NW, cargate_SW, cargate_SE]
pedestrian_OD_matrix = 3 * np.array([[0, 5, 5, 10, 5, 5], [5, 0, 5, 5, 10, 5], [5, 5, 0, 5, 5, 10], [10, 5, 5, 0, 5, 5], [5, 10, 5, 5, 0, 5], [5, 5, 10, 5, 5, 0]])
car_OD_matrix        = 5 * np.array([[0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0]])

crossing_EW_displacement = 6 * unit_E

waypoint_NE_posn = centre_E + (ptngate_NS_displacement - crossing_EW_displacement)
waypoint_NW_posn = centre_W + (ptngate_NS_displacement + crossing_EW_displacement)
waypoint_SW_posn = centre_W - (ptngate_NS_displacement - crossing_EW_displacement)
waypoint_SE_posn = centre_E - (ptngate_NS_displacement + crossing_EW_displacement)
waypoint_N_posn  = centre_true + ptngate_NS_displacement
waypoint_S_posn  = centre_true - ptngate_NS_displacement

waypoint_NE   = SMTApp.create_waypoint(10, SMGS.Circle(waypoint_NE_posn, gate_radius), SMAD.AgentType.PEDESTRIAN)
waypoint_NW   = SMTApp.create_waypoint(11, SMGS.Circle(waypoint_NW_posn, gate_radius), SMAD.AgentType.PEDESTRIAN)
waypoint_SW   = SMTApp.create_waypoint(12, SMGS.Circle(waypoint_SW_posn, gate_radius), SMAD.AgentType.PEDESTRIAN)
waypoint_SE   = SMTApp.create_waypoint(13, SMGS.Circle(waypoint_SE_posn, gate_radius), SMAD.AgentType.PEDESTRIAN)
waypoint_N    = SMTApp.create_waypoint(14, SMGS.Circle(waypoint_N_posn, gate_radius), SMAD.AgentType.PEDESTRIAN)
waypoint_S    = SMTApp.create_waypoint(15, SMGS.Circle(waypoint_S_posn, gate_radius), SMAD.AgentType.PEDESTRIAN)
ptn_waypoints = [waypoint_NE, waypoint_N, waypoint_NW, waypoint_SW, waypoint_S, waypoint_SE]

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
# E-W symmetric pedestrian connections
#
SMTN.symconnect(waypoint_NE,  waypoint_N)
SMTN.symconnect(waypoint_N,  waypoint_NW)
SMTN.symconnect(waypoint_SW,  waypoint_S)
SMTN.symconnect(waypoint_S,  waypoint_SE)

#
# N-S symmetric pedestrian connections
#
SMTN.symconnect(waypoint_NE,  waypoint_SE)
SMTN.symconnect(waypoint_N,   waypoint_S)
SMTN.symconnect(waypoint_NW,  waypoint_SW)

#
# E-W one-way car connections
#
cargate_NW.connect(cargate_NE)
cargate_SE.connect(cargate_SW)

car_waypoints = []

ptn_topology = SMM.PartialTopology(ptn_centroids, ptn_waypoints, [], SMAD.AgentType.PEDESTRIAN)
car_topology = SMM.PartialTopology(car_centroids, car_waypoints, [], SMAD.AgentType.CAR)
partial_topologies = [ptn_topology, car_topology]

# partial_topologies = [SMM.PartialTopology(ptn_centroids, waypoints, 0), SMM.PartialTopology(car_centroids, [], 3)]


# topology = mnn.Graph()
# topology.add_nodelist(centroids + waypoints)


wall_depth  = 4.  # in metres
wall_radius = 0.5 * wall_depth

building_NS_extradisplacement = (gate_radius + safety_margin + wall_radius) * unit_N
# building_NS_displacement = ptngate_NS_displacement + building_NS_extradisplacement
building_EW_displacement = (gate_radius + safety_margin + wall_radius) * unit_E

bldng_NE_pt_0  = ptngate_NE_centre_posn    + building_NS_extradisplacement # building_offset * unit_N
bldng_NE_pt_1  = np.array([ptngate_NE_centre_posn[0], centre_N[1]])
bldng_NE_pt_2  = centre_N           + building_EW_displacement
bldng_NE_pt_3  = centre_true + ptngate_NS_displacement + building_NS_extradisplacement + building_EW_displacement # building_offset * (unit_E + unit_N)
bldng_NE_pts   = [bldng_NE_pt_0, bldng_NE_pt_1, bldng_NE_pt_2, bldng_NE_pt_3]
bldng_NE_shape = SMGS.Polygon(bldng_NE_pts, wall_radius)

bldng_NW_pt_0  = centre_true + ptngate_NS_displacement + building_NS_extradisplacement - building_EW_displacement # building_offset * (unit_E - unit_N)
bldng_NW_pt_1  = centre_N     - building_EW_displacement # building_offset * unit_E
bldng_NW_pt_2  = np.array([ptngate_NW_centre_posn[0], centre_N[1]])
bldng_NW_pt_3  = ptngate_NW_centre_posn    + building_NS_extradisplacement # building_offset * unit_N
bldng_NW_pts   = [bldng_NW_pt_0, bldng_NW_pt_1, bldng_NW_pt_2, bldng_NW_pt_3]
bldng_NW_shape = SMGS.Polygon(bldng_NW_pts, wall_radius)

bldng_SW_pt_0  = centre_S     - building_EW_displacement # building_offset * unit_E
bldng_SW_pt_1  = centre_true - ptngate_NS_displacement - (building_NS_extradisplacement + building_EW_displacement) # building_offset * (unit_E + unit_N)
bldng_SW_pt_2  = ptngate_SW_centre_posn    - building_NS_extradisplacement # building_offset * unit_N
bldng_SW_pt_3  = np.array([ptngate_SW_centre_posn[0], centre_S[1]])
bldng_SW_pts   = [bldng_SW_pt_0, bldng_SW_pt_1, bldng_SW_pt_2, bldng_SW_pt_3]
bldng_SW_shape = SMGS.Polygon(bldng_SW_pts, wall_radius)

bldng_SE_pt_0  = np.array([ptngate_SE_centre_posn[0], centre_S[1]])
bldng_SE_pt_1  = ptngate_SE_centre_posn - building_NS_extradisplacement # building_offset * unit_N
bldng_SE_pt_2  = centre_true - ptngate_NS_displacement - building_NS_extradisplacement + building_EW_displacement # + building_offset * (unit_E - unit_N)
bldng_SE_pt_3  = centre_S  + building_EW_displacement # building_offset * unit_E
bldng_SE_pts   = [bldng_SE_pt_0, bldng_SE_pt_1, bldng_SE_pt_2, bldng_SE_pt_3]
bldng_SE_shape = SMGS.Polygon(bldng_SE_pts, wall_radius)

bldng_NE       = SMZZ.create_obstacle(24, bldng_NE_shape)
bldng_NW       = SMZZ.create_obstacle(25, bldng_NW_shape)
bldng_SW       = SMZZ.create_obstacle(26, bldng_SW_shape)
bldng_SE       = SMZZ.create_obstacle(27, bldng_SE_shape)


#
# Declare a few other zones:
#   two (or possibly four) footpaths;
#   two streetscapes; and
#   three crossings. 
#
footpath_NW_NE_shape = SMGS.Segment(ptngate_NW_centre_posn - gate_radius * unit_E, ptngate_NE_centre_posn + gate_radius * unit_E, gate_radius)
footpath_SW_SE_shape = SMGS.Segment(ptngate_SW_centre_posn - gate_radius * unit_E, ptngate_SE_centre_posn + gate_radius * unit_E, gate_radius)

roadway_NW_shape  = SMGS.Segment(cargate_NW_posn - (gate_radius + cargate_radius) * unit_E, waypoint_NW_posn - ptngate_NS_extradisplacement - (gate_radius + cargate_radius) * unit_E, cargate_radius)
roadway_NNW_shape = SMGS.Segment(waypoint_NW_posn - ptngate_NS_extradisplacement + (gate_radius + cargate_radius) * unit_E, waypoint_N_posn - ptngate_NS_extradisplacement - (gate_radius + cargate_radius) * unit_E, cargate_radius)
roadway_NNE_shape = SMGS.Segment(waypoint_N_posn - ptngate_NS_extradisplacement + (gate_radius + cargate_radius) * unit_E, waypoint_NE_posn - ptngate_NS_extradisplacement - (gate_radius + cargate_radius) * unit_E, cargate_radius)
roadway_NE_shape  = SMGS.Segment(waypoint_NE_posn - ptngate_NS_extradisplacement + (gate_radius + cargate_radius) * unit_E, cargate_NE_posn + (gate_radius + cargate_radius) * unit_E, cargate_radius)

roadway_SW_shape  = SMGS.Segment(cargate_SW_posn - (gate_radius + cargate_radius) * unit_E, waypoint_SW_posn + ptngate_NS_extradisplacement - (gate_radius + cargate_radius) * unit_E, cargate_radius)
roadway_SSW_shape = SMGS.Segment(waypoint_SW_posn + ptngate_NS_extradisplacement + (gate_radius + cargate_radius) * unit_E, waypoint_S_posn + ptngate_NS_extradisplacement - (gate_radius + cargate_radius) * unit_E, cargate_radius)
roadway_SSE_shape = SMGS.Segment(waypoint_S_posn + ptngate_NS_extradisplacement + (gate_radius + cargate_radius) * unit_E, waypoint_SE_posn + ptngate_NS_extradisplacement - (gate_radius + cargate_radius) * unit_E, cargate_radius)
roadway_SE_shape  = SMGS.Segment(waypoint_SE_posn + ptngate_NS_extradisplacement + (gate_radius + cargate_radius) * unit_E, cargate_SE_posn + (gate_radius + cargate_radius) * unit_E, cargate_radius)

crossing_W_shape = SMGS.Segment(waypoint_NW_posn - gate_radius * unit_N, waypoint_SW_posn + gate_radius * unit_N, gate_radius)
crossing_C_shape = SMGS.Segment(waypoint_N_posn  - gate_radius * unit_N, waypoint_S_posn  + gate_radius * unit_N, gate_radius)
crossing_E_shape = SMGS.Segment(waypoint_NE_posn - gate_radius * unit_N, waypoint_SE_posn + gate_radius * unit_N, gate_radius)

ctrcrossing_NW_centre = waypoint_NW_posn - ptngate_NS_extradisplacement
ctrcrossing_NE_centre = waypoint_NE_posn - ptngate_NS_extradisplacement
ctrcrossing_SW_centre = waypoint_SW_posn + ptngate_NS_extradisplacement
ctrcrossing_SE_centre = waypoint_SE_posn + ptngate_NS_extradisplacement

ctrcrossing_halfL = gate_radius + cargate_radius
epsilon           = 0.25
ctrcrossing_halfW = cargate_radius - epsilon

ctrcrossing_NW_shape = SMGS.Segment(ctrcrossing_NW_centre - ctrcrossing_halfL * unit_E, ctrcrossing_NW_centre + ctrcrossing_halfL * unit_E, ctrcrossing_halfW)
ctrcrossing_NE_shape = SMGS.Segment(ctrcrossing_NE_centre - ctrcrossing_halfL * unit_E, ctrcrossing_NE_centre + ctrcrossing_halfL * unit_E, ctrcrossing_halfW)

ctrcrossing_SW_shape  = SMGS.Segment(ctrcrossing_SW_centre - ctrcrossing_halfL * unit_E, ctrcrossing_SW_centre + ctrcrossing_halfL * unit_E, ctrcrossing_halfW)
ctrcrossing_SE_shape  = SMGS.Segment(ctrcrossing_SE_centre - ctrcrossing_halfL * unit_E, ctrcrossing_SE_centre + ctrcrossing_halfL * unit_E, ctrcrossing_halfW)


footpath_N = SMZZ.create_footpath(28, footpath_NW_NE_shape)
footpath_S = SMZZ.create_footpath(29, footpath_SW_SE_shape)

roadway_NW  = SMZZ.create_roadway(30, roadway_NW_shape)
roadway_NNW = SMZZ.create_roadway(31, roadway_NNW_shape)
roadway_NNE = SMZZ.create_roadway(32, roadway_NNE_shape)
roadway_NE  = SMZZ.create_roadway(33, roadway_NE_shape)

roadway_SW  = SMZZ.create_roadway(34, roadway_SW_shape)
roadway_SSW = SMZZ.create_roadway(35, roadway_SSW_shape)
roadway_SSE = SMZZ.create_roadway(36, roadway_SSE_shape)
roadway_SE  = SMZZ.create_roadway(37, roadway_SE_shape)

crossing_W = SMZZ.create_pelican_crossing(38, crossing_W_shape)
crossing_C = SMZZ.create_crossing(39, crossing_C_shape)
crossing_E = SMZZ.create_pelican_crossing(40, crossing_E_shape)

ctrcrossing_NW = SMZZ.create_counterpelican_turning(41, ctrcrossing_NW_shape, unit_N, unit_E)
ctrcrossing_SW = SMZZ.create_counterpelican_turning(42, ctrcrossing_SW_shape, -unit_N, -unit_E)
ctrcrossing_NE = SMZZ.create_counterpelican_turning(43, ctrcrossing_NE_shape, unit_N, unit_E)
ctrcrossing_SE = SMZZ.create_counterpelican_turning(44, ctrcrossing_SE_shape, -unit_N, -unit_E)

OD_matrices = {0: pedestrian_OD_matrix, 3: car_OD_matrix}

buildings        = [bldng_NE, bldng_NW, bldng_SW, bldng_SE]
footpaths        = [footpath_N, footpath_S]
roadways         = [roadway_NW, roadway_NNW, roadway_NNE, roadway_NE, roadway_SW, roadway_SSW, roadway_SSE, roadway_SE]
crossings        = [crossing_W, crossing_C, crossing_E]
countercrossings = [ctrcrossing_NW, ctrcrossing_SW, ctrcrossing_NE, ctrcrossing_SE]

if __name__ == "__main__":
    zones = buildings + footpaths + roadways + crossings + countercrossings

    plaza           = SMM.Model(width, height, screen_size, scale_factor, T_simulation, SMF.T_step, 4, partial_topologies, OD_matrices, zones, [], [], [])

    SML.log_static_components(plaza.centroids, plaza.waypoints, plaza.zone_config)
    logger = SML.Logger()
    plaza.register_logger(logger)

    for i in range(L_simulation):
        # print("Simulation step ", i, ": start")
        plaza.spawn_agents()
        plaza.advance_timestep()
        assert(isinstance(plaza, SMM.Model))
        assert(isinstance(view, SVV.View))
        plaza.draw(view)
        plaza.despawn_agents()
        view.flip()
        # print("Simulation step ", i, ": finish")
