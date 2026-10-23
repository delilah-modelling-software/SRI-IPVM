import sys
import numpy as np
import scipy as sp
import pygame as pg

import view.view as SVV
import model.numerical.defs as SMND
# import model.force as SMF
import model.distributor as SMDis
import model.topology.node as SMTN
import model.topology.centroid as SMTC
import model.topology.topology as SMTT
import model.agency.defs as SMAD
import model.model as SMM
import model.geometry.shape as SMGS
import model.zone.zone as SMZZ
import model.logger as SML

width  = 60
height = 60

scale_factor   = 14
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width, screen_height)

# layer_flags = SVV.ViewLayerFlagset.GEOMETRY | SVV.ViewLayerFlagset.TOPOLOGY | SVV.ViewLayerFlagset.MACROSCOPIC | SVV.ViewLayerFlagset.MICROSCOPIC
# layer_flags = SVV.ViewLayerFlagset.GEOMETRY | SVV.ViewLayerFlagset.TOPOLOGY | SVV.ViewLayerFlagset.MICROSCOPIC | SVV.ViewLayerFlagset.ATTRACTION
layer_flags = SVV.ViewLayerFlagset.GEOMETRY | SVV.ViewLayerFlagset.TOPOLOGY | SVV.ViewLayerFlagset.MICROSCOPIC | SVV.ViewLayerFlagset.REPULSION
view = SVV.View(width, height, screen_size, scale_factor, layer_flags)

centre_true = np.array([width / 2, height / 2])

corner_NE = np.array([width - 5, height - 5])
corner_NW = np.array([  5, height - 5])
corner_SW = np.array([  5,   5])
corner_SE = np.array([width - 5,   5])

centre_E = np.array([width - 5, centre_true[1]])
centre_N = np.array([centre_true[0], height - 5])
centre_W = np.array([  5, centre_true[1]])
centre_S = np.array([centre_true[0],   5])

# T_step       = 0.5 # 0.125    # Duration of each simulation time-step
# T_reaction   = smf.T_step
T_simulation = 3 * 60 # 15 * 60  # 15 minutes
L_simulation = int(T_simulation / SMND.T_step)

gate_diameter = 7.
gate_radius   = 0.5 * gate_diameter

spawner_NE = SMDis.DistributorBeta(0., 10, 10)
spawner_NW = SMDis.DistributorBeta(0., 5, 5)
spawner_SW = SMDis.DistributorBeta(0., 10, 10)
spawner_SE = SMDis.DistributorBeta(0., 5, 5)

gate_NE = SMTC.Centroid(0, SMGS.Circle(corner_NE, gate_radius), SVV.colour(0, 0, 170), SVV.colour(0, 0, 255), spawner_NE, SMAD.AgentType.PEDESTRIAN)
gate_NW = SMTC.Centroid(1, SMGS.Circle(corner_NW, gate_radius), SVV.colour(114, 56, 0), SVV.colour(170, 85, 0), spawner_NW, SMAD.AgentType.PEDESTRIAN)
gate_SW = SMTC.Centroid(2, SMGS.Circle(corner_SW, gate_radius), SVV.colour(170, 0, 0), SVV.colour(255, 0, 0), spawner_SW, SMAD.AgentType.PEDESTRIAN)
gate_SE = SMTC.Centroid(3, SMGS.Circle(corner_SE, gate_radius), SVV.colour(114, 0, 56), SVV.colour(170, 0, 85), spawner_SE, SMAD.AgentType.PEDESTRIAN)

ptn_centroids = [gate_NE, gate_NW, gate_SW, gate_SE]
ptn_OD_matrix = np.array([[0, 50, 100, 50], [50, 0, 50, 100], [100, 50, 0, 50], [50, 100, 50, 0]])

SMTN.omniconnect(ptn_centroids)

ptn_waypoints = []
ptn_topo_spec = SMTT.AgentNodeConfig(ptn_centroids, ptn_waypoints, [], SMAD.AgentType.PEDESTRIAN)
topo_specs    = [ptn_topo_spec]
# ptn_topology = SMM.PartialTopology(ptn_centroids, [], [], SMAD.AgentType.PEDESTRIAN)
# partial_topologies = [ptn_topology]

# topology = mnn.Graph()
# topology.add_nodelist(centroids)

unit_E = np.array([1, 0])
unit_N = np.array([0, 1])

fountain_shape = SMGS.Circle(centre_true, 2)
fountain       = SMZZ.create_obstacle(0, fountain_shape)

statue_NE_shape = SMGS.Circle(centre_true + 4 * (unit_E + unit_N), 1.)
statue_NW_shape = SMGS.Circle(centre_true - 4 * (unit_E - unit_N), 1.)
statue_SW_shape = SMGS.Circle(centre_true - 4 * (unit_E + unit_N), 1.)
statue_SE_shape = SMGS.Circle(centre_true + 4 * (unit_E - unit_N), 1.)
statue_NE    = SMZZ.create_obstacle(1, statue_NE_shape)
statue_NW    = SMZZ.create_obstacle(2, statue_NW_shape)
statue_SW    = SMZZ.create_obstacle(3, statue_SW_shape)
statue_SE    = SMZZ.create_obstacle(4, statue_SE_shape)

# statuette_NE_shape = smsh.Circle(centre_true + 6 * (unit_E + unit_N), 0.5)
# statuette_NW_shape = smsh.Circle(centre_true - 6 * (unit_E - unit_N), 0.5)
# statuette_SW_shape = smsh.Circle(centre_true - 6 * (unit_E + unit_N), 0.5)
# statuette_SE_shape = smsh.Circle(centre_true + 6 * (unit_E - unit_N), 0.5)
# A_statuette  = 10.
# D_statuette  = 0.2
# statuette_NE = SMZZ.create_obstacle(5, statuette_NE_shape)
# statuette_NW = SMZZ.create_obstacle(6, statuette_NW_shape)
# statuette_SW = SMZZ.create_obstacle(7, statuette_SW_shape)
# statuette_SE = SMZZ.create_obstacle(8, statuette_SE_shape)

# wall_depth  = 0.2   # 20cm
# wall_radius = 0.5 * wall_depth
wall_salient_NE = centre_true + 6 * (unit_E + unit_N)
wall_salient_SW = centre_true - 6 * (unit_E + unit_N)
wall_salient_NW = centre_true - 6 * (unit_E - unit_N)
wall_salient_SE = centre_true + 6 * (unit_E - unit_N)

rt3 = np.sqrt(3)
unit_ENE = (2 * unit_E + unit_N) / rt3
unit_ESE = (2 * unit_E - unit_N) / rt3
unit_NNE = (2 * unit_N + unit_E) / rt3
unit_NNW = (2 * unit_N - unit_E) / rt3

wall_NE_SSW_shape = SMGS.Segment(wall_salient_NE, wall_salient_NE - 0.5 * rt3 * unit_NNE, 0.1)
wall_SW_NNE_shape = SMGS.Segment(wall_salient_SW, wall_salient_SW + 0.5 * rt3 * unit_NNE, 0.1)
wall_NW_ESE_shape = SMGS.Segment(wall_salient_NW, wall_salient_NW + 0.5 * rt3 * unit_ESE, 0.1)
wall_SE_WSW_shape = SMGS.Segment(wall_salient_SE, wall_salient_SE - 0.5 * rt3 * unit_ESE, 0.1)
# wall_NE_WSW_shape = smgs.Segment(wall_salient_NE, wall_salient_NE - 0.5 * rt3 * unit_ENE, 0.1)
# wall_SW_ENE_shape = smgs.Segment(wall_salient_SW, wall_salient_SW + 0.5 * rt3 * unit_ENE, 0.1)

A_wall      = 10.
D_wall      =  0.2
wall_NE_SSW = SMZZ.create_obstacle( 9, wall_NE_SSW_shape)
wall_SW_NNE = SMZZ.create_obstacle(11, wall_SW_NNE_shape)
wall_NW_ESE = SMZZ.create_obstacle(10, wall_NW_ESE_shape)
wall_SE_WNW = SMZZ.create_obstacle(12, wall_SE_WSW_shape)

# wall_NE_WSW = sst.Repulsor( ?, A_wall, D_wall, wall_NE_WSW_shape)
# wall_SW_ENE = sst.Repulsor( ?, A_wall, D_wall, wall_SW_ENE_shape)

obstacles = [fountain, statue_NE, statue_NW, statue_SW, statue_SE, wall_NE_SSW, wall_SW_NNE, wall_NW_ESE, wall_SE_WNW]


OD_matrices = {SMAD.AgentType.PEDESTRIAN: ptn_OD_matrix}

#
# @todo: Add pedestrian spawning. Each centroid should spawn uniformly over T_simulation
#

if __name__ == "__main__":
    plaza           = SMM.Model(width, height, screen_size, scale_factor, T_simulation, 4, topo_specs, OD_matrices, obstacles, [], [], [])

    SML.log_static_components(plaza.centroids, plaza.waypoints, plaza.activity_nodes, plaza.zone_config)
    logger = SML.Logger()
    plaza.register_logger(logger)

    # pg.init()
    screen_display = pg.display.set_mode(screen_size)

    for i in range(L_simulation):
        # print("Simulation step ", i, ": start")
        plaza.spawn_agents()
        plaza.advance_timestep()
        plaza.draw(view)
        plaza.despawn_agents()
        pg.display.flip()
    
        # pg.time.wait(1)
        # print("Simulation step ", i, ": finish")
        # plaza.log()
