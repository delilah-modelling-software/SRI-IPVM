import sys
import shared_spaces_types as sst
import numpy as np
import scipy as sp
import scipy.stats as ss
import pygame as pg

import shared_spaces_types as sst
import multimodal_network_view as mnv

width  = 60
height = 60

scale_factor   = 14
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width,screen_height)

view = mnv.View(width, height, screen_size, scale_factor, 1.)

centre_true = np.array([width / 2, height / 2])

corner_NE = np.array([width - 5, height - 5])
corner_NW = np.array([  5, height - 5])
corner_SW = np.array([  5,   5])
corner_SE = np.array([width - 5,   5])

centre_E = np.array([width - 5, centre_true[1]])
centre_N = np.array([centre_true[0], height - 5])
centre_W = np.array([  5, centre_true[1]])
centre_S = np.array([centre_true[0],   5])

T_step       = 0.5 # 0.125    # Duration of each simulation time-step
T_reaction   = T_step
T_simulation = 3 * 60 # 15 * 60  # 15 minutes
L_simulation = int(T_simulation / T_step)

gate_diameter = 7.
gate_radius   = 0.5 * gate_diameter

spawner_NE = sst.Betaspawner(0., 10, 10)
spawner_NW = sst.Betaspawner(0., 5, 5)
spawner_SW = sst.Betaspawner(0., 10, 10)
spawner_SE = sst.Betaspawner(0., 5, 5)
# spawner_SE = sst.ClusterSpawner()

gate_NE = sst.Centroid(0, corner_NE, gate_radius, mnv.colour(0, 0, 170), mnv.colour(0, 0, 255), view, spawner_NE)
gate_NW = sst.Centroid(1, corner_NW, gate_radius, mnv.colour(114, 56, 0), mnv.colour(170, 85, 0), view, spawner_NW)
gate_SW = sst.Centroid(2, corner_SW, gate_radius, mnv.colour(170, 0, 0), mnv.colour(255, 0, 0), view, spawner_SW)
gate_SE = sst.Centroid(3, corner_SE, gate_radius, mnv.colour(114, 0, 56), mnv.colour(170, 0, 85), view, spawner_SE)

centroids = [gate_NE, gate_NW, gate_SW, gate_SE]
# OD_matrix = np.zeros((4, 4))
OD_matrix = np.array([[0, 100, 200, 100], [100, 0, 100, 200], [200, 100, 0, 100], [100, 200, 100, 0]])

unit_E = np.array([1, 0])
unit_N = np.array([0, 1])

fountain_shape = sst.Circle(centre_true, 2)
A_fountain     = 10.
D_fountain     = 0.2
fountain       = sst.Repulsor(0, A_fountain, D_fountain, fountain_shape)

statue_NE_shape = sst.Circle(centre_true + 4 * (unit_E + unit_N), 1.)
statue_NW_shape = sst.Circle(centre_true - 4 * (unit_E - unit_N), 1.)
statue_SW_shape = sst.Circle(centre_true - 4 * (unit_E + unit_N), 1.)
statue_SE_shape = sst.Circle(centre_true + 4 * (unit_E - unit_N), 1.)
A_statue     = 10.
D_statue     = 0.2
statue_NE    = sst.Repulsor(1, A_statue, D_statue, statue_NE_shape)
statue_NW    = sst.Repulsor(2, A_statue, D_statue, statue_NW_shape)
statue_SW    = sst.Repulsor(3, A_statue, D_statue, statue_SW_shape)
statue_SE    = sst.Repulsor(4, A_statue, D_statue, statue_SE_shape)

# statuette_NE_shape = sst.Circle(centre_true + 6 * (unit_E + unit_N), 0.5)
# statuette_NW_shape = sst.Circle(centre_true - 6 * (unit_E - unit_N), 0.5)
# statuette_SW_shape = sst.Circle(centre_true - 6 * (unit_E + unit_N), 0.5)
# statuette_SE_shape = sst.Circle(centre_true + 6 * (unit_E - unit_N), 0.5)
# A_statuette  = 10.
# D_statuette  = 0.2
# statuette_NE = sst.Repulsor(5, A_statuette, D_statuette, statuette_NE_shape)
# statuette_NW = sst.Repulsor(6, A_statuette, D_statuette, statuette_NW_shape)
# statuette_SW = sst.Repulsor(7, A_statuette, D_statuette, statuette_SW_shape)
# statuette_SE = sst.Repulsor(8, A_statuette, D_statuette, statuette_SE_shape)

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

wall_NE_SSW_shape = sst.Segment(wall_salient_NE, wall_salient_NE - 0.5 * rt3 * unit_NNE, 0.1)
wall_SW_NNE_shape = sst.Segment(wall_salient_SW, wall_salient_SW + 0.5 * rt3 * unit_NNE, 0.1)
wall_NW_ESE_shape = sst.Segment(wall_salient_NW, wall_salient_NW + 0.5 * rt3 * unit_ESE, 0.1)
wall_SE_WSW_shape = sst.Segment(wall_salient_SE, wall_salient_SE - 0.5 * rt3 * unit_ESE, 0.1)
# wall_NE_WSW_shape = sst.Segment(wall_salient_NE, wall_salient_NE - 0.5 * rt3 * unit_ENE, 0.1)
# wall_SW_ENE_shape = sst.Segment(wall_salient_SW, wall_salient_SW + 0.5 * rt3 * unit_ENE, 0.1)

A_wall      = 10.
D_wall      =  0.2
wall_NE_SSW = sst.Repulsor( 9, A_wall, D_wall, wall_NE_SSW_shape)
wall_SW_NNE = sst.Repulsor(11, A_wall, D_wall, wall_SW_NNE_shape)
wall_NW_ESE = sst.Repulsor(10, A_wall, D_wall, wall_NW_ESE_shape)
wall_SE_WNW = sst.Repulsor(12, A_wall, D_wall, wall_SE_WSW_shape)

# wall_NE_WSW = sst.Repulsor( ?, A_wall, D_wall, wall_NE_WSW_shape)
# wall_SW_ENE = sst.Repulsor( ?, A_wall, D_wall, wall_SW_ENE_shape)

obstacles = [fountain, statue_NE, statue_NW, statue_SW, statue_SE, wall_NE_SSW, wall_SW_NNE, wall_NW_ESE, wall_SE_WNW]

#
# @todo: Add pedestrian spawning. Each centroid should spawn uniformly over T_simulation
#

if __name__ == "__main__":
    streams = []
    stream_idx = 0
    for i in range(4):
        for j in range(4):
            n_ped = OD_matrix[i, j]
            if int(n_ped) == 0:
                continue

            src          = centroids[i]
            dst          = centroids[j]
            spawner      = src.streamspawner
            sample_times = spawner.generate_spawning_times(T_simulation, T_step, n_ped)
            src.register_spawning_list(dst, sample_times)

    #
    # @ todo: Add some more attractors and obscurers (perhaps some lamp-posts, etc.).
    #

    plaza           = sst.Model(width, height, screen_size, scale_factor, T_step, 4, centroids, [], obstacles, [], [], [])
    plaza.MRCA_flag = False

    pg.init()
    screen_display = pg.display.set_mode(screen_size)

    for i in range(L_simulation):
        print("Simulation step ", i, ": start")
        plaza.spawn_pedestrians()
        plaza.advance_timestep()
        plaza.draw(view)
        plaza.despawn_pedestrians()
        pg.display.flip()
        # pg.time.wait(1)
        print("Simulation step ", i, ": finish")
        # plaza.log()
