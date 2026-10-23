import sys
import shared_spaces_types as sst
import numpy as np
import scipy as sp
import scipy.stats as ss
import pygame as pg

import shared_spaces_types as sst

width  = 110
height =  36

scale_factor   = 14
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width,screen_height)

view = sst.View(width, height, screen_size, scale_factor, pedestrian_upscale_factor = 1.)

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

gate_diameter = 3.
gate_radius   = 0.5 * gate_diameter

spawner_NE = sst.Betaspawner(0., 10, 10)
spawner_N  = sst.Uniformspawner()
spawner_NW = sst.Betaspawner(0., 5, 5)
spawner_SW = sst.Betaspawner(0., 10, 10)
spawner_S  = sst.Uniformspawner()
spawner_SE = sst.Betaspawner(0., 5, 5)
# spawner_SE = sst.ClusterSpawner()

unit_E = np.array([1, 0])
unit_N = np.array([0, 1])

gate_NE = sst.Centroid(0, centre_E + 6 * unit_N, gate_radius, pg.Color(  0,   0, 170), pg.Color(  0,   0, 255), view, spawner_NE)
gate_N  = sst.Centroid(1, centre_N,  gate_radius, pg.Color(114,  56,   0), pg.Color(170,  85,   0), view, spawner_NE)
gate_NW = sst.Centroid(2, centre_W + 6 * unit_N, gate_radius, pg.Color( 56, 114,   0), pg.Color( 85, 170,   0), view, spawner_NW)
gate_SW = sst.Centroid(3, centre_W - 6 * unit_N, gate_radius, pg.Color(170,   0,   0), pg.Color(255,   0,   0), view, spawner_SW)
gate_S  = sst.Centroid(4, centre_S,  gate_radius, pg.Color(114,   0,  56), pg.Color(170,   0,  85), view, spawner_SE)
gate_SE = sst.Centroid(5, centre_E - 6 * unit_N, gate_radius, pg.Color( 56,   0, 114), pg.Color( 85,   0, 170), view, spawner_SE)

centroids = [gate_NE, gate_N, gate_NW, gate_SW, gate_S, gate_SE]
OD_matrix = np.array([[0, 50, 50, 100, 50, 50], [50, 0, 50, 50, 100, 50], [50, 50, 0, 50, 50, 100], [100, 50, 50, 0, 50, 50], [50, 100, 50, 50, 0, 50], [50, 50, 100, 50, 50, 0]])

wall_depth  = 0.2   # 20cm
wall_radius = 0.5 * wall_depth

#
# @todo: Add pedestrian spawning. Each centroid should spawn uniformly over T_simulation
#

if __name__ == "__main__":
    streams = []
    stream_idx = 0
    for i in range(6):
        for j in range(6):
            n_ped = OD_matrix[i, j]
            if int(n_ped) == 0:
                continue

            src          = centroids[i]
            dst          = centroids[j]
            spawner      = src.streamspawner
            sample_times = spawner.generate_spawning_times(T_simulation, T_step, n_ped)
            src.register_spawning_list(dst, sample_times)

    obstacles = []

    #
    # @ todo: Add some more attractors and obscurers (perhaps some lamp-posts, etc.).
    #

    plaza           = sst.Model(width, height, screen_size, scale_factor, T_step, 4, centroids, obstacles, [], [], [])
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
