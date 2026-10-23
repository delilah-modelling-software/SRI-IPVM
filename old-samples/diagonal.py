import sys
import shared_spaces_types as sst
import numpy as np
import scipy as sp
import scipy.stats as ss
import pygame as pg

import shared_spaces_types as sst

width  = 110
height = 110

scale_factor   = 6
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width,screen_height)

view = sst.View(width, height, screen_size, scale_factor)

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

spawner_NE = sst.Uniformspawner()
spawner_SW = sst.Uniformspawner()
spawner_NW = sst.Betaspawner(0., 10, 10)
spawner_SE = sst.Betaspawner(0., 5, 5)
# spawner_SE = sst.ClusterSpawner()

gate_NE = sst.Centroid(0, corner_NE, gate_radius, pg.Color(0, 0, 85), pg.Color(0, 0, 255), view, spawner_NE)
gate_NW = sst.Centroid(1, corner_NW, gate_radius, pg.Color(57, 28, 0), pg.Color(170, 85, 0), view, spawner_NW)
gate_SW = sst.Centroid(2, corner_SW, gate_radius, pg.Color(85, 0, 0), pg.Color(255, 0, 0), view, spawner_SW)
gate_SE = sst.Centroid(3, corner_SE, gate_radius, pg.Color(57, 0, 28), pg.Color(170, 0, 85), view, spawner_SE)

centroids = [gate_NE, gate_NW, gate_SW, gate_SE]
# OD_matrix = np.zeros((4, 4))
OD_matrix = np.array([[0, 300, 600, 300], [300, 0, 300, 600], [600, 300, 0, 300], [300, 600, 300, 0]])

unit_E = np.array([1, 0])
unit_N = np.array([0, 1])

wall_depth  = 0.2   # 20cm
wall_radius = 0.5 * wall_depth

#
# @todo: Add pedestrian spawning. Each centroid should spawn uniformly over T_simulation
#

if __name__ == "__main__":
    args = sys.argv[1:]

    streams     = []
    stream_mask = int(args[0])
    stream_idx  = 0
    for i in range(4):
        if (stream_mask >> i) & 0x1:
            distribution = args[1 + stream_idx * 3]
            timeshift    = float(args[2 + stream_idx * 3])
            quantum      = int(args[3 + stream_idx * 3])
            stream_idx  += 1

            stream         = sst.Stream(stream_idx, i, (i + 2) % 4, distribution, timeshift)
            if distribution == "cluster":
                OD_matrix[stream.src_id, stream.dst_id] = quantum
            else:
                psr           = quantum / 3600
                n_pedestrians = int(T_simulation * psr)
                OD_matrix[stream.src_id, stream.dst_id] = n_pedestrians

            streams.append(stream)

    # psr             = 600. / 3600
    # n_pedestrians   = int(T_simulation * psr)
    # OD_matrix[0, 2] = 24 # n_pedestrians
    # OD_matrix[1, 3] = 24 # n_pedestrians
    # OD_matrix[2, 0] = 24 # n_pedestrians
    # OD_matrix[3, 1] = 24 # n_pedestrians

    for stream in streams:
        cur_np = int(OD_matrix[stream.src_id, stream.dst_id])
        if cur_np == 0:
            continue
        # print("Spawning pedestrians from src {} to dst {}".format(i, j))
        src = centroids[stream.src_id]
        dst = centroids[stream.dst_id]

        spawning_list = []
        if stream.type == "cluster":
            for k in range(int(cur_np)):
                spawning_list.append(stream.T_shift)
        elif stream.type == "uniform":
            sample_times = ss.uniform.rvs(loc = 0, scale = T_simulation, size = cur_np)
            for k in range(int(cur_np)):
                step = int(sample_times[k] / T_step)
                spawning_list.append(step)
                # print("Spawning pedestrians for ({}, {}) @ time-step {}".format(i, j, step0))
        else: # stream.type == "beta":
            sample_times = ss.beta.rvs(10, 10, loc = 0, scale = T_simulation, size = cur_np)
            for k in range(int(cur_np)):
                t_sample = np.fmod(sample_times[k] + stream.T_shift, T_simulation)
                step = int(sample_times[k] / T_step)
                spawning_list.append(step)
                # print("Spawning pedestrians for ({}, {}) @ time-step {}".format(i, j, step0))
        spawning_list.sort()
        src.register_spawning_list(dst, spawning_list)

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
        pg.time.wait(20)
        print("Simulation step ", i, ": finish")
        # plaza.log()
