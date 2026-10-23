import shared_spaces_types as sst
import numpy as np
import scipy as sp
import scipy.stats as ss
import pygame as pg

width  = 210
height = 210

scale_factor   = 3
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width,screen_height)

# rhc_surface    = pg.Surface(screen_size, screen_display)
# lhc_surface

#
# @todo: Three-stage render. Right-hand coordinates, vertical flip, screen-blit.
#     Render everything to rhc_surface using draw calls;
#     lhc_surface = pg.transform.flip(rhc_surface, flip_x = False, flip_y = True)
#     Call screen_surface.blit(lhc_surface, (0, 0))
#

centre_true = np.array([105, 105])

corner_NE = np.array([205, 205])
corner_NW = np.array([  5, 205])
corner_SW = np.array([  5,   5])
corner_SE = np.array([205,   5])

centre_E = np.array([205, 105])
centre_N = np.array([105, 205])
centre_W = np.array([  5, 105])
centre_S = np.array([105,   5])

T_step       = 0.5 # 0.125    # Duration of each simulation time-step
T_simulation = 60 # 15 * 60  # 15 minutes
L_simulation = int(T_simulation / T_step)
L_history    = 4        # Length of pedestrian history *and* number of time-steps that matches human reaction time.
T_reaction   = L_history * T_step   # 0.5s
T_relaxation = 2.       # Duration required by pedestrian to reaccelerate to full speed

gate_diameter = 6.
gate_radius   = 0.5 * gate_diameter

gate_E = sst.Centroid(0, centre_E, gate_radius, pg.Color(0, 0, 85), pg.Color(0, 0, 255), screen_size, scale_factor)
gate_N = sst.Centroid(1, centre_N, gate_radius, pg.Color(0, 85, 85), pg.Color(0, 170, 170), screen_size, scale_factor)
gate_W = sst.Centroid(2, centre_W, gate_radius, pg.Color(85, 0, 0), pg.Color(255, 0, 0), screen_size, scale_factor)
gate_S = sst.Centroid(3, centre_S, gate_radius, pg.Color(85, 0, 85), pg.Color(170, 0, 170), screen_size, scale_factor)
# gate_E = sst.Centroid(0, centre_E, gate_radius, pg.Color(96, 64, 32), pg.Color(192, 128,  64), screen_size, scale_factor)
# gate_N = sst.Centroid(1, centre_N, gate_radius, pg.Color(96, 32, 64), pg.Color(192,  64, 128), screen_size, scale_factor)
# gate_W = sst.Centroid(2, centre_W, gate_radius, pg.Color(96, 96,  0), pg.Color(192, 192,   0), screen_size, scale_factor)
# gate_S = sst.Centroid(3, centre_S, gate_radius, pg.Color(96,  0, 96), pg.Color(192,   0, 192), screen_size, scale_factor)

centroids = [gate_E, gate_N, gate_W, gate_S]
OD_matrix = np.zeros((4, 4))

unit_E = np.array([1, 0])
unit_N = np.array([0, 1])

wall_depth  = 0.2   # 20cm
wall_radius = 0.5 * wall_depth

#
# @todo: Add pedestrian spawning. Each centroid should spawn uniformly over T_simulation
#

psr             = 600. / 3600
n_pedestrians   = int(T_simulation * psr)
OD_matrix[0, 2] = n_pedestrians
OD_matrix[1, 3] = n_pedestrians
OD_matrix[2, 0] = n_pedestrians
OD_matrix[3, 1] = n_pedestrians

for i in range(4):
    for j in range(4):
        cur_np = int(OD_matrix[i][j])
        if cur_np == 0:
            continue
        # print("Spawning pedestrians from src {} to dst {}".format(i, j))
        src = centroids[i]
        dst = centroids[j]
        spawning_list = []
        for k in range(cur_np):
            step0 = int(ss.uniform.rvs(loc = 0, scale = T_simulation) / T_step)
            spawning_list.append(step0)
            # print("Spawning pedestrians for ({}, {}) @ time-step {}".format(i, j, step0))
        spawning_list.sort()
        src.register_spawning_list(dst, spawning_list)

#
# @todo: Construct wall_{NNW, WNW, WSW, SSW, SSE, ESE} and append to obstacles
# @todo: Add some circular obstacles too.
#
A_wall   = 10.
D_wall   =  0.2
wall_ENE = sst.Segment(centre_E + gate_radius * unit_N, corner_NE, wall_radius, screen_size, scale_factor)
wall_NNE = sst.Segment(centre_N + gate_radius * unit_E, corner_NE, wall_radius, screen_size, scale_factor)

obstacle_wall_ENE = sst.Repulsor(0, A_wall, D_wall, wall_ENE)
obstacle_wall_NNE = sst.Repulsor(1, A_wall, D_wall, wall_NNE)

railing_depth  = 0.04   # 4cm 
railing_radius = 0.5 * railing_depth
railing_inset  = 3.     # 3m

fountain_radius  = 5.
roundabout_width = 5.

A_fountain     = 10.    # fountain repulsion amplitude
D_fountain     = 0.2    # fountain repulsion reduction characteristic distance
fountain_shape = sst.Circle(centre_true, fountain_radius, screen_size, scale_factor)
fountain       = sst.Repulsor(8, A_fountain, D_fountain, fountain_shape)

A_railing = 10.
D_railing =  0.2
railing_E = \
    sst.Segment(
        centre_true + (fountain_radius + roundabout_width + railing_inset) * unit_E,
        centre_E - railing_inset * unit_E,
        railing_radius,
	screen_size,
	scale_factor
    )
obstacle_railing_E = sst.Repulsor(9, A_railing, D_railing, railing_E)

A_pillar = 10.
D_pillar = 0.2
pillar_radius = 0.25 * railing_inset
pillar_E = sst.Circle(centre_E - (gate_radius + pillar_radius) * unit_E, pillar_radius, screen_size, scale_factor)
obstacle_pillar_E = sst.Repulsor(13, A_railing, D_railing, pillar_E)

obstacles = [obstacle_wall_ENE, obstacle_wall_NNE, obstacle_railing_E, obstacle_pillar_E]

#
# @ todo: Add some more attractors and obscurers (perhaps some trees, etc.).
#

plaza = sst.Model(width, height, screen_size, scale_factor, T_step, 4, centroids, obstacles, [], [], [])

pg.init()
screen_display = pg.display.set_mode(screen_size)

view = sst.View(width, height, screen_size, scale_factor)

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
