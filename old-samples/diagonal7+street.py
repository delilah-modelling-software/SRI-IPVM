import sys
import numpy as np
import scipy as sp
import scipy.stats as ss
import pygame as pg

# import shared_spaces_types as sst
import multimodal_network.view as mnv
import multimodal_network.model as mnm
import multimodal_network.node as mnn
import multimodal_network.spawner as mnsp
import multimodal_network.shape as mnsh
import multimodal_network.object as mno
import multimodal_network.agent as mna

width  = 110
height =  36

scale_factor   = 14
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width,screen_height)

view = mnv.View(width, height, screen_size, scale_factor, pedestrian_upscale_factor = 1.)

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

spawner_NE = mnsp.Betaspawner(0., 10, 10)
spawner_N  = mnsp.Uniformspawner()
spawner_NW = mnsp.Betaspawner(0., 5, 5)
spawner_SW = mnsp.Betaspawner(0., 10, 10)
spawner_S  = mnsp.Uniformspawner()
spawner_SE = mnsp.Betaspawner(0., 5, 5)
# spawner_SE = sst.ClusterSpawner()

unit_E = np.array([1, 0])
unit_N = np.array([0, 1])

gate_NE = mnn.Centroid(0, centre_E + 6 * (unit_N - unit_E), gate_radius, mnv.colour(  0,   0, 170), mnv.colour(  0,   0, 255), view, spawner_NE)
gate_N  = mnn.Centroid(1, centre_N,  gate_radius, mnv.colour(114,  56,   0), mnv.colour(170,  85,   0), view, spawner_NE)
gate_NW = mnn.Centroid(2, centre_W + 6 * (unit_N + unit_E), gate_radius, mnv.colour( 56, 114,   0), mnv.colour( 85, 170,   0), view, spawner_NW)
gate_SW = mnn.Centroid(3, centre_W - 6 * (unit_N - unit_E), gate_radius, mnv.colour(170,   0,   0), mnv.colour(255,   0,   0), view, spawner_SW)
gate_S  = mnn.Centroid(4, centre_S,  gate_radius, mnv.colour(114,   0,  56), mnv.colour(170,   0,  85), view, spawner_SE)
gate_SE = mnn.Centroid(5, centre_E - 6 * (unit_N + unit_E), gate_radius, mnv.colour( 56,   0, 114), mnv.colour( 85,   0, 170), view, spawner_SE)

cargate_diameter = 3.
cargate_radius   = 0.5 * gate_diameter

# carspawner_E = mnsp.Betaspawner(0., 15, 15)
# carspawner_W = mnsp.Betaspawner(0., 10, 10)
carspawner_E = mnsp.Uniformspawner()
carspawner_W = mnsp.Uniformspawner()

cargate_NE = mnn.Centroid(6, centre_E + cargate_radius * unit_N, cargate_radius, mnv.colour(170,   0, 170), mnv.colour(255,   0, 255), view, carspawner_E, agent_type = mna.Vehicle)
cargate_NW = mnn.Centroid(7, centre_W + cargate_radius * unit_N, cargate_radius, mnv.colour(170,   0, 170), mnv.colour(255,   0, 255), view, carspawner_E, agent_type = mna.Vehicle)
cargate_SW = mnn.Centroid(8, centre_W - cargate_radius * unit_N, cargate_radius, mnv.colour(  0, 170, 170), mnv.colour(  0,   0, 255), view, carspawner_W, agent_type = mna.Vehicle)
cargate_SE = mnn.Centroid(9, centre_E - cargate_radius * unit_N, cargate_radius, mnv.colour(  0, 170, 170), mnv.colour(  0,   0, 255), view, carspawner_W, agent_type = mna.Vehicle)

centroids            = [gate_NE, gate_N, gate_NW, gate_SW, gate_S, gate_SE, cargate_NE, cargate_NW, cargate_SW, cargate_SE]
car_centroids        = centroids[6:] # [cargate_NE, cargate_NW, cargate_SW, cargate_SE]
pedestrian_OD_matrix = np.array([[0, 25, 25, 50, 25, 25], [25, 0, 25, 25, 50, 25], [25, 25, 0, 25, 25, 50], [50, 25, 25, 0, 25, 25], [25, 50, 25, 25, 0, 25], [25, 25, 50, 25, 25, 0]])
car_OD_matrix        = np.array([[0, 0, 0, 0], [10, 0, 0, 0], [0, 0, 0, 0], [0, 0, 10, 0]])

waypoint_N = mnn.Waypoint(0, centre_true + 6 * unit_N, gate_radius, view)
waypoint_S = mnn.Waypoint(1, centre_true - 6 * unit_N, gate_radius, view)

wall_depth  = 0.2   # 20cm
wall_radius = 0.5 * wall_depth
wall_ENE_INE_shape = mnsh.Segment(gate_NE.posn - gate_NW.radius * (unit_E - unit_N), waypoint_N.posn + gate_NE.radius * (unit_E + unit_N), 0.1)
wall_WNW_INW_shape = mnsh.Segment(gate_NW.posn + gate_NW.radius * (unit_E + unit_N), waypoint_N.posn - gate_NW.radius * (unit_E - unit_N), 0.1)
wall_WSW_ISW_shape = mnsh.Segment(gate_SW.posn + gate_SW.radius * (unit_E - unit_N), waypoint_S.posn - gate_SW.radius * (unit_E + unit_N), 0.1)
wall_ESE_ISE_shape = mnsh.Segment(gate_SE.posn - gate_SE.radius * (unit_E + unit_N), waypoint_S.posn + gate_SE.radius * (unit_E - unit_N), 0.1)
wall_INE_NNE_shape = mnsh.Segment(waypoint_N.posn + gate_NE.radius * (unit_E + unit_N), gate_N.posn + gate_N.radius * (unit_E - unit_N), 0.1)
wall_INW_NNW_shape = mnsh.Segment(waypoint_N.posn - gate_NW.radius * (unit_E - unit_N), gate_N.posn - gate_N.radius * (unit_E + unit_N), 0.1)
wall_ISW_SSW_shape = mnsh.Segment(waypoint_S.posn - gate_SW.radius * (unit_E + unit_N), gate_S.posn - gate_S.radius * (unit_E - unit_N), 0.1)
wall_ISE_SSE_shape = mnsh.Segment(waypoint_S.posn + gate_NE.radius * (unit_E - unit_N), gate_S.posn + gate_S.radius * (unit_E + unit_N), 0.1)

A_wall       = 10.
D_wall       =  0.2
wall_ENE_INE = mno.Repulsor( 0, A_wall, D_wall, wall_ENE_INE_shape)
wall_WNW_INW = mno.Repulsor( 1, A_wall, D_wall, wall_WNW_INW_shape)
wall_WSW_ISW = mno.Repulsor( 2, A_wall, D_wall, wall_WSW_ISW_shape)
wall_ESE_ISE = mno.Repulsor( 3, A_wall, D_wall, wall_ESE_ISE_shape)
wall_INE_NNE = mno.Repulsor( 4, A_wall, D_wall, wall_INE_NNE_shape)
wall_INW_NNW = mno.Repulsor( 5, A_wall, D_wall, wall_INW_NNW_shape)
wall_ISW_SSW = mno.Repulsor( 6, A_wall, D_wall, wall_ISW_SSW_shape)
wall_ISE_SSE = mno.Repulsor( 7, A_wall, D_wall, wall_ISE_SSE_shape)

#
# @todo: Add pedestrian spawning. Each centroid should spawn uniformly over T_simulation
#

# for i in range(4):
    # print("CarCentroid[{}] = ".format(i), car_centroids[i].posn)

if __name__ == "__main__":
    streams = []
    stream_idx = 0
    for i in range(6):
        for j in range(6):
            n_ped = pedestrian_OD_matrix[i, j]
            if int(n_ped) == 0:
                continue

            src          = centroids[i]
            dst          = centroids[j]
            spawner      = src.streamspawner
            sample_times = spawner.generate_spawning_times(T_simulation, T_step, n_ped)
            src.register_spawning_list(dst, sample_times)
            
    for i in range(4):
        for j in range(4):
            n_cars = car_OD_matrix[i, j]
            if int(n_cars) == 0:
                continue
                
            src          = car_centroids[i]
            dst          = car_centroids[j]
            # print("Main - registering a list of {} cars to travel from {} to {}".format(n_cars, src.posn, dst.posn))
            spawner      = src.streamspawner
            sample_times = spawner.generate_spawning_times(T_simulation, T_step, n_cars)
            src.register_spawning_list(dst, sample_times)

    obstacles = [] # [wall_ENE_INE, wall_WNW_INW, wall_WSW_ISW, wall_ESE_ISE, wall_INE_NNE, wall_INW_NNW, wall_ISW_SSW, wall_ISE_SSE]
    waypoints = [waypoint_N, waypoint_S]

    #
    # @ todo: Add some more attractors and obscurers (perhaps some lamp-posts, etc.).
    #

    plaza           = mnm.Model(width, height, screen_size, scale_factor, T_step, 4, centroids, waypoints, obstacles, [], [], [])
    plaza.MRCA_flag = False

    # pg.init()
    # screen_display = pg.display.set_mode(screen_size)

    for i in range(L_simulation):
        # print("Simulation step ", i, ": start")
        plaza.spawn_agents()
        plaza.advance_timestep()
        plaza.draw(view)
        plaza.despawn_agents()
        view.flip()
        # pg.display.flip()
        # pg.time.wait(1)
        # print("Simulation step ", i, ": finish")
        # plaza.log()
