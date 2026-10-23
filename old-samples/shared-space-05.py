import sys
import numpy as np
import scipy as sp
import scipy.stats as ss

# import shared_spaces_types as sst
import multimodal_network.view as mnv
import multimodal_network.model as mnm
import multimodal_network.node as mnn
# import multimodal_network.spawner as mnsp
import multimodal_network.distributor as mnd
import multimodal_network.shape as mnsh
import multimodal_network.object as mno
import multimodal_network.agent as mna

width  = 110
height =  48

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
centre_N = np.array([centre_true[0], height - 3])
centre_W = np.array([  5, centre_true[1]])
centre_S = np.array([centre_true[0],   3])

T_step       = 0.5 # 0.125    # Duration of each simulation time-step
T_reaction   = T_step
T_simulation = 3 * 60 # 15 * 60  # 15 minutes
L_simulation = int(T_simulation / T_step)

gate_diameter = 3.
gate_radius   = 0.5 * gate_diameter

# spawner_NE = mnsp.Betaspawner(0., 10, 10)
# spawner_N  = mnsp.Uniformspawner()
# spawner_NW = mnsp.Betaspawner(0., 5, 5)
# spawner_SW = mnsp.Betaspawner(0., 10, 10)
# spawner_S  = mnsp.Uniformspawner()
# spawner_SE = mnsp.Betaspawner(0., 5, 5)
# spawner_SE = sst.ClusterSpawner()

spawner_NE = mnd.DistributorBeta(0., 10, 10)
spawner_N  = mnd.DistributorUniform()
spawner_NW = mnd.DistributorBeta(0., 5, 5)
spawner_SW = mnd.DistributorBeta(0., 10, 10)
spawner_S  = mnd.DistributorUniform()
spawner_SE = mnd.DistributorBeta(0., 5, 5)

unit_E = np.array([1, 0])
unit_N = np.array([0, 1])

gate_NE = mnn.Centroid(0, centre_E + 6 * unit_N, gate_radius, mnv.colour(  0,   0, 170), mnv.colour(  0,   0, 255), view, spawner_NE)
gate_N  = mnn.Centroid(1, centre_N,  gate_radius, mnv.colour(114,  56,   0), mnv.colour(170,  85,   0), view, spawner_NE)
gate_NW = mnn.Centroid(2, centre_W + 6 * unit_N, gate_radius, mnv.colour( 56, 114,   0), mnv.colour( 85, 170,   0), view, spawner_NW)
gate_SW = mnn.Centroid(3, centre_W - 6 * unit_N, gate_radius, mnv.colour(170,   0,   0), mnv.colour(255,   0,   0), view, spawner_SW)
gate_S  = mnn.Centroid(4, centre_S,  gate_radius, mnv.colour(114,   0,  56), mnv.colour(170,   0,  85), view, spawner_SE)
gate_SE = mnn.Centroid(5, centre_E - 6 * unit_N, gate_radius, mnv.colour( 56,   0, 114), mnv.colour( 85,   0, 170), view, spawner_SE)

cargate_diameter = 3.
cargate_radius   = 0.5 * gate_diameter

# carspawner_E = mnsp.Betaspawner(0., 15, 15)
# carspawner_W = mnsp.Betaspawner(0., 10, 10)
# carspawner_E = mnsp.Uniformspawner()
# carspawner_W = mnsp.Uniformspawner()

carspawner_E = mnd.DistributorUniform()
carspawner_W = mnd.DistributorUniform()

cargate_NE = mnn.Centroid(6, centre_E + 2 * cargate_radius * unit_N, cargate_radius, mnv.colour(170,   0, 170), mnv.colour(255,   0, 255), view, carspawner_E, agent_type = mna.Vehicle)
cargate_NW = mnn.Centroid(7, centre_W + 2 * cargate_radius * unit_N, cargate_radius, mnv.colour(170,   0, 170), mnv.colour(255,   0, 255), view, carspawner_E, agent_type = mna.Vehicle)
cargate_SW = mnn.Centroid(8, centre_W - 2 * cargate_radius * unit_N, cargate_radius, mnv.colour(  0, 170, 170), mnv.colour(  0,   0, 255), view, carspawner_W, agent_type = mna.Vehicle)
cargate_SE = mnn.Centroid(9, centre_E - 2 * cargate_radius * unit_N, cargate_radius, mnv.colour(  0, 170, 170), mnv.colour(  0,   0, 255), view, carspawner_W, agent_type = mna.Vehicle)

ptn_centroids        = [gate_NE, gate_N, gate_NW, gate_SW, gate_S, gate_SE]
car_centroids        = [cargate_NE, cargate_NW, cargate_SW, cargate_SE]
pedestrian_OD_matrix = np.array([[0, 25, 25, 50, 25, 25], [25, 0, 25, 25, 50, 25], [25, 25, 0, 25, 25, 50], [50, 25, 25, 0, 25, 25], [25, 50, 25, 25, 0, 25], [25, 25, 50, 25, 25, 0]])
car_OD_matrix        = np.array([[0, 0, 0, 0], [10, 0, 0, 0], [0, 0, 0, 0], [0, 0, 10, 0]])

waypoint_NE = mnn.Waypoint(10, centre_E + 6 * (unit_N - unit_E), gate_radius, view)
waypoint_NW = mnn.Waypoint(11, centre_W + 6 * (unit_N + unit_E), gate_radius, view)
waypoint_SW = mnn.Waypoint(12, centre_W - 6 * (unit_N - unit_E), gate_radius, view)
waypoint_SE = mnn.Waypoint(13, centre_E - 6 * (unit_N + unit_E), gate_radius, view)
waypoint_N = mnn.Waypoint(14, centre_true + 6 * unit_N, gate_radius, view)
waypoint_S = mnn.Waypoint(15, centre_true - 6 * unit_N, gate_radius, view)
waypoints  = [waypoint_NE, waypoint_N, waypoint_NW, waypoint_SW, waypoint_S, waypoint_SE]

mnn.symconnect(gate_NE,  waypoint_NE)
mnn.symconnect(gate_NW,  waypoint_NW)
mnn.symconnect(gate_SW,  waypoint_SW)
mnn.symconnect(gate_SE,  waypoint_SE)

mnn.symconnect(gate_N,  waypoint_N)
mnn.symconnect(gate_S,  waypoint_S)
mnn.omniconnect([waypoint_NE, waypoint_N, waypoint_NW, waypoint_SW, waypoint_S, waypoint_SE])
cargate_NW.connect(cargate_NE)
cargate_SE.connect(cargate_SW)

partial_topologies = [mnm.PartialTopology(ptn_centroids, waypoints, 0), mnm.PartialTopology(car_centroids, [], 3)]

# mnn.symconnect(gate_NE, waypoint_N)
# mnn.symconnect(gate_NE, gate_NW)
# mnn.symconnect(gate_NE, gate_SW)
# mnn.symconnect(gate_NE, waypoint_S)
# mnn.symconnect(gate_NE, gate_SE)
# mnn.symconnect(waypoint_N, gate_NW)
# mnn.symconnect(waypoint_N, gate_SW)
# mnn.symconnect(waypoint_N, waypoint_S)
# mnn.symconnect(waypoint_N, gate_SE)
# mnn.symconnect(gate_NW, gate_SW)
# mnn.symconnect(gate_NW, waypoint_S)
# mnn.symconnect(gate_NW, gate_SE)
# mnn.symconnect(gate_SW, waypoint_S)
# mnn.symconnect(gate_SW, gate_SE)
# mnn.symconnect(waypoint_S, gate_SE)


# topology = mnn.Graph()
# topology.add_nodelist(centroids + waypoints)

building_offset = gate_radius + 3.

wall_depth  = 4.  # 2m
wall_radius = 0.5 * wall_depth

bldng_NE_pt_0  = gate_NE.posn    + building_offset * unit_N
bldng_NE_pt_1  = np.array([gate_NE.posn[0], gate_N.posn[1]])
bldng_NE_pt_2  = gate_N.posn     + building_offset * unit_E
bldng_NE_pt_3  = waypoint_N.posn + building_offset * (unit_E + unit_N)
bldng_NE_pts   = [bldng_NE_pt_0, bldng_NE_pt_1, bldng_NE_pt_2, bldng_NE_pt_3]
bldng_NE_shape = mnsh.Polygon(bldng_NE_pts, wall_radius)

bldng_NW_pt_0  = waypoint_N.posn - building_offset * (unit_E - unit_N)
bldng_NW_pt_1  = gate_N.posn     - building_offset * unit_E
bldng_NW_pt_2  = np.array([gate_NW.posn[0], gate_N.posn[1]])
bldng_NW_pt_3  = gate_NW.posn    + building_offset * unit_N
bldng_NW_pts   = [bldng_NW_pt_0, bldng_NW_pt_1, bldng_NW_pt_2, bldng_NW_pt_3]
bldng_NW_shape = mnsh.Polygon(bldng_NW_pts, wall_radius)

bldng_SW_pt_0  = gate_S.posn     - building_offset * unit_E
bldng_SW_pt_1  = waypoint_S.posn - building_offset * (unit_E + unit_N)
bldng_SW_pt_2  = gate_SW.posn    - building_offset * unit_N
bldng_SW_pt_3  = np.array([gate_SW.posn[0], gate_S.posn[1]])
bldng_SW_pts   = [bldng_SW_pt_0, bldng_SW_pt_1, bldng_SW_pt_2, bldng_SW_pt_3]
bldng_SW_shape = mnsh.Polygon(bldng_SW_pts, wall_radius)

bldng_SE_pt_0  = np.array([gate_SE.posn[0], gate_S.posn[1]])
bldng_SE_pt_1  = gate_SE.posn    - building_offset * unit_N
bldng_SE_pt_2  = waypoint_S.posn + building_offset * (unit_E - unit_N)
bldng_SE_pt_3  = gate_S.posn     + building_offset * unit_E
bldng_SE_pts   = [bldng_SE_pt_0, bldng_SE_pt_1, bldng_SE_pt_2, bldng_SE_pt_3]
bldng_SE_shape = mnsh.Polygon(bldng_SE_pts, wall_radius)


# wall_ENE_INE_shape = mnsh.Segment(gate_NE.posn + building_offset * unit_N, waypoint_N.posn + building_offset * (unit_E + unit_N), wall_radius)
wall_WNW_INW_shape = mnsh.Segment(gate_NW.posn + building_offset * unit_N, waypoint_N.posn - building_offset * (unit_E - unit_N), wall_radius)
wall_WSW_ISW_shape = mnsh.Segment(gate_SW.posn - building_offset * unit_N, waypoint_S.posn - building_offset * (unit_E + unit_N), wall_radius)
wall_ESE_ISE_shape = mnsh.Segment(gate_SE.posn - building_offset * unit_N, waypoint_S.posn + building_offset * (unit_E - unit_N), wall_radius)

# wall_INE_NNE_shape = mnsh.Segment(waypoint_N.posn + building_offset * (unit_E + unit_N), gate_N.posn + building_offset * unit_E, wall_radius)
wall_INW_NNW_shape = mnsh.Segment(waypoint_N.posn - building_offset * (unit_E - unit_N), gate_N.posn - building_offset * unit_E, wall_radius)
wall_ISW_SSW_shape = mnsh.Segment(waypoint_S.posn - building_offset * (unit_E + unit_N), gate_S.posn - building_offset * unit_E, wall_radius)
wall_ISE_SSE_shape = mnsh.Segment(waypoint_S.posn + building_offset * (unit_E - unit_N), gate_S.posn + building_offset * unit_E, wall_radius)

A_wall       = 10.
D_wall       =  0.2
# wall_ENE_INE = mno.Repulsor( 0, A_wall, D_wall, wall_ENE_INE_shape)
wall_WNW_INW = mno.Repulsor( 1, A_wall, D_wall, wall_WNW_INW_shape)
wall_WSW_ISW = mno.Repulsor( 2, A_wall, D_wall, wall_WSW_ISW_shape)
wall_ESE_ISE = mno.Repulsor( 3, A_wall, D_wall, wall_ESE_ISE_shape)
# wall_INE_NNE = mno.Repulsor( 4, A_wall, D_wall, wall_INE_NNE_shape)
wall_INW_NNW = mno.Repulsor( 5, A_wall, D_wall, wall_INW_NNW_shape)
wall_ISW_SSW = mno.Repulsor( 6, A_wall, D_wall, wall_ISW_SSW_shape)
wall_ISE_SSE = mno.Repulsor( 7, A_wall, D_wall, wall_ISE_SSE_shape)

bldng_NE       = mno.Repulsor( 8, A_wall, D_wall, bldng_NE_shape)
bldng_NW       = mno.Repulsor( 9, A_wall, D_wall, bldng_NW_shape)
bldng_SW       = mno.Repulsor(10, A_wall, D_wall, bldng_SW_shape)
bldng_SE       = mno.Repulsor(10, A_wall, D_wall, bldng_SE_shape)

OD_matrices = {0: pedestrian_OD_matrix, 3: car_OD_matrix}

#
# @todo: Add pedestrian spawning. Each centroid should spawn uniformly over T_simulation
#

for i in range(4):
    print("CarCentroid[{}] = ".format(i), car_centroids[i].posn)

if __name__ == "__main__":

    obstacles = [bldng_NE, bldng_NW, bldng_SW, bldng_SE]
    # obstacles = [wall_ENE_INE, wall_WNW_INW, wall_WSW_ISW, wall_ESE_ISE, wall_INE_NNE, wall_INW_NNW, wall_ISW_SSW, wall_ISE_SSE]
    
    plaza           = mnm.Model(width, height, screen_size, scale_factor, T_simulation, T_step, 4, partial_topologies, OD_matrices, obstacles, [], [], [])
    plaza.MRCA_flag = False

    #
    # @ todo: Add some more attractors and obscurers (perhaps some lamp-posts, etc.).
    #

    # print("Pre-Simulation: route-log")
    # topology = plaza.topology
    # for i in range(len(plaza.centroids)):
        # node    = plaza.centroids[i]
        # for j in range(i + 1, len(plaza.centroids)):
            # ctrnode = plaza.centroids[j]
            # route_ij = topology.construct_route(node.id, ctrnode.id)
            # if route_ij != None:
                # print("Route from {} to {} is {}".format(node.id, ctrnode.id, route_ij))
    
    # plaza.draw_topology(view)
    # view.flip()
    # mnv.delay(2000)

    for i in range(L_simulation):
        print("Simulation step ", i, ": start")
        plaza.spawn_agents()
        plaza.advance_timestep()
        assert(isinstance(plaza, mnm.Model))
        assert(isinstance(view, mnv.View))
        plaza.draw(view)
        plaza.despawn_agents()
        view.flip()
        print("Simulation step ", i, ": finish")
        # plaza.log()
