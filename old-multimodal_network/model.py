import numpy as np
import scipy as sp
import scipy.stats as ss

from collections import deque

import multimodal_network.view as mnv
import multimodal_network.node as mnn
import multimodal_network.geometry as mng
import multimodal_network.agent as mna
import multimodal_network.object as mno


#
# The following values for A, D are taken from Anvari et al. (2015)
#     PvP   0.7 +- 0.25      2.25 +- 1
#     PvC   5   +- 1.        3 +- 1
#     CvP   6   +- 1.        5 +- 1
#     CvC   8   +- 1.       12 +- 1
# @todo: Organise these parameters into 2D tables for an arbitrary number of agent classes, e.g.:
#     0 -> pedestrian
#     1 -> bicycle
#     2 -> motorcycle
#     3 -> car
#     4 -> truck
#     5 -> bus
#     6 -> tram
#     A[0, 0] = A_PvP
#     A[0, 3] = A_PvC
#     etc.
# @todo: Replace parameters by normal-ish distributions: A -> N(A_mu, A_sigma); etc.
#


# if HelbingMolnar1995:
    # T_step = 2.
# else:
    # T_step = 0.5


A_PvP =  0.7
D_PvP =  2.25

A_PvC =  5.
D_PvC =  3.

A_CvP =  6.
D_CvP =  5.

A_CvC =  8.
D_CvC = 12.

A       = np.zeros((6, 6))
A[0, 0] =  0.7
A[0, 3] =  5.
A[3, 0] =  6.
A[3, 3] =  8.

D       = np.zeros((6, 6))
D[0, 0] =  2.25
D[0, 3] =  3.
D[3, 0] =  5.
D[3, 3] = 12.

#
# @todo: lambda_03 is a personal hypothesis and lambda_30 is fairly callous, so it would be good to find better estimates (preferably empirical).
#   Perhaps Schoenauer et al., Anvari et al., or Johora & Mueller provide better answers? 
#
lambda_interagent = np.zeros((6, 6))
lambda_interagent[0, 0] = 0.5
lambda_interagent[0, 3] = 0.5
lambda_interagent[3, 0] = 0.2
lambda_interagent[3, 3] = 0.2

#
# d = min distance between centre[-line]s - sum(radii | halfstrokewidths)
# A, D are properties of the repulsor
# Note: Passing 1/D instead may allow reduction of #divisions
#
def calc_repulsion(d, A, D): # distance, amplitude, characteristic distance
    return A * np.exp(-d / D)

#
# d = min distance between centre[-line]s - sum(radii | halfstrokewidths)
# t = time since pedestrian (or driver) perceived attractor
# A, D, T are properties of the attractor
# Note: Passing 1/D, 1/T instead may allow reduction of #divisions
#
def calc_attraction(d, t, A, D, T): # distance, time, amplitude, characteristic distance, characteristic duration
    #
    # @todo: Example: callers should consider making T a random function of the pedestrians or pedestrian+attractor concerned.
    #
    # @todo: Is attraction exponentially decaying in both space and time? Or should the below formula be more complicated?
    # @todo: Consider more complicated formulae, e.g. a generic separable formula f(d) * g(t)
    #       hence, def calc_attraction(d, t, f, g): return f(d) * g(t)
    # @todo: Consider truly generic formulae, e.g f(d, t)
    #
    return A * np.exp(-(d / D + t / T))

class PartialTopology:
    def __init__(self, centroids, waypoints, agent_type = mna.Pedestrian):
        self.centroids  = centroids
        self.waypoints  = waypoints
        self.agent_type = agent_type

class Model:
    def __init__(self, width, height, screen_size, scale_factor, T_simulation, T_step, reaction_factor, partial_topologies, OD_matrices, obstacles = [], attractors = [], obscurers = [], relationships = [], background_colour = mnv.colour(128, 128, 128)):
        # self.cell_map = {}

        self.w            = width
        self.h            = height
        self.T_simulation = T_simulation
        self.T_step       = T_step
        self.T_react      = reaction_factor * T_step

        topology = {}
        centroids = []
        waypoints = []

        self.centroid_idx_offsets = {}
        for pt in partial_topologies:
            centroids += pt.centroids
            waypoints += pt.waypoints

            self.centroid_idx_offsets[pt.agent_type] = len(centroids)
            
            topology[pt.agent_type] = mnn.Graph()
            topology[pt.agent_type].add_nodelist(pt.centroids + pt.waypoints)
            topology[pt.agent_type].construct_all_routes()

            # print("Model::__init__ agent-type {} has {} centroids".format(pt.agent_type, len(centroids)))
            odm = OD_matrices[pt.agent_type]
            # print("Model::__init__ agent-type {} has a {}x{} OD-matrix".format(pt.agent_type, odm.shape[0], odm.shape[1]))
            for i in range(len(pt.centroids)):
                src = pt.centroids[i]
                for j in range(len(pt.centroids)):
                    # print("Model::__init__ agent-type {} i = {}, j = {}".format(pt.agent_type, i, j))
                    dst      = pt.centroids[j]
                    n_agents = odm[i, j]
                    if n_agents != 0:
                        # print("Model::__init__ agent-type {}[{}, {}]: {} agents travel from {} to {}".format(pt.agent_type, i, j, n_agents, src.id, dst.id))
                        src.register_stream(dst.id, topology[pt.agent_type], self.T_simulation, self.T_step, n_agents)
        
        self.centroids = centroids
        self.waypoints = waypoints
        self.topology  = topology

        self.obstacles     = obstacles
        self.attractors    = attractors
        self.relationships = relationships

        self.MRCA_flag = False

        # self.active_pedestrians   = {}
        # self.inactive_pedestrians = deque()
        self.active_vehicles      = {}
        self.n_vehicles           = 0
        self.inactive_vehicles    = deque()
        self.next_pedestrian_id   = 0
        self.next_vehicle_id      = 0
        self.cur_step             = 0

        #
        # @todo: Decompose model into cells.
        #     Cut model area into tessellating cells (squares, but regular hexagons and triangles could also work).
        #     Construct 2D array of cell-ids.
        #     Move centroids, small obstacles (and attractors, obscurers), and pedestrians into cells.
        #     As pedestrians move, it will be necessary to move them between cells.
        #
        self.grid = mng.Grid(width, height, 2., 2.)
        
        self.colour = background_colour

    def spawn_pedestrians(self):
        # print("Model::spawn_pedestrians - step ", self.cur_step, ": start")
        for centroid in self.centroids:
            centroid.spawn_pedestrians(self)
        # print("Model::spawn_pedestrians - step ", self.cur_step, ": finish")
        
    def despawn_pedestrians(self):
        self.grid.despawn_pedestrians()
        # for cell in self.ped_cells:
            # cell.despawn_pedestrians()
        return

    def spawn_vehicles(self):
        # print("Model::spawn_vehicles - step ", self.cur_step, ": start")
        for centroid in self.centroids:
            centroid.spawn_vehicles(self)
        # print("Model::spawn_vehicles - step ", self.cur_step, ": finish")
        
    def despawn_vehicles(self):
        # print("Model::despawn_vehicles - start: old n_vhcl", self.n_vehicles)
        for vhcl_id in self.inactive_vehicles:
            if vhcl_id not in self.active_vehicles:
                continue
            del(self.active_vehicles[vhcl_id])
            self.n_vehicles -= 1
        self.inactive_vehicles = []
        # print("Model::despawn_vehicles - finish: new n_vhcl", self.n_vehicles)
        return

    def spawn_agents(self):
        for centroid in self.centroids:
            centroid.spawn_agents(self)
            
        # self.spawn_pedestrians()
        # self.spawn_vehicles()
        return

    def despawn_agents(self):
        self.despawn_pedestrians()
        self.despawn_vehicles()
        return

    def add_pedestrian(self, p):
        # print("Model::add_pedestrian - start: ", self.next_pedestrian_id, p.posn)
        self.next_pedestrian_id += 1
        self.grid.add_pedestrian(p)
        return

    def add_vehicle(self, vhcl):
        self.next_vehicle_id += 1
        self.active_vehicles[vhcl.id] = vhcl
        self.n_vehicles += 1
        return

    def add_agent(self, agent, agent_type):
        # print("Model::add_agent {} is a {}".format(agent.id, agent_type))
        if agent_type == mna.Pedestrian:
            self.add_pedestrian(agent)
        else:
            self.add_vehicle(agent)
        return

    def transfer_migrants(self):
        self.grid.transfer_migrants()
        return
        
    def update_pedestrian_kinematics(self, p):
        self.grid.update_pedestrian_kinematics(p)
        return

    def remove_vehicle(self, id):
        del self.active_vehicles[id]
        return

    def draw_topology(self, view):
        for agent_type, graph in self.topology.items():
            # graph = self.topology
            for node_id, node in graph.nodes.items():
                for ctrnode_id in node.adjacency_list:
                    # print("Model::draw_topology - edge between {} and {}".format(node_id, ctrnode_id))
                    counternode = graph.nodes[ctrnode_id]
                    # view.draw_thin_line(mnv.black, node.posn, counternode.posn)
                    view.draw_line(mnv.black, node.posn, counternode.posn, 0.06125 * (1 + agent_type))
        return

    def draw(self, view):
        # pg.draw.rect(pg.display.get_surface(), self.colour, pg.Rect((0., 0.), (view.scale_factor * self.w, view.height - view.scale_factor * self.h)))
        # pg.draw.rect(view.display.get_surface(), self.colour, view.get_viewport_rect())
        view.draw_viewport_rect(self.colour)
        # for cell in self.ped_cells:
            # cell.draw_background(view) # density-coloured rectangles + optional diagonal lines
        self.grid.draw_background(view)
        for centroid in self.centroids:
            centroid.draw(view) # Dull-coloured circles
        for waypoint in self.waypoints:
            waypoint.draw(view) # Dark grey circles
        self.draw_topology(view)
        for obstacle in self.obstacles:
            # print("Model::draw obstacle", obstacle.id)
            assert(isinstance(self, Model))
            assert(isinstance(view, mnv.View))
            assert(isinstance(obstacle, mno.Repulsor))
            obstacle.draw(view) # [White fill + Black stroke] shapes
        for attractor in self.attractors:
            attractor.draw(view)    # [Gold fill + Black stroke] shapes
        # for cell in self.ped_cells:
            # cell.draw_foreground(view) # optional dashed gridlines + cell's pedestrians.draw() calls
        self.grid.draw_foreground(view)

        #
        # @todo: Add support for obscurers, cells, and/or relationships.
        #
        
        # for obscurer in self.obscurers:
            # obscurer.draw(view)     # [Black polka dot pattern fill] shapes
        # for cell in self.ped_cells:
            # cell.draw(view) # optional dashed gridlines + cell's pedestrians.draw() calls
        for k, v in self.active_vehicles.items():
            v.draw(view)
        # for relationship in self.relationships:
            # relationship.draw(view) # [Gold] connecting lines between pedestrian-pairs

    def update_vehicle_kinematics(self, vhcl, delta_s):
        old_posn = vhcl.posn
        new_posn = old_posn + delta_s

        vhcl.posn += delta_s

        #
        # @todo: Need to have a vhcl.cur_direction = unit(vhcl.cur_dst - vhcl.cur_src)
        #
        vhcl.v    += self.T_step * vhcl.f_total

        cur_src_idx = vhcl.route[vhcl.routing_idx]
        cur_src = vhcl.nodegraph.nodes[cur_src_idx]

        cur_dst_idx = vhcl.route[vhcl.routing_idx + 1]
        cur_dst = vhcl.nodegraph.nodes[cur_dst_idx]

        #
        # @todo: Replace this crude projection with something more subtle!
        #
        goal_dirn = mna.calc_unit_vector(cur_dst.posn - cur_src.posn)
        vhcl.v = np.dot(vhcl.v, goal_dirn) * goal_dirn

        if np.dot(vhcl.v, goal_dirn) < 0:
            vhcl.v = np.zeros(2)
        
        v_norm     = mna.calc_norm(vhcl.v)
        vhcl.dirn  = vhcl.v / v_norm
        if v_norm > vhcl.v_max:
            vhcl.v = vhcl.v_max * vhcl.dirn
            
        delta_finish = cur_dst.posn - vhcl.posn 
        d_sqr_dst    = np.dot(delta_finish, delta_finish)
        is_near_dst  = d_sqr_dst < cur_dst.radius**2
        
        travel_goal  = cur_dst.posn - cur_src.posn
        travel       = vhcl.posn - cur_src.posn
        has_travelled_past_dst = np.dot(travel, travel_goal) > np.dot(travel_goal, travel_goal)
        if is_near_dst or has_travelled_past_dst:
            vhcl.routing_idx += 1
            if vhcl.routing_idx + 1 == len(vhcl.route):
                self.inactive_vehicles.append(vhcl.id)
        return

    def advance_timestep(self):
        # print("Model::advance_timestep - step ", self.cur_step, ": start")
        self.grid.update_pedestrian_forces(self.obstacles, self.active_vehicles)

        for k, vhcl in self.active_vehicles.items():
            #
            # @todo: Compute resultant force
            #
            vhcl.f_total = np.zeros(2)
            vhcl.f_total += vhcl.calculate_driving_force()
            for l, w in self.active_vehicles.items():
                if w.id == vhcl.id:
                    continue

                #
                # @todo: The Vehicle::calculate_intercar_repulsion function
                #   needs to be fixed.
                #
                f_vw          = vhcl.calculate_intercar_repulsion(w, self.MRCA_flag)
                vhcl.f_total += f_vw
                
            #
            # @todo: the Vehicle::calculate_total_pedestrian_avoidance_force
            #   function needs to exist and choose all cars in all cells close
            #   enough and to the front of the vehicle.
            #
            # vhcl.f_total += vhcl.calculate_total_pedestrian_avoidance_force()

            #
            # @todo: Move into Vehicle::calculate_total_pedestrian_avoidance
            #   and probably then Grid::...
            #
            old_posn = vhcl.posn
            new_posn = vhcl.posn + self.T_step * vhcl.v
            i_old, j_old = self.grid.get_cell_coordinates(old_posn)
            i_new, j_new = self.grid.get_cell_coordinates(new_posn)
            i_min = min(i_old, i_new)
            i_max = max(i_old, i_new)
            j_min = min(j_old, j_new)
            j_max = max(j_old, j_new)
            for k in range(i_min - 1, i_max + 2):
                if k < 0 or k >= self.grid.m:
                    continue
                for l in range(j_min - 1, j_max + 2):
                    if l < 0 or l >= self.grid.n:
                        continue
                    countercell = self.grid.cells[k][l]
                    for q_id in countercell.active_pedestrians.keys():
                        q = countercell.active_pedestrians[q_id] 
                        # k_tmp, l_tmp = self.get_cell_coordinates(q.posn)
                        # if k_tmp != k or l_tmp != l:
                            # print("Model::update_pedestrian_forces - countercell[{}, {}] contains pedestrian {} at {} expected to be in cell[{}, {}]".format(k, l, q.id, q.posn, k_tmp, l_tmp))
                        # assert(k_tmp == k)
                        # assert(l_tmp == l)
                        f_vq = vhcl.calculate_pedestrian_repulsion(q, False)
                        # print("Model::advance_timestep - interpedestrian repulsion force (p.id, q.id, f_R): ", p.id, q.id, p.f_total)
                        vhcl.f_total += f_vq
                        assert(np.dot(vhcl.f_total, vhcl.f_total) >= 0.)
                    # print("Model::advance_timestep - total interpedestrian repulsion force (id, f_R): ", p.id, p.f_total)
            


            cur_src_idx = vhcl.route[vhcl.routing_idx]
            cur_src = vhcl.nodegraph.nodes[cur_src_idx]

            cur_dst_idx = vhcl.route[vhcl.routing_idx + 1]
            cur_dst = vhcl.nodegraph.nodes[cur_dst_idx]
            
            u          = cur_dst.posn - cur_src.posn
            u         /= np.sqrt(np.dot(u, u))
            vhcl.f_total  = np.dot(vhcl.f_total, u) * u
            
            #
            # @todo: Start supporting vehicle-obstacle interactions!
            #
            for b in self.obstacles:
                pass
                # f_vb       = vhcl.calculate_obstacle_repulsion(b, self.MRCA_flag)
                # vhcl.f_total += f_vb

        self.cur_step += 1

        #
        # Movement stage
        #
        self.grid.update_pedestrian_kinematics(self.T_step)
        self.transfer_migrants()

        for k, vhcl in self.active_vehicles.items():
            #
            # @todo: Compute new position and add to position history.
            # @todo: Move between cells if necessary.
            #
            # vhcl.posn += self.T_step * v.v
            # vhcl.v    += self.T_step * v.f_total

            self.update_vehicle_kinematics(vhcl, self.T_step * vhcl.v)
        # print("Model::advance_timestep - step ", self.cur_step, ": finish")
        return
