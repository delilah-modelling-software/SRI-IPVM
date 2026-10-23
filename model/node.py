import numpy as np
import scipy as sp
import scipy.stats as ss

from collections import deque

import view.view as svv
import model.distributor as smd
import model.agent.agent as smaa

class Stream:
    def __init__(self, id, src_id, dst_id, type, T_shift):
        self.id      = id
        self.src_id  = src_id
        self.dst_id  = dst_id
        self.type    = type
        self.T_shift = T_shift

#
# @todo: Consider generating waypoints and adding connections based on a visibility graph.
#
class Node: # intersections, gates, doorways, elevators, transporters, stopped [public transport] vehicles
    def __init__(self, id, posn, r, colour, view, agent_type = smaa.Pedestrian):
        self.id = id
        
        self.posn       = posn
        self.radius     = r
        self.colour     = colour
        self.view       = view
        self.agent_type = agent_type
        self.adjacency_list = []
        self.visited_flag   = False
        return
    
    def register_adjacency_list(self, adjacent_nodes):
        self.adjacency_list += adjacent_nodes
        return
    
    def connect(self, other):
        # print("Node::connect - {} -> {}".format(self.id, other.id))
        self.adjacency_list.append(other.id)
        return
    
    def set_visited_flag(self):
        self.visited_flag = True
        return

    def clear_visited_flag(self):
        self.visited_flag = False
        return

    def get_desc_list(self):
        # return [self.id, self.posn, self.radius]
        return [self.id, self.posn[0], self.posn[1], self.radius]

def symconnect(node1, node2):
    node1.connect(node2)
    node2.connect(node1)
    return

def omniconnect(nodelist):
    for i in range(len(nodelist)):
        node = nodelist[i]
        for j in range(i + 1, len(nodelist)):
            # print("mnn.omniconnect - {} <-> {}".format(i, j))
            counternode = nodelist[j]
            symconnect(node, counternode)
    return

def interconnect(nodelist1, nodelist2):
    for i in range(len(nodelist1)):
        node = nodelist1[i]
        for j in range(len(nodelist2)):
            # print("mnn.interconnect - {} <-> {}".format(i, j))
            counternode = nodelist2[j]
            symconnect(node, counternode)
    return

#
#
# @todo: If 2^8 nodes are insufficient, upgrade to a larger power-of-2 or consider
#   using a non-power-of-2 as the arithmetic-codent.
# Implementation Note: More than 2^16 - 1 nodes
#   would be problematic for an integer word-size of 32 bits.
#
max_node_count = 1 << 8
def calc_route_id(src_id, dst_id):
    return src_id * max_node_count + dst_id

class Graph:
    def __init__(self):
        self.nodes  = {}
        self.routes = {}
        return

    def add_node(self, node):
        self.nodes[node.id] = node
        return

    def add_nodelist(self, nodelist):
        for node in nodelist:
            self.add_node(node)
        return

    #
    # A* rather than Dijkstra
    #
    def construct_route(self, src_id, dst_id):
        # print("Graph::construct_route over {} from {} to {}".format(self.nodes.keys(), src_id, dst_id))
        src = self.nodes[src_id]
        dst = self.nodes[dst_id]

        cur_node_set  = set([src_id])
        prev_node_map = {}

        #
        # Note: If this code is ever converted to C++, g_map can be an std::hashmap,
        # but f_map should be implemented as an std::map, because it is implemented
        # as a sorted tree and will remove the per-iteration re-sorting unnecessary.
        #
        g_map = {src_id: 0}
        delta = dst.posn - src.posn
        h     = np.sqrt(np.dot(delta, delta))
        f_map = {src_id: h}
        # print("Graph::construct_route post-setup with g_map {}, f_map {}".format(g_map, f_map))

        while len(cur_node_set) > 0:
            # print("Graph::construct_route choosing element from {} with lowest estimated distance from {}".format(cur_node_set, f_map))
            
            f_pairs = list(f_map.items())
            f_pairs.sort(key = lambda pair: -pair[1])
            # sort(f_items, key = lambda item: item[1])

            cur_item   = f_pairs.pop()
            cur_id     = cur_item[0] 

            # print("Graph::construct_route deleting {} from {} ".format(cur_id, f_map))
            del f_map[cur_id]

            cur_node_set.remove(cur_id)
            if cur_id == dst_id:
                # print("Graph::construct_route found route - last two nodes are {} and {}".format(prev_node_map[cur_id], cur_id))
                route = [dst_id]
                while cur_id != src_id:
                    cur_id = prev_node_map[cur_id]
                    route.append(cur_id)
                route.reverse()
                # print("Route constructed", route)
                return route
            
            cur_node = self.nodes[cur_id]
            # print("Graph::construct_route node {} has neighbours {}".format(cur_id, cur_node.adjacency_list))
            for next_id in cur_node.adjacency_list:
                next_node  = self.nodes[next_id]
                delta      = next_node.posn - cur_node.posn
                d_cur_next = np.sqrt(np.dot(delta, delta))
                g_score    = g_map[cur_id] + d_cur_next
                # print("Graph::construct_route consideration of {} with g_score {}".format(next_id, g_score))
                if next_id not in g_map or g_score < g_map[next_id]:
                    prev_node_map[next_id] = cur_id
                    
                    g_map[next_id] = g_score
                    delta = dst.posn - next_node.posn
                    h = np.sqrt(np.dot(delta, delta))
                    f_map[next_id] = g_score + h
                    # print("Graph::construct_route post-addition with g_map {}, f_map {}".format(g_map, f_map))

                    #
                    # Note: this assumes set-addition is idempotent,
                    # which it should be!
                    #
                    cur_node_set.add(next_id)
            # end-for
        # end-while
        # print("Route has not been constructed between {} and {}".format(src_id, dst_id))

    def construct_all_routes(self):
        # node_mappings = self.nodes.items()
        for src_id, src_node in self.nodes.items():
        # for i in range(len(node_mappings)):
            # Waypoints are not valid sources ...
            # src_id, src_node = node_mappings[i]
            if isinstance(src_node, Waypoint):
                continue
            
            for dst_id, dst_node in self.nodes.items():
                # ... nor valid destinations.
                # dst_id, dst_node = node_mappings[i]
                if dst_id == src_id or isinstance(dst_node, Waypoint):
                    continue

                route = self.construct_route(src_id, dst_id)
                self.routes[calc_route_id(src_id, dst_id)] = route

    def find_route(self, src_id, dst_id):
        # print("Graph::find_route from {} to {}".format(src_id, dst_id))
        return self.routes[calc_route_id(src_id, dst_id)]

#
# @todo: Each Centroid should have:
#       an arrival-time probability distribution (similar to Spawner);
#       a set of streams (differentiated by the destination)
#   This suggests an algorithm:
#       forall stream
#           construct route from O -> D using Graph::construct_route
#           get the number of arrivals from the OD matrix
#           use the arrival distribution to generate arrival times
#           register the arrival times
#       forall stream
#           queue up agents for the current time
#           spawn agents with routes
#
class Centroid(Node): # gates, doorways, elevators, transporters, stopped [public transport] vehicles
    # def __init__(self, id, posn, r, colour, agent_colour, view, streamspawner, agent_type = smaa.Pedestrian):
    def __init__(self, id, posn, r, colour, agent_colour, view, distributor, agent_type = smaa.Pedestrian):
        super(Centroid, self).__init__(id, posn, r, colour, view, agent_type)

        self.agent_colour = agent_colour
        # self.streamspawner = streamspawner
        self.distributor = distributor

        # self.spawning_table = {}
        self.stream_table = {}
        
        if self.agent_type != smaa.Pedestrian:
            self.queue_length  = 0
            self.head_agent_id = -1

        return

    def register_stream(self, dst_id, graph, T_simulation, T_step, n_agents):
        # print("Centroid::register_stream from {} to {} with {} agents".format(self.id, dst_id, n_agents))
        route = graph.find_route(self.id, dst_id)
        times = self.distributor.generate_values(T_simulation, T_step, n_agents)
        self.stream_table[dst_id] = (route, times)
        # print("Centroid::register_stream registered {}, {}".format(route, times))
        assert(route != None or n_agents == 0)
        return
    
    def spawn_pedestrians(self, model):
        # print("Centroid::spawn_pedestrians - step ", model.cur_step, ": start")
        assert(self.agent_type == smaa.Pedestrian)
        if self.agent_type == smaa.Pedestrian:
            # print("Centroid[{}]::spawn_pedestrians - step ".format(self.id), model.cur_step, ": start")
            for dst_id in self.stream_table.keys():
                # print("Centroid::spawn_pedestrians - spawning_table[", dst_id, "]")
                item = self.stream_table[dst_id]
                # print(item)
                route, spawning_list = self.stream_table[dst_id]
                # print("Centroid::spawn_pedestrians - from {} to {} along {}".format(self.id, dst_id, route))

                spawning_sentinel = 0
                while spawning_sentinel < len(spawning_list) and spawning_list[spawning_sentinel] == model.cur_step:
                    spawning_sentinel += 1
                if spawning_sentinel <= 0:
                    continue
                self.stream_table[dst_id] = (route, spawning_list[spawning_sentinel:])

                for i in range(spawning_sentinel):
                    src   = self
                    dst   = model.centroids[dst_id]
                    delta = dst.posn - self.posn
                    dirn  = delta / np.sqrt(np.dot(delta, delta))

                    perp              = np.array([-dirn[1], dirn[0]])
                    deviant_magnitude = self.radius * (ss.beta.rvs(2, 0.5) * 2 - 1)
                    posn              = self.posn + deviant_magnitude * perp

                    p = smaa.Pedestrian(model.next_pedestrian_id, model.T_react, model.topology[0], route, posn, dirn, self.agent_colour, dst.agent_colour, self.view)
                    # p = smaa.Pedestrian(model.next_pedestrian_id, model.T_react, self, dst, posn, dirn, self.agent_colour, dst.agent_colour, self.view)
                    # print("Model::spawn_pedestrians - new pedestrian ((id, posn, dirn))", p.id, p.posn, p.dirn)

                    model.add_agent(p, self.agent_type)
                    # model.add_pedestrian(p)
                    
                    # model.next_pedestrian_id += 1
                    # model.active_pedestrians[p.id] = p
            # print("Centroid::spawn_pedestrians - step ", model.cur_step, ": finish")
        # print("Centroid::spawn_pedestrians - step ", model.cur_step, ": finish")
        return

    def spawn_vehicles(self, model):
        assert(self.agent_type == smaa.Vehicle)
        if self.agent_type == smaa.Vehicle:
            # print("Centroid[{}]::spawn_vehicles - step ".format(self.id), model.cur_step, ": start")
            for dst_id in self.stream_table.keys():
                # print("Centroid[{}]::spawn_vehicles - spawning_table[{}]".format(self.id, dst_id))
                item = self.stream_table[dst_id]
                # print(item)
                route, spawning_list = self.stream_table[dst_id]
                # print("Centroid::spawn_pedestrians - from {} to {} along {}".format(self.id, dst_id, route))

                # print("Centroid[{}]::spawn_vehicles - spawning_table[{}]".format(self.id, dst_id))
                # spawning_list = self.stream_table[dst_id]

                spawning_sentinel = 0
                while spawning_sentinel < len(spawning_list) and spawning_list[spawning_sentinel] == model.cur_step:
                    spawning_sentinel += 1

                if self.head_agent_id != -1:
                    assert(self.head_agent_id >= 0)

                    vhcl = model.active_vehicles[self.head_agent_id]
                    dx   = vhcl.posn - self.posn

                    intercar_distance = 1.8
                    # intercar_distance = 0.4
                    if np.dot(dx, dx) >= (0.5 * vhcl.length + intercar_distance)**2:
                        self.head_agent_id = -1

                if spawning_sentinel <= 0:
                    continue

                # print("Centroid[{}]::spawn_vehicles - spawning_table[{}] should enqueue {} vehicles".format(self.id, dst_id, spawning_sentinel))

                self.stream_table[dst_id] = route, spawning_list[spawning_sentinel:]

                self.queue_length += spawning_sentinel
                # print("Centroid[{}]::spawn_vehicles - there are {} vehicles in the spawning queue".format(self.id, self.queue_length))


                if self.head_agent_id == -1 and self.queue_length >= 1:
                    self.queue_length -= 1
                    src   = self
                    dst   = model.centroids[dst_id]
                    delta = dst.posn - self.posn
                    dirn  = delta / np.sqrt(np.dot(delta, delta))

                    perp              = np.array([-dirn[1], dirn[0]])
                    deviant_magnitude = 0. # self.radius * (ss.beta.rvs(2, 0.5) * 2 - 1)
                    posn              = self.posn + deviant_magnitude * perp

                    #
                    # @todo: After introducing a vehicle-type enumeration, replace 3 with CAR
                    #
                    self.head_agent_id = model.next_vehicle_id
                    vhcl = smaa.Vehicle(self.head_agent_id, 3, model.T_react, model.topology[3], route, posn, dirn, 4.8, 1.8, self.agent_colour, self.view)
                    # vhcl = smaa.Vehicle(self.head_agent_id, 3, model.T_react, self, dst, posn, dirn, 4.8, 1.8, self.agent_colour, self.view)
                    # print("Centroid::spawn_vehicles - new vehicle ((id, posn, dirn))", vhcl.id, vhcl.posn, vhcl.dirn)

                    model.add_agent(vhcl, self.agent_type)
                    # model.add_vehicle(vhcl)

                continue
            
                #
                # @todo: Consider removing this loop.
                #
                for i in range(spawning_sentinel):
                    src   = self
                    dst   = model.centroids[dst_id]
                    delta = dst.posn - self.posn
                    dirn  = delta / np.sqrt(np.dot(delta, delta))

                    perp              = np.array([-dirn[1], dirn[0]])
                    deviant_magnitude = 0. # self.radius * (ss.beta.rvs(2, 0.5) * 2 - 1)
                    posn              = self.posn + deviant_magnitude * perp

                    #
                    # @todo:
                    #
                    intercar_distance = 0.4
                    queue_magnitude = i * (4.8 + intercar_distance)
                    posn -= queue_magnitude * dirn

                    #
                    # @todo: After introducing a vehicle-type enumeration, replace 3 with CAR
                    #
                    vhcl = smaa.Vehicle(model.next_vehicle_id, 3, model.T_react, self, dst, posn, dirn, 4.8, 1.8, self.agent_colour, self.view)
                    print("Centroid::spawn_vehicles - new vehicle ((id, posn, dirn))", vhcl.id, vhcl.posn, vhcl.dirn)

                    model.add_agent(vhcl, self.agent_type)
                    # model.add_vehicle(vhcl)

            # print("Centroid::spawn_vehicles - step ", model.cur_step, ": finish")
        # print("Centroid::spawn_vehicles - step ", model.cur_step, ": finish")
        return

    def spawn_agents(self, model):
        # print("Centroid::spawn_agents - step ", model.cur_step, ": start")        
        if self.agent_type == smaa.Vehicle:
            self.spawn_vehicles(model)
        else:
            self.spawn_pedestrians(model)            
        return
    
    def draw(self, view):
        view.fill_circle(self.colour, self.posn, self.radius)
        return
#
# Note: Centroid could be extended to act as a Portal between Models, i.e. each Portal within a Model
# would be paired with a Portal within another Model. Exiting a Model via a Portal would also mean
# entering the Model associated with the counter-Portal. This could be especially useful if the
# Portals were spatiotemporally separated by a mascroscopically modelled region, preferably with
# a minimum delay >= the communication time between the computer systems running the Models.
#

class Waypoint(Node): # gates, doorways, elevators, transporters, stopped [public transport] vehicles
    def __init__(self, id, posn, r, view, agent_type = smaa.Pedestrian):
        super(Waypoint, self).__init__(id, posn, r, svv.darker_grey, view, agent_type)
        return

    def draw(self, view):
        view.fill_circle(self.colour, self.posn, self.radius)
        return
