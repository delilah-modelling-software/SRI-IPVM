import numpy as np
import scipy as sp
import scipy.stats as ss

# from collections import deque

import view.view as svv
import model.distributor as smd
import model.numerical.defs as smnd
import model.agency.defs as smad
import model.agency.agent as smaa
import model.agency.pedestrian as smap
import model.agency.vehicle as smav

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
    # Use A* to efficiently calculate the shortest route between two nodes.
    #
    def construct_route(self, src_id, dst_id):
        # print("Graph::construct_route over {} from {} to {}".format(self.nodes.keys(), src_id, dst_id))
        src = self.nodes[src_id]
        dst = self.nodes[dst_id]

        cur_node_set  = set([src_id])
        prev_node_map = {}

        #
        # Note: If this code is ever converted to C++, g_map can be an std::hashmap,
        # but f_map should be implemented as an std::map (i.e. a sorted tree), because
        # per-iteration re-sorting would be unnecessary.
        #
        g_map = {src_id: 0}
        delta = dst.approach_shape.posn - src.approach_shape.posn
        h     = np.sqrt(np.dot(delta, delta))
        f_map = {src_id: h}
        # print("Graph::construct_route post-setup with g_map {}, f_map {}".format(g_map, f_map))

        while len(cur_node_set) > 0:
            # print("Graph::construct_route choosing element from {} with lowest estimated distance from {}".format(cur_node_set, f_map))
            
            f_pairs = list(f_map.items())
            f_pairs.sort(key = lambda pair: -pair[1])

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
            
            #
            # NB: the estimated distances are based on internodal distances rather than the arclengths of
            # internodal edges. This is not expected to cause problems for routing, but any temptation to save
            # these estimates and reuse them for unrelated purposes should be resisted.
            #
            cur_node = self.nodes[cur_id]
            for next_id in cur_node.adjacency_tbl:
                next_node  = self.nodes[next_id]
                delta      = next_node.approach_shape.posn - cur_node.approach_shape.posn
                d_cur_next = np.sqrt(np.dot(delta, delta))
                g_score    = g_map[cur_id] + d_cur_next
                # print("Graph::construct_route consideration of {} with g_score {}".format(next_id, g_score))
                if next_id not in g_map or g_score < g_map[next_id]:
                    prev_node_map[next_id] = cur_id
                    
                    g_map[next_id] = g_score
                    delta = dst.approach_shape.posn - next_node.approach_shape.posn
                    h = np.sqrt(np.dot(delta, delta))
                    f_map[next_id] = g_score + h
                    # print("Graph::construct_route post-addition with g_map {}, f_map {}".format(g_map, f_map))

                    cur_node_set.add(next_id)
            # end-for
        # end-while
        # print("Route has not been constructed between {} and {}".format(src_id, dst_id))

        #
        # If we've exited the loop without constructing a route, then:
        #     there isn't one (most likely); or
        #     there is a bug in the code (significantly less likely but possible).
        # The caller, if it encounters this situation, should probably check that
        # the node-graph has been connected properly as a first step.
        #
        return None

    def construct_all_routes(self):
        # node_mappings = self.nodes.items()
        for src_id, src_node in self.nodes.items():
            #
            # @todo: Decide whether all activity nodes are to be valid
            # origins and destinations or only some of them.
            #
            # Waypoints are neither valid sources ...
            #
            
            if not src_node.is_possible_terminus():
                continue
            
            for dst_id, dst_node in self.nodes.items():
                #
                # ... nor valid destinations.
                # Nor self-loops for that matter.
                #
                if not dst_node.is_possible_terminus():
                    continue
                
                # if dst_id == src_id or isinstance(dst_node, Waypoint):
                    # continue

                route = self.construct_route(src_id, dst_id)
                self.routes[calc_route_id(src_id, dst_id)] = route

    def find_route(self, src_id, dst_id):
        # print("Graph::find_route from {} to {}".format(src_id, dst_id))
        return self.routes[calc_route_id(src_id, dst_id)]
