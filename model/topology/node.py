# import numpy as np
# import scipy as sp
# import scipy.stats as ss

# from collections import deque

# import view.view as svv
# import model.distributor as smd
# import model.numerical.defs as smnd
import model.agency.defs as smad
import model.topology.edge as smte
# import model.agency.agent as smaa
# import model.agency.pedestrian as smap
# import model.agency.vehicle as smav

#
# @todo: Consider generating waypoints and adding connections based on a visibility graph.
#
# The Node class and its subclasses are intended to capture the scene topology
# (and its relationship to the underlying geometry).
#
# Places which generate/produce/spawn agents, dissolve/consume/despawn agents, or
# where agents are restricted in motion at a medium- to large-scale should have
# Node objects associated with them.
#
# Examples include:
#     intersections, gates, doorways, elevators, transporters,
#     [preferably stopped] [public transport] vehicles
#
class Node:
    def __init__(self, id, shape, colour, agent_type): #  = smad.AgentType.PEDESTRIAN):
        self.id = id
        
        self.approach_shape = shape
        self.colour         = colour
        self.agent_type     = agent_type
        self.adjacency_tbl  = {}
        self.visited_flag   = False
        return
    
    def requires_waiting(self, agent):
        return False
    
    def requires_exiting(self):
        return False

    def get_waiting_details(self, agent):
        return None

    # def register_adjacency_list(self, adjacent_nodes):
        # self.adjacency_list += adjacent_nodes

        # return
    
    def connect(self, other, priority = None):
        # print("Node::connect - {} -> {}".format(self.id, other.id))

        # self.adjacency_list.append(other.id)
        if other.id not in self.adjacency_tbl:
            self.adjacency_tbl[other.id] = smte.Edge(self, other, priority)

        return

    def get_edge(self, other):
        if other.id not in self.adjacency_tbl:
            return None

        return self.adjacency_tbl[other.id]
    
    def is_possible_terminus(self):
        return False
    
    def get_active_focal_shape(self, agent):
        return None
    
    def set_visited_flag(self):
        self.visited_flag = True
        return

    def clear_visited_flag(self):
        self.visited_flag = False
        return

    def get_desc_list(self):
        # return [self.id, self.shape.posn, self.shape.radius]
        return [self.id, self.approach_shape.posn[0], self.approach_shape.posn[1], self.approach_shape.radius]

def symconnect(node1, node2, priority):
    node1.connect(node2, priority)
    node2.connect(node1, priority)
    return

def omniconnect(nodelist, priority):
    for i in range(len(nodelist)):
        node = nodelist[i]
        for j in range(i + 1, len(nodelist)):
            # print("mnn.omniconnect - {} <-> {}".format(i, j))
            counternode = nodelist[j]
            symconnect(node, counternode, priority)
    return

def interconnect(nodelist1, nodelist2, priority):
    for i in range(len(nodelist1)):
        node = nodelist1[i]
        for j in range(len(nodelist2)):
            # print("mnn.interconnect - {} <-> {}".format(i, j))
            counternode = nodelist2[j]
            symconnect(node, counternode, priority)
    return

def seqconnect(nodelist, priority, symmetric = False):
    n = len(nodelist)
    if n < 1:
        return
    
    if symmetric:
        for i in range(n - 1):
            symconnect(nodelist[i], nodelist[i + 1], priority)
    else:
        for i in range(n - 1):
            nodelist[i].connect(nodelist[i + 1], priority)

#
# @todo: Use Waypoint as the basis of an ActivityZone
#
# class Waypoint(Node): # gates, doorways, elevators, transporters, stopped [public transport] vehicles
    # def __init__(self, id, posn, r, view, agent_type = smad.PEDESTRIAN):
        # super(Waypoint, self).__init__(id, posn, r, svv.darker_grey, view, agent_type)
        # return

    # def draw(self, view):
        # view.fill_circle(self.colour, self.posn, self.radius)
        # return
