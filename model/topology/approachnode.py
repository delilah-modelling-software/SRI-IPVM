import numpy as np
# import scipy as sp
# import scipy.stats as ss
# import pygame as pg

# from collections import deque

import view.defs as svd
import view.view as svv
# import model.model as smm
import model.agency.defs as smad
import model.agency.reaction as smar
import model.numerical.defs as smnd
import model.agency.agent as smaa
import model.force as smf
import model.geometry.shape as smgs

import model.topology.node as smtn

class ApproachNode(smtn.Node):
    def __init__(self, id, approach_shape, focal_shape, colour, focal_colour, agent_type, zone, waiting_influence_fn):
        super(ApproachNode, self).__init__(id, approach_shape, colour, agent_type)

        self.focal_shape = focal_shape
        self.focal_colour = focal_colour
        self.zone       = zone

        self.waiting_influence_fn = waiting_influence_fn

    def requires_waiting(self, agent):
        if self.zone == None:
            return False
        
        return self.zone.currently_excludes(agent)
    
    def get_waiting_duration(self, t0):
        return self.zone.period

    def get_waiting_details(self, agent):
        return self.zone.period, None

    def get_active_focal_shape(self, agent):
        return self.focal_shape
    
    def draw(self, view):
        self.approach_shape.draw(self.colour, view)
        text = "W {}".format(self.id)
        view.draw_text(text, svd.white, self.approach_shape.posn, True)

        # print("Zone::draw", self.id)
        self.approach_shape.draw(self.colour, view)
        if self.focal_shape != None and self.requires_waiting():
            assert(self.focal_colour != None)
            self.focal_shape.draw(self.focal_colour, view)
        return

#
# @todo: To make this work properly, we need a way for agents to evaluate their
# choices, e.g.:
#     every activity-zone has a [dis]utility vector;
#     and every agent has a preference weight vector; and
#     agents calculate the inner product to generate a scalar value.
# This technique is widely used in ML (recommender systems, etc.) and elsewhere,
# and Hoogendoorn+Bovy (2003) disclosed it for pedestrian spaces.
#
def create_waypoint(id, approach_shape, agent_type):
    return ApproachNode(id, approach_shape, None, svd.darker_grey, None, agent_type, None, None)

def create_round_waypoint(id, centre, e, halfwidth, agent_type):
    approach_shape = smgs.Circle(centre, halfwidth)
    return create_waypoint(id, approach_shape, agent_type)

def create_flat_waypoint(id, centre, e, halfwidth, halfdepth, agent_type):
    approach_shape = smgs.create_segment(centre, e, halfwidth, halfdepth)
    return create_waypoint(id, approach_shape, agent_type)

def create_oblong_waypoint(id, centre, e, halflength, halfwidth, r_corner, agent_type):
    approach_shape = smgs.create_rect(centre, e, halflength, halfwidth, r_corner)
    return create_waypoint(id, approach_shape, agent_type)

def create_signalised_approach_node(id, approach_shape, stop_shape, zone, agent_type):
    return ApproachNode(id, approach_shape, stop_shape, svd.darker_grey, svd.gold, agent_type, zone, smf.calc_PV_waiting)

