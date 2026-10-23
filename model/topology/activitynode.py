import numpy as np
# import scipy as sp
# import scipy.stats as ss
# import pygame as pg

# from collections import deque

import view.defs as svd
import view.view as svv
# import model.model as smm
import model.agency.defs as smad
import model.numerical.defs as smnd
import model.agency.agent as smaa
import model.force as smf
import model.geometry.shape as smgs
# import model.waiting as smw

import model.topology.node as smtn

#
# @todo: Define activity class later?
#
# class Activity
#     class SimpleActivity(Activity)
#     class ComplexActivity(Activity)
#

# class Event:
    # def __init__(object, method, argument_value):
        # object.register_observer(method, argument)
        # return

#
# To construct an activity table, we add entries to an empty table:
#    each entry requires a description,
#    a minimum duration, and an optional event.
#
# To execute an activity at an activity zone which has been initialised
# with this activity table entry, the agent must navigate to the activity
# zone, then wait there for the minimum duration, and may need to wait longer
# there until an event occurs.
#
# def register_activity_table_entry(tbl, desc, min_duration, event = None):
    # tbl[desc] = (duration, event)


#
# Or handcode the activity table, e.g.:
#     cafe_tbl     = {"order_coffee": (30, None), "await_coffee": (120, None), "drink_coffee": (60, None)}
#     crossing_waitnode = {"await_greensignal": (0, (crossing_zone, regime, "PTN_Start"))}
#
# In the case of signalised crossings in particular, a method-call must be made:
#     crossing_zone.register_regime_observer(crossing_waitnode, "PTN_Start")
# resulting in crossing_zone.get_current_regime() comparing the result of its
# get_current_regime() method against its prev_regime datum, updating the
# prev_regime datum and alerting observers if it has.
#
# Each crossing_zone will then alert its observing crossing_waitnodes which will alert their observing pedestrians.
# Alternatively, waiting nodes can have states, similar to their associated crossing zones.
#

waypoint_activity_tbl = {"Reroute": (0, None)}
cafe_activity_tbl     = {"TakeawayCoffee": (30, None), "OnsiteCoffee": (120, None)}
bar_activity_tbl      = {"Swill": (30, None), "Drink": (120, None)}
garden_activity_tbl   = {"Dogwalking": (5, None), "Sightseeing": (1, None)} # 30s, 60s, 90s, 120s, 150s

class ActivityNode(smtn.Node):
    def __init__(self, id, approach_shape, fill, activity_tbl, agent_type, waiting_model = None):
        super(ActivityNode, self).__init__(id, approach_shape, fill, agent_type)

        self.activity_tbl = activity_tbl # {}
        # for activity in activities:
            # self.activity_tbl[activity.id] = activity

        self.waiting_model = waiting_model

        self.focal_shape_tbl = {}

    def is_possible_terminus(self):
        return True
    
    def register_focus(self, activity_name, focus):
        self.focal_shape_tbl[activity_name] = focus
        return
    
    def requires_waiting(self, agent):
        return True
    
    def get_waiting_details(self, agent):
    # def get_waiting_details(self, desc):
        agent_activity_name = agent.activity_schedule[0]
        if agent_activity_name not in self.activity_tbl:
            return None
        return self.activity_tbl[agent_activity_name]

    def requires_exiting(self):
        return True

    def get_active_focal_shape(self, agent):
        agent_activity_name = agent.activity_schedule[0]
        return self.focal_shape_tbl[agent_activity_name][0]
    
    def draw(self, view):
        # print("Zone::draw", self.id)
        self.approach_shape.draw(self.colour, view)
        text = "A {}".format(self.id)
        view.draw_text(text, svd.white, self.approach_shape.posn, True)
        for name in self.focal_shape_tbl:
            focus = self.focal_shape_tbl[name]
            shape, fill = focus
            shape.draw(fill, view)
        return

#
# @todo: To make this work properly, we need a way for agents to evaluate their
# choices, e.g.:
#     every activity-zone has a [dis]utility vector;
#     and every agent has a preference weight vector; and
#     agents calculate the inner product to generate a scalar value.
# This technique is widely used in ML (recommender systems, etc.) and elsewhere.
# This is also the method proposed by Hoogebdoorn and Bovy 2003 for Nomad, their variational shared space model.
#
# def create_waypoint(id, shape):
    # return ActivityNode(id, shape, svd.darker_grey, waypoint_activity_tbl)

# def create_signalised_approach_zone(id, shape, agent_type):

def create_cafe(id, shape, fill, foci, barriers = []):
    node = ActivityNode(id, shape, fill, cafe_activity_tbl, smad.AgentType.PEDESTRIAN)
    node.register_focus("TakeawayCoffee", foci[0])
    node.register_focus("OnsiteCoffee",   foci[1])
    return node

def create_bar(id, shape, fill, foci, barriers = []):
    node = ActivityNode(id, shape, fill, bar_activity_tbl, smad.AgentType.PEDESTRIAN)
    node.register_focus("Swill", foci[0])
    node.register_focus("Drink", foci[1])
    return node

def create_cafebar(id, shape, ):
    return ActivityNode(id, shape, ...)

def create_bistro(id, shape):
    return ActivityNode(id, shape, ...)

def create_restaurant(id, shape):
    return ActivityNode(id, shape, ...)

def create_garden(id, shape, fill, foci, barriers = []):
    node = ActivityNode(id, shape, fill, garden_activity_tbl, smad.AgentType.PEDESTRIAN)
    node.register_focus("Sightseeing", foci[0])
    return node
