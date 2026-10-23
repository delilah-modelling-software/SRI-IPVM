#
# Agent and two subclasses, Pedestrian and Vehicle
#
# @todo: Move some of the Vehicle functionality into a Car subclass
# @todo: Add other vehicle subclasses, e.g. Bicycle 
#

import numpy as np
# import scipy as sp
# import scipy.stats as ss

# from collections import deque

# import view.view as svv
# import model.agency.defs as smad
# import model.interaction.interaction as smii
# import model.geometry.shape as smgs
# import model.geometry.grid as smgg
# import model.defs as smd
# import model.force as smf
# import model.numerical.defs as smnd
# import model.topology.approachnode as SMTApp
# import model.topology.activitynode as SMTAct

#
# States
#
from enum import IntEnum

#
# Goal-states (relationship between an agent and their current destination node):
#   APPROACHING     direct navigation towards current destination;
#   TURNING         curved navigation towards current destination;
#   CONVERGING      stopping at a focal area of current destination;
#   WAITING         staying stopped at a focal area of current destination;
#   EXITING         departing a focal area of the current destination;
#
class GoalStateId(IntEnum):
    # APPROACHING =  0,
    # TURNING     =  1,
    # CONVERGING  =  2,
    # WAITING     =  3,
    # EXITING     =  4
    NAVIGATING  =  0,
    CONVERGING  =  1,
    WAITING     =  2,
    EXITING     =  3

#
# Conflict-states
#   MODULATING  minimal reactive response to conflict, e.g. SFM or CFM.
#   VEERING     proactive response, evading by veering without slowing or stopping.
#   YIELDING    proactive response, evading by slowing or stopping.
#   AWAITING    proactive response, evading by stopping at a node prior to the conflict.
#
class ConflictStateId(IntEnum):
    MODULATING = 0,
    VEERING    = 1,
    YIELDING   = 2
    AWAITING   = 3

class Stage:
    def __init__(self, shape, spd_limit):
        self.shape        = shape
        self.spd_limit = spd_limit

class TurningInfo:
    def __init__(self, stages, goal_dirn, target_dirn, Lq1, Lq2, L_chord):
        self.stages      = stages
        self.goal_dirn   = goal_dirn
        self.target_dirn = target_dirn
        self.Lq1         = Lq1
        self.Lq2         = Lq2
        self.L_chord     = L_chord

class GoalState:
    def __init__(self, node_id):
        #
        # Each agent has state related to an attractive node such as a waiting time.
        #
        self.state_id     = GoalStateId.NAVIGATING
        self.node_id      = node_id
        self.waiting_t1   = None
        self.turning_info = None

    def start_navigating_state(self, node_id, turning_info = None):
        #
        # Each agent has state related to an attractive node such as a waiting time.
        #
        self.state_id     = GoalStateId.NAVIGATING
        self.node_id      = node_id
        self.waiting_t1   = None
        self.turning_info = turning_info

    # def start_turning_state(self, node_id, stages, goal_dirn, target_dirn, Lq1, Lq2, L_chord):
    # def start_turning_state(self, node_id, stages, target_dirn):
        #
        # Each agent has state related to an attractive node such as its target facing direction and
        # intermediate stage information.
        #
        # self.state_id    = GoalStateId.TURNING
        # self.node_id     = node_id
        # self.stages      = stages
        # self.n_stages    = len(stages)
        # self.goal_dirn   = goal_dirn
        # self.target_dirn = target_dirn
        # self.Lq1         = Lq1
        # self.Lq2         = Lq2
        # self.L_chord     = L_chord

    def start_converging_state(self, node_id):
        #
        # Each agent has state related to an attractive node such as a waiting time.
        #
        self.state_id   = GoalStateId.CONVERGING
        self.node_id    = node_id
        self.waiting_t1 = None

    def start_waiting_state(self, node_id, t0, T_wait):
        #
        # Each agent has state related to an attractive node such as a waiting time.
        #
        self.state_id   = GoalStateId.WAITING
        self.node_id    = node_id
        # assert(T_wait != None)
        self.waiting_t1 = t0 + T_wait

    def start_exiting_state(self):
        # assert(False)

        #
        # Each agent has state related to an attractive node such as a waiting time.
        #
        self.state_id   = GoalStateId.EXITING
        # self.node_id    = node_id
        self.waiting_t1 = None


class ConflictState:
    def __init__(self):
        self.state_id          = ConflictStateId.MODULATING
        self.major_ctragent_id = -1
        self.t0                = None
        self.T_curve           = None
        self.t1                = None
        self.ctragent_x0       = None
        self.ctragent_x1       = None
        self.x_curve_pts       = None
        self.x_des             = None
        self.v_des             = None        
        return

    def clear_state(self):
        self.state_id          = ConflictStateId.MODULATING
        self.major_ctragent_id = -1
        self.conflict_t0       = None
        self.conflict_t1       = None
        self.ctragent_x0       = None
        self.ctragent_x1       = None
        self.x_curve_pts       = None
        self.x_des             = None
        self.v_des             = None        
        return

    def start_veering_state(self, major_ctragent_id, t0, T_curve, ctragent_segment, x_des, v_des, x_curve_pts):
        # assert(False)
        self.state_id          = ConflictStateId.VEERING
        self.major_ctragent_id = major_ctragent_id
        self.t0                = t0
        self.T_curve           = T_curve
        self.t1                = t0 + T_curve

        self.ctragent_x0       = None
        self.ctragent_x1       = None
        if ctragent_segment != None:
            self.ctragent_x0       = ctragent_segment[0]
            self.ctragent_x1       = ctragent_segment[1]

        self.x_curve_pts       = x_curve_pts
        self.x_des             = x_des
        self.v_des             = v_des        
        return

    def DEPRECATED_start_yielding_state(self, major_ctragent_id, t0, T_curve, ctragent_segment, x_des):
        self.state_id          = ConflictStateId.YIELDING
        self.major_ctragent_id = major_ctragent_id
        self.t0                = t0
        self.T_curve           = T_curve
        self.t1                = t0 + T_curve
        self.ctragent_x0       = ctragent_segment[0]
        self.ctragent_x1       = ctragent_segment[1]
        # self.x_curve_pts       = x_curve_pts
        self.x_curve_pts       = None
        self.x_des             = x_des
        self.v_des             = np.zeros(2)   
        return

    def DEPRECATED_start_yielding_state(self, s):
        self.state_id = ConflictStateId.YIELDING
        self.s        = s

    def start_yielding_state(self, major_ctragent_id, stop_shape, s):
        self.state_id   = ConflictStateId.YIELDING
        self.stop_shape = stop_shape
        self.s          = s

    def start_awaiting_state(self, major_ctragent_id, dst_node):
        self.state_id          = ConflictStateId.AWAITING
        self.major_ctragent_id = major_ctragent_id
        # dst_shape              = dst_node.approach_shape
        # self.x_des             = dst_shape.posn
        self.stop_shape        = dst_node.approach_shape
        return

    def get_desc_list(self):
        return [self.state_id, self.major_ctragent_id, self.t0, self.T_curve, self.t1, self.ctragent_x0, self.ctragent_x1, self.x_curve_pts, self.x_des, self.v_des]
