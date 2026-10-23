import numpy as np
# import scipy as sp
# import scipy.stats as ss

from enum import IntEnum

class ReactionType(IntEnum):
    IGNORE   = 0, # Do nothing
    REPEL    = 1, # SFM
    FOLLOW   = 2, # v_max := v*(s, v, v - v_leader)
    VEER     = 3, # v_alpha^* in ORCA (i.e. not in VO, RVO, or GRVO)
    YIELD    = 4, # x_alpha^* not in MH_beta, v_alpha^* = 0
    AWAIT    = 5  # x_alpha^* in X_dst_alpha, v_alpha^* = 0


class Reaction:
    def __init__(self, type):
        self.reaction_type = type
        return

class RepelReaction(Reaction):
    def __init__(self, s, relposn, relv):
        super().__init__(ReactionType.REPEL)
        self.s       = s
        self.relposn = relposn
        self.relv    = relv
        return

#
# NB: Not currently used.
#
class FollowReaction(Reaction):
    # def __init__(self, s, v, delta_v):
    def __init__(self, s):
        super().__init__(ReactionType.FOLLOW)
        self.s       = s
        # self.v       = v
        # self.delta_v = delta_v
        return

class VeerReaction(Reaction):
    def __init__(self, s, t_conflict, RVO):
        super().__init__(ReactionType.VEER)
        self.t_conflict = t_conflict
        self.s          = s
        self.RVO        = RVO
        return

class YieldReaction(Reaction):
    def __init__(self, stop_shape, s):
        super().__init__(ReactionType.YIELD)
        self.stop_shape = stop_shape
        self.s          = s
        return

class AwaitReaction(Reaction):
    def __init__(self):
        super().__init__(ReactionType.AWAIT)
        return
