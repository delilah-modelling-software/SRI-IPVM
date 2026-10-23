import numpy as np
# import scipy as sp
# import scipy.stats as ss

# from collections import deque

# import view.view as svv
# import model.distributor as smd
import model.numerical.defs as smnd
import model.topology.node as smtn
# import model.agency.agent as smaa
# import model.agency.pedestrian as smap
# import model.agency.vehicle as smav

from enum import IntEnum
class Relationship(IntEnum):
    INDEPENDENT = 0,
    DIVERGING   = 1,
    MERGING     = 2,
    SHARING     = 3,
    CROSSING    = 4

class Edge:
    def __init__(self, src, dst, priority):
        self.src         = src
        self.dst         = dst
        self.priority    = priority

        self.agent_idmap = {}

    def classify_relationship(self, edge):
        if self.src.id == edge.src.id and self.dst.id == edge.dst.id:
            return Relationship.SHARING
        elif self.src.id == edge.src.id:
            return Relationship.DIVERGING
        elif self.dst.id == edge.dst.id:
            return Relationship.MERGING

        if self.src.id == edge.dst.id or self.dst.id == edge.src.id:
            return Relationship.INDEPENDENT

        dx_alpha = self.dst.approach_shape.posn - self.src.approach_shape.posn
        dx_beta  = edge.dst.approach_shape.posn - edge.src.approach_shape.posn

        L_alpha_chord, alpha_chord_dirn = smnd.calc_polar_decomp(dx_alpha)
        L_beta_chord,  beta_chord_dirn  = smnd.calc_polar_decomp(dx_beta)

        net_cos = np.dot(alpha_chord_dirn, beta_chord_dirn)
        if (net_cos - 1)**2 < smnd.EPS_m_sqr or (net_cos + 1)**2 < smnd.EPS_m_sqr:
            return Relationship.INDEPENDENT

        det = smnd.calc_det2x2(alpha_chord_dirn, beta_chord_dirn)
        assert(det**2 >= smnd.EPS_m_sqr)
        rcp_det = 1 / det

        dx_src        = edge.src.approach_shape.posn - self.src.approach_shape.posn
        L_alpha_cross = smnd.calc_det2x2(dx_src, beta_chord_dirn)  * rcp_det
        if L_alpha_cross < 0 or L_alpha_cross > L_alpha_chord:
            return Relationship.INDEPENDENT

        L_beta_cross  = -smnd.calc_det2x2(alpha_chord_dirn, dx_src) * rcp_det
        if L_beta_cross < 0 or L_beta_cross > L_beta_chord:
            return Relationship.INDEPENDENT

        return Relationship.CROSSING

    def register_agent(self, agent_id, next):
        assert(not next or agent_id not in self.agent_idmap)
        self.agent_idmap[agent_id] = next
        return

    def deregister_agent(self, agent_id):
        assert(agent_id in self.agent_idmap)
        del self.agent_idmap[agent_id]
        return
