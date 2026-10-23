import numpy as np
# import scipy as sp
# import scipy.stats as ss
# import pygame as pg

# from collections import deque

import multimodal_network.view as mnv
import multimodal_network.model as mnm
import multimodal_network.agent as mna
# import shape as mns

#
# Repulsors repel people away from them.
#
class Repulsor:
    def __init__(self, id, A, D, shape):
        self.id    = id
        self.A     = A
        self.D     = D
        self.shape = shape
        return
    
    def calc_repulsive_force(self, agent):
        # print("Repulsor::calc_repulsive_force - obstcl {} against agent {}({})".format(self.id, agent.id, agent.agent_type))
        f_R = np.zeros(2)

        #
        # Each shape
        #
        info = self.shape.get_closest_point_info(agent.posn + agent.T_react * agent.v)
        # info = self.shape.get_closest_point_info(agent.posn)
        if info == None:
            return np.zeros(2)

        #
        # @todo: Grand unified obstacle repulsion
        #
        x_omega, nv, tv = info
        # print("Repulsor::calc_repulsive_force - closest point {}, normal {}, tangent {}".format(x_omega, nv, tv))

        if agent.agent_type == 0:
            delta = x_omega - agent.posn
            dist  = mna.calc_norm(delta)
            f_R   = mnm.calc_repulsion(dist - (self.shape.radius + agent.radius), self.A, self.D) * nv
            
        return f_R

    def draw(self, view):
        # print("Repulsor::draw", self.id)
        self.shape.draw(mnv.white, view)
        return
#
# Attractors attract people (esp. pedestrians) to them. The most attractive thing for
# most vehicles is likely to be an empty lane or a parking space, dependent on the driver.
# The most attractive thing for pedestrians is quite variable, e.g. shop windows, trees, fountains,
# statuary, pools.
#
class Attractor:
    def __init__(self, id, A, D, T, shape):
        self.id    = id
        self.A     = A
        self.D     = D
        self.T     = T
        self.shape = shape
        return

    def draw(self, view):
        self.shape.draw(black_and_gold, view)
        return

#
# Obscurers prevent visual perception of other pedestrians, vehicles, etc.
# This will impair the effect of Attractors and Repulsors. If an object is to
# serve multiple-duty, the code would be simplest if the radius of the Obstruction
# were smaller than that of the coincident Attractor/Repulsor.
#
# @todo: Obscurers should also introduce uncertainty, esp. in pedestrian v vehicle scenarios.
# @todo: A more advanced implementation may require some vehicles (esp. trucks) to obscure
#     visibility too. If these vehicles are non-stationary, then efficient implementation may
#     be significantly more difficult.
#
class Obscurer:
    def __init__(self, id, shape):
        self.id    = id
        self.shape = shape
        return

