#
# Agent and two subclasses, Pedestrian and Vehicle
#
# @todo: Move some of the Vehicle functionality into a Car subclass
# @todo: Add other vehicle subclasses, e.g. Bicycle 
#

import numpy as np
import scipy as sp
import scipy.stats as ss

from collections import deque

import multimodal_network.view as mnv
import multimodal_network.shape as mnsh

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
PEDESTRIAN = 0
BICYCLE    = 1
MOTORCYCLE = 2
CAR        = 3
TRUCK      = 4
BUS        = 5
TRAM       = 6

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

#
# Mollification
#
mollification = True
eps_m_sqr = 0.0001

if mollification:
    def calc_norm(v):
        return np.sqrt(np.dot(v, v) + eps_m_sqr)
else:
    def calc_norm(v):
        return np.sqrt(np.dot(v, v))

def calc_unit_vector(v):
    return v / calc_norm(v)

A_pedestrian = 2.1
D_pedestrian = 0.3

# epsilon_v_sqr = 0.0625 * 0.0625

class Agent:
    #
    # @todo: Pass route instead of src, dst.
    #   cur_src = src = route[0]
    #   cur_dst = route[1]
    #   dst     = route[-1]
    #
    def __init__(self, id, agent_type, T_react, nodegraph, route, posn, dirn, view):
        self.nodegraph   = nodegraph
        self.route       = route
        self.routing_idx = 0
        # self.src_idx     = route[ 0]
        # self.dst_idx     = route[-1]
        # self.cur_src_idx = route[ 0]
        # self.cur_dst_idx = route[ 1]
        
        cur_src_idx = self.route[self.routing_idx]
        cur_dst_idx = self.route[self.routing_idx + 1]
        
        cur_src_node = self.nodegraph.nodes[cur_src_idx]
        cur_dst_node = self.nodegraph.nodes[cur_dst_idx]

        true_src_idx = self.route[ 0]
        true_dst_idx = self.route[-1]

        true_src_node = self.nodegraph.nodes[true_src_idx]
        true_dst_node = self.nodegraph.nodes[true_dst_idx]
        
    # def __init__(self, id, agent_type, T_react, src, dst, posn, dirn, view):
        self.id         = id
        self.agent_type = agent_type
        self.T_react    = T_react
        # self.src        = src
        # self.dst        = dst
        self.posn       = posn
        self.dirn       = dirn
        self.view       = view

        assert(np.dot(self.dirn, self.dirn) >= 0)

        # self.cur_src = self.src
        # self.cur_dst = self.dst

        #
        # @todo: Add a personalised interactivity substructure comprising
        #   A ~ N(A_mu[agent_type, *], A_sigma[agent_type, *]
        #   D ~ N(D_mu[agent_type, *], D_sigma[agent_type, *]
        #

        #
        # @todo: Add a personalised dynamic profile comprising
        #   m ~ N(75, ??)
        #   v_pref ~ N(1.34, ??)
        #   a_socmax ~ N(??, ??)
        #   T_react ~ N(0.5, ??)
        #
        #

        return

class Pedestrian(Agent):
    def __init__(self, id, T_react, nodegraph, route, posn, dirn, src_colour, dst_colour, view):
        super(Pedestrian, self).__init__(id, 0, T_react, nodegraph, route, posn, dirn, view)
    # def __init__(self, id, T_react, src, dst, posn, dirn, src_colour, dst_colour, view):
        # super(Agent, self).__init__()
        # super(Pedestrian, self).__init__(id, 0, T_react, src, dst, posn, dirn, view)

        self.src_colour = src_colour
        self.dst_colour = dst_colour
        
        self.f_total = np.zeros(2)
        self.v       = np.zeros(2)

        self.v_ped          = 1.33
        self.v_max          = 1.3 * self.v_ped # (Helbing and Molnar 1995)
        self.T_relax        = 0.3 # 0.3s, matches Anvari et al. (2015) rather than the 0.5 from (Helbing and Molnar 1995)

        self.T_SRCA_horizon = 0.5
        self.T_MRCA_horizon = 2.
        self.T_LRCA_horizon = 4.
        self.radius         = 0.2
        return

    def calculate_driving_force(self):
        cur_dst_idx = self.route[self.routing_idx + 1]
        cur_dst = self.nodegraph.nodes[cur_dst_idx]

        ds   = cur_dst.posn - self.posn
        u    = calc_unit_vector(ds) # v / np.sqrt(np.dot(v, v))
        f_D = (self.v_ped * u - self.v) / self.T_relax
        # print("Pedestrian:calculate_driving_force (id, v_desired, v, f_D): ", self.id, self.v_ped * u, self.v, f_D)
        return f_D

    def calculate_interpedestrian_repulsion(self, q, MRCA_flag):
        self.T_horizon = self.T_SRCA_horizon
        # if MRCA_flag:
            # self.T_horizon = self.T_MRCA_horizon

        #
        # @todo: Moving max_s into the Model and refining it may be a good idea.
        #
        ds = q.posn - self.posn
        max_s = 2. # 1.4 * 1.33 * self.T_horizon**2
        if np.dot(ds, ds) > max_s * max_s:
            return np.zeros(2)

        if np.dot(ds, ds) <= (self.radius + q.radius)**2:
            impact_posn = 0.5 * (self.posn + q.posn)
            # print("Impact between pedestrian {} and pedestrian {} at {}".format(self.id, q.id, impact_posn))

        dv = q.v - self.v

        #
        # @todo: Remove if this isn't helpful.
        #
        t_min     = 0.5

        delta    = ds + t_min * dv
        distance = calc_norm(delta) # np.sqrt(np.dot(delta, delta)) 
        u        = delta / distance
        # if distance < self.radius + q.radius:
            # print("Conflict (expected impact) between {} and {} at {}".format(self.id, q.id, 0.5 * (self.posn + q.posn)))
        # f_R      = calc_repulsion(distance - (self.radius + q.radius), self.A, self.D) * -u
        f_R      = calc_repulsion(distance - (self.radius + q.radius), A[0, 0], D[0, 0]) * -u

        #
        # @todo: Find justifying literature for rejecting component of f_R
        #   perpendicular to dv.
        #
        dv_sqr  = np.dot(dv, dv)
        dv_norm = np.sqrt(dv_sqr)
        DV_EPSILON = 0.25 * 0.25
        if dv_norm > DV_EPSILON: # MRCA_flag and t_min > self.T_SRCA_horizon:
            #
            # @todo: The following line of code was causing problems (NaN-generation, etc.). Consider
            #   whether there are better/safer ways of achieving the same goal. If not, delete with a clear conscience.
            #
            # f_R /= t_min / self.T_SRCA_horizon

            #
            # @todo: Reject component of f_R parallel to dv
            #
            unit_dv = dv / dv_norm
            f_R -= np.dot(f_R, unit_dv) * unit_dv
            # f_R = np.dot(f_R, unit_dv) * unit_dv

        #
        # Attenuate by visibility
        #
        cos_pq      = np.dot(u, self.dirn)
        vis_factor  = 0.5 + 0.25 * (1 + cos_pq)
        f_R        *= vis_factor

        # print("Pedestrian:calculate_repulsion (id, q.id, f_R): ", self.id, q.id, f_R)
        return f_R

    def calculate_vehicle_avoidance_force(self, vhcl):
        f_R = np.zeros(2)

        self.T_horizon = self.T_SRCA_horizon
        # if MRCA_flag:
            # self.T_horizon = self.T_MRCA_horizon

        #
        # @todo: Introduce a max_s into the Model for pedestrian-car interactions?
        #
        ds = vhcl.posn - self.posn
        dv = vhcl.v - self.v

        # print("Pedestrian::calc_vhcl_avdnc_force pedestrian {} at x = {} v = {} ... ".format(self.id, self.posn, self.v))
        # print("... wants to avoid vhcl {} at x = {} v = {} ... ".format(vhcl.id, vhcl.posn, vhcl.v))
        # print("... and ds = {} dv = {}".format(ds, dv))

        t_min     = 0.5
        delta       = ds + t_min * dv
        distance    = calc_norm(delta) # np.sqrt(np.dot(delta, delta)) 
        u           = delta / distance
        # print("delta = {} and delta^2 = {}".format(delta, np.dot(delta, delta)))
        # print("true distance = {} and mollified distance = {}".format(np.sqrt(np.dot(delta, delta)), distance))

        #
        # Begin testing stolen+modified code
        #
        
        # print("u = {} and u^2 = {}".format(u, np.dot(u, u)))
        # assert(np.dot(vhcl.dirn, vhcl.dirn) <= 1.)
        # assert(np.dot(u, u)                 <= 1.)

        # print("vhcl dirn {} vs pdstrn posn relative to vhcl {}".format(vhcl.dirn, -u))
        c_vhcl = np.dot(vhcl.dirn, -u)
        # assert(-1. <= c_vhcl and c_vhcl <= 1.)
        # c_vhcl = -np.dot(vhcl.dirn, ds) / distance

        l_car = 0.5 * vhcl.length
        w_car = 0.5 * vhcl.width
        assert(l_car >= w_car)
        e_car = np.sqrt(l_car * l_car - w_car * w_car) / l_car

        r_car = w_car / np.sqrt(1 - (c_vhcl * e_car)**2)

        # wsqr  = w_car**2
        # lsqr  = l_car**2
        # csqr  = c_vhcl**2

        # if csqr <= 0.5:
            # r_car = w_car * np.sqrt(lsqr / (lsqr - csqr * (lsqr - wsqr)))
        # else:
            # r_car = l_car * np.sqrt(wsqr / (wsqr + (1 - csqr) * (lsqr - wsqr)))
        
        # print("Pedestrian::calc_vhcl_avoidance_force - car has e {}".format(e_car))
        # print("Pedestrian::calc_vhcl_avoidance_force - car has csqr {} ".format(csqr))
        # print("Pedestrian::calc_vhcl_avoidance_force - car has radius {} which is expected to be between w {} and l {}".format(r_car, w_car, l_car))
        assert(w_car <= r_car)
        assert(r_car <= l_car)


        #
        # End testing stolen+modified code
        #

        vhcl_radius = r_car
        # if distance < self.radius + vhcl_radius:
            # print("Conflict (expected impact) between {} and {} at {}".format(self.id, v.id, 0.5 * (self.posn + v.posn)))
        f_R      = calc_repulsion(distance - (self.radius + vhcl_radius), A[0, 3], D[0, 3]) * -u

        # dv_sqr  = np.dot(dv, dv)
        dv_norm = calc_norm(dv) # dv_norm = np.sqrt(dv_sqr)
        # DV_EPSILON = 0.25 * 0.25
        if dv_norm**2 > eps_m_sqr: # MRCA_flag and t_min > self.T_SRCA_horizon:
            #
            # @todo: The following line of code was causing problems (NaN-generation, etc.). Consider
            #   whether there are better/safer ways of achieving the same goal. If not, delete with a clear conscience.
            #
            # f_R /= t_min / self.T_SRCA_horizon

            #
            # @todo: Reject component of f_R parallel to dv
            #
            unit_dv = dv / dv_norm
            f_R -= np.dot(f_R, unit_dv) * unit_dv
            # f_R = np.dot(f_R, unit_dv) * unit_dv

        #
        # Attenuate by visibility
        #
        cos_pc      = np.dot(u, self.dirn)
        lambda_pc   = lambda_interagent[0, 3]
        vis_factor  = lambda_pc + (1 - lambda_pc) * 0.5 * (1 + cos_pc)
        f_R        *= vis_factor

        # print("Pedestrian:calculate_vehicle_avoidance (id, v.id, f_R): ", self.id, v.id, f_R)
        return f_R

    def calculate_circular_obstacle_repulsion(self, b, MRCA_flag = False):
        A_b = b.A
        D_b = b.D

        ds = b.shape.posn - self.posn
        dv = -self.v

        t_min     = 0.
        T_horizon = self.T_SRCA_horizon
        if MRCA_flag:
            T_horizon = self.T_MRCA_horizon

        #
        # @todo: If this doesn't help, remove and try something else.
        #
        t_min     = 0.5
        delta    = ds + t_min * dv
        distance = calc_norm(delta) # np.sqrt(np.dot(delta, delta)) 
        u        = delta / distance
        # if distance < self.radius + b.shape.radius:
            # print("Collision between pedestrian {} and obstacle {}", self.id, b.id)
        f_R      = calc_repulsion(distance - (self.radius + b.shape.radius), A_b, D_b) * -u
        if MRCA_flag and t_min > self.T_SRCA_horizon:
            # f_R /= t_min / self.T_SRCA_horizon
            #
            # @todo: Reject component of f_R parallel to dv
            #
            unit_dv = calc_unit_vector(dv) # dv / np.sqrt(np.dot(dv, dv))
            f_R -= np.dot(f_R, unit_dv) * unit_dv

        # f_R = np.zeros(2)
        return f_R

    def calculate_linear_obstacle_repulsion(self, b, MRCA_flag = False):
        #
        # @todo: Redesign and recode this functionality?
        # @todo: Add a composite obstacle type. Find the nearest component of the obstacle during the
        # near-future and use its repulsion rule. By combining linear and circular obstacles, arbitrary
        # shapes can be supported.
        #
        
        A_b = b.A
        D_b = b.D

        v = self.v
        w = b.shape.posn2 - b.shape.posn1

        x0 = b.shape.posn1
        x1 = b.shape.posn2
        if np.dot(v, w) <= 0:
            w = -w
            x0 = b.shape.posn2
            x1 = b.shape.posn1
        u = self.posn - b.shape.posn1
            
        unit_w = calc_unit_vector(w) # w / np.sqrt(np.dot(w, w))
        
        #
        # Calculate the projection and rejection of u, v w.r.t. w
        #
        u_par  = np.dot(u, unit_w) * unit_w
        u_perp = u - u_par
        u_perp_sqr = np.dot(u_perp, u_perp)
        v_par  = np.dot(v, unit_w) * unit_w
        v_perp = v - v_par
        v_perp_sqr = np.dot(v_perp, v_perp)

        t_min     = 0.
        T_horizon = self.T_SRCA_horizon
        if MRCA_flag:
            T_horizon = self.T_MRCA_horizon

        #
        # @todo: If this doesn't help, remove and try something else.
        #
        t_min     = 0.5
        delta_par  = u_par + t_min * v_par
        delta_perp = u_perp + t_min * v_perp
        delta      = delta_par + delta_perp

        distance   = calc_norm(delta) # np.sqrt(np.dot(delta, delta)) 
        unit_delta = delta_perp / distance
        # if distance < self.radius + b.shape.radius:
            # print("Collision between pedestrian {} and obstacle {}", self.id, b.id)
        f_R      = calc_repulsion(distance - (self.radius + b.shape.radius), A_b, D_b) * unit_delta
        # if MRCA_flag and t_min > self.T_SRCA_horizon:
            # f_R /= t_min / self.T_SRCA_horizon
            #
            # @todo: Reject component of f_R parallel to dv
            #
            # unit_dv = calc_unit_vector(dv) # dv / np.sqrt(np.dot(dv, dv))
            #f_R -= np.dot(f_R, unit_dv) * unit_dv

        # f_R = np.zeros(2)
        return f_R

    def calculate_obstacle_repulsion(self, b, MRCA_flag = False):
        if isinstance(b.shape, mnsh.Circle):
            return self.calculate_circular_obstacle_repulsion(b, MRCA_flag)
        elif isinstance(b.shape, mnsh.Segment):
            return self.calculate_linear_obstacle_repulsion(b, MRCA_flag)
        else:
            return np.zeros(2)


    def calculate_total_force(self):
        return

    def update_kinematics(self, delta_t):
        self.posn += delta_t * self.v
        # print("Pedestrian::update_kinematics - ptn {} at {} moving at vel {} with total force {}".format(self.id, self.posn, self.v, self.f_total))
        self.v    += delta_t * self.f_total
        v_norm     = calc_norm(self.v)
        self.dirn  = self.v / v_norm
        # print("Pedestrian[{}]::update_kinematics moving from {} at {} in direction {}".format(self.id, self.posn, self.v, self.dirn))
        if not (0 <= np.dot(self.dirn, self.dirn) and np.dot(self.dirn, self.dirn) <= 1.):
            print("Pedestrian {} has posn {}, vel {}, and dirn {}".format(self.id, self.posn, self.v, self.dirn))
        # assert(0 <= np.dot(self.dirn, self.dirn))
        # assert(np.dot(self.dirn, self.dirn) <= 1.)
        assert(0 <= np.dot(self.v, self.v))
        if v_norm > self.v_max:
            self.v = self.v_max * self.dirn
        # assert(0 <= np.dot(self.v, self.v))
        assert(np.dot(self.v, self.v) <= self.v_max**2)
        assert(0 <= np.dot(self.f_total, self.f_total))
            
    def draw(self, view):
        # print("Pedestrian::draw (id, posn, r)", self.id, self.posn, self.radius)
        # view.draw_circle(self.src_colour, self.posn, self.radius)
        # view.draw_line(white, self.posn, self.posn + self.radius * self.dirn, 0)
        spd = calc_norm(self.v) # np.sqrt(np.dot(self.v, self.v))

        view = self.view
        view.fill_circle(self.src_colour, self.posn, view.pedestrian_upscale_factor * self.radius)
        # view.stroke_circle(black, self.posn, 2.5 * self.radius)
        view.draw_line(mnv.white, self.posn, self.posn + view.pedestrian_upscale_factor * spd * self.radius * self.dirn, 0)
        return

class Vehicle(Agent):
    def __init__(self, id, agent_type, T_react, nodegraph, route, posn, dirn, length, width, colour, view):
        super(Vehicle, self).__init__(id, agent_type, T_react, nodegraph, route, posn, dirn, view)
    # def __init__(self, id, agent_type, T_react, src, dst, posn, dirn, length, width, colour, view):
        # super(Vehicle, self).__init__(id, agent_type, T_react, src, dst, posn, dirn, view)
        
        self.length = length # 9.6m
        self.width  = width  # 3.6m

        self.colour = colour

        #
        # @todo: A starting velocity should probably be passed in.
        #
        self.f_total = np.zeros(2)
        self.v       = np.zeros(2)

        self.v_max          = 1.39 # 5 km/h = 25/18 m/s, probably safest.
        # self.v_max          = 2.78 # 10 km/h = 25/9 m/s, probably even safer.
        # self.v_max          = 5.56 # 20 km/h = 50/9 m/s, probably safer.
        # self.v_max          = 8.33 # 30 km/h = 25/3 m/s, matches Anvari (2015), good enough for now.
        self.T_relax        = 2.4  # 2.4s, matches Anvari et al. (2015)
        self.T_brake        = 0.77 # 0.77s, matches Anvari et al. (2015) 
        self.T_SRCA_horizon = 0.5
        self.T_MRCA_horizon = 2.
        self.T_LRCA_horizon = 4.

        self.f_total = np.zeros(2)
        return

    def calculate_driving_force(self):
        cur_dst_idx = self.route[self.routing_idx + 1]
        cur_dst     = self.nodegraph.nodes[cur_dst_idx]
        
        ds      = cur_dst.posn - self.posn
        # ds      = self.dst.posn - self.posn
        unit_ds = calc_unit_vector(ds) # ds / np.sqrt(np.dot(ds, ds))

        #
        # @todo: Experiment with using v_legal instead.
        #
        f_D = (self.v_max * unit_ds - self.v) / self.T_relax
        # print("Vehicle:calculate_driving_force (id, v_desired, v, f_D): ", self.id, self.v_max * unit_ds, self.v, f_D)
        return f_D

    #
    # @todo: This function probably needs to be specialised into a pair of functions, e.g.:
    #     calculate_intercar_repulsion_section; and
    #     calculate_intercar_repulsion_node,
    # where the first function would simplify most of the calculations significantly. 
    #
    def calculate_intercar_repulsion(self, car, MRCA_flag):
        ds = car.posn - self.posn
        dv = car.v    - self.v

        #
        # @todo: Introduce a multimodal_network.node.Queue class. Have all vehicles
        # enter the centroids via such a queue so that coincident car-pairs do not recur.
        #
        distance = calc_norm(ds) # np.sqrt(np.dot(ds, ds))
        u        = ds / distance


        f_R = np.zeros(2)
        # return f_R

        #
        # Calculate a social force based purely on current distance and accumulate.
        #
        max_s = 2 * (self.v_max + car.v_max)
        if np.dot(ds, ds) < max_s * max_s:
        # if np.dot(ds, ds) > max_s * max_s:
            c_self = np.dot(self.dirn, ds) / distance

            #
            # Note: the formula for p has been drastically simplified from Anvari et al. (2012).
            # Instead of p = abs(arccos(c_self)) < pi/18 and abs(arccos(c_self) + arccos(c_other) - pi ) < pi/18),
            #   p = abs(arccos(c_self)) < pi/18 and abs(arccos(c_self) - pi) < pi/18
            #   optimised to p = c_self^2 > cos(pi/18)
            # Note: the formula q = abs(arccos(c_self)) < pi/6 has also been optimised
            # to q = c_self > cos(pi/6)
            #
            Theta     = np.dot(dv, dv) > eps_m_sqr
            p = np.dot(self.v, car.v) > eps_m_sqr and c_self**2 > (np.cos(np.pi / 18))**2
            q = c_self**2 > 0.75

            l_self = 0.5 * self.length
            l_car  = 0.5 * car.length
            
            if p:
                #
                # CFM (Car-Following Model) only
                # Simplified:
                #   p * Theta * calc_repulsion(...) * ... -> calc_repulsion(...) * ...
                #   r_self -> l_self, r_car -> l_car
                #
                cur_dst_idx = self.route[self.routing_idx + 1]
                cur_dst = self.nodegraph.nodes[cur_dst_idx]
                s_desired    = cur_dst.posn - self.posn
                dirn_desired = calc_unit_vector(s_desired) # s_desired / np.sqrt(np.dot(s_desired, s_desired))

                #
                # @todo: Anvari et al. 2015 distinguishes these characteristic distances from
                # the D used in the car-car SFM, but values do not appear to have been determined for them.
                # However, if we refer to Helbing and Tilch (1998), cited by Anvari et al. (2015),
                # we find:
                #   D_accel_interact   =  5.59m
                #   D_braking_interact = 98.78m
                #   T_accel_interact   =  2.45s
                #   T_braking_interact =  0.77s
                # In addition:
                #   T_react            =  0.74s
                #
                D_accel_interact   =  5.59 # D[3, 3]
                D_braking_interact = 98.78 # D[3, 3]
                
                f_R += calc_repulsion(distance - (l_self + l_car), A[3, 3], D_accel_interact) * self.v_max * dirn_desired / self.T_relax
                if Theta:
                    ideal_distance = 1.38
                    cur_spd        = calc_norm(self.v) # np.sqrt(np.dot(self.v, self.v))
                    f_R += calc_repulsion(distance + self.T_react * cur_spd - (l_self + l_car + ideal_distance), 1, D_braking_interact) * -dv / self.T_brake
            elif q:
                #
                # SFM (Social Force Model) only
                #
                c_other = -np.dot(car.dirn, ds) / distance

                w_self = 0.5 * self.width
                assert(l_self >= w_self)
                e_self = np.sqrt(l_self * l_self - w_self * w_self) / l_self
                r_self = w_self / np.sqrt(1 - (e_self * c_self)**2)
                
                w_car = 0.5 * car.width
                assert(l_car >= w_car)
                e_car = np.sqrt(l_car * l_car - w_car * w_car) / l_car
                r_car = w_car / np.sqrt(1 - (e_car * c_self)**2)

                #
                # @todo: Initialise lambda to something sensible from one of Anvari's papers if at all possible.
                #
                lambda_self       = lambda_interagent[3, 3]
                perception_factor = lambda_self + (1 - lambda_self) * 0.5 * (1 + c_self)
                
                #
                # Simplified: (1 - p) * q * ... -> ...
                #
                f_R += perception_factor * calc_repulsion(distance - (r_self + r_car), A[3, 3], D[3, 3]) * -u
                # f_R += (1 - p) * q * perception_factor * calc_repulsion(distance - (r_self + r_car), A[3, 3], D[3, 3]) * -u
                # f_R += (1 - p) * q * perception_factor * calc_repulsion(distance - (r_self + r_car), self.A, self.D) * -u

        return f_R    

    def calculate_pedestrian_repulsion(self, pedestrian, MRCA_flag):
        cur_dst_idx = self.route[self.routing_idx + 1]
        cur_dst     = self.nodegraph.nodes[cur_dst_idx]

        f_R = np.zeros(2)
        # return f_R
        
        ds = pedestrian.posn - self.posn
        dv = pedestrian.v    - self.v

        t_min    = 0.5
        # t_min    = 0.
        delta    = ds + t_min * dv
        distance = calc_norm(delta) # np.sqrt(np.dot(delta, delta)) 
        u        = delta / distance

        max_s = 2 * (self.v_max + pedestrian.v_max)
        if np.dot(ds, ds) < max_s * max_s:
            c_self = np.dot(self.dirn, delta) / distance
            # c_self = np.dot(self.dirn, ds) / distance
            if c_self**2 > 1.:
                print("Vehicle::calculate_pedestrian_repulsion (dirn, delta, distance)", self.dirn, delta, distance)
            assert(c_self**2 <= 1.)

            l_self = 0.5 * self.length
            w_self = 0.5 * self.width
            assert(l_self >= w_self)
            e_self = np.sqrt(l_self * l_self - w_self * w_self) / l_self
            assert(e_self <= 1.)

            r_self = w_self / np.sqrt(1 - (c_self * e_self)**2)

            dist_minus_radii = distance - (r_self + pedestrian.radius)
            if dist_minus_radii < 0.:
                return f_R

            tmp_f  = calc_repulsion(dist_minus_radii, A[3, 0], D[3, 0]) * -u
            if np.dot(tmp_f, tmp_f) >= 0.:
                f_R = tmp_f

            e_nav = calc_unit_vector(cur_dst.posn - self.posn)
            f_R   = np.dot(f_R, e_nav) * e_nav 

        #
        # Attenuate by visibility
        #
        cos_pq      = np.dot(u, self.dirn)
        vis_factor  = 0.5 + 0.25 * (1 + cos_pq)
        f_R        *= vis_factor

        # print("Vehicle:calculate_pedestrian_repulsion (id, q.id, f_R): ", self.id, q.id, f_R)
        return f_R


        #
        # Calculate a social force based purely on current distance and accumulate.
        #
        max_s = 2 * (self.v_legal + car.v_legal)
        if np.dot(ds, ds) > max_s * max_s:
            c_self = np.dot(self.dirn, ds) / distance
            # s_self = np.sqrt(1 - c_self * c_self)

            #
            # @todo: After copying the following code to a Vehicle::calculate_pedestrian_repulsion
            # function, we should be able to simplify to r_self = 0.5 * self.length, r_car = 0.5 * car.length
            #

            l_self = 0.5 * self.length
            w_self = 0.5 * self.width
            assert(l_self >= w_self)
            e_self = np.sqrt(l_self * l_self - w_self * w_self) / l_self
            r_self = w_self / np.sqrt(1 - (e_self * c_self)**2)
            
            #
            # @todo: Initialise lambda to something sensible from one of Anvari's papers if at all possible.
            #
            lambda_self       = 0.2
            perception_factor = lambda_self + (1 - lambda_self) * 0.5 * (1 + c_self)            
            
            dist_minus_radii = distance - (r_self + pedestrian.radius)
            if dist_minus_radii < 0.:
                return f_R

            f_R += perception_factor * calc_repulsion(distance - (r_self + ped.radius), A[3, 0], D[3, 0]) * -u
        return f_R
        

    def draw(self, view):
        # print("Vehicle::draw (id, posn, v)", self.id, self.posn, self.v)
        spd = calc_norm(self.v) # np.sqrt(np.dot(self.v, self.v))

        major_axis = 0.5 * self.length * self.dirn
        minor_axis = 0.5 * self.width * np.array([-self.dirn[1], self.dirn[0]])

        # view.draw_line(self.colour, self.posn - 0.5 * self.length * self.dirn, self.posn + 0.5 * self.length * self.dirn, 0.5 * self.width)
        view.fill_ellipse(self.colour, self.posn, major_axis, minor_axis)
        view.draw_line(mnv.white, self.posn, self.posn + 0.5 * (spd / self.v_max) * self.length * self.dirn, 0)
        return
