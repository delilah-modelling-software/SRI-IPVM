#
# Agent and two subclasses, Pedestrian and Vehicle
#
# @todo: Move some of the Vehicle functionality into a Car subclass
# @todo: Add other vehicle subclasses, e.g. Bicycle 
#

import numpy as np
np.seterr(invalid='raise')

# import scipy as sp
# import scipy.stats as ss

# from collections import deque

from model.interaction.interactor import Interactor

import view.defs as svd
import model.agency.defs as smad
import model.agency.reaction as smar
import model.interaction.codec as smic
# import model.interaction.interaction as smii
import model.agency.reaction as smir
import model.agency.state as smas
import model.geometry.shape as smgs
import model.geometry.grid as smgg
import model.defs as smd
import model.configuration as smc
import model.numerical.defs as smnd
import model.numerical.advancing as smna
import model.topology.approachnode as smtApp
import model.topology.activitynode as smtAct

# def estimate_central_hull_point(x_curve_pts, y_line_pts):
    # z_curve_pts = smnd.construct_curve_line_difference(x_curve_pts, y_line_pts)

    # return smnd.evaluate_cubic_spline(z_curve_pts, 0.5)

class Agent(Interactor):
    #
    # Done(was @todo): Pass route instead of src, dst.
    #   cur_src_idx = true_src_idx = route[0]
    #   cur_dst_idx  = route[1]
    #   true_dst_idx = route[-1]
    #
    def __init__(self, id, agent_type, nodegraph, route, posn, dirn):
        super(Agent, self).__init__(id, True)

        self.nodegraph   = nodegraph
        self.route       = route
        self.routing_idx = 0

        # smd.dbgprint("Agent::__init__: agent {} will approach route[{}]".format(id, self.routing_idx))

        cur_src_id = self.route[self.routing_idx]
        cur_dst_id = self.route[self.routing_idx + 1]
        
        #
        # Each agent has state related to an attractive node such as a waiting time.
        #
        self.goal_state = smas.GoalState(cur_dst_id)
        
        #
        # An agent may also have state related to counteragents: such as a primary
        # counteragent (@todo: extend to a small set of them), time-of-conflict,
        # expected future position of counteragent, and the agent's own desired
        # future position and velocity
        #
        self.conflict_state = smas.ConflictState()

        #
        # An agent may also be following and/or leading other agents.
        #

        smd.dbgprint("Agent::__init__: agent {} is currently travelling between nodes {} and {} along route {}".format(id, cur_src_id, cur_dst_id, route))
        # self.cur_src_node = nodegraph.nodes[cur_src_id]
        # self.cur_dst_node = nodegraph.nodes[cur_dst_id]

        self.true_src_id = self.route[ 0]
        self.true_dst_id = self.route[-1]

        self.true_src_node = self.nodegraph.nodes[self.true_src_id]
        self.true_dst_node = self.nodegraph.nodes[self.true_dst_id]
        
        # self.id            = id
        self.agent_type    = agent_type
        # self.T_react       = T_react

        self.posn        = posn
        self.v           = np.zeros(2)
        # self.a           = np.zeros(2)
        self.I_total     = np.zeros(2)
        self.F_collision = np.zeros(2)
        self.F_total     = np.zeros(2)
        self.dirn        = dirn
        assert((smnd.calc_norm_sqr(self.dirn) - 1) < smnd.EPS_m_sqr)

        self.F_collision = np.zeros(2)

        self.max_angular_velocity = 0.

        #
        # There isn't an active_counterzone_idset (zones are assumed known to all agents).
        #
        self.interagent_conflicts      = {smar.ReactionType.REPEL: {}, smar.ReactionType.FOLLOW: {}, smar.ReactionType.VEER: {}, smar.ReactionType.YIELD: {}, smar.ReactionType.AWAIT: {}}
        self.contrazone_interactions   = {} # {smar.ReactionType.REPEL: {}, smar.ReactionType.VEER: {}, smar.ReactionType.YIELD: {}}
        self.active_counteragent_idset = set()

        #
        # @todo: Add a personalised interactivity substructure comprising
        #   A[*] ~ N(smf.A_mu[agent_type, *], smf.A_sigma[agent_type, *]
        #   D[*] ~ N(smf.D_mu[agent_type, *], smf.D_sigma[agent_type, *]
        # forall counteragent-types
        #

        #
        # @todo: Add a personalised dynamic profile comprising
        #   m ~ N(75, ??)
        #   v_pref ~ N(1.34, ??)
        #   a_socmax ~ N(??, ??)
        #   T_react ~ N(0.5, ??)
        #

        #
        # Consider giving each agent a personal temporal horizon, e.g.:
        #   T_hrzn ~ N(2, 2/3) or
        #   T_hrzn ~ B(alpha, beta) * max_T_hrzn s.t. median << mean
        #

        self.activity_schedule = None
        self.activity_focussed = False
        self.t_spawn           = None

        return

    def register_model(self, model):
        self.model = model
        return

    def register_leader(self, leader, s):
        return

    def deregister_leader(self, leader, s):
        return

    def register_activity(self, activity_name):
        if self.activity_schedule != None:
            self.activity_schedule.append(activity_name)
        else:
            self.activity_schedule = [activity_name]
        self.activity_focussed = True
        return

    def complete_activity(self, activity_name):
        assert(self.activity_schedule != None)
        assert(self.activity_schedule != [])
        assert(self.activity_schedule[0] == activity_name)
        if len(self.activity_schedule) == 1:
            self.activity_schedule = None
        else:
            self.activity_schedule = self.activity_schedule[1:]
    
    def get_directional_radius(self, u):
        return 0.

    def get_length_axis(self):
        return 0.

    def get_width_axis(self, u):
        return 0.

    def get_target_speed(self):
        return 0.

    #
    # Compute the agent's position and velocity parallel to a particular
    # direction.
    #
    def get_parallel_position_and_motion(self, dirn):
        x = np.dot(self.posn, dirn)
        u = np.dot(self.v, dirn)
        return x, u

    #
    # Compute the agent's position and velocity perpendicular to a particular
    # direction.
    #
    def get_perpendicular_position_and_motion(self, dirn):
        perp_dirn = smnd.rotate_90ACW(dirn)

        y = np.dot(self.posn, perp_dirn)
        v = np.dot(self.v,    perp_dirn)
        return y, v

    def construct_cubic_spline(self):
        det      = 0
        e_target = None

        if self.goal_state.state_id == smas.GoalStateId.NAVIGATING:
            turning_info = self.goal_state.turning_info
            if self.goal_state.turning_info != None:
                e_target = turning_info.target_dirn
                det = smnd.calc_det2x2(self.dirn, e_target)

        cur_dst = self.get_current_dst_node()
        x0 = self.posn
        x3 = cur_dst.approach_shape.posn

        if det ** 2 >= smnd.EPS_m_sqr:
            L_chord = smnd.calc_norm(x3 - x0)
            L_side  = (1 / 3) * L_chord
            x1      = x0 + L_side * self.dirn
            x2      = x3 - L_side * e_target
        else:
            dx     = (1/3) * (x3 - x0)
            x1     = x0 + dx
            x2     = x3 - dx

        return [x0, x1, x2, x3]


    #
    # Four methods to return
    #   previous source node (None if there isn't one);
    #   current source node;
    #   current destination node; and
    #   next destination node (or None if there isn't one).
    #
    def get_prev_src_node(self):
        assert(self.routing_idx >= 0)
        assert(self.routing_idx + 1 < len(self.route))
        if self.routing_idx > 0:
            prev_src_idx = self.route[self.routing_idx - 1]
            prev_src     = self.nodegraph.nodes[prev_src_idx]
            return prev_src
        else:
            return None

    def get_current_src_node(self):
        assert(self.routing_idx >= 0)
        assert(self.routing_idx + 1 < len(self.route))

        cur_src_idx = self.route[self.routing_idx]
        cur_src     = self.nodegraph.nodes[cur_src_idx]
        return cur_src

    def get_current_dst_node(self):
        assert(self.routing_idx >= 0)
        assert(self.routing_idx + 1 < len(self.route))

        cur_dst_idx = self.route[self.routing_idx + 1]
        cur_dst     = self.nodegraph.nodes[cur_dst_idx]
        return cur_dst

    def get_next_dst_node(self):
        assert(self.routing_idx >= 0)
        assert(self.routing_idx + 1 < len(self.route))
        if self.routing_idx + 2 < len(self.route):
            next_dst_idx = self.route[self.routing_idx + 2]
            next_dst     = self.nodegraph.nodes[next_dst_idx]
            return next_dst
        else:
            return None

    def must_wait(self, node):
        awaiting = \
                self.conflict_state.state_id == smas.ConflictStateId.AWAITING or \
                len(self.interagent_conflicts[smar.ReactionType.AWAIT]) != 0
        return awaiting or node.requires_waiting(self)

    def assess_zone_privilege(self):
        return False

    def assess_zone_exclusions(self, zone_cfg, max_horizon):
        # smd.dbgprint("Agent::assess_zone_exclusions of agent {} at {} moving at {}".format(self.id, self.posn, self.v))
        horizon = max_horizon
        zone_exclusion_info_list = []

        # definite_exclusion_zone_idset = zone_cfg.definite_exclusion_zones[self.agent_type]

        possible_exclusion_zone_idset = zone_cfg.possible_exclusion_zones[self.agent_type]
        if len(possible_exclusion_zone_idset) == 0:
            return max_horizon, []
       
        zone_tbl = zone_cfg.zone_tbl

        exclusive_contrazones = self.contrazone_interactions & possible_exclusion_zone_idset
        for zone_id in exclusive_contrazones:
            zone = zone_tbl[zone_id]

            #
            # If the agent is predicted to cross into a forbidden zone, find when
            # and truncate the agent's temporal horizon to the minimum of that
            # time and the current horizon.
            #
            if zone.currently_excludes(self.agent_type):
                x_omega, future_posn, d, normal, tangent, t_min = zone.shape.get_nearest_approach_info(self.posn, self.v, max_horizon)
                
                smd.dbgprint("Agent::assess_zone_exclusions: agent {} will approach zone {} most closely at {} at time {}".format(self.id, zone.id, future_posn, t_min))
                smd.dbgprint("Agent::assess_zone_exclusions: zone {} has shape {}".format(zone.id, zone.shape))
                if isinstance(zone.shape, smgs.Segment):
                    smd.dbgprint("Agent::assess_zone_exclusions".format(zone.shape.posn1, zone.shape.posn2, zone.shape.radius))
                smd.dbgprint("Agent::assess_zone_exclusions: agent {}({}) should be repelled by {} in dirn {} but not dirn {}".format(self.id, self.agent_type, zone.id, normal, tangent))

                if t_min >= 0 and t_min < max_horizon and zone.shape.inside(future_posn):
                    cur_zone_info = (t_min, zone.id)
                    zone_exclusion_info_list.append(cur_zone_info)

        if len(zone_exclusion_info_list) == 0:
            return max_horizon, []

        zone_exclusion_info_list.sort(key=lambda zone_info: zone_info[0])
        interagent_horizon = zone_exclusion_info_list[0][0]

        #
        # This should allow the self-agent to be repelled by some or all of the relevant exclusion-zones, and omit repulsion from agents within those zones.
        #
        return interagent_horizon, zone_exclusion_info_list

    #
    # Initially, this will probably be used by each agent independently, but
    # could be used per bi- (or even multi-) agent interaction instead.
    #
    def assess_zone_priority(self, ctragent, zone_tbl, posn):
        if self.agent_type == ctragent.agent_type:
            return 0
        
        total_rel_priority = 0
        for zone_id in self.contrazone_interactions:
            zone = zone_tbl[zone_id]
            if len(zone.priorities) == 0:
                continue
            
            relative_priority = zone.assess_relative_priority(self, ctragent) 
            if relative_priority != 0 and zone.shape.inside(posn):
                total_rel_priority += relative_priority
                
        return total_rel_priority

    def register_interagent_collision(self, ctragent, s, relv, nrml, tngnt, momentum_factor):
        # assert(self.agent_type != smad.AgentType.CAR)
        # assert(ctragent.agent_type != smad.AgentType.CAR)
        # assert(self.agent_type != smad.AgentType.CAR or ctragent.agent_type != smad.AgentType.CAR)

        if np.dot(nrml, ctragent.posn - self.posn) > 0:
            # print("Agent::register_interagent_collision between {} and {} with overlap {}".format(self.id, ctragent.id, -s))
            # print("Agent::register_interagent_collision between positions {} and {} with normal {}".format(self.posn, ctragent.posn, nrml))
            nrml = -nrml
            # assert(False)

        #
        # NB: The interaction/dispatcher is taking responsibility for logging collisions, so this method doesn't need to.
        #
        # print("Agent::register_interagent_collision between {} and {} with overlap {}".format(self.id, ctragent.id, -s))

        assert(s < 0)
        neg_s = -s
        
        k     = 120000
        elastic_F = momentum_factor * k * neg_s * nrml

        kappa = 240000
        friction_F = momentum_factor * kappa * neg_s * np.dot(relv, tngnt) * tngnt

        self.F_collision += elastic_F + friction_F
        return

    def register_contrazone_interaction(self, zone_id, interaction):
        # print("Agent::register_contrazone_interaction agent {} vs zone {}".format(self.id, zone_id))
        if zone_id in self.contrazone_interactions:
            old_interaction = self.contrazone_interactions[zone_id]
            old_interaction.merge_interaction(interaction)
            return

        if interaction.t_min > smc.T_master_horizon:
            return
            
        self.contrazone_interactions[zone_id] = interaction
        # self.contrazone_interactions.add(zone_id)
        return

    def register_contrazone_conflict(self, zone_id, interaction, t_min):
        # print("Agent::register_contrazone_interaction agent {} vs zone {}".format(self.id, zone_id))
        self.contrazone_interactions[zone_id] = interaction
        return

    def is_concerned(self, ctragent, rel_posn, rel_v):
        return True

    def register_interagent_interaction(self, interaction, reaction):
        assert(reaction != None)
        assert(interaction.t_conflict != None or reaction.reaction_type == smar.ReactionType.AWAIT)

        #
        # Add counteragent's id to the list of known potential conflicts
        # @todo: Consider renaming as known_agent_idset and giving the set a maximum size based on a
        #   subitizing principle - if 4 is the biological limit of subitization [q.v. Kutter et al. 2023] for most
        #   human beings, then it is likely to be a good default limit. Might be worth adding a hyperparameter
        #   smc.max_n_conflicts
        #

        assert(reaction.reaction_type != smir.ReactionType.IGNORE)
        if reaction.reaction_type == smir.ReactionType.FOLLOW:
            assert(False)
            self.register_leader(reaction.s)
            return

        ctragent = interaction.ctragent
        if self != interaction.agent:
            assert(self == interaction.ctragent)
            ctragent = interaction.agent

        if False: # ctragent.id in self.active_counteragent_idset:
            # old_interaction, old_reaction = self.interagent_interactions[ctragent.id]
            # old_interaction.merge_interaction(interaction, old_reaction, reaction)
            return

        if interaction.t_conflict != None and interaction.t_conflict > smc.T_master_horizon:
            return


        self.active_counteragent_idset.add(ctragent.id)

        assert(reaction.reaction_type in self.interagent_conflicts)
        interaction_tbl = self.interagent_conflicts[reaction.reaction_type]
        interaction_tbl[ctragent.id] = interaction, reaction

        return

    def calc_interagent_influence(self, ctragent):
        assert(False)
        print("Agent::calc_interagent_influence".format(self.id, ctragent.id))
        return np.zeros(2)

    def apply_navigating_policy(self):
        assert(self.conflict_state.state_id == smas.ConflictStateId.MODULATING)
        assert(self.goal_state.state_id in {smas.GoalStateId.NAVIGATING, smas.GoalStateId.EXITING})

        if self.goal_state.state_id == smas.GoalStateId.NAVIGATING and self.goal_state.turning_info != None:
            turning_info = self.goal_state.turning_info
            if len(turning_info.stages) > 0:
                cur_stage = turning_info.stages[0]
                target_shape = cur_stage.shape
                max_spd = min(self.get_target_speed(), cur_stage.spd_limit)
                if len(turning_info.stages) > 1:
                    # assert(False)
                    next_stage = turning_info.stages[1]
                    max_spd = min(max_spd, next_stage.spd_limit)
                if len(turning_info.stages) > 2:
                    # assert(False)
                    next_stage = turning_info.stages[2]
                    max_spd = min(max_spd, next_stage.spd_limit)
            else:
                cur_dst_idx = self.route[self.routing_idx + 1]
                cur_dst = self.nodegraph.nodes[cur_dst_idx]
                target_shape = cur_dst.approach_shape
                max_spd = self.get_target_speed()

            x_omega, dist, nv, tv = target_shape.get_nearest_point_info(self.posn)
            v_desired = max_spd * smnd.calc_unit_vector(x_omega - self.posn)
            I_nav = (v_desired - self.v) / self.T_relax

            return I_nav
        else:
            cur_dst_idx = self.route[self.routing_idx + 1]
            cur_dst = self.nodegraph.nodes[cur_dst_idx]

            x_omega, dist, nv, tv = cur_dst.approach_shape.get_nearest_point_info(self.posn)

            v_desired = self.get_target_speed() * smnd.calc_unit_vector(x_omega - self.posn)
            I_nav = (v_desired - self.v) / self.T_relax

            return I_nav

        return

    def apply_waiting_policy(self):
        assert(self.goal_state.state_id in {smas.GoalStateId.CONVERGING, smas.GoalStateId.WAITING})

        cur_dst_idx = self.route[self.routing_idx + 1]
        cur_dst = self.nodegraph.nodes[cur_dst_idx]
        assert(self.goal_state.node_id == cur_dst.id)

        active_focal_shape = None
        if isinstance(cur_dst, smtAct.ActivityNode):
            agent_activity_name = self.activity_schedule[0]
            active_focal_shape, _ = cur_dst.focal_shape_tbl[agent_activity_name]
        else:
            assert(isinstance(cur_dst, smtApp.ApproachNode))
            active_focal_shape = cur_dst.focal_shape

        # print("Agent::apply_waiting_policy - agent {} at node {} with shape {}".format(self.id, cur_dst.id, active_focal_shape))
        
        x_omega, dist, nv, tv = active_focal_shape.get_nearest_point_info(self.posn)

        x_pref = x_omega
        ds = x_omega - self.posn
        spd_pref = self.get_target_speed()
        #
        # Johansson et al. (2015)
        #
        r = 4 * self.T_relax * self.get_target_speed()
        if dist > r:
            #
            # Preferred Position
            #
            v_pref = spd_pref * (-nv)
        # elif active_focal_shape.inside(self.posn + smnd.T_step * self.v):
            #
            # Preferred Velocity
            #
            # v_pref = np.zeros(2)
        else:
            v_pref = spd_pref * ds / r
            
        return x_pref, v_pref


    def assess_possible_veer(self, ctragent, x_curve_pts, y_line_pts, e_evade, d_crit):
        print("Agent::assess_possible_detour - agent {} might detour via {} around {} travelling between {}".format(self, x_curve_pts, ctragent, y_line_pts))
        # assert(False)
        
        z_curve_pts = smnd.construct_curve_line_difference(x_curve_pts, y_line_pts)

        #
        # Map the control points to a single dimension, then construct the derivative quadratic spline (i.e. parabola).
        #
        w_pts  = []
        for i in range(len(z_curve_pts)):
            z = z_curve_pts[i]
            w = np.dot(z, e_evade)
            w_pts.append(w)

        assert(len(w_pts) == 4)

        dw_pts = []
        for i in range(len(w_pts) - 1):
            dw_pts.append(2 * (w_pts[i + 1] - w_pts[i]))

        assert(len(dw_pts) == 3)

        #
        # Now solve a quadratic eqn to find the minimum distance. The sign of the discriminant (disc) determines the number of internal extrema:
        #   -ve => no internal soln => min d will occur at a parametric boundary (t in {0., 1.})
        #   0   => \exists! soln    => min d will occur at -b / 2a
        #   +ve => \exists two internal solns => min d will occur at t0 or t1
        #
        a    = dw_pts[0] - 2 * dw_pts[1] + dw_pts[2]
        b    = 2 * (dw_pts[1] - dw_pts[0])
        c    = dw_pts[0]
        disc = b**2 - 4 * a * c
        centre = -b / 2 * a
        if disc < 0:
            t0 = 0
            t1 = 1
        elif disc > 0:
            disc_rt = np.sqrt(disc)
            t0 = centre - disc_rt
            t1 = centre + disc_rt
        else:
            assert(disc == 0)
            t0 = t1 = centre

        assert(t0 <= t1)
        if t0 >= 1:
            t0 = t1 = 1
            w0 = smnd.evaluate_cubic_spline(w_pts, t0)
            rel_safeness = w0 - d_crit
        elif t1 <= 0:
            t0 = t1 = 0
            w1 = smnd.evaluate_cubic_spline(w_pts, t1)
            rel_safeness = w1 - d_crit
        else:
            if t0 < 0:
                t0 = 0
            if t1 > 1:
                t1 = 1
            w0 = smnd.evaluate_cubic_spline(w_pts, t0)
            w1 = smnd.evaluate_cubic_spline(w_pts, t1)

            rel_safeness = min(w0, w1) - d_crit
        
        # midpoint_relposn = smnd.evaluate_cubic_spline(z_curve_pts, 0.5)
        # midpoint_relposn = estimate_central_hull_point(x_curve_pts, y_line_pts)
        # u             = smnd.calc_unit_vector(midpoint_relposn)
        # r_total       = self.get_directional_radius(u) + ctragent.get_directional_radius(-u)
        # rel_safeness  = smnd.calc_norm(midpoint_relposn) - (smd.d_safe_IA + r_total)

        return rel_safeness

    # def assess_possible_swerve(self, ctragent, x_curve_pts, y_line_pts):
    def assess_possible_detour(self, ctragent, x_curve_pts, y_line_pts):
        print("Agent::assess_possible_detour - agent {} might detour via {} around {} travelling between {}".format(self, x_curve_pts, ctragent, y_line_pts))
        assert(False)
        
        z_curve_pts = smnd.construct_curve_line_difference(x_curve_pts, y_line_pts)
        midpoint_relposn = smnd.evaluate_cubic_spline(z_curve_pts, 0.5)
        # midpoint_relposn = estimate_central_hull_point(x_curve_pts, y_line_pts)
        u             = smnd.calc_unit_vector(midpoint_relposn)
        r_total       = self.get_directional_radius(u) + ctragent.get_directional_radius(-u)
        rel_safeness  = smnd.calc_norm(midpoint_relposn) - (smd.d_safe_IA + r_total)

        return midpoint_relposn, rel_safeness

    # def construct_GRVO(cone, v_alpha, v_beta, T, w):
        # convex = dst.convex
        
        # shape = smgs.construct_tcone(X_beta, X_alpha, T, convex)
        # shape.apex = (1 - w) * v_alpha + w * v_beta

        # return shape


    def init_best_possible_swerve(self, ctragent, interaction, t0):
        print("Agent::init_best_possible_swerve - agents {}, {} interacting with rel_posn {} at time {} to last for {}".format(self.id, ctragent.id, interaction.rel_posn, t0, interaction.t_min))

        d_margin = smd.d_safe_CZ
        T_swerve  = interaction.t_min

        # hl_self  = self.get_length_axis()
        # hw_self  = self.get_width_axis()
        # hl_other = ctragent.get_length_axis()
        # hw_other = ctragent.get_width_axis()

        spd_other, u1 = smnd.calc_polar_decomp(ctragent.v)
        u2            = smnd.rotate_90ACW(u1)

        #
        # Augment half-length and half-width of counteragent with half-width of self.
        #
        # cfd = hl_other * u1
        # cld = hw_other * u2

        #
        # @todo: Build the velocity cone as:
        #   T_shift = smd.T_SRCA_horizon
        #   T_rem   = T_swerve - T_shift
        #   src.translate(T_shift * v_alpha)
        #   src.translate(T_shift * v_beta)
        #   construct_tcone(src, dst, T_rem)
        #
        assert(T_swerve > smd.T_SRCA_horizon)

        vel_cone = None
        if interaction.posn_cone != None:
            cone = interaction.posn_cone
            print("Agent::init_best_possible_swerve interaction's posn-cone at apex {}".format(cone.apex))
            print("Agent::init_best_possible_swerve interaction's posn-cone has bounding directions {}, {}".format(cone.e_left, cone.e_right))

            #
            # @todo: Consider using an RVO or GRVO later.
            #
            vel_cone = cone
            # vel_cone = construct_GRVO(cone, ctragent.v, T_swerve, 1)
            # vel_cone = cone.generate_GRVO(self.v, ctragent.v, T_swerve, 0.5)
            # vel_cone = cone.generate_GRVO(self.v, ctragent.v, T_swerve, w)

            u_VO, _, nrml_VO, tngt_VO = vel_cone.get_nearest_point_info(self.v - ctragent.v)
            # u_VO += ctragent.v
            
        # ctragent_VO = self.generate_velocity_obstacle(ctragent, interaction.rel_posn, T_swerve)
        # print("Agent::init_best_possible_swerve derived a velocity obstacle {}".format(ctragent_VO))

        # u_VO, _, nrml_VO, tngt_VO = ctragent_VO.get_nearest_point_info(self.v)
        # u_VO, _, nrml_VO, tngt_VO = ctragent_VO.get_nearest_point_info(self.v - ctragent.v)
        # u_VO += ctragent.v

        cur_dst_idx = self.route[self.routing_idx + 1]
        cur_dst     = self.nodegraph.nodes[cur_dst_idx]
       
        x_omega, dist, nv, tv = cur_dst.approach_shape.get_nearest_point_info(self.posn)

        v_des           = self.get_target_speed() * smnd.calc_unit_vector(x_omega - self.posn)
        v_des_proj_norm = np.dot(v_des - u_VO, nrml_VO)
        
        if v_des_proj_norm < 0:
            v_des -= v_des_proj_norm * nrml_VO
            print("Agent::init_best_possible_swerve clipped the desired velocity to {}".format(v_des))

        #
        # @todo: Consider whether to reintroduce this extension along the tangent later:
        #   by replacing the projection with a rotation onto the boundary of the VO, it
        #   would preserve speed (similar, possibly identical to CS-ORCA [Durand 2018]).
        #
        if 0:
            spd_des, dirn_des = smnd.calc_polar_decomp(v_des)

            cos_psi = np.dot(dirn_des, tngt_VO)
            a = 1
            b = 2 * cos_psi * spd_des
            c = spd_des**2 - (self.get_target_speed())**2

            disc = b**2 - 4 * a * c
            s = (-b + np.sqrt(disc))/(2 * a)

            v_des += s * tngt_VO

            print("Agent::init_best_possible_swerve used the cosine rule and quadratic formula to derive a slightly better velocity {}".format(v_des))

            
            
            # spd_self, w1 = smnd.calc_polar_decomp(self.v)
            # w2 = smnd.rotate_90ACW(w1)
            # v_des = 

        self.conflict_state.start_swerving_state(set([ctragent.id]), set(), t0, T_swerve, v_des, ctragent_VO)
            
        return

    def determine_veering_criteria(self, ctragent):
        # assert(False)
        
        ctragent_perpdirn = smnd.rotate_90ACW(ctragent.dirn)
        
        relposn = self.posn - ctragent.posn
        rel_posn_prl  = np.dot(relposn, ctragent.dirn)
        rel_posn_perp = np.dot(relposn, ctragent_perpdirn)
        
        if rel_posn_prl > ctragent.rect_length:
            #
            # self is before ctragent
            #

            if rel_posn_perp > ctragent.rect_width:
                # assert(False)

                #
                # self is to the left of ctragent
                # Left spline crosses before, right spline passes on the left
                #
                e_evade_left  = ctragent.dirn
                e_evade_right = ctragent_perpdirn
                d_crit_left   = ctragent.rect_length
                d_crit_right  = ctragent.rect_width
                s_safe_left   = smc.d_safe_IA_before
                s_safe_right  = smc.d_safe_IA_beside
                
            elif rel_posn_perp < -ctragent.rect_width:
                # assert(False)

                #
                # self is to the right of ctragent
                # Left spline passes on the right, right spline crosses before
                #
                e_evade_left  = -ctragent_perpdirn
                e_evade_right = ctragent.dirn
                d_crit_left   = ctragent.rect_width
                d_crit_right  = ctragent.rect_length
                s_safe_left   = smc.d_safe_IA_beside
                s_safe_right  = smc.d_safe_IA_before

            else:
                # assert(False)

                #
                # self is directly before ctragent
                # Both splines cross before
                #
                e_evade_left  = ctragent.dirn
                e_evade_right = ctragent.dirn
                d_crit_left   = ctragent.rect_length
                d_crit_right  = ctragent.rect_length
                s_safe_left   = smc.d_safe_IA_before
                s_safe_right  = smc.d_safe_IA_before

        elif rel_posn_prl < -ctragent.rect_length:
            #
            # self is behind ctragent
            #

            if rel_posn_perp > ctragent.rect_width:
                #
                # self is to the left of ctragent
                # Left spline passes on the left, right spline crosses behind
                #

                e_evade_left  = ctragent_perpdirn
                e_evade_right = -ctragent.dirn
                d_crit_left   = ctragent.rect_width
                d_crit_right  = ctragent.rect_length
                s_safe_left   = smc.d_safe_IA_beside
                s_safe_right  = smc.d_safe_IA_behind
            elif rel_posn_perp < -ctragent.rect_width:
                #
                # self is to the right of ctragent
                # Left spline crosses behind, right spline passes on the right
                #

                e_evade_left  = -ctragent.dirn
                e_evade_right = -ctragent_perpdirn
                d_crit_left   = ctragent.rect_length
                d_crit_right  = ctragent.rect_width
                s_safe_left   = smc.d_safe_IA_behind
                s_safe_right  = smc.d_safe_IA_beside

            else:
                # assert(False)

                #
                # self is directly behind ctragent
                # Both splines cross behind
                #
                e_evade_left  = -ctragent.dirn
                e_evade_right = -ctragent.dirn
                d_crit_left   = ctragent.rect_length
                d_crit_right  = ctragent.rect_length
                s_safe_left   = smc.d_safe_IA_behind
                s_safe_right  = smc.d_safe_IA_behind

        else:
            # assert(False)

            #
            # self is (probably, hopefully) beside ctragent
            #
            if rel_posn_perp > ctragent.rect_width:
                # assert(False)

                #
                # self is to the left of ctragent
                # Both splines pass on the left.
                #
                e_evade_left  = ctragent_perpdirn
                e_evade_right = ctragent_perpdirn
                d_crit_left   = ctragent.rect_width
                d_crit_right  = ctragent.rect_width
                s_safe_left   = smc.d_safe_IA_beside
                s_safe_right  = smc.d_safe_IA_beside
            elif rel_posn_perp < -ctragent.rect_width:
                # assert(False)

                #
                # self is to the right of ctragent
                # Both splines pass on the right.
                #
                e_evade_left  = -ctragent_perpdirn
                e_evade_right = -ctragent_perpdirn
                d_crit_left   = ctragent.rect_width
                d_crit_right  = ctragent.rect_width
                s_safe_left   = smc.d_safe_IA_beside
                s_safe_right  = smc.d_safe_IA_beside
            else:
                #
                # self is inside ctragent
                # Should not happen (collision code should handle it first),
                # but if it does we will be returning None.
                #
                # @todo: See if we can *really* eliminate this case.
                #
                
                # print("Agent {} at {} is too close to counteragent {} at {}".format(self.id, self.posn, ctragent.id, ctragent.posn))
                # assert(False)
                return None
        return e_evade_left, e_evade_right, d_crit_left, d_crit_right, s_safe_left, s_safe_right

    #
    # @todo: Replace this function with a discrete choice model of your choice.
    #
    def choose_between_viable_veers(self, ctragent, s_min_left, s_min_right, s_safe_left, s_safe_right, t0, T_left, T_right):
        if s_min_left > s_safe_left and s_min_right > s_safe_right:
            if T_left < T_right:
                return 0
            else:
                return 1
        elif s_min_left > s_safe_left:
            return 0
        elif s_min_right > s_safe_right:
            return 1
        else:
            if T_left < T_right:
                return 0
            else:
                return 1
        return -1
    
    def init_best_possible_simple_veer(self, ctragent, interaction, reaction, t0):
        print("Agent::init_best_possible_veer - agent {} investigating veer around ctragent {}".format(self.id, ctragent.id))
        
        T_curve           = interaction.t_conflict
        # x_self_future     = self.posn + T_curve * self.v
        x_ctragent_future = ctragent.posn + T_curve * ctragent.v

        spd_other, u1 = smnd.calc_polar_decomp(ctragent.v)
        u2      = smnd.rotate_90ACW(u1)

        #
        # @todo: Pedestrians approaching stationary cars from behind are a problem.
        #

        relv_cone = interaction.relv_cone
        if relv_cone == None and reaction.reaction_type == smar.ReactionType.VEER:
            relv_cone = reaction.RVO

        if relv_cone == None:
            return
        
        relv              = self.v - ctragent.v
        relspd, unit_relv = smnd.calc_polar_decomp(relv)

        # assert(np.dot(unit_relv, smnd.rotate_90ACW(relv_cone.e_left) < 0)
        # assert(np.dot(unit_relv, smnd.rotate_90ACW(relv_cone.e_right) > 0)

               
        relv_apex = ctragent.v
        target_v_left  = relv_apex + relv_cone.l_left * relv_cone.e_left
        target_v_right = relv_apex + relv_cone.l_right * relv_cone.e_right

        chord_left  = T_curve * target_v_left
        chord_right = T_curve * target_v_right

        target_posn_left  = self.posn + chord_left
        target_posn_right = self.posn + chord_right

        chord_length_left, chord_e_left  = smnd.calc_polar_decomp(chord_left)
        chord_length_right, chord_e_right = smnd.calc_polar_decomp(chord_right)

        third_factor      = 1 / 3
        side_length_left  = third_factor * chord_length_left
        side_length_right = third_factor * chord_length_right

        pt0 = self.posn

        pt1_L = self.posn + side_length_left  * self.dirn
        pt1_R = self.posn + side_length_right * self.dirn

        pt3_L = target_posn_left
        pt3_R = target_posn_right

        pt2_L = pt3_L - side_length_left * chord_e_left
        pt2_R = pt3_R - side_length_right * chord_e_right

        spline_left  = [pt0, pt1_L, pt2_L, pt3_L]
        spline_right = [pt0, pt1_R, pt2_R, pt3_R]
        splines      = [spline_left, spline_right]
        ctrline      = [ctragent.posn, x_ctragent_future]

        omega_max = self.max_angular_velocity
        spd_max   = self.v_max

        print("omega_max {}, spd_max {}".format(omega_max, spd_max))
               
        E_left,  L_left,  T_left  = smnd.analyse_cubic_spline(spline_left,  omega_max, spd_max, T_curve)
        E_right, L_right, T_right = smnd.analyse_cubic_spline(spline_right, omega_max, spd_max, T_curve)

        vc = self.determine_veering_criteria(ctragent)
        if vc == None:
            print("Agent::init_best_possible_veer - agent {} vs ctragent {} yields no criteria".format(self.id, ctragent.id))
            return
        e_evade_left, e_evade_right, d_crit_left, d_crit_right, s_safe_left, s_safe_right = vc

        print("Agent::init_best_possible_veer - agent {} vs ctragent {} yields usable criteria".format(self.id, ctragent.id))
        # assert(False)
        
        #
        # Choose whichever curve will deliver the better score. If that score
        # is negative, then switch to the STOPPING state, etc.
        if T_left <= T_curve and T_right <= T_curve:
            # assert(False)
            
            #
            # @todo: Add biases here. These biases could be:
            #   personal, e.g. daring/prudent when considering whether to
            #       cross before or behind a vehicle;
            #   social, e.g. lateral bias for pedestrian-vs-counteragent conflicts;
            #   legal, e.g. lateral bias for vehicle-vs-vehicle conflicts.
            #

            #
            # @todo: Assess both possible swerves, and then use safety_decision_fn(self, T_A, T_B, safeness_A, safeness_B) to decide
            # based on agent-specific priorities.
            #

            s_min_left  = self.assess_possible_veer(ctragent, spline_left,  ctrline, e_evade_left, d_crit_left)
            s_min_right = self.assess_possible_veer(ctragent, spline_right, ctrline, e_evade_right, d_crit_right)

            self.model.register_candidate_spline([self.id, ctragent.id, spline_left,  (E_left,  L_left,  T_left),  (e_evade_left,  d_crit_left,  s_safe_left),  s_min_left])
            self.model.register_candidate_spline([self.id, ctragent.id, spline_right, (E_right, L_right, T_right), (e_evade_right, d_crit_right, s_safe_right), s_min_right])

            # print("Agent::init_best_possible_veer {} v {} via {} has E/L/T {}, e/d/s {}, s_min {}".format(self.id, ctragent.id, spline_left,  (E_left,  L_left,  T_left),  (e_evade_left,  d_crit_left,  s_safe_left),  s_min_left))
            # print("Agent::init_best_possible_veer {} v {} via {} has E/L/T {}, e/d/s {}, s_min {}".format(self.id, ctragent.id, spline_right, (E_right, L_right, T_right), (e_evade_right, d_crit_right, s_safe_right), s_min_right))
            # assert(False)

            if s_min_left > 0 and s_min_right > 0:
                veer_idx = self.choose_between_viable_veers(ctragent, s_min_left, s_min_right, s_safe_left, s_safe_right, t0, T_left, T_right)
                if veer_idx >= 0:
                    if veer_idx == 0:
                        x_desired   = target_posn_left 
                        v_desired   = target_v_left
                        self.conflict_state.start_veering_state(ctragent.id, t0, T_curve, ctrline, x_desired, v_desired, spline_left)
                    else:
                        assert(veer_idx == 1)
                        x_desired   = target_posn_right 
                        v_desired   = target_v_right
                        self.conflict_state.start_veering_state(ctragent.id, t0, T_curve, ctrline, x_desired, v_desired, spline_right)
            elif s_min_left > 0:
                x_desired   = target_posn_left 
                v_desired   = target_v_left
                self.conflict_state.start_veering_state(ctragent.id, t0, T_curve, ctrline, x_desired, v_desired, spline_left)
            elif s_min_right > 0:
                x_desired   = target_posn_right 
                v_desired   = target_v_right
                self.conflict_state.start_veering_state(ctragent.id, t0, T_curve, ctrline, x_desired, v_desired, spline_right)
            return

        elif T_left <= T_curve:
            x_desired   = target_posn_left 
            v_desired   = target_v_left
            y_line_pts  = [ctragent.posn, x_ctragent_future]

            s_min = self.assess_possible_veer(ctragent, spline_left, ctrline, e_evade_left, d_crit_left)

            self.model.register_candidate_spline([self.id, ctragent.id, spline_left,  (E_left,L_left, T_left),     (e_evade_left,  d_crit_left,  s_safe_left), s_min])
            self.model.register_candidate_spline([self.id, ctragent.id, spline_right, (E_right, L_right, T_right), (e_evade_right, d_crit_right, s_safe_right), None])

            # print("Agent::init_best_possible_veer {} v {} via {} has E/L/T {}, e/d/s {}, s_min {}".format(self.id, ctragent.id, spline_left,  (E_left,  L_left,  T_left),  (e_evade_left,  d_crit_left,  s_safe_left),  s_min))
            # print("Agent::init_best_possible_veer {} v {} via {} has E/L/T {}, e/d/s {}, s_min {}".format(self.id, ctragent.id, spline_right, (E_right, L_right, T_right), (e_evade_right, d_crit_right, s_safe_right), None))
            # assert(False)

            if s_min >= 0.:
                # assert(False)
                
                print("Agent::init_best_possible_detour - agent {} will avoid ctragent {} over {}s by curving defined by {}".format(self.id, ctragent.id, T_left, spline_left))
                self.conflict_state.start_veering_state(ctragent.id, t0, T_curve, ctrline, x_desired, v_desired, spline_left)
                return

        elif T_right <= T_curve:
            x_desired   = target_posn_right 
            v_desired   = target_v_right
            y_line_pts  = [ctragent.posn, x_ctragent_future]

            s_min = self.assess_possible_veer(ctragent, spline_right, ctrline, e_evade_right, d_crit_right)

            self.model.register_candidate_spline([self.id, ctragent.id, spline_left,  (E_left,  L_left,  T_left),  (e_evade_left,  d_crit_left,  s_safe_left), None])
            self.model.register_candidate_spline([self.id, ctragent.id, spline_right, (E_right, L_right, T_right), (e_evade_right, d_crit_right, s_safe_right), s_min])

            print("Agent::init_best_possible_veer {} v {} via {} has E/L/T {}, e/d/s {}, s_min {}".format(self.id, ctragent.id, spline_left,  (E_left,  L_left,  T_left),  (e_evade_left,  d_crit_left,  s_safe_left),  None))
            print("Agent::init_best_possible_veer {} v {} via {} has E/L/T {}, e/d/s {}, s_min {}".format(self.id, ctragent.id, spline_right, (E_right, L_right, T_right), (e_evade_right, d_crit_right, s_safe_right), s_min))

            if s_min >= 0.:
                print("Agent::init_best_possible_veer - agent {} will avoid ctragent {} over {}s by curving defined by {}".format(self.id, ctragent.id, T_right, spline_right))
                self.conflict_state.start_veering_state(ctragent.id, t0, T_curve, ctrline, x_desired, v_desired, spline_right)
                return
        else:
            self.model.register_candidate_spline([self.id, ctragent.id, spline_left,  (E_left,  L_left,  T_left),  (e_evade_left,  d_crit_left,  s_safe_left),  None])
            self.model.register_candidate_spline([self.id, ctragent.id, spline_right, (E_right, L_right, T_right), (e_evade_right, d_crit_right, s_safe_right), None])

            print("Agent::init_best_possible_veer {} v {} via {} has E/L/T {}, e/d/s {}, s_min {}".format(self.id, ctragent.id, spline_left,  (E_left,  L_left,  T_left),  (e_evade_left,  d_crit_left,  s_safe_left),  None))
            print("Agent::init_best_possible_veer {} v {} via {} has E/L/T {}, e/d/s {}, s_min {}".format(self.id, ctragent.id, spline_right, (E_right, L_right, T_right), (e_evade_right, d_crit_right, s_safe_right), None))

            print("")
            x_agent_future = self.posn + T_curve * self.v
            stop_shape = smgs.Circle(x_agent_future, self.radius)
            s = T_curve * smnd.calc_norm(self.v)
            self.conflict_state.start_yielding_state(ctragent.id, stop_shape, s)

        print("Agent::init_best_possible_veer - agent {} cannot avoid ctragent {} in {}s by either curve defined by {} or {}".format(self.id, ctragent.id, T_curve, spline_left, spline_right))
        return

    def init_best_possible_complex_veer(self, veerable_items, t0):
        primary_interaction, primary_reaction = veerable_items[0][1] # [0]

        # T_primary = primary_interaction.t_conflict
        T_primary = primary_reaction.t_conflict
        assert(T_primary == t0)

        v_max = self.v_max
        e1    = self.dirn
        e2    = smnd.rotate_90ACW(self.dirn)

        #
        # Instead of constructing a polygon in the agent's velocity-space, we will just perform multiple projections
        # (<= 1 per halfplane), so we won't need extra halfplanes to force the construction of a bounded polygon.
        #
        halfplanes = []

        # hp0 = smgs.Halfplane(-v_max * e1, e1)
        # hp1 = smgs.Halfplane(-v_max * e2, e2)
        # hp2 = smgs.Halfplane(v_max * e1, -e1)
        # hp3 = smgs.Halfplane(v_max * e2, -e2)
        # halfplanes = [hp0, hp1, hp2, hp3]

        for veerable_item in veerable_items:
            interaction, reaction = veerable_item[1]

            t                 = reaction.t_conflict
            assert(t >= T_primary)

            id1, id2          = smic.interagent_codec.decode(interaction.combined_id)
            major_ctragent_id = self.id ^ id1 ^ id2
            assert(major_ctragent_id == id2)

            ctragent          = self.model.active_agents[major_ctragent_id]

            relv_cone = interaction.relv_cone
            if relv_cone == None and reaction.reaction_type == smar.ReactionType.VEER:
                relv_cone = reaction.RVO

            if relv_cone == None:
                continue

            relv = self.v - ctragent.v
            relspd, unit_relv = smnd.calc_polar_decomp(relv)

            relv_apex = ctragent.v
            target_v_left = relv_apex + relv_cone.l_left * relv_cone.e_left
            target_v_right = relv_apex + relv_cone.l_right * relv_cone.e_right

            FIX_2025_11_11 = True
            if FIX_2025_11_11:
                n_left  = smnd.rotate_90ACW(relv_cone.e_left)
                n_right = -smnd.rotate_90ACW(relv_cone.e_right)
                if np.dot(relv, n_left) > np.dot(relv, n_right):
                    halfplane = smgs.Halfplane(target_v_left, smnd.rotate_90ACW(relv_cone.e_left))
                    halfplanes.append(halfplane)
                else:
                    halfplane = smgs.Halfplane(target_v_right, -smnd.rotate_90ACW(relv_cone.e_right))
                    halfplanes.append(halfplane)
            else:
                if smnd.calc_norm_sqr(target_v_left - self.v) < smnd.calc_norm_sqr(target_v_right - self.v):
                    halfplane = smgs.Halfplane(target_v_left, smnd.rotate_90ACW(relv_cone.e_left))
                    halfplanes.append(halfplane)
                else:
                    halfplane = smgs.Halfplane(target_v_right, -smnd.rotate_90ACW(relv_cone.e_right))
                    halfplanes.append(halfplane)


        #
        # Instead of computing a target_v as the closest velocity within a velocity polygon constructed by intersecting all the
        # velocity halfplanes.
        #
        veer_required = False
        target_v      = self.v
        for halfplane in halfplanes:
            mu = np.dot(target_v - halfplane.posn, halfplane.normal)
            if mu < 0:
                target_v      -= mu * halfplane.normal
                veer_required  = True

        #
        # If there was no solution (i.e. the intersection of the halfplanes is the empty set), then we must return
        # without specifying a veer.
        #
        if veer_required:
            for halfplane in halfplanes:
                mu = np.dot(target_v - halfplane.posn, halfplane.normal)
                if mu < 0 and mu**2 >= smnd.EPS_m_sqr:
                    return True

        if not veer_required:
            return False

        #
        # DEPRECATED
        #
        # v_polygon = smgs.construct_polygon(halfplanes)
        # assert not v_polygon.contains(self.v)
        # target_v_info = v_polygon.get_nearest_point_info(self.v)
        # target_v      = target_v_info[0]

        chord_vector = T_primary * target_v

        target_posn = self.posn + chord_vector

        chord_length, chord_e = smnd.calc_polar_decomp(chord_vector)

        third_factor = 1 / 3
        side_length = third_factor * chord_length

        pt0 = self.posn

        pt1 = self.posn + side_length * self.dirn

        pt3 = target_posn

        pt2 = pt3 - side_length * chord_e

        spline = [pt0, pt1, pt2, pt3]

        omega_max = self.max_angular_velocity
        spd_max   = self.v_max
        E, L, T   = smnd.analyse_cubic_spline(spline, omega_max, spd_max, T_primary)

        self.conflict_state.start_veering_state(ctragent.id, t0, T_primary, None, target_posn, target_v,
                                                spline)

        return True

    def init_best_possible_stop(self, ctragent, interaction, t0):
        assert(False)
        
        #
        # @todo: Clarify that this is a netto displacement (boundary-to-boundary)
        # rather than brutto (position-to-position) distance.
        #
        d_margin = smc.d_safe_IA_maximum

        # id1, id2 = smii.interagent_codec.decode_interaction(interaction.combined_id)
        # major_ctragent_id = self.id ^ id1 ^ id2
        # ctragent          = agent_tbl[major_ctragent_id]
        T_curve           = interaction.t_min
        if T_curve <= 0:
            assert(False)
            return
        
        x_self_future     = self.posn + T_curve * self.v
        x_ctragent_future = ctragent.posn + T_curve * ctragent.v

        hl_self  = self.get_length_axis()
        hw_other = ctragent.get_width_axis()

        #
        # @todo: Divide by abs(sin(theta)) where cos(theta) = np.dot(self.dirn, ctragent.dirn)
        #
        cos_dirn = np.dot(self.dirn, ctragent.dirn)
        d_stopping_margin = (d_margin + hw_other + hl_self) / np.sqrt(1 - cos_dirn**2)
        # d_stopping_margin = d_margin + hw_other + hl_self

        # x_desired = x_ctragent_future - d_stopping_margin * self.dirn
        x_desired = x_self_future - d_stopping_margin * self.dirn
        # print("Agent::init_best_possible_stop".format(self.id, self.posn, x_desired))
        if np.dot(x_desired - self.posn, self.dirn) <= 0:
            assert(False)
            
            # print("Agent::init_best_possible_stop - failure! {} at {} wanted to stop at {}".format(self.id, self.posn, x_desired))
            return

        print("Agent::init_best_possible_stop - {} at {} will try to stop at {}".format(self.id, self.posn, x_desired))

        #
        # @todo: There should be a braking relaxation time. 
        #
        T_brk = self.T_relax
        if self.agent_type != smad.AgentType.PEDESTRIAN:
            T_brk_vhcl = 0.77 # 0.77s, matches Anvari et al. (2015)
            T_brk = max(T_brk, T_brk_vhcl)
            
        a = 2 * np.dot(x_ctragent_future - x_self_future, self.dirn) / (T_curve - self.model.T_react)**2
        if a < -self.v_max / T_brk:
            #
            # Cannot stop in time but can decelerate as much as possible to reduce injury and improve chance of
            # cooperation sufficing to avoid incident.
            #
            
            # a = self.v_ped / T_brk
            a = -self.v_max / T_brk
            x_desired = self.posn + T_curve * self.v - 0.5 * a * T_curve**2
        
        v_desired = np.zeros(2)

        assert(T_curve == interaction.t_min)
        # ctrl_point_factor = T_curve / 3

        # print("Agent::init_best_possible_stop".format(self.id, self.posn, x_desired))
        # assert(np.dot(x_desired - self.posn, self.dirn) > 0)

        # assert(False)

        self.conflict_state.start_yielding_state(ctragent.id, t0, T_curve, [ctragent.posn, x_ctragent_future], x_desired)
        return

    def select_avoidance_policy(self, agent_tbl, zone_cfg, t0):
        interaction = None
        reaction = None

        if len(self.active_counteragent_idset) == 0:
            return

        #
        # Prioritise interactions requiring halting behaviour.
        # @todo: Do we need to consider "len(yielding_subtbl) != 0 and len(awaiting_subtbl) != 0" as a separate case?!
        #
        awaiting_subtbl    = self.interagent_conflicts[smar.ReactionType.AWAIT]
        yielding_subtbl    = self.interagent_conflicts[smar.ReactionType.YIELD]
        if len(yielding_subtbl) != 0 and len(awaiting_subtbl) != 0:
            yielding_items = list(yielding_subtbl.items())
            # yielding_items.sort(key=lambda yielding_item: yielding_item[1].s)
            yielding_items.sort(key=lambda yielding_item: yielding_item[1][1].s)

            yielding_item = yielding_items[0]
            interaction, reaction = yielding_item[1]

            id1, id2 = smic.interagent_codec.decode(interaction.combined_id)
            major_ctragent_id = self.id ^ id1 ^ id2

            s = reaction.s

            dst_node = self.get_current_dst_node()
            print("Agent::select_avoidance_policy agent {} at t0 {}".format(self.id, t0))
            d_goal = smnd.calc_norm(dst_node.approach_shape.posn - self.posn)

            if s < d_goal:
                self.conflict_state.start_yielding_state(major_ctragent_id, reaction.stop_shape, s)
            else:
                self.conflict_state.start_awaiting_state(major_ctragent_id, dst_node)

            return
        elif len(yielding_subtbl) != 0:
            yielding_items = list(yielding_subtbl.items())
            yielding_items.sort(key=lambda yielding_item: yielding_item[1][1].s)

            yielding_item = yielding_items[0]
            interaction, reaction = yielding_item[1]

            id1, id2 = smic.interagent_codec.decode(interaction.combined_id)
            major_ctragent_id = self.id ^ id1 ^ id2

            # self.conflict_state.start_yielding_state(major_ctragent_id, reaction.s)
            self.conflict_state.start_yielding_state(major_ctragent_id, reaction.stop_shape, reaction.s)

            return
        elif len(awaiting_subtbl) != 0:
            dst_node = self.get_current_dst_node()

            awaiting_items = list(awaiting_subtbl.items())

            awaiting_item = awaiting_items[0]
            interaction, reaction = awaiting_item[1]

            id1, id2 = smic.interagent_codec.decode(interaction.combined_id)
            major_ctragent_id = self.id ^ id1 ^ id2

            self.conflict_state.start_awaiting_state(major_ctragent_id, dst_node)

            return

        #
        # Now determine whether veering is necessary.
        #
        veering_subtbl    = self.interagent_conflicts[smar.ReactionType.VEER]
        if len(veering_subtbl) != 0:
            veerable_items = list(veering_subtbl.items())
            veerable_items.sort(key=lambda veerable_item: veerable_item[1][1].t_conflict)

            assert len(veerable_items) < 4
            if len(veerable_items) < 2:
                veerable_item = veerable_items[0]
                interaction, reaction = veerable_item[1]
                # t0 = interaction.t_conflict
                t0 = reaction.t_conflict
                # assert interaction.t_conflict == reaction.t_conflict

                id1, id2 = smic.interagent_codec.decode(interaction.combined_id)
                major_ctragent_id = self.id ^ id1 ^ id2
                ctragent = self.model.active_agents[major_ctragent_id]

                self.init_best_possible_simple_veer(ctragent, interaction, reaction, t0)
                if self.conflict_state.state_id in {smas.ConflictStateId.MODULATING, smas.ConflictStateId.VEERING}:
                    # assert(False)
                    return

                relposn = ctragent.posn - self.posn
                relv = ctragent.v - self.v

                s_min = self.radius + ctragent.rect_length + smc.d_safe_IA_maximum
                t_min = -np.dot(relposn, relv) / smnd.calc_norm_sqr(relv)
                stop_posn = ctragent.posn + t_min * ctragent.v - s_min * ctragent.dirn
                stop_shape = smgs.Circle(stop_posn, ctragent.rect_width)

                s_arc = smnd.calc_norm(stop_posn - self.posn)

                self.conflict_state.start_yielding_state(major_ctragent_id, stop_shape, s_arc)
            else: # len(veerable_items) == 2:
                veerable_items.sort(key=lambda veerable_item: veerable_item[1][1].t_conflict)

                primary_veerable_item = veerable_items[0]
                interaction, reaction = primary_veerable_item[1]
                # t0 = interaction.t_conflict
                t0 = reaction.t_conflict

                veer_required = self.init_best_possible_complex_veer(veerable_items, t0)
                if not veer_required:
                    return

                # if self.conflict_state.state_id in {smas.ConflictStateId.MODULATING, smas.ConflictStateId.VEERING}:
                if self.conflict_state.state_id == smas.ConflictStateId.VEERING:
                    # assert(False)
                    return

                id1, id2 = smic.interagent_codec.decode(interaction.combined_id)
                major_ctragent_id = self.id ^ id1 ^ id2
                ctragent = self.model.active_agents[major_ctragent_id]

                relposn = ctragent.posn - self.posn
                relv = ctragent.v - self.v

                s_min = self.radius + ctragent.rect_length + smc.d_safe_IA_maximum
                t_min = -np.dot(relposn, relv) / smnd.calc_norm_sqr(relv)
                stop_posn = ctragent.posn + t_min * ctragent.v - s_min * ctragent.dirn
                stop_shape = smgs.Circle(stop_posn, ctragent.rect_width)

                s_arc = smnd.calc_norm(stop_posn - self.posn)

                self.conflict_state.start_yielding_state(major_ctragent_id, stop_shape, s_arc)


            #
            # @todo:
            #
            return


    def clear_avoidance_policy(self):
        self.conflict_state.clear_state()
        
    #
    # @todo: Consider integrating the reconsideration of an avoidance policy if
    # it is not working, e.g.:
    #   because the agent has been prevented by crowd density from fulfilling the policy; or
    #   because the major counteragent has confounded expectations by accelerating, decelerating, or swerving.
    #
    def apply_avoidance_policy(self, t):
        #
        # VEERING (formerly SWERVING) is currently VO-based (FS93/BLM08), DETOURING is trajectory-based (Pascucci+ 2015),
        # YIELDING (formerly STOPPING) is analogous to WAITING (Johansson+ 2015).
        #
        spd_pref = self.get_target_speed()
        v_pref = None
        if self.conflict_state.state_id == smas.ConflictStateId.AWAITING:
            cur_dst = self.get_current_dst_node()
            stop_shape = cur_dst.approach_shape
            npi = None
            if not stop_shape.contains(self.posn) and not stop_shape.contains(self.posn + self.get_length_axis() * self.dirn):
                npi = stop_shape.get_nearest_point_info(self.posn)
                x_stop, d_stop, nv_stop, tv_stop = npi

                rel_posn_stop = x_stop - self.posn
                d_sqr = smnd.calc_norm_sqr(rel_posn_stop)
                r = 4 * self.T_relax * self.get_target_speed()
                if d_sqr <= r ** 2:
                    v_pref = spd_pref * rel_posn_stop / r
                else:
                    v_pref = spd_pref * smnd.calc_unit_vector(rel_posn_stop)
            else:
                v_pref = np.zeros(2)
                # npi = (self.posn, 0, -self.dirn, smnd.rotate_90ACW(self.dirn))
                # cur_src = self.get_current_src_node()
                # start_shape = cur_src.approach_shape
                # canonical_dirn = smnd.calc_unit_vector(stop_shape.posn - start_shape.posn)
                # npi = (self.posn, 0, -canonical_dirn, smnd.rotate_90ACW(-canonical_dirn))

        elif self.conflict_state.state_id == smas.ConflictStateId.YIELDING:

            npi = self.conflict_state.stop_shape.get_nearest_point_info(self.posn)
            assert (npi != None)
            x_stop, d_stop, nv_stop, tv_stop = npi
            if smc.STABILISED_YIELDING:
                if np.dot(self.dirn, x_stop - self.posn) < 0:
                    x_stop = self.posn
                    nv_stop = -self.dirn

            target_shape = None
            if self.goal_state.state_id == smas.GoalStateId.NAVIGATING and self.goal_state.turning_info != None:
                turning_info = self.goal_state.turning_info

                if len(turning_info.stages) > 0:
                    cur_stage = turning_info.stages[0]
                    goal_shape = cur_stage.shape
                else:
                    dst        = self.get_current_dst_node()
                    goal_shape = dst.approach_shape

                x_goal, d_goal, nv_goal, tv_goal = goal_shape.get_nearest_point_info(self.posn)
                if d_goal < d_stop:
                    target_shape = goal_shape

            spd_pref = self.get_target_speed()
            r        = 4 * self.T_relax * self.get_target_speed()
            if target_shape != None:
                s = self.conflict_state.s
                if s <= r:
                    rel_posn_goal = x_goal - self.posn
                    stop_dirn     = smnd.calc_unit_vector(rel_posn_goal)
                    v_pref        = (s / r) * spd_pref * stop_dirn
                else:
                    v_pref = spd_pref * (-nv_stop)
            else:
                rel_posn_stop   = x_stop - self.posn
                if d_stop <= r:
                    v_pref = spd_pref * rel_posn_stop / r
                else:
                    v_pref = spd_pref * (-nv_stop)

            FIX_2025_12_10 = True
            if np.dot(v_pref, self.dirn) < 0.:
                v_pref = np.zeros(2)

        elif self.conflict_state.state_id == smas.ConflictStateId.VEERING:
            x_curve_pts = self.conflict_state.x_curve_pts
            t0          = self.conflict_state.t0
            T_curve     = self.conflict_state.T_curve
            
            s = (t - t0) / T_curve

            x_next = smnd.evaluate_cubic_spline(x_curve_pts, s)
            # if np.dot(self.dirn, x_next - self.posn) < np.cos(self.max_angular_momentum * T_step):
                # self.agent_state = CounteragentRelatedState.STOPPING
                            
            v_pref = self.get_target_speed() * smnd.calc_unit_vector(x_next - self.posn)
        else:
            cur_dst = self.get_current_dst_node()
            goal_posn = cur_dst.approach_shape.posn
            v_pref = self.get_target_speed() * smnd.calc_unit_vector(goal_posn - self.posn)

        I_nav     = (v_pref - self.v) / self.T_relax
        I_soc = np.zeros(2)
        if self.conflict_state.state_id not in {smas.ConflictStateId.AWAITING, smas.ConflictStateId.YIELDING}:
            repulsion_tbl = self.interagent_conflicts[smar.ReactionType.REPEL]
            for ctragent_id in repulsion_tbl:
                interaction_reaction_pair = repulsion_tbl[ctragent_id]
                interaction, reaction     = interaction_reaction_pair

                agent_tbl = self.model.active_agents
                if interaction.t_conflict < smc.T_SRCA_horizon and ctragent_id in agent_tbl:
                    ctragent = agent_tbl[ctragent_id]
                    I_soc += self.calc_interagent_influence(ctragent)

        return I_nav, I_soc

    #
    # @todo: Source the minor influences differently,
    # 	i.e. interaction, reaction in self.interagent_conflicts[smar.ReactionType.REPEL]
    # @todo: Include zone repulsion?!
    #
    def accumulate_minor_influences(self, agent_tbl, zone_cfg, horizon):
        I_soc = np.zeros(2)
        repulsion_subtbl = self.interagent_conflicts[smar.ReactionType.REPEL]
        if len(repulsion_subtbl) != 0:
            for ctragent_id in repulsion_subtbl:
                interaction, reaction = repulsion_subtbl[ctragent_id]
                if interaction.t_conflict != None and interaction.t_conflict < horizon and ctragent_id in agent_tbl:
                    ctragent = agent_tbl[ctragent_id]

                    #
                    # @todo: If the agent has an interesting goal-state (CONVERGING, TURNING)
                    # or conflict-state (STOPPING), x_alpha_future may be
                    # overestimating the agent's speed. If the agent is in one of
                    # those states, then clipping to the conflict_state.x_des may
                    # be worthwhile. The SWERVING conflict-state may be even worse,
                    # in that x_alpha_future may need to be calculated along the
                    # spline.
                    #
                    x_alpha_future = self.posn + interaction.t_conflict * self.v
                    mdl = self.model
                    if x_alpha_future[0] < 0 or x_alpha_future[0] >= mdl.w:
                        continue
                    if x_alpha_future[1] < 0 or x_alpha_future[1] >= mdl.h:
                        continue
                    i, j = mdl.grid.get_cell_coordinates(x_alpha_future)
                    # print("Agent::accumulate_minor_influences - future agent posn {} is in cell[{}, {}]".format(x_alpha_future, i, j))
                    cell = mdl.grid.cells[i][j]
                    ctragent_must_wait = False
                    for zone_id in cell.zone_coverage_idset:
                        zone = zone_cfg.zone_tbl[zone_id]
                        if zone.shape.contains(x_alpha_future):
                            if zone.excludes(ctragent):
                                ctragent_must_wait = True
                                break
                    if ctragent_must_wait:
                        continue
                    
                    I_soc += self.calc_interagent_influence(ctragent)

        return I_soc


    def DEPRECATED_find_soonest_major_conflict(self, agent_tbl, zone_cfg, t):
        assert(False)

        possible_conflicts = list(self.interagent_interactions.items())
        possible_conflicts.sort(key = lambda item: item[1][0].t_conflict)

        #
        # Find soonest major counteragent interaction or determine
        # there isn't one.
        #
        mci_idset   = set()
        conflict_t1 = 2 * smc.T_master_horizon
        for i in range(len(possible_conflicts)):
            ctragent_id, interaction_reaction_pair = possible_conflicts[i]
            interaction, reaction = interaction_reaction_pair

            #
            # @todo: Verify!
            #
            if ctragent_id not in agent_tbl or ctragent_id not in self.interagent_interactions:
                continue

            # assert(False)
            
            if self.id < ctragent_id:
                assert(self.agent_type == interaction.agent.agent_type)
                ctragent_type = interaction.ctragent.agent_type
            else:
                assert(self.agent_type == interaction.ctragent.agent_type)
                ctragent_type = interaction.agent.agent_type

            major     = ctragent_type == smad.AgentType.CAR or self.agent_type != ctragent_type
            proactive = interaction.t_conflict > smc.T_SRCA_horizon or ctragent_id in self.interagent_interactions

            near, converging, impacted = interaction.current_importance
            impacting                  = interaction.impactful and not impacted
            if major: # and proactive: # and impacting:
                mci_idset.add(ctragent_id)
                conflict_t1 = interaction.t_conflict
                break

        return conflict_t1, mci_idset

        
    def DEPRECATED_update_influences_while_modulating(self, agent_tbl, zone_cfg, t):
        if len(self.interagent_interactions) != 0:
            assert(False)
            
            print("Agent::resolve_total_influence - agent {} will handle {} IAIs and {} CZIs".format(self.id, len(self.interagent_interactions), len(self.contrazone_interactions)))

            conflict_t1, major_ctragent_idset = self.find_soonest_major_conflict(agent_tbl, zone_cfg, t)

            if len(major_ctragent_idset) != 0:
                assert(len(major_ctragent_idset) <= 1)
                
                self.conflict_t1          = conflict_t1
                # self.major_ctragent_idset = major_ctragent_idset
                
                print("Agent::resolve_total_influence - agent {} is now conflict-dominated vs ctragent {}".format(self.id, self.major_ctragent_idset))

                # self.select_avoidance_policy(agent_tbl, zone_cfg, self.major_ctragent_idset, t)
                self.select_avoidance_policy(agent_tbl, zone_cfg, t)

        return


    def DEPRECATED_update_influences_while_veering(self, agent_tbl, zone_cfg, t):
        assert(len(set(self.interagent_interactions) - set(agent_tbl)) == 0)

        self.major_ctragent_idset &= set(self.interagent_interactions)
        if len(self.major_ctragent_idset) > 0:
            #
            # Find a trumpworthy (in the tricktaking rather than authoritarian sense) major conflict.
            #
            # NB: It is possible that the alternate counteragents will be the original counteragents.
            # The code is *deliberately* not guarding against this case; if the conflict time is earlier,
            # recalculation is required anyway.
            #
            alt_conflict_t1, alt_ctragent_idset = self.find_soonest_major_conflict(agent_tbl, zone_cfg, t)
            if len(alt_ctragent_idset) > 0 and alt_conflict_t1 < self.conflict_state.t1:
                # assert(False)
                self.clear_avoidance_policy()
                self.update_influences_while_modulating(agent_tbl, zone_cfg, t)
        else:
            self.clear_avoidance_policy()
            self.update_influences_while_modulating(agent_tbl, zone_cfg, t)

    def DEPRECATED_update_influences_while_yielding(self, agent_tbl, zone_cfg, t):
        assert(len(set(self.interagent_interactions) - set(agent_tbl)) == 0)

        self.major_ctragent_idset &= set(self.interagent_interactions)
        if len(self.major_ctragent_idset) > 0:
            #
            # Find a trumpworthy (in the tricktaking rather than authoritarian sense) major conflict.
            #
            # NB: It is possible that the alternate counteragents will be the original counteragents.
            # The code is *deliberately* not guarding against this case; if the conflict time is earlier,
            # recalculation is required anyway.
            #
            alt_conflict_t1, alt_ctragent_idset = self.find_soonest_major_conflict(agent_tbl, zone_cfg, t)
            if len(alt_ctragent_idset) > 0 and alt_conflict_t1 < self.conflict_state.t1:
                # assert(False)
                self.clear_avoidance_policy()
                self.update_influences_while_modulating(agent_tbl, zone_cfg, t)
        else:
            self.clear_avoidance_policy()
            self.update_influences_while_modulating(agent_tbl, zone_cfg, t)


    def DEPRECATED_update_influences(self, agent_tbl, zone_cfg, t):
        assert(False)

        #
        # @todo: If this assert triggers, lose the invalid interagent interactions.
        #
        assert(len(set(self.interagent_interactions) - set(agent_tbl)) == 0)
        idset = set(agent_tbl) & set(self.interagent_interactions)
        self.interagent_interactions = {id: self.interagent_interactions[id] for id in idset}
    
        if self.conflict_state.state_id == smas.ConflictStateId.MODULATING:
            self.update_influences_while_modulating(agent_tbl, zone_cfg, t)
        elif self.conflict_state.state_id == smas.ConflictStateId.VEERING:
            self.update_influences_while_veering(agent_tbl, zone_cfg, t)
        else:
            assert(self.conflict_state.state_id == smas.ConflictStateId.YIELDING)
            self.update_influences_while_yielding(agent_tbl, zone_cfg, t)

    #
    # @todo: Reintroduce agent-state-dependent reception of influence.
    #
    def update_influences(self, agent_tbl, zone_cfg, t):
        self.clear_avoidance_policy()
        self.select_avoidance_policy(agent_tbl, zone_cfg, t)

        return

    def clip_influence(self, I):
        return I

    def resolve_total_influence(self, agent_tbl, zone_cfg, t):
        if self.conflict_state.state_id != smas.ConflictStateId.MODULATING:
            print("Agent::resolve_total_influence - agent {} is conflict-dominated ({})".format(self.id, self.conflict_state.state_id))

            I_nav, I_soc = self.apply_avoidance_policy(t)
        elif self.goal_state.state_id in {smas.GoalStateId.CONVERGING, smas.GoalStateId.WAITING}:
            print("Agent::resolve_total_influence - agent {} is goal-dominated ({})".format(self.id, self.goal_state.state_id))
            assert(self.goal_state.state_id in {smas.GoalStateId.CONVERGING, smas.GoalStateId.WAITING})
            x_pref, v_pref = self.apply_waiting_policy()
            I_nav = (v_pref - self.v) / self.T_relax

            #
            # @todo: Accumulate SFM-like influence for the appropriate horizon.
            #
            horizon = smc.T_master_horizon
            if self.agent_type == smad.AgentType.PEDESTRIAN:
                # horizon = min(horizon, 1)
                # horizon = min(horizon, smc.T_SRCA_horizon)
                horizon = smc.T_SRCA_horizon

            I_soc = self.accumulate_minor_influences(agent_tbl, zone_cfg, horizon)
        else:
            assert(self.goal_state.state_id in {smas.GoalStateId.NAVIGATING, smas.GoalStateId.EXITING})
            I_nav = self.apply_navigating_policy()
            horizon = smc.T_master_horizon
            # if self.agent_type == smad.AgentType.PEDESTRIAN:
                # horizon = min(horizon, 1)
                # horizon = min(horizon, smc.T_SRCA_horizon)
            I_soc = self.accumulate_minor_influences(agent_tbl, zone_cfg, horizon)

        self.update_influences(agent_tbl, zone_cfg, t)

        I_total = self.clip_influence(I_nav + I_soc)
        return I_total

    def DEPRECATED_clip_intrinsic_force(self, I):
        #
        # @todo: Clip the total force to something physically plausible for the agent to generate.
        # This 
        #
        # I_norm, I_dirn = smnd.calc_polar_decomp(self.I_total)
        # 
        # self.F_total   = smnd.clip_magnitude(I_norm, self.F_max) * I_dirn

        return I * self.mass

    def DEPRECATED_calc_total_impact_force(self):
        assert(False)
        return self.F_collision

    def calc_total_force(self, agent_tbl, zone_cfg, t):
        #
        # Determine what the agent can and will do.
        #
        smd.dbgprint("Agent:calc_total_force ready to execute self.calc_total_influence()")
        self.I_total = self.resolve_total_influence(agent_tbl, zone_cfg, t)

        # F_intrinsic = self.clip_intrinsic_force(self.I_total)
        # F_intrinsic = self.mass * self.clip_intrinsic_force(self.I_total)
        F_intrinsic = self.mass * self.I_total

        #
        # @todo: Verify that impact will trump influence!
        # assert(F_impact approx== 0 or F_impact >> F_intrinsic)
        # Consider using magnitude of F_intrinsic to inhibit self.I_total
        #
        print("Agent::calc_total_force - agent {} has intrinsic {} and  impact {} forces".format(self.id, F_intrinsic, self.F_collision))
        # assert(smnd.calc_norm_sqr(self.F_collision) < 2 * smnd.EPS_m_sqr or smnd.calc_norm_sqr(self.F_collision) > smnd.calc_norm_sqr(F_intrinsic))
        self.F_total = F_intrinsic # + self.F_collision
        self.a = self.F_total / self.mass

        self.F_collision = np.zeros(2)
        
        return
    
    #
    # Note: was originally called Agent::get_kinematic_desc_list(), but that
    # name has been deprecated with extreme prejudice.
    #
    def get_desc_list(self):
        cur_src_idx = self.route[self.routing_idx]
        if cur_src_idx == self.route[-1]:
            cur_dst_idx = -1
        else:
            cur_dst_idx = self.route[self.routing_idx + 1]

        return [self.id, self.agent_type, cur_src_idx, self.conflict_state.state_id, self.goal_state.state_id, cur_dst_idx, self.posn[0], self.posn[1], self.v[0], self.v[1], self.F_total[0], self.F_total[1]]

    #
    # Calculate a bounding box assuming the agent will not turn.
    #
    def calc_simple_motion_bbox(self, horizon):
        #
        # @todo: Consider adding support for a cubic spline, using the convex hull property, i.e.:
        #     calculate the bounding box of the four control points.
        #
        posn1 = self.posn
        posn2 = self.posn + horizon * self.v

        return smgg.BoundingBox(posn1, posn2)

    def calc_motion_bboxen(self, horizon):
        cur_dst_id = self.route[self.routing_idx + 1]
        cur_dst    = self.nodegraph.nodes[cur_dst_id]

        if self.goal_state.state_id in {smas.GoalStateId.CONVERGING, smas.GoalStateId.WAITING}:
            if isinstance(cur_dst, smtAct.ActivityNode):
                active_focal_shape = cur_dst.get_active_focal_shape(self)
                x_omega, d, nv, tv = active_focal_shape.get_nearest_point_info(self.posn)
            else:
                dst_shape          = cur_dst.approach_shape
                x_omega, d, nv, tv = dst_shape.get_nearest_point_info(self.posn)
            mbox               = smgg.BoundingBox(self.posn, x_omega)
        else:
            mbox               = self.calc_simple_motion_bbox(horizon)

        return [mbox]

    def clip_velocity(self):
        return

    def is_reversible(self):
        return True

    def advance_navigation(self):
        if self.routing_idx + 2 == len(self.route): # cur_dst_idx = agent.route[-1]:
            assert(False)
            #
            # @todo: Deactivate.
            #
            inactive_agents.add(self.id)
            assert(self.route[self.routing_idx + 1] == self.true_dst_id)
            assert(self.nodegraph.nodes[self.route[self.routing_idx + 1]] == self.true_dst_node)
        # elif self.conflict_state.state_id == smas.ConflictStateId.AWAITING:
            # return
        else:
            assert(self.routing_idx + 2 < len(self.route))
            self.routing_idx += 1

            if self.routing_idx + 2 < len(self.route) and self.agent_type == smad.AgentType.CAR:
                # cur_src_id = self.route[self.routing_idx]
                # cur_src     = self.nodegraph.nodes[cur_src_id]
                cur_src     = self.get_current_src_node()

                # cur_dst_id = self.route[self.routing_idx + 1]
                # cur_dst     = self.nodegraph.nodes[cur_dst_id]
                cur_dst     = self.get_current_dst_node()

                # next_dst_id = self.route[self.routing_idx + 2]
                # next_dst     = self.nodegraph.nodes[next_dst_id]
                next_dst     = self.get_next_dst_node()

                dx                   = cur_dst.approach_shape.posn - cur_src.approach_shape.posn
                L_chord, goal_dirn   = smnd.calc_polar_decomp(dx)
                cos_theta   = np.dot(self.dirn, goal_dirn)
                if cos_theta >= smad.cos_theta_max[smad.AgentType.CAR]:
                    self.goal_state.start_navigating_state(self.route[self.routing_idx + 1])
                    return

                target_dirn    = smnd.calc_unit_vector(next_dst.approach_shape.posn - cur_dst.approach_shape.posn)
                cos_phi        = np.dot(self.dirn, target_dirn)
                if cos_phi >= cos_theta:
                    # assert(False)
                    self.goal_state.start_navigating_state(self.route[self.routing_idx + 1])
                    return

                num1    = smnd.calc_det2x2(dx, target_dirn)
                num2    = smnd.calc_det2x2(self.dirn, dx)
                dnm     = smnd.calc_det2x2(self.dirn, target_dirn)
                rcp_dnm = 1 / dnm
                Lq1     = num1 * rcp_dnm
                Lq2     = num2 * rcp_dnm
                if Lq1 <= 0. or Lq2 <= 0.:
                    self.goal_state.start_navigating_state(self.route[self.routing_idx + 1])
                    return

                r_circular   = min(Lq1, Lq2) * np.sqrt((1 + cos_phi)/(1 - cos_phi))
                spd_circular = np.sqrt(smc.a_max_cfgl * r_circular)
                print("Agent.advance_navigation: {}'s circular approximation would have radius {} hence speed {}".format(self.id, r_circular, spd_circular))
                
                # Lc1 = Lc2 = (1/3) * L_chord
                # Lc1 = Lc2 = 0.29 * L_chord
                Lc1 = 0.44 * L_chord
                Lc2 = 0.33 * L_chord
                # Lc1 = 0.63 * (Lq1 / (Lq1 + Lq2)) * L_chord
                # Lc2 = 0.58 * (Lq2 / (Lq1 + Lq2)) * L_chord

                u0 = Lc1 * self.dirn
                u2 = Lc2 * target_dirn

                icp1 = cur_src.approach_shape.posn + u0
                icp2 = cur_dst.approach_shape.posn - u2

                spline = np.array([cur_src.approach_shape.posn, icp1, icp2, cur_dst.approach_shape.posn])
                # diff_spline = 3 * [u0, dx - (u0 + u2), u2]
                diff_spline = 3 * (spline[1:4] - spline[0:3])
                # dbldiff_spline = 6 * [dx - 2 * u0 - u2, 2 * u2 + u0 - dx] 
                dbldiff_spline = 2 * (diff_spline[1:3] - diff_spline[0:2]) 
                print("Agent.advance_navigation: car {} has spline {}".format(self.id, spline))
                print("Agent.advance_navigation: car {} has first derivative spline {}".format(self.id, diff_spline))
                print("Agent.advance_navigation: car {} has second derivative spline {}".format(self.id, dbldiff_spline))

                stages   = []
                n_stages = 8
                for i in range(1, n_stages):
                    param_ftr = i / n_stages
                    x          = smnd.evaluate_cubic_spline(spline, param_ftr)
                    x_prime    = smnd.evaluate_quadratic_spline(spline, param_ftr)
                    x_dblprime = smnd.evaluate_line(dbldiff_spline, param_ftr)

                    print("Agent.advance_navigation: car {} has {}st/th stage at {} with parametric derivatives {} and {}".format(self.id, i, x, x_prime, x_dblprime))

                    nmrtr   = smnd.calc_det2x2(x_prime, x_dblprime)
                    dnm_sqr = (smnd.calc_norm_sqr(x_prime))**3
                    dnm     = np.sqrt(dnm_sqr)

                    print("Agent.advance_navigation: car {} has cross-product of derivatives {} and cube of parametric derivative {}".format(self.id, nmrtr, dnm))

                    kappa       = np.abs(nmrtr / dnm)
                    r_curvature = 1 / kappa
                    # spd_des     = min(self.v_max, np.sqrt(smc.a_max_cfgl * r_curvature))
                    spd_des     = min(self.v_max, np.sqrt(smc.a_max_cfgl * r_curvature), smc.omega_max_CAR * r_curvature)

                    print("Agent.advance_navigation: car {} has absolute curvature {} => curvature radius {}".format(self.id, kappa, r_curvature))
                    print("Agent.advance_navigation: car {} has three speed limits - intrinsic {}, centrifugal {}, and geometric {}".format(self.id, self.v_max, np.sqrt(smc.a_max_cfgl * r_curvature), smc.omega_max_CAR * r_curvature))

                    stage = smas.Stage(smgs.Circle(x, 0.5 * self.get_width_axis()), spd_des)
                    stages.append(stage)

                turning_info = smas.TurningInfo(stages, goal_dirn, target_dirn, Lq1, Lq2, L_chord)

                self.goal_state.start_navigating_state(self.route[self.routing_idx + 1], turning_info)
                # for stage in self.goal_state.stages:
                    # print("Agent.advance_navigation: car {} has stage at {} with radius {} and speed limit {}".format(self.id, stage.shape.posn, stage.shape.radius, stage.spd_limit))

                # self.goal_state.stages = self.goal_state.stages[1:]

                # assert(False)
                # assert(smnd.calc_det2x2(self.dirn, target_dirn) < 0)
                # assert(smnd.calc_det2x2(self.dirn, target_dirn) > 0)
            else:
                cur_dst_id = self.route[self.routing_idx + 1]
                # cur_dst     = self.nodegraph.nodes[cur_dst_id]
                self.goal_state.start_navigating_state(cur_dst_id)

    def clear_conflict_tables(self):
        for rt in smar.ReactionType:
            self.interagent_conflicts[rt] = {}
        if self.agent_type == smad.AgentType.CAR and self.leader_id != None:
            if self.leader_id in self.model.active_agents:
                leader = self.model.active_agents[self.leader_id]

                leader_cd = leader.get_current_dst_node()
                own_cd = self.get_current_dst_node()
                own_nd = self.get_next_dst_node()
                # if own_cd == leader_cd or own_nd == leader_cd:
                    # return
            self.leader_id = None
        self.leader_s  = None
        return

    def update_kinematics(self, t, delta_t, inactive_agents):
        smd.dbgprint("Agent:update_kinematics for agent {} starts with posn {}, velocity {}, acceleration {}, and will use delta_t {}".format(self.id, self.posn, self.v, self.F_total, delta_t))

        print("Agent:update_kinematics for agent {} starts with posn {}, velocity {}, acceleration {}, and will use delta_t {}".format(self.id, self.posn, self.v, self.F_total, delta_t))
        assert(np.dot(self.posn, self.posn) >= smnd.EPS_m_sqr)

        assert(smnd.calc_norm_sqr(self.a * self.mass - self.F_total) < 2 * smnd.EPS_m_sqr)
        new_posn, new_v = smna.advance_SIE(delta_t, self.posn, self.v, self.a)

        if self.agent_type == smad.AgentType.CAR:
            if np.dot(new_v, self.dirn) < 0.:
                new_v  = np.zeros(2)
                new_posn = self.posn

        self.posn = new_posn
        self.v    = new_v

        if self.agent_type == smad.AgentType.PEDESTRIAN:
            self.dirn  = smnd.calc_unit_vector(self.v)

        self.clip_velocity()

        cur_dst_id = self.route[self.routing_idx + 1]
        cur_dst     = self.nodegraph.nodes[cur_dst_id]

        cur_src_id = self.route[self.routing_idx]
        cur_src    = self.nodegraph.nodes[cur_src_id]

        intergoal_dirn = smnd.calc_unit_vector(cur_dst.approach_shape.posn - cur_src.approach_shape.posn)
                    
        if self.goal_state.state_id == smas.GoalStateId.NAVIGATING:
            if self.goal_state.turning_info != None:
                turning_info = self.goal_state.turning_info
                assert (self.agent_type == smad.AgentType.CAR)
                if len(turning_info.stages) > 0:
                    cur_stage = turning_info.stages[0]
                    target_shape = cur_stage.shape

                    x_omega, d, nv, tv = target_shape.get_nearest_point_info(self.posn)
                    delta_finish = x_omega - self.posn
                    # delta_edge   = cur_dst.approach_shape.posn - cur_src.approach_shape.posn
                    d_sqr_dst = np.dot(delta_finish, delta_finish)

                    on = d_sqr_dst < target_shape.radius ** 2
                    inside = smnd.calc_norm_sqr(target_shape.posn - self.posn) < smnd.calc_norm_sqr(
                        target_shape.posn - x_omega)
                    close = on or inside
                    # reached      = self.agent_type != smad.AgentType.PEDESTRIAN and np.dot(self.dirn, delta_finish) <= self.get_length_axis() + target_shape.radius
                    # reached      = self.agent_type != smad.AgentType.PEDESTRIAN and np.dot(self.dirn, delta_finish) <= 0 # self.rect_length + target_shape.radius
                    # reached      = self.agent_type != smad.AgentType.PEDESTRIAN and np.dot(-nv, delta_finish) <= self.rect_length + target_shape.radius
                    # reached      = self.agent_type != smad.AgentType.PEDESTRIAN and np.dot(cur_dst.approach_shape.posn - self.posn, delta_finish) <= 0 # self.rect_length + target_shape.radius
                    # reached      = self.agent_type != smad.AgentType.PEDESTRIAN and np.dot(intergoal_dirn, delta_finish) <= 0 # self.rect_length + target_shape.radius

                    # reached0 = self.agent_type != smad.AgentType.PEDESTRIAN
                    reached1 = np.dot(self.dirn, delta_finish) <= 0
                    reached2 = np.dot(intergoal_dirn, delta_finish) <= 0
                    reached = reached1 or reached2
                    if close or reached:
                        turning_info.stages = turning_info.stages[1:]
                else:
                    x_omega, d, nv, tv = cur_dst.approach_shape.get_nearest_point_info(self.posn)
                    delta_finish = x_omega - self.posn
                    # delta_edge   = cur_dst.approach_shape.posn - cur_src.approach_shape.posn
                    d_sqr_dst = np.dot(delta_finish, delta_finish)

                    on = d_sqr_dst < cur_dst.approach_shape.radius ** 2
                    inside = smnd.calc_norm_sqr(cur_dst.approach_shape.posn - self.posn) < smnd.calc_norm_sqr(
                        cur_dst.approach_shape.posn - x_omega)
                    close = on or inside
                    # reached      = self.agent_type != smad.AgentType.PEDESTRIAN and np.dot(self.dirn, delta_finish) <= self.get_length_axis() + cur_dst.approach_shape.radius
                    reached = self.agent_type != smad.AgentType.PEDESTRIAN and np.dot(self.dirn,
                                                                                      delta_finish) <= self.rect_length + cur_dst.approach_shape.radius

                    if cur_dst.requires_waiting(self):
                        if close:
                            self.goal_state.start_converging_state(cur_dst_id)
                    elif (close or reached) and not self.must_wait(cur_dst):
                        # assert (not self.must_wait(cur_dst))
                        self.advance_navigation()
                    else:
                        pass
            else:
                x_omega, d, nv, tv      = cur_dst.approach_shape.get_nearest_point_info(self.posn)
                delta_finish = x_omega - self.posn
                delta_edge   = cur_dst.approach_shape.posn - cur_src.approach_shape.posn
                assert(np.dot(self.dirn, delta_edge) >= 0. or self.is_reversible())
                d_sqr_dst    = np.dot(delta_finish, delta_finish)

                on           = d_sqr_dst < cur_dst.approach_shape.radius**2
                inside       = smnd.calc_norm_sqr(cur_dst.approach_shape.posn - self.posn) < smnd.calc_norm_sqr(cur_dst.approach_shape.posn - x_omega)
                close        = on or inside
                # reached      = self.agent_type != smad.AgentType.PEDESTRIAN and np.dot(self.dirn, delta_finish) <= self.get_length_axis() + cur_dst.approach_shape.radius
                reached      = self.agent_type != smad.AgentType.PEDESTRIAN and np.dot(self.dirn, delta_finish) <= 0.5 * self.rect_length + cur_dst.approach_shape.radius
                if self.must_wait(cur_dst):
                # if cur_dst.requires_waiting(self) or self.conflict_state.state_id == smas.ConflictStateId.AWAITING:
                    if close:
                        self.goal_state.start_converging_state(cur_dst_id)
                elif close or reached:
                    assert(not self.must_wait(cur_dst))
                    assert(self.conflict_state.state_id != smas.ConflictStateId.AWAITING)
                    assert(len(self.interagent_conflicts[smar.ReactionType.AWAIT]) == 0)
                    if self.routing_idx + 2 == len(self.route): # cur_dst_idx = agent.route[-1]:
                        #
                        # @todo: Deactivate.
                        #
                        inactive_agents.add(self.id)
                        assert(cur_dst_id == self.true_dst_id)
                        assert(cur_dst == self.true_dst_node)
                    else:
                        self.advance_navigation()
        elif self.goal_state.state_id == smas.GoalStateId.CONVERGING:
            if not self.must_wait(cur_dst):
                self.advance_navigation()
            elif cur_dst.requires_waiting(self):
                active_focal_shape = cur_dst.get_active_focal_shape(self)
                if active_focal_shape == None:
                    active_focal_shape = cur_dst.approach_shape
                # assert(cur_dst.focal_shape_tbl != None)
                # assert(cur_dst.focal_shape_tbl != [])

                # agent_activity_name = self.activity_schedule[0]
                # assert(cur_dst.focal_shape_tbl[0] == agent_activity_name)
                # active_focal_shape, _ = cur_dst.focal_shape_tbl[agent_activity_name]
                
                x_omega, d, nv, tv      = active_focal_shape.get_nearest_point_info(self.posn)
                delta_finish = x_omega - self.posn
                # delta_edge   = cur_dst.shape.posn - cur_src.shape.posn
                d_sqr_dst    = np.dot(delta_finish, delta_finish)

                close        = d_sqr_dst < active_focal_shape.radius**2
                if close:
                    #
                    # @todo: t = Now, T_wait = cur_dst.get_waiting_duration()
                    #     \infty for ApproachNode # cur_dst.requires_waiting(self) will release the agent from its grasp.
                    #     a non-negative real number for ActivityNode # e.g. the result of using scipy.stats to generate
                    #         a random value from an activity-appropriate distribution.
                    #
                    
                    # t     = Input parameter
                    # agent_activity_name = self.activity_schedule[0]
                    T_wait, _ = cur_dst.get_waiting_details(self) 
                    self.goal_state.start_waiting_state(cur_dst_id, t, T_wait)
                    # self.goal_state.state_id = smas.GoalState.WAITING
        elif self.goal_state.state_id == smas.GoalStateId.WAITING:
            #
            # @todo: Check the countdown and put the agent back into circulation iff t >= t1
            #
            if not self.must_wait(cur_dst):
                self.routing_idx += 1
                
                cur_dst_id = self.route[self.routing_idx + 1]
                # cur_dst     = self.nodegraph.nodes[cur_dst_id]

                # self.goal_state.start_approaching_state(cur_dst_id)
                self.advance_navigation()

                if isinstance(cur_dst, smtAct.ActivityNode):
                    agent_activity_name = self.activity_schedule[0]
                    self.complete_activity(agent_activity_name)
                    assert(False)

            elif t > self.goal_state.waiting_t1:
                if cur_dst.requires_exiting():
                    self.goal_state.start_exiting_state()
                    assert(self.goal_state.node_id != None)
                else:
                    self.advance_navigation()
        else:
            assert(self.goal_state.state_id == smas.GoalStateId.EXITING)
            assert(self.agent_type != smad.AgentType.CAR)
            
            x_omega, d, nv, tv      = cur_dst.approach_shape.get_nearest_point_info(self.posn)
            delta_finish = x_omega - self.posn
            # delta_edge   = cur_dst.approach_shape.posn - cur_src.approach_shape.posn
            d_sqr_dst    = np.dot(delta_finish, delta_finish)

            on           = d_sqr_dst < cur_dst.approach_shape.radius**2
            inside       = smnd.calc_norm_sqr(cur_dst.approach_shape.posn - self.posn) < smnd.calc_norm_sqr(cur_dst.approach_shape.posn - x_omega)
            close        = on or inside
            # reached      = self.agent_type != smad.AgentType.PEDESTRIAN and np.dot(self.dirn, delta_finish) <= self.get_length_axis() + cur_dst.approach_shape.radius
            reached      = self.agent_type != smad.AgentType.PEDESTRIAN and np.dot(self.dirn, delta_finish) <= self.rect_length + cur_dst.approach_shape.radius
            if close or reached:
                self.advance_navigation()

        if self.conflict_state.state_id == smas.ConflictStateId.YIELDING:
            # assert(False)
            print("Agent::update_kinematics - agent {} at {} is moving at {}m/s and should be stopping at {} ".format(self.id, self.posn, self.v, self.conflict_state.x_des))
            
        assert(self.goal_state.state_id in {smas.GoalStateId.NAVIGATING, smas.GoalStateId.CONVERGING, smas.GoalStateId.WAITING, smas.GoalStateId.EXITING})

        self.clear_conflict_tables()
        return

    def estimate_arclength_to_curdst(self):
        dst                 = self.nodegraph.nodes[self.goal_state.node_id]
        dx                  = dst.approach_shape.posn - self.posn
        L_chord, chord_dirn = smnd.calc_polar_decomp(dx)
        L_arc = L_chord

        if self.goal_state.state_id == smas.GoalStateId.NAVIGATING and self.goal_state.turning_info != None:
            turning_info = self.goal_state.turning_info

            e_target = turning_info.target_dirn
            det      = smnd.calc_det2x2(self.dirn, e_target)
            if det ** 2 >= smnd.EPS_m_sqr:
                L_side = (1/3) * L_chord
                x1     = self.posn               + L_side * self.dirn
                x2     = dst.approach_shape.posn - L_side * e_target

                L_poly = 2 * L_side + np.sqrt(smnd.calc_norm_sqr(x2 - x1))
                L_arc  = 0.5 * (L_chord + L_poly)

        return L_arc

    def estimate_arclength_of_current_edge(self):
        src_idx = self.routing_idx

        assert(src_idx >= 0)
        assert(src_idx + 1 < len(self.route))

        src_id = self.route[src_idx]
        src = self.nodegraph.nodes[src_id]

        cur_dst_id = self.route[src_idx + 1]
        cur_dst = self.nodegraph.nodes[cur_dst_id]

        dx = cur_dst.approach_shape.posn - src.approach_shape.posn
        L_chord, goal_dirn = smnd.calc_polar_decomp(dx)

        L_arc = L_chord
        MH = None

        if src_idx + 2 == len(self.route):
            return L_arc

        cos_theta = np.dot(self.dirn, goal_dirn)

        next_dst_id = self.route[src_idx + 2]
        next_dst     = self.nodegraph.nodes[next_dst_id]
        target_dirn = smnd.calc_unit_vector(next_dst.approach_shape.posn - cur_dst.approach_shape.posn)
        cos_phi = np.dot(self.dirn, target_dirn)
        if cos_phi <= cos_theta or self.agent_type != smad.AgentType.CAR or cos_theta >= smad.cos_theta_max[smad.AgentType.CAR]:
            return L_arc

        det      = smnd.calc_det2x2(self.dirn, target_dirn)
        if det ** 2 >= smnd.EPS_m_sqr:
            L_side = (1/3) * L_chord
            x1     = src.approach_shape.posn + L_side * self.dirn
            x2     = cur_dst.approach_shape.posn - L_side * target_dirn

            L_poly = 2 * L_side + np.sqrt(smnd.calc_norm_sqr(x2 - x1))
            L_arc  = 0.5 * (L_chord + L_poly)

        return L_arc

    def estimate_arclength_of_next_edge(self):
        src_idx = self.routing_idx

        assert(src_idx >= 0)
        assert(src_idx + 2 < len(self.route))

        src_id = self.route[src_idx]
        src = self.nodegraph.nodes[src_id]

        cur_dst_id = self.route[src_idx + 1]
        cur_dst = self.nodegraph.nodes[cur_dst_id]

        next_dst_id = self.route[src_idx + 2]
        next_dst     = self.nodegraph.nodes[next_dst_id]

        dx = next_dst.approach_shape.posn - cur_dst.approach_shape.posn
        L_chord, goal_dirn = smnd.calc_polar_decomp(dx)

        L_arc = L_chord
        MH = None

        if src_idx + 3 == len(self.route):
            return L_arc

        cos_theta = np.dot(self.dirn, goal_dirn)

        beyondnext_dst_id = self.route[src_idx + 3]
        beyondnext_dst    = self.nodegraph.nodes[beyondnext_dst_id]

        target_dirn = smnd.calc_unit_vector(beyondnext_dst.approach_shape.posn - next_dst.approach_shape.posn)
        cos_phi = np.dot(self.dirn, target_dirn)
        if cos_phi <= cos_theta or self.agent_type != smad.AgentType.CAR or cos_theta >= smad.cos_theta_max[smad.AgentType.CAR]:
            return L_arc

        det      = smnd.calc_det2x2(self.dirn, target_dirn)
        if det ** 2 >= smnd.EPS_m_sqr:
            L_side = (1/3) * L_chord

            x1     = cur_dst.approach_shape.posn  + L_side * self.dirn
            x2     = next_dst.approach_shape.posn - L_side * target_dirn

            L_poly = 2 * L_side + np.sqrt(smnd.calc_norm_sqr(x2 - x1))
            L_arc  = 0.5 * (L_chord + L_poly)

        return L_arc

    def DEPRECATED_predict_navigation_trajectory(self, width, MH_required):
        dst                 = self.nodegraph.nodes[self.goal_state.node_id]
        dx                  = dst.approach_shape.posn - self.posn
        L_chord, chord_dirn = smnd.calc_polar_decomp(dx)
    
        L_arc = L_chord
        MH    = None
        if self.goal_state.state_id == smas.GoalStateId.NAVIGATING and self.goal_state.turning_info != None:
            turning_info = self.goal_state.turning_info

            e_target = turning_info.target_dirn
            det      = smnd.calc_det2x2(self.dirn, e_target)
            if det ** 2 >= smnd.EPS_m_sqr:
                L_side = (1/3) * L_chord
                x1     = self.posn               + L_side * self.dirn
                x2     = dst.approach_shape.posn - L_side * e_target

                L_poly = 2 * L_side + np.sqrt(smnd.calc_norm_sqr(x2 - x1))
                L_arc  = 0.5 * (L_chord + L_poly)
                if MH_required:
                    MH = smgs.Polygon([self.posn, x1, x2, dst.approach_shape.posn], width)

        if MH_required and MH == None:
            MH = smgs.Segment(self.posn, dst.approach_shape.posn, width)
    
        return L_arc, chord_dirn, MH

    def DEPRECATED_compute_next_navigation_trajectory(self, width, MH_required):
        src_idx = self.routing_idx + 1

        assert(src_idx >= 0)
        assert(src_idx + 1 < len(self.route))

        src_id = self.route[src_idx]
        src = self.nodegraph.nodes[src_id]

        cur_dst_id = self.route[src_idx + 1]
        cur_dst = self.nodegraph.nodes[cur_dst_id]

        dx = cur_dst.approach_shape.posn - src.approach_shape.posn
        L_chord, goal_dirn = smnd.calc_polar_decomp(dx)

        L_arc = L_chord
        MH = None
        if src_idx + 2 == len(self.route):
            return L_arc, goal_dirn, MH
        else:
            cos_theta = np.dot(self.dirn, goal_dirn)

            next_dst_id = self.route[src_idx + 2]
            next_dst     = self.nodegraph.nodes[next_dst_id]
            target_dirn = smnd.calc_unit_vector(next_dst.approach_shape.posn - cur_dst.approach_shape.posn)
            cos_phi = np.dot(self.dirn, target_dirn)
            if cos_phi <= cos_theta or self.agent_type != smad.AgentType.CAR or cos_theta >= smad.cos_theta_max[smad.AgentType.CAR]:
                return L_arc, goal_dirn, MH

            det      = smnd.calc_det2x2(self.dirn, target_dirn)
            if det ** 2 >= smnd.EPS_m_sqr:
                L_side = (1/3) * L_chord
                x1     = src.approach_shape.posn + L_side * self.dirn
                x2     = cur_dst.approach_shape.posn - L_side * target_dirn

                L_poly = 2 * L_side + np.sqrt(smnd.calc_norm_sqr(x2 - x1))
                L_arc  = 0.5 * (L_chord + L_poly)
                if MH_required:
                    MH = smgs.Polygon([self.posn, x1, x2, cur_dst.approach_shape.posn], width)

            return L_arc, goal_dirn, MH

        assert(False)
        return None

    def draw_conflict_trajectory(self, view):
        if self.conflict_state.state_id == smas.ConflictStateId.MODULATING:
            return
        
        #
        # Get spline, draw spline with < subpixel accuracy.
        #
        # view.draw_cubic_spline(svv.white, self.conflict_state.x_curve_pts, 0.25, 0.5 * (2 + 0.5 * self.agent_type))

        #
        # @todo: Draw arrowheads?
        #
        x_curve_pts = self.conflict_state.x_curve_pts
        if self.conflict_state.state_id == smas.ConflictStateId.VEERING:
            # view.draw_line(svd.blue, x_curve_pts[0], x_curve_pts[1], 1)
            # view.draw_line(svd.blue,  x_curve_pts[1], x_curve_pts[2], 0.5)
            # view.draw_line(svd.blue, x_curve_pts[2], x_curve_pts[3], 0)
            # view.draw_cubic_spline(svd.blue, self.conflict_state.x_curve_pts, 0.25, 0.5 * (2 + 0.5 * self.agent_type))
            view.draw_cubic_spline(svd.blue, self.conflict_state.x_curve_pts, 0.25, 0)
        elif self.conflict_state.state_id == smas.ConflictStateId.YIELDING:
            # view.draw_line(svd.yellow, self.posn, self.conflict_state.x_des, 0)
            view.draw_line(svd.yellow, self.posn, self.conflict_state.stop_shape.posn, 0)
        elif self.conflict_state.state_id == smas.ConflictStateId.AWAITING:
            cur_dst = self.get_current_dst_node()
            x_omega, dist, nv, tv = cur_dst.approach_shape.get_nearest_point_info(self.posn)
            view.draw_line(svd.red, self.posn, x_omega, 0)
