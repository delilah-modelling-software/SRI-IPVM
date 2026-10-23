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

import view.defs as svd
import view.view as svv
import model.defs as smd
import model.configuration as smc
import model.agency.defs as smad
import model.agency.reaction as smar
import model.agency.state as smas
import model.agency.agent as smaa
import model.agency.following as smaf
import model.geometry.grid as smgg
import model.geometry.shape as smgs
import model.force as smf
import model.numerical.defs as smnd
import model.topology.activitynode as smtAct

class LeaderInfo:
    def __init__(self, id, s):
        self.id = id
        self.s  = s

class Vehicle(smaa.Agent):
    def __init__(self, id, agent_type, nodegraph, route, posn, dirn, length, width, radius, colour):
        super(Vehicle, self).__init__(id, agent_type, nodegraph, route, posn, dirn)
        
        self.rect_length = length
        self.rect_width  = width
        self.radius      = radius

        ENHANCEMENT_2025_11_26 = True
        if ENHANCEMENT_2025_11_26:
            expected_lane_width  = 3
            self.elliptic_length = self.rect_length + expected_lane_width
            self.elliptic_width  = expected_lane_width
        else:
            social_factor        = np.sqrt(2)
            self.elliptic_length = social_factor * self.rect_length
            self.elliptic_width  = social_factor * self.rect_width

        self.colour = colour

        self.v_max = smc.CFM_v_max

        #
        # @todo: A starting velocity should probably be passed in.
        #
        # self.DEPRECATED_f_total = np.zeros(2)
        # self.v       = np.zeros(2)
        self.v = self.v_max * self.dirn

        self.T_relax        = 2.4  # 2.4s, matches Anvari et al. (2015)
        # self.T_brake        = 0.77 # 0.77s, matches Anvari et al. (2015)

        #
        # Helbing, Farkas, Vicsek (2000) used 75kg as the mean pedestrian mass.
        # I'm using 2 tonnes as an estimate for average vehicle mass.
        #
        self.mass = 2000

        self.leader_id = None
        self.leader_s  = None
        # self.leader_info = None

        return

    def register_leader(self, leader, s):
        # assert(False)
        if self.leader_s == None or s < self.leader_s:
            self.leader_id = leader.id
            self.leader_s  = s
        # leader.follower_id = self.id
        return

    def deregister_leader(self, leader, s):
        self.leader_info = None
        return

    def get_directional_radius(self, u):
        c = np.dot(u, self.dirn)

        # print("Vehicle::get_directional_radius: cosine between directions {} and {} is {}".format(self.dirn, u, c))
        # assert(c**2 <= 1.)

        l_self = 0.5 * self.elliptic_length
        w_self = 0.5 * self.elliptic_width
        assert(l_self >= w_self)
        e_self = np.sqrt(l_self * l_self - w_self * w_self) / l_self
        assert(e_self <= 1.)

        r = w_self / np.sqrt(1 - (c * e_self)**2)

        return r
    
    def get_length_axis(self):
        return 0.5 * self.elliptic_length

    def get_width_axis(self):
        return 0.5 * self.elliptic_width

    def get_target_speed(self):
        v_nav = self.v_max
        if self.goal_state.state_id == smas.GoalStateId.NAVIGATING and self.goal_state.turning_info != None:
            if len(self.goal_state.turning_info.stages) > 0:
                v_nav = self.goal_state.turning_info.stages[0].spd_limit

        if self.leader_id == None or self.leader_id not in self.model.active_agents:
            return v_nav

        cur_spd = np.dot(self.v, self.dirn)

        ctragent = self.model.active_agents[self.leader_id]
        l_sum   = 0.5 * (self.rect_length + ctragent.rect_length)
        # s       = np.dot(ctragent.posn - self.posn, self.dirn) - l_sum
        s       =  self.leader_s
        # if s < smc.d_safe_IA_maximum:
            # return 0

        return v_nav

    def calc_motion_bboxen(self, horizon):
        # cur_dst_id = self.route[self.routing_idx + 1]
        # cur_dst    = self.nodegraph.nodes[cur_dst_id]
        cur_dst = self.get_current_dst_node()

        if self.goal_state.state_id in {smas.GoalStateId.CONVERGING, smas.GoalStateId.WAITING}:
            if isinstance(cur_dst, smtAct.ActivityNode):
                active_focal_shape = cur_dst.get_active_focal_shape(self)
                x_omega, d, nv, tv = active_focal_shape.get_nearest_point_info(self.posn)
            else:
                dst_shape          = cur_dst.approach_shape
                x_omega, d, nv, tv = dst_shape.get_nearest_point_info(self.posn)
            mbox               = smgg.BoundingBox(self.posn, x_omega)
            return [mbox]
        else:
            mboxen = []

            dst_shape = cur_dst.approach_shape
            x_omega = dst_shape.posn
            L_chord = smnd.calc_norm(x_omega - self.posn)
            L_arc   = None

            if self.goal_state.state_id == smas.GoalStateId.NAVIGATING and self.goal_state.turning_info != None:
                turning_info = self.goal_state.turning_info

                mbox        = smgg.BoundingBox(self.posn, x_omega)
                target_dirn = turning_info.target_dirn

                det    = smnd.calc_det2x2(self.dirn, target_dirn)
                if np.dot(det, det) < smnd.EPS_m_sqr:
                    L_arc = L_chord
                else:
                    dx = x_omega - self.posn
                    rcp_det = 1 / det
                    L_seg1 = smnd.calc_det2x2(dx, target_dirn) * rcp_det
                    L_seg2 = smnd.calc_det2x2(self.dirn, dx)   * rcp_det
                    L_arc = (1/3) * (L_seg1 + L_seg2 + 2 * L_chord)

            else:
                if horizon * self.get_target_speed() <= L_chord:
                    mbox = self.calc_simple_motion_bbox(horizon)
                    return [mbox]
                else:
                    mbox = smgg.BoundingBox(self.posn, x_omega)
                    L_arc = L_chord

            mboxen.append(mbox)

            #
            # If the vehicle is too close to its current destination, the next destination will also
            # need to be considered.
            #
            if horizon * self.get_target_speed() > L_arc:
                # next_dst_id    = self.route[self.routing_idx + 1]
                # next_dst       = self.nodegraph.nodes[cur_dst_id]

                next_dst       = self.get_next_dst_node()
                if next_dst != None:
                    next_dst_shape = next_dst.approach_shape
                    x_next         = next_dst_shape.posn
                    mbox           = smgg.BoundingBox(x_omega, x_next)
                    mboxen.append(mbox)

            return mboxen

    def is_concerned(self, ctragent, rel_posn, rel_v):
        if ctragent.agent_type == smad.AgentType.PEDESTRIAN:
            ctrptn = ctragent
            
            max_s = 2 * (self.v_max + ctrptn.v_max)
            if np.dot(rel_posn, rel_posn) > (max_s + 0.5 * (self.rect_length + ctrptn.radius))**2:
                return False
            
            u      = smnd.calc_unit_vector(rel_posn)
            c_self = np.dot(self.dirn, u)
            #
            # Would cos^2(pi/12) be better. Avoid transcendental eval if at all possible.
            #
            if c_self < smc.cos_IAI_concern:
                return False
            
        #
        # Since ctragent is a vehicle or is a pedestrian within the
        # angle-of-concern, the agent is concerned.
        #
        return True
    
    def calc_contrapedestrian_deceleration(self, ctrptn):
        cur_dst_idx = self.route[self.routing_idx + 1]
        cur_dst     = self.nodegraph.nodes[cur_dst_idx]

        dx = ctrptn.posn - self.posn
        dv = ctrptn.v    - self.v

        delta = dx + self.model.T_react * dv
        d1, u = smnd.calc_polar_decomp(delta)

        if not self.is_concerned(ctrptn, dx, dv):
            return np.zeros(2)

        s_ftr = d1 - (0.5 * self.rect_length + ctrptn.radius)
        I_soc = smf.calc_interagent_repulsion(s_ftr, self.agent_type, ctrptn.agent_type) * -self.dirn

        #
        # Attenuate by visibility
        #
        assert(self.conflict_state.state_id == smas.ConflictStateId.MODULATING or ctrptn.id not in self.major_ctragent_idset)
        
        cos_pq      = np.dot(u, self.dirn)
        vis_factor  = 0.5 + 0.25 * (1 + cos_pq)
        I_soc        *= vis_factor

        print("Vehicle:calculate_pedestrian_repulsion (vhcl {}, ctrptn {}, I {}): ".format(self.id, ctrptn.id, I_soc))
        return I_soc

    #
    # SFM, as used in Anvari+(2012, 2014, 2015)
    #
    def calc_intervehicular_influence_SFM(self, ctrvhcl):
        d_ideal =  1.8

        dx = ctrvhcl.posn - self.posn
        # dv = ctrvhcl.v    - self.v

        d0, u = smnd.calc_polar_decomp(dx) # np.sqrt(np.dot(dx, dx))
        c_self = np.dot(self.dirn, u)
        c_other = -np.dot(ctrvhcl.dirn, u)

        l_self = 0.5 * self.elliptic_length
        l_car  = 0.5 * ctrvhcl.elliptic_length

        w_self = 0.5 * self.elliptic_width
        assert(l_self >= w_self)
        e_self = np.sqrt(l_self * l_self - w_self * w_self) / l_self
        r_self = w_self / np.sqrt(1 - (e_self * c_self)**2)
                
        w_car = 0.5 * ctrvhcl.elliptic_width
        assert(l_car >= w_car)
        e_car = np.sqrt(l_car * l_car - w_car * w_car) / l_car
        r_car = w_car / np.sqrt(1 - (e_car * c_self)**2)

        #
        # @todo: Initialise lambda to something sensible from one of Anvari's papers if at all possible.
        #
        lambda_self       = smf.lambda_mu[smad.AgentType.CAR, smad.AgentType.CAR]
        perception_factor = lambda_self + (1 - lambda_self) * 0.5 * (1 + c_self)
                
        #
        # Simplified: (1 - p) * q * ... -> ...
        #
        I_soc = perception_factor * smf.calc_interagent_repulsion(d0 - (r_self + r_car + d_ideal), self.agent_type, ctrvhcl.agent_type) * -u
        assert(np.dot(dx, I_soc) <= 0.)
            
        return I_soc
        

    def calc_intervehicular_influence(self, ctrvhcl):
        dx = ctrvhcl.posn - self.posn
        dv = ctrvhcl.v    - self.v

        d0, u   = smnd.calc_polar_decomp(dx) # np.sqrt(np.dot(dx, dx))
        c_self  = np.dot(self.dirn, u)
        c_other = np.dot(ctrvhcl.dirn, u)

        p = c_self >= smc.cos_IAI_QP and c_other >= smc.cos_IAI_QP
        # p = np.dot(self.v, ctrvhcl.v) > 0 and c_self**2 > (np.cos(np.pi / 18))**2
        q = c_self**2 > 0.75

        if p:           
            scalar_v    = np.dot(u, self.v)
            scalar_vdes = self.v_max
            scalar_dv   = -np.dot(u, dv)
            scalar_s    = np.dot(u, dx) - (self.rect_length + ctrvhcl.rect_length)
            a_lead      = np.dot(u, ctrvhcl.F_total) / ctrvhcl.mass

            if scalar_s <= 0:
                return np.zeros(2)

            # a_des   = smaf.follow_GFM(scalar_s, scalar_v, scalar_vdes, scalar_dv, a_lead)
            # a_des   = smaf.follow_IDM(scalar_s, scalar_v, scalar_vdes, scalar_dv, a_lead)
            # a_des   = smaf.follow_IIDM(scalar_s, scalar_v, scalar_vdes, scalar_dv, a_lead)
            a_des   = smaf.follow_EIDM(scalar_s, scalar_v, scalar_vdes, scalar_dv, a_lead)
            # a_des   = smaf.follow_FVDM(scalar_s, scalar_v, scalar_vdes, scalar_dv, a_lead)
            I_soc    = a_des * self.dirn
            return I_soc
        elif q:
            return self.calc_intervehicular_influence_SFM(ctrvhcl)
        else:
            return np.zeros(2)
    

    def calc_interagent_influence(self, ctragent):
        if ctragent.agent_type == smad.AgentType.PEDESTRIAN:
            #
            # Decelerate (and possibly stop).
            #
            # @todo: calc_total_force() may need to prevent cars
            # from linearly scaling braking force to handle multiple pedestrians.
            #
            # @todo: Influence from multiple counterpedestrians may need to be
            # limited in the calc_total_influence method beforehand.
            #
            # assert(False)
            return self.calc_contrapedestrian_deceleration(ctragent)
        elif ctragent.agent_type == smad.AgentType.CAR:
            # assert(False)
            return self.calc_intervehicular_influence(ctragent)

    def DEPRECATED_apply_navigating_policy(self):
        assert(self.conflict_state.state_id == smas.ConflictStateId.MODULATING)
        assert(self.goal_state.state_id in {smas.GoalStateId.NAVIGATING, smas.GoalStateId.EXITING})

        cur_dst_idx = self.route[self.routing_idx + 1]
        cur_dst = self.nodegraph.nodes[cur_dst_idx]
        print("Vehicle::apply_navigating_policy: agent {} travelling towards node {}".format(self.id, cur_dst.id))
        # print("Vehicle::apply_navigating_policy: agent {} travelling towards node {} at {}".format(self.id, cur_dst.id, cur_dst.approach_shape.posn))

        assert(len(self.route) > self.routing_idx + 1)
        if len(self.route) > self.routing_idx + 2:
            # assert()
            # print()
            next_dst_idx = self.route[self.routing_idx + 2]
            next_dst     = self.nodegraph.nodes[next_dst_idx]
            next_internode_dirn = smnd.calc_unit_vector(next_dst.approach_shape.posn - cur_dst.approach_shape.posn)
            print("Vehicle::apply_navigating_policy old dirn {} to new dirn {}".format(self.dirn, next_internode_dirn))
            print("Vehicle::apply_navigating_policy self.posn {}, goal posn {}".format(self.posn, cur_dst.approach_shape.posn))


            #
            # Use cubic (quadratic with degree elevation) spline interpolation to find an intermediate target.
            #
            if np.dot(self.dirn, next_internode_dirn) < smad.cos_theta_max[smad.AgentType.CAR]:
                dx = cur_dst.approach_shape.posn - self.posn
                
                rcp_dnm = 1 / smnd.calc_det2x2(self.dirn, next_internode_dirn)
                num1    = smnd.calc_det2x2(dx, next_internode_dirn)
                num2    = smnd.calc_det2x2(self.dirn, dx)
                qcp_A = self.posn + num1 * rcp_dnm * self.dirn
                qcp_B = cur_dst.approach_shape.posn - num2 * rcp_dnm * next_internode_dirn
                max_radius = max(num1 * rcp_dnm, num2 * rcp_dnm)
                print("Vehicle::apply_navigating_policy qcp {} = {}, max radius {}".format(qcp_A, qcp_B, max_radius))
                assert(False)

                side_length = (1/3) * smnd.calc_norm(dx)

                u0 = side_length * self.dirn
                u2 = side_length * next_internode_dirn
                icp1 = self.posn + u0
                icp2 = cur_dst.approach_shape.posn - u2
                # u1 = icp2 - icp1

                #
                # Note: self.posn + 1.5 * u0 == cur_dst.approach_shape.posn - 1.5 * u2
                #
                spline = [self.posn, icp1, icp2, cur_dst.approach_shape.posn]
                # side1_length = np.sqrt(np.dot(u1, u1))
                E,  L,  T  = smnd.analyse_cubic_spline(spline, smc.omega_max_CAR, self.v_max, None)

                assert(E > 0)
                assert(L > 0)

                kappa_mean = np.sqrt(2 * E / L)
                r_mean     = 1 / (kappa_mean * smc.omega_max_CAR)

                spd_des_mean =  smc.omega_max_CAR * r_mean
                spd_sqr      = np.dot(self.v, self.v)
                spd          = np.sqrt(spd_sqr)

                est_time_ftr = self.T_relax
                
                if spd_des_mean**2 > spd_sqr:
                    spd_des = min(spd_des_mean, self.get_target_speed())

                    est_spd = spd_des + spd
                    
                    param_ftr = est_spd * est_time_ftr / L
                elif spd_des_mean**2 < spd_sqr:
                    spd_des_min = spd + 2 * (spd_des_mean - spd)
                    spd_des     = min(spd_des_min, self.get_target_speed())
                    
                    est_spd = spd_des + spd
                    
                    param_ftr = est_spd * est_time_ftr / L
                else:
                    spd_des = min(spd, self.get_target_speed())
                    est_spd = spd_des + spd
                    param_ftr = est_spd * est_time_ftr / L

                assert(param_ftr > 0)
                # assert(param_ftr <= 1)
                param_ftr = min(param_ftr, 1)
                x_des = smnd.evaluate_cubic_spline(spline, param_ftr)
                v_des = spd_des * smnd.calc_unit_vector(x_des - self.posn)
                I_nav = (v_des - self.v)/ self.T_relax
                
                return I_nav
                
            
      
        x_omega, dist, nv, tv = cur_dst.approach_shape.get_nearest_point_info(self.posn)
        v_desired = self.get_target_speed() * smnd.calc_unit_vector(x_omega - self.posn)
            
        I_nav     = (v_desired - self.v) / self.T_relax

        return I_nav

    def DEPRECATED_apply_navigating_policy(self):
        assert(self.conflict_state.state_id == smas.ConflictStateId.MODULATING)
        assert(self.goal_state.state_id in {smas.GoalStateId.NAVIGATING, smas.GoalStateId.EXITING})
        assert(self.routing_idx + 1 < len(self.route))

        if self.routing_idx + 2 == len(self.route):
            return super(Vehicle, self).apply_navigating_policy()

        cur_src_idx = self.route[self.routing_idx]
        cur_src     = self.nodegraph.nodes[cur_src_idx]

        cur_dst_idx = self.route[self.routing_idx + 1]
        cur_dst     = self.nodegraph.nodes[cur_dst_idx]
        dx                   = cur_dst.approach_shape.posn - self.posn
        L_chord, goal_dirn   = smnd.calc_polar_decomp(cur_dst.approach_shape.posn - self.posn)
        cos_theta   = np.dot(self.dirn, goal_dirn)
        if cos_theta >= smad.cos_theta_max[smad.AgentType.CAR]:
            return super(Vehicle, self).apply_navigating_policy()

        next_dst_idx   = self.route[self.routing_idx + 2]
        next_dst       = self.nodegraph.nodes[next_dst_idx]
        target_dirn    = smnd.calc_unit_vector(next_dst.approach_shape.posn - cur_dst.approach_shape.posn)
        cos_phi        = np.dot(self.dirn, target_dirn)
        if cos_phi >= cos_theta:
            return super(Vehicle, self).apply_navigating_policy()

            
        assert(np.dot(self.dirn, goal_dirn) < smad.cos_theta_max[smad.AgentType.CAR])
        sin_phi = smnd.calc_det2x2(self.dirn, target_dirn)
        csc_phi  = 1 / sin_phi
        num1     = smnd.calc_det2x2(dx, target_dirn)
        num2     = smnd.calc_det2x2(self.dirn, dx)
        if num1 < 0 or num2 < 0:
            print("Vehicle::apply_navigating_policy numerators {} and {}, denominator {}".format(num1, num2, sin_phi))
            return super(Vehicle, self).apply_navigating_policy()
        
        #
        # Lengths of the triangle defining a quadratic (parabolic) spline.
        #
        Lq1 = num1 * csc_phi
        Lq2 = num2 * csc_phi        

        print("Vehicle::apply_navigating_policy car {} has Cramer numerators {} and {}, Cramer denominator {} = sin(phi) hence reciprocal denominator {} = csc(theta)".format(self.id, num1, num2, sin_phi, csc_phi))
        print("Vehicle::apply_navigating_policy car {} therefore has pre- and post-qcp lengths {} and {}".format(self.id, Lq1, Lq2))
        
        #
        # Extreme debugging.
        #
        if 1:
            qcp_A       = self.posn + (num1 * csc_phi) * self.dirn
            qcp_B       = cur_dst.approach_shape.posn - (num2 * csc_phi) * target_dirn
            err_qcp_sqr = smnd.calc_norm_sqr(qcp_A - qcp_B)
            assert(err_qcp_sqr <= smnd.EPS_m_sqr)

        cot_halfphi = np.sqrt((1 + cos_phi)/(1 - cos_phi))
        # print("Vehicle::apply_navigating_policy car {} also has cos(theta) {} and tan(theta/2) {}".format(self.id, cos_theta, tan_halftheta))

        r_max      = min(Lq1, Lq2) * cot_halfphi
        v_max_turn = np.sqrt(smc.a_max_cfgl * r_max)
        T_turn     = np.arccos(cos_phi) / smc.omega_max_CAR

        # Lc1 = (1/3) * L_chord
        # Lc2 = (1/3) * L_chord
        Lc1 = (2/3) * Lq1
        Lc2 = (2/3) * Lq2

        u0 = Lc1 * self.dirn
        u2 = Lc2 * target_dirn
        icp1 = self.posn + u0
        icp2 = cur_dst.approach_shape.posn - u2

        spline = [self.posn, icp1, icp2, cur_dst.approach_shape.posn]

        spd_sqr = np.dot(self.v, self.v)
        spd     = np.sqrt(spd_sqr)

        # geometric_v_max = np.sqrt(max_spd_sqr)
        # v_max           = min(self.v_max, geometric_v_max)
        E,  L,  T_move       = smnd.analyse_cubic_spline(spline, smc.omega_max_CAR, spd, None)
        assert((L - T_move * spd)**2 < smnd.EPS_m_sqr)
        
        spd_max = min(self.v_max, v_max_turn)
        spd_des = min(spd, v_max_turn)
        if T_turn > T_move:
            #
            # T_move must be increased
            #
            T_move  = T_turn_min
            spd_des = min(L / T_move, spd_max)
        elif (T_turn - T_move)**2 < smnd.EPS_m_sqr:
            skip
        else:
            assert(T_turn < T_move)
            #
            # T_move can be increased
            #
            T_turn = T_move
            spd_des = min(L / T_turn, spd_max)

        est_time_ftr = smc.T_step
        
        param_ftr = spd_des * est_time_ftr / L
        param_ftr = min(param_ftr, 1)
        # assert(param_ftr > 0)

        x_des = smnd.evaluate_cubic_spline(spline, param_ftr)
        v_des = spd_des * smnd.calc_unit_vector(x_des - self.posn)        
        I_nav = (v_des - self.v)/ self.T_relax

        return I_nav

    def init_best_possible_stop(self, ctragent, interaction, t0):
        #
        # @todo: Clarify that this is a netto displacement (boundary-to-boundary)
        # rather than brutto (position-to-position) distance.
        #
        d_margin = smc.d_safe_IA_maximum

        # id1, id2 = smii.interagent_codec.decode_interaction(interaction.combined_id)
        # major_ctragent_id = self.id ^ id1 ^ id2
        # ctragent          = agent_tbl[major_ctragent_id]
        T_curve           = interaction.t_conflict
        if T_curve <= 0:
            # assert(False)
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
            # assert(False)
            
            # print("Agent::init_best_possible_stop - failure! {} at {} wanted to stop at {}".format(self.id, self.posn, x_desired))
            return

        print("Agent::init_best_possible_stop - {} at {} will try to stop at {}".format(self.id, self.posn, x_desired))

        #
        # @todo: There should be a braking relaxation time. 
        #
        T_brk_vhcl = 0.77 # 0.77s, matches Anvari et al. (2015)
        T_brk = max(self.T_relax, T_brk_vhcl)

        # assert(False)
                    
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

    def clip_influence(self, I):
        # assert(False)
        v_new = self.v + I * smc.T_step
        spd_new, dirn_new = smnd.calc_polar_decomp(v_new)

        cos_theta = np.dot(dirn_new, self.dirn)
        if cos_theta <= 0:
            spd_new = 0
            dirn_new = self.dirn
            cos_theta = 1
            I_new = -self.v / smc.T_step
            v_new = np.zeros(2)
            return I_new

        if spd_new > self.v_max:
            spd_new = self.v_max

        max_theta = smad.theta_max[smad.AgentType.CAR]
        cos_max_theta = smad.cos_theta_max[smad.AgentType.CAR]
        sin_max_theta = smad.sin_theta_max[smad.AgentType.CAR]
        if spd_new * smc.T_step < smc.L_wheelbase * max_theta:
            # assert(False)
            max_theta = spd_new * smc.T_step / smc.L_wheelbase
            cos_max_theta = np.cos(max_theta)
            sin_max_theta = np.sin(max_theta)

        if cos_theta < cos_max_theta:
            perp_dirn = smnd.rotate_90ACW(self.dirn)
            sin_theta = np.dot(dirn_new, perp_dirn)
            if sin_theta < 0:
                perp_dirn = -perp_dirn
                sin_theta = -sin_theta

            # assert(sin_theta > sin_max_theta)

            theta = max_theta
            cos_theta = cos_max_theta
            sin_theta = min(sin_theta, sin_max_theta)


            dirn_new = cos_theta * self.dirn + sin_theta * perp_dirn

            if self.goal_state.state_id == smas.GoalStateId.WAITING or self.conflict_state.state_id == smas.ConflictStateId.AWAITING:
                dst = self.get_current_dst_node()
                # dirn_ideal = smnd.calc_unit_vector(dst.approach_shape.posn - self.posn)
                cur_dst = self.get_current_dst_node()
                cur_src = self.get_current_src_node()
                intergoal_dirn = smnd.calc_unit_vector(cur_dst.approach_shape.posn - cur_src.approach_shape.posn)

                # if np.dot(dirn_ideal, dirn_new) < np.dot(dirn_ideal, self.dirn):
                if np.dot(intergoal_dirn, dirn_new) < np.dot(intergoal_dirn, self.dirn):
                    # assert np.dot(dirn_ideal, dirn_new) >= 0.
                    dirn_new = self.dirn

        v_new = spd_new * dirn_new
        I_new = (v_new - self.v) / smc.T_step

        if self.leader_id != None and self.leader_id in self.model.active_agents:
            # assert(self.leader_s >= 0)
            # assert(self.leader_s > 0)

            leader = self.model.active_agents[self.leader_id]
            a_pred = np.dot(I_new, self.dirn)

            spd_lead = smnd.calc_norm(leader.v)
            a_lead   = smnd.calc_norm(leader.I_total)

            a_des = smaf.follow_EIDM(self.leader_s, spd_new, self.v_max, spd_new - spd_lead, a_lead)
            if a_pred > a_des:
                I_new -= (a_pred - a_des) * self.dirn

            # scalar_v    = np.dot(u, self.v)
            # scalar_vdes = self.v_max
            # scalar_dv   = -np.dot(u, dv)
            # a_lead      = np.dot(u, ctrvhcl.F_total) / ctrvhcl.mass

            # a_des   = smaf.follow_GFM(scalar_s, scalar_v, scalar_vdes, scalar_dv, a_lead)
            # a_des   = smaf.follow_IDM(scalar_s, scalar_v, scalar_vdes, scalar_dv, a_lead)
            # a_des   = smaf.follow_IIDM(scalar_s, scalar_v, scalar_vdes, scalar_dv, a_lead)
            # a_des   = smaf.follow_EIDM(scalar_s, scalar_v, scalar_vdes, scalar_dv, a_lead)
            # a_des   = smaf.follow_FVDM(scalar_s, scalar_v, scalar_vdes, scalar_dv, a_lead)
            # I_soc    = a_des * self.dirn

        print("Vehicle::clip_intrinsic_force v {} = {} * {}".format(v_new, spd_new, dirn_new))
        print("Vehicle::clip_intrinsic_force I {} = ({} - {}) / {}".format(I_new, v_new, self.v, smc.T_step))

        return I_new

    def OLDDEPRECATED_clip_intrinsic_force(self, I):
        v = self.v + I * smc.T_step
        print("Vhcl::clip_intrinsic_force - {} has velocity {} and wants velocity {}".format(self.id, self.v, v))

        AV_is_excessive = np.dot(v, self.dirn)**2 < smnd.calc_norm_sqr(v) * cos_max_delta_angle**2
        LV_is_excessive = smnd.calc_norm_sqr(v) > self.v_max**2
        if not AV_is_excessive and not LV_is_excessive:
            return I
        
        new_spd, new_dirn = smnd.calc_polar_decomp(v)

        # cos_theta = np.dot(new_dirn, self.dirn)

        #
        # @todo: Calculate these (and return I if possible) prior to polar decomposition:
        #    AV_is_excessive = np.dot(v, self.dirn)**2 < cos_sqr_max_delta_angle == cos_max_delta_angle**2
        #    LV_is_excessive = smnd.calc_norm_sqr(v) > self.v_ped_sqr == self.v_ped**2
        # AV_is_excessive = (cos_theta < cos_max_delta_angle)
        # LV_is_excessive = (new_spd > self.v_ped)
        # if not AV_is_excessive and not LV_is_excessive:
            # return I

        if AV_is_excessive:
            print("Vhcl::clip_intrinsic_force - {} has excessive desired AV whose cos is {} < {}".format(self.id, np.dot(new_dirn, self.dirn), cos_max_delta_angle))
            perp_dirn = smnd.rotate_90ACW(self.dirn)
            if np.dot(perp_dirn, new_dirn) < 0:
                perp_dirn = -perp_dirn
            new_dirn = cos_max_delta_angle * self.dirn + sin_max_delta_angle * perp_dirn
            # assert(False)

        #
        # @todo: Rethink this code later.
        #
        if LV_is_excessive:
            # print("Ptn::clip_intrinsic_force - {} has excessive desired LV {}".format(self.id, new_spd))

            newer_spd = self.v_max
            assert(newer_spd <= new_spd)
            # newer_spd = smnd.clip_magnitude_softmax(new_spd, self.v_ped, self.v_max)
            
            # print("Ptn::clip_intrinsic_force - {} has clipped desired LV (spd) from {} to {}".format(self.id, new_spd, newer_spd))
            new_spd = newer_spd
            # assert(new_spd >= self.v_ped)
            # assert(new_spd <= self.v_max)

        # print("Ptn::clip_intrinsic_force - {} is clipping new velocity from {} to {}".format(self.id, v, new_spd * new_dirn))

        v = new_spd * new_dirn
        I = (v - self.v) / smc.T_step
        return I * self.mass

    def DEPRECATED_clip_intrinsic_force(self, I):
        v_new             = self.v + I * smc.T_step
        spd_new, dirn_new = smnd.calc_polar_decomp(v_new)

        if spd_new > self.v_max:
            spd_new = self.v_max

        cos_max_delta_angle = smad.cos_omega_max[smad.AgentType.CAR]
        if np.dot(dirn_new, self.dirn) < cos_max_delta_angle:
            perp_dirn = smnd.rotate_90ACW(self.dirn)
            if np.dot(perp_dirn, dirn_new) < 0:
                perp_dirn = -perp_dirn
            sin_max_delta_angle = smad.sin_omega_max[smad.AgentType.CAR]
            dirn_new = cos_max_delta_angle * self.dirn + sin_max_delta_angle * perp_dirn

        v = spd_new * dirn_new
        I = (v - self.v) / smc.T_step
        return I * self.mass

    def DEPRECATED_clip_intrinsic_force(self, I):
        # assert(False)
        v_new             = self.v + I * smc.T_step
        spd_new, dirn_new = smnd.calc_polar_decomp(v_new)

        cos_theta = np.dot(dirn_new, self.dirn)
        if cos_theta <= 0:
            spd_new  = 0
            dirn_new = self.dirn
            cos_theta = 1
            I_new = -self.v / smc.T_step
            v_new = np.zeros(2)
            return self.mass * I_new

        assert(np.dot(v_new, self.v) >= 0.)

        if spd_new > self.v_max:
            spd_new = self.v_max

        max_theta     = smad.theta_max[smad.AgentType.CAR]
        cos_max_theta = smad.cos_theta_max[smad.AgentType.CAR]
        sin_max_theta = smad.sin_theta_max[smad.AgentType.CAR]
        if spd_new * smc.T_step < smc.L_wheelbase * max_theta:
            # assert(False)
            max_theta     = spd_new * smc.T_step / smc.L_wheelbase
            cos_max_theta = np.cos(max_theta)
            sin_max_theta = np.sin(max_theta)

        if cos_theta < cos_max_theta:
            theta     = max_theta
            cos_theta = cos_max_theta
            sin_theta = sin_max_theta

            perp_dirn = smnd.rotate_90ACW(self.dirn)
            if np.dot(perp_dirn, dirn_new) < 0:
                perp_dirn = -perp_dirn
            dirn_new = cos_theta * self.dirn + sin_theta * perp_dirn

        v_new = spd_new * dirn_new
        I_new = (v_new - self.v) / smc.T_step
        print("Vehicle::clip_intrinsic_force v {} = {} * {}".format(v_new, spd_new, dirn_new))
        print("Vehicle::clip_intrinsic_force I {} = ({} - {}) / {}".format(I_new, v_new, self.v, smc.T_step))
        
        return self.mass * I_new

        #
        # DEPRECATED
        # @todo: Consider moving this code into clip_velocity to reduce desired speed and turning angle.
        #
        perp_dirn = smnd.rotate_90ACW(self.dirn)
        if np.dot(perp_dirn, dirn_new) < 0:
            perp_dirn = -perp_dirn
        sin_theta = np.dot(dirn_new, perp_dirn)
        assert(sin_theta >= 0)
        assert(cos_theta > 0)
        assert(cos_theta**2 >= smnd.EPS_m_sqr)
        if sin_theta * spd_new**2 > cos_theta * smc.L_wheelbase * smc.a_max_cfgl:
            assert(False)
            tan_theta = sin_theta / cos_theta

            assert(tan_theta != 0)
            assert(spd_new != 0)
            assert(tan_theta > 0)
            assert(spd_new > 0)
            reduction_factor = smc.L_wheelbase * smc.a_max_cfgl / tan_theta * spd_new**2
            assert(reduction_factor >= 0)

            cbrt_red_factor = np.power(red_factor, 1/3)
            spd_new        *= cbrt_red_factor
            tan_theta      *= cbrt_red_factor
            cos_theta = np.sqrt(1 / (1 + tan_theta**2))
            sin_theta = cos_theta * tan_theta
            dirn_new = cos_theta * self.dirn + sin_theta * perp_dirn

            #
            # END-DEPRECATED
            #
            

    def is_reversible(self):
        return False

    def clip_velocity(self):
        print("Vehicle::clip_velocity starting - vhcl {} at {} will move with v = {}".format(self.id, self.posn, self.v))
        
        cur_dst_idx = self.route[self.routing_idx + 1]
        cur_dst     = self.nodegraph.nodes[cur_dst_idx]

        cur_src_idx = self.route[self.routing_idx]
        cur_src     = self.nodegraph.nodes[cur_src_idx]

        #
        # Disabled (2024-06-12)
        #

        # if self.goal_state.state_id == smas.GoalStateId.NAVIGATING or cur_dst.focal_shape == None:
            # self.dirn = smnd.calc_unit_vector(cur_dst.approach_shape.posn - self.posn)
        # else:
            # self.dirn = smnd.calc_unit_vector(cur_dst.focal_shape.posn - self.posn)

        L_goal, goal_dirn = smnd.calc_polar_decomp(cur_dst.approach_shape.posn - cur_src.approach_shape.posn)

        # print("Vehicle::clip_velocity direction - vhcl {} is moving towards {} in direction {}".format(self.id, cur_dst.posn, self.dirn))

        if np.dot(self.v, goal_dirn) <= 0 and not self.is_reversible():
        # if np.dot(self.v, self.dirn) <= 0 and not self.is_reversible():
            # print("Vehicle::clip_velocity direction - vhcl {} <v , dirn> = <{}, {}> = {}".format(self.id, self.v, self.dirn, np.dot(self.v, self.dirn)))
            self.v   = np.zeros(2)
            # self.state = smaa.AgentState.STOPPED
            return

        if np.dot(self.v, self.v) < smnd.EPS_m_sqr:
            self.v   = np.zeros(2)
            return

        #
        # Disabled: 2024-06-13
        #
        # self.v = np.dot(self.v, self.dirn) * self.dirn

        v_norm, unit_v     = smnd.calc_polar_decomp(self.v)

        assert(v_norm**2 > smnd.EPS_m_sqr)
        if np.dot(self.dirn, unit_v) < smad.cos_theta_max[smad.AgentType.CAR]:
            perp_dirn  = smnd.rotate_90ACW(self.dirn)
            delta_dirn = unit_v - self.dirn
            if np.dot(perp_dirn, delta_dirn) < 0:
                perp_dirn = -perp_dirn
            unit_v = smad.cos_theta_max[smad.AgentType.CAR] * self.dirn + smad.sin_theta_max[smad.AgentType.CAR] * perp_dirn

        if self.goal_state.state_id == smas.GoalStateId.WAITING or self.conflict_state.state_id == smas.ConflictStateId.AWAITING:
            cur_dst = self.get_current_dst_node()
            # ideal_dirn = smnd.calc_unit_vector(cur_dst.approach_shape.posn - self.posn)
            intergoal_dirn = smnd.calc_unit_vector(cur_dst.approach_shape.posn - cur_src.approach_shape.posn)

            # if np.dot(unit_v, ideal_dirn) >= np.dot(self.dirn, ideal_dirn):
            if np.dot(unit_v, intergoal_dirn) >= np.dot(self.dirn, intergoal_dirn) and self.goal_state.turning_info == None:
                self.dirn = intergoal_dirn
        else:
            self.dirn = unit_v
        # assert(np.dot(self.dirn, goal_dirn) >= 0.)

        if v_norm > self.v_max:
            v_norm = self.v_max

        self.v = v_norm * self.dirn

        # if v_norm > self.v_max:
            # assert(False)
            # self.v = self.v_max * self.dirn

        # print("Vehicle::clip_velocity ending - vhcl {} will move with v {}".format(self.id, self.posn, self.v))

        return

    def draw_social_aura(self, view):
        side_dirn = np.array([-self.dirn[1], self.dirn[0]])
        major_axis = 0.5 * self.elliptic_length * self.dirn
        minor_axis = 0.5 * self.elliptic_width  * side_dirn

        clr = svd.white
        if self.goal_state.state_id == smas.GoalStateId.WAITING:
            clr = svd.light_green
        elif self.goal_state.state_id == smas.GoalStateId.CONVERGING:
            clr = svd.light_blue

        view.fill_ellipse(clr, self.posn, major_axis, minor_axis)

    def draw_physical_area(self, view):
        spd = smnd.calc_norm(self.v)
        if not spd >= 0.:
            spd = 0.

        side_dirn = np.array([-self.dirn[1], self.dirn[0]])

        true_halflength = 0.5 * self.rect_length
        true_halfwidth  = 0.5 * self.rect_width

        fwd_vector  = true_halflength * self.dirn
        side_vector = true_halfwidth * side_dirn

        spine_fwd = self.posn + fwd_vector
        spine_bwd = self.posn - fwd_vector

        pts = [spine_fwd + side_vector, spine_bwd + side_vector, spine_bwd - side_vector, spine_fwd - side_vector]

        clr = svd.dark_blue
        if self.conflict_state.state_id == smas.ConflictStateId.VEERING:
            clr = svd.dark_green
        elif self.conflict_state.state_id == smas.ConflictStateId.YIELDING:
            clr = svd.light_red
        elif self.conflict_state.state_id == smas.ConflictStateId.AWAITING:
            clr = svd.red
        view.fill_polygon(clr, pts)
        # view.fill_polygon(self.colour, pts)

        view.draw_line(self.colour, self.posn, self.posn + (spd / self.v_max) * true_halflength * self.dirn, 0)

        text = "{}".format(self.id)
        # view.draw_text(text, svd.black, self.posn - 0.25 * major_axis - np.array([1, 0]))
        view.draw_text(text, svd.white, self.posn - 0.125 * fwd_vector - np.array([0.75, 0]), True)


        if self.goal_state.state_id == smas.GoalStateId.WAITING and self.conflict_state == smas.ConflictStateId.YIELDING:
            # view.draw_line(svd.red, self.posn - 0.5 * major_axis - 0.5 * minor_axis, self.posn - 0.5 * major_axis + 0.5 * minor_axis, 0.125 * self.elliptic_width)

            view.draw_line(svd.red, self.posn - 0.75 * fwd_vector - 0.75 * side_vector, self.posn - 0.75 * fwd_vector + 0.75 * side_vector, 0.2)

        elif self.goal_state.state_id == smas.GoalStateId.WAITING:
            # view.draw_line(svd.magenta, self.posn - 0.5 * major_axis - 0.5 * minor_axis, self.posn - 0.5 * major_axis + 0.5 * minor_axis, 0.125 * self.elliptic_width)
            view.draw_line(svd.magenta, self.posn - 0.75 * fwd_vector - 0.75 * side_vector, self.posn - 0.75 * fwd_vector + 0.75 * side_vector, 0.2)

        elif self.conflict_state.state_id == smas.ConflictStateId.AWAITING:
            # view.draw_line(svd.orange, self.posn - 0.5 * major_axis - 0.5 * minor_axis, self.posn - 0.5 * major_axis + 0.5 * minor_axis, 0.125 * self.elliptic_width)
            view.draw_line(svd.orange, self.posn - 0.75 * fwd_vector - 0.75 * side_vector, self.posn - 0.75 * fwd_vector + 0.75 * side_vector, 0.2)

        elif self.conflict_state.state_id == smas.ConflictStateId.YIELDING:
            # view.draw_line(svd.yellow, self.posn - 0.5 * major_axis - 0.5 * minor_axis, self.posn - 0.5 * major_axis + 0.5 * minor_axis, 0.125 * self.elliptic_width)
            view.draw_line(svd.yellow, self.posn - 0.75 * fwd_vector - 0.75 * side_vector, self.posn - 0.75 * fwd_vector + 0.75 * side_vector, 0.2)

        if self.goal_state.state_id == smas.GoalStateId.NAVIGATING and self.goal_state.turning_info != None:
            turning_info = self.goal_state.turning_info

            prev_spd_lim = np.dot(self.v, self.dirn)
            for i in range(len(turning_info.stages)):
                stage = turning_info.stages[i]
                cur_spd_lim = stage.spd_limit
                
                colour = svd.green
                if prev_spd_lim < cur_spd_lim:
                    colour = svd.red
                stage.shape.draw(colour, view)
                prev_spd_lim = cur_spd_lim

            if self.routing_idx + 2 < len(self.route):
                next_dst_idx = self.route[self.routing_idx + 2]
                next_dst = self.nodegraph.nodes[next_dst_idx]

            cur_src_idx = self.route[self.routing_idx]
            cur_src = self.nodegraph.nodes[cur_src_idx]

            if self.routing_idx + 1 == len(self.route):
                return

            cur_dst_idx = self.route[self.routing_idx + 1]
            cur_dst = self.nodegraph.nodes[cur_dst_idx]

            goal_dirn = smnd.calc_unit_vector(cur_dst.approach_shape.posn - cur_src.approach_shape.posn)
            future_goal_dirn = smnd.calc_unit_vector(next_dst.approach_shape.posn - cur_dst.approach_shape.posn)

            goal_sin = smnd.calc_det2x2(goal_dirn, future_goal_dirn)

            if goal_sin**2 > smnd.EPS_m_sqr:
                if goal_sin > 0:
                    # view.fill_circle(svd.yellow, self.posn + 0.5 * major_axis + 0.5 * minor_axis, 0.125 * self.elliptic_width)
                    # view.fill_circle(svd.yellow, self.posn + 0.5 * major_axis + 0.5 * minor_axis, 0.2)
                    view.fill_circle(svd.yellow, self.posn + 0.75 * fwd_vector + 0.75 * side_vector, 0.2)
                elif goal_sin < 0:
                    # view.fill_circle(svd.yellow, self.posn + 0.5 * major_axis - 0.5 * minor_axis, 0.125 * self.elliptic_width)
                    # view.fill_circle(svd.yellow, self.posn + 0.5 * major_axis - 0.5 * minor_axis, 0.2)
                    view.fill_circle(svd.yellow, self.posn + 0.75 * fwd_vector - 0.75 * side_vector, 0.2)
        return
