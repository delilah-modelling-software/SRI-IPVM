#
# Agent and two subclasses, Pedestrian and Vehicle
#
# @todo: Move some of the Vehicle functionality into a Car subclass
# @todo: Add other vehicle subclasses, e.g. Bicycle 
#

import numpy as np
np.seterr(invalid='raise')

# import scipy as sp
import scipy.stats as ss

# from collections import deque

import view.defs as svd
import view.view as svv
import model.agency.defs as smad
import model.agency.agent as smaa
import model.agency.state as smas
import model.geometry.shape as smgs
import model.configuration as smc
import model.force as smf
import model.numerical.defs as smnd

#
# @todo: Replace this with a function of vehicle speed, as-per Anvari et al. 2012, 2013, 2015
#

class Pedestrian(smaa.Agent):
    def __init__(self, id, nodegraph, route, posn, dirn, src_colour, dst_colour):
        super(Pedestrian, self).__init__(id, smad.AgentType.PEDESTRIAN, nodegraph, route, posn, dirn)

        self.src_colour = src_colour
        self.dst_colour = dst_colour
        
        self.DEPRECATED_f_total = np.zeros(2)
        # self.v       = np.zeros(2)

        self.v_ped          = 1.34
        self.v_max          = 1.3 * self.v_ped # (Helbing and Molnar 1995)
        self.T_relax        = 0.3 # 0.3s, matches Anvari et al. (2015) rather than the 0.5 from (Helbing and Molnar 1995)

        #
        # Helbing, Farkas, Vicsek (2000) used 75kg as the mean pedestrian mass.
        #
        self.mass = 75

        self.max_angular_velocity = 0.25 * np.pi

        self.radius         = 0.2
        return

    def get_directional_radius(self, u):
        return self.radius

    def get_length_axis(self):
        return self.radius

    def get_width_axis(self):
        return self.radius

    def get_target_speed(self):
        return self.v_ped

    # def register_interagent_collision(self, ctragent, s, nrml, tngnt, momentum_factor):
        #
        # @todo: Add forces:
        #   Normal elastic force and tangential frictional force.
        #       n = -e
        #       r_total = r_alpha + r_beta
        #       m_factor_alpha = 2 * m_beta / (m_alpha + m_beta)
        #       m_factor_beta = 2 * m_alpha / (m_alpha + m_beta)
        #       F_alpha += m_factor_alpha  * k * (-s) * n
        #       F_beta  += m_factor_beta * k * (-s) * e
        #
        #   Tangential frictional force
        #       t        = smnd.rotate_90ACW(e)
        #       v_rel    = v_beta - v_alpha
        #       friction = kappa * (r_total - s) * np.dot(v_rel, t)
        #       m_factor_alpha = 2 * m_beta / (m_alpha + m_beta)
        #       m_factor_beta = 2 * m_alpha / (m_alpha + m_beta)
        #       F_alpha -= m_factor_alpha  * friction * t 
        #       F_beta  += m_factor_beta * friction * t
        #
        # As-per HFV 2000, k = 1.2 * 10**5 and kappa = 2.4 * 10**5
        #

        #
        # NB: The interaction/dispatcher is taking responsibility for logging collisions, so this method doesn't need to.
        #
        # print("Pedestrian::register_interagent_collision between {} and {} with overlap {}".format(self.id, ctragent.id, -s))
        
        # return

    def calc_reactive_circular_interpedestrian_repulsion(self, ctragent):
        dx     = ctragent.posn - self.posn
        dx_sqr = smnd.calc_norm_sqr(dx)
        s_max = 2.
        sum_radii = self.radius + ctragent.radius
        if dx_sqr > (s_max + sum_radii)**2:
            return np.zeros(2)
        
        d0, dirn0 = smnd.calc_polar_decomp(dx)
        I_soc = smf.calc_interagent_repulsion(d0 - sum_radii, self.agent_type, ctragent.agent_type) * -dirn0

        cosine      = np.dot(dirn0, self.dirn)
        vis_factor  = 0.5 + 0.25 * (1 + cosine)
        I_soc      *= vis_factor

        return I_soc

    def calc_proactive_circular_interpedestrian_repulsion(self, ctragent):
        dx_now = ctragent.posn - self.posn
        dx_sqr = smnd.calc_norm_sqr(dx_now)
        s_max = 2.
        sum_radii = self.radius + ctragent.radius
        if dx_sqr > (s_max + sum_radii)**2:
            return np.zeros(2)
        
        d0, dirn0 = smnd.calc_polar_decomp(dx_now)

        dv        = ctragent.v - self.v
        dx_soon   = dx_now + self.model.T_react * dv
        d1, dirn1 = smnd.calc_polar_decomp(dx_soon)

        I_soc = smf.calc_interagent_repulsion(d1 - sum_radii, self.agent_type, ctragent.agent_type) * -dirn1

        cosine      = np.dot(dirn0, self.dirn)
        vis_factor  = 0.5 + 0.25 * (1 + cosine)
        I_soc      *= vis_factor

        return I_soc

    def calc_elliptical_interpedestrian_repulsion(self, ctragent):
        dx_now     = ctragent.posn - self.posn
        dx_sqr = smnd.calc_norm_sqr(dx_now)
        s_max = 2.
        sum_radii = self.radius + ctragent.radius
        if dx_sqr > (s_max + sum_radii)**2:
            return np.zeros(2)
        
        d0, dirn0 = smnd.calc_polar_decomp(dx_now)

        dv        = ctragent.v - self.v
        dx_soon   = dx_now + self.model.T_react * dv
        d1, dirn1 = smnd.calc_polar_decomp(dx_soon)

        T_react = self.model.T_react
        b       = 0.5 * np.sqrt((d0 + d1)**2 - T_react**2 * smnd.calc_norm_sqr(dv))
        I_soc   = -smf.calc_repulsion(b, 0.04, 3.22) * ((d0 + d1)/ (2 * b)) * 0.5 * (dirn0 + dirn1)

        ENHANCEMENT_2025_11_19 = False
        if ENHANCEMENT_2025_11_19:
            #
            # @todo: What follows is a hacky stochastic method to avoid head-on collisions.
            #   A better solution would probably result from re-examining the following:
            #       Guo (2014), which proposed an additional sliding force for head-on approaches;
            #       Lee et al. (2016), which proposed additional forces for following and head-on collision avoidance;
            #       Chen et al. (2018), a review paper which referenced the above papers and others.
            #
            rel_spd, dv_dirn = smnd.calc_polar_decomp(dv)
            if rel_spd**2 and np.dot(dv_dirn, dirn0) > 0.:
                side_dirn = smnd.rotate_90ACW(self.dirn)
                perpendicularity = np.dot(dirn0, side_dirn)
                if perpendicularity**2 < smnd.EPS_m_sqr:
                    #
                    # Head-on collision avoidance
                    #
                    I_sidewards = 0.5 * smnd.calc_norm(I_soc) * (ss.beta.rvs(2.5, 2.) * 2 - 1) * side_dirn
                    I_soc += I_sidewards

        cosine      = np.dot(dirn0, self.dirn)
        vis_factor  = 0.5 + 0.25 * (1 + cosine)
        I_soc      *= vis_factor

        return I_soc

    #
    # Devised to try to get the best of both worlds:
    #   reactive circular repulsion over short timescales; and
    #   proactive elliptical repulsion over longer timescales,
    # with a continuous transition between the two.
    #
    # @todo: Consider a cubic blending function, e.g.:
    #   [(1-t)^3 + 3t(1-t)]I1 + [3t^2(1-t) + t^3]I2
    # would yield a C^1 function of t
    #
    def calc_semireactive_interpedestrian_repulsion(self, ctragent):
        # print("Pedestrian::calc_interpedestrian_repulsion".format(self.id, ctragent.id))

        dx     = ctragent.posn - self.posn
        dx_sqr = np.dot(dx, dx)
        s_max = 2.
        sum_radii = self.radius + ctragent.radius
        if dx_sqr > (s_max + sum_radii)**2:
            return np.zeros(2)

        d0, dirn0 = smnd.calc_polar_decomp(dx)
        # print("".format())

        dv    = ctragent.v - self.v
        delta = dx + self.model.T_react * dv
        d1, dirn1 = smnd.calc_polar_decomp(delta)
        # sum_radii = self.radius + ctragent.radius

        T_react = self.model.T_react
        b = 0.5 * np.sqrt((d0 + d1)**2 - T_react**2 * smnd.calc_norm_sqr(dv))

        #
        # Blended circular and elliptical SFMs
        #
        t_min = -np.dot(dx, dv) / smnd.calc_norm_sqr(dv)
        if t_min <= 0: # self.model.T_react:
            # I_soc -= smf.calc_repulsion(b, 0.04, 3.22) * ((d0 + d1)/ (2 * b)) * 0.5 * (dirn0 + dirn1)
            I_soc = smf.calc_interagent_repulsion(d0 - sum_radii, self.agent_type, ctragent.agent_type) * -dirn0
        elif t_min >= T_react: # else
            I_soc = smf.calc_repulsion(b, 0.04, 3.22) * ((d0 + d1)/ (2 * b)) * 0.5 * -(dirn0 + dirn1)
        else:
            I1 = smf.calc_interagent_repulsion(d0 - sum_radii, self.agent_type, ctragent.agent_type) * -dirn0
            I2 = smf.calc_repulsion(b, 0.04, 3.22) * ((d0 + d1)/ 2 * b ) * 0.5 * -(dirn0 + dirn1)
            I_soc = I1 + (t_min / T_react) * (I2 - I1)

        #
        # Note that the visibility factor is based on the current facing and current relative position.
        #
        cosine      = np.dot(dirn0, self.dirn)
        vis_factor  = 0.5 + 0.25 * (1 + cosine)
        I_soc      *= vis_factor

        return I_soc

    #
    # Devised to try to get the best of both worlds:
    #   proactive elliptical repulsion over shorter timescales,
    #   proactive circular repulsion over longer timescales; and
    # with a continuous transition between the two.
    #
    # @todo: Consider a cubic blending function, e.g.:
    #   [(1-t)^3 + 3t(1-t)]I1 + [3t^2(1-t) + t^3]I2
    # would yield a C^1 function of t
    #
    def calc_semiproactive_interpedestrian_repulsion(self, ctragent):
        # print("Pedestrian::calc_interpedestrian_repulsion".format(self.id, ctragent.id))

        dx     = ctragent.posn - self.posn
        dx_sqr = np.dot(dx, dx)
        s_max = 2.
        sum_radii = self.radius + ctragent.radius
        if dx_sqr > (s_max + sum_radii)**2:
            return np.zeros(2)

        d0, dirn0 = smnd.calc_polar_decomp(dx)
        # print("".format())

        dv    = ctragent.v - self.v
        delta = dx + self.model.T_react * dv
        d1, dirn1 = smnd.calc_polar_decomp(delta)
        # sum_radii = self.radius + ctragent.radius

        T_react = self.model.T_react
        b = 0.5 * np.sqrt((d0 + d1)**2 - T_react**2 * smnd.calc_norm_sqr(dv))

        #
        # Blended circular and elliptical SFMs
        #
        t_min = -np.dot(dx, dv) / smnd.calc_norm_sqr(dv)
        if t_min <= 0: # self.model.T_react:
            # I_soc -= smf.calc_repulsion(b, 0.04, 3.22) * ((d0 + d1)/ (2 * b)) * 0.5 * (dirn0 + dirn1)
            I_soc = smf.calc_repulsion(b, 0.04, 3.22) * ((d0 + d1)/ (2 * b)) * 0.5 * -(dirn0 + dirn1)
        elif t_min >= T_react: # else
            I_soc = smf.calc_interagent_repulsion(d1 - sum_radii, self.agent_type, ctragent.agent_type) * -dirn1
        else:
            I1 = smf.calc_repulsion(b, 0.04, 3.22) * ((d0 + d1)/ 2 * b ) * 0.5 * -(dirn0 + dirn1)
            I2 = smf.calc_interagent_repulsion(d1 - sum_radii, self.agent_type, ctragent.agent_type) * -dirn1
            I_soc = I1 + (t_min / T_react) * (I2 - I1)

        #
        # Note that the visibility factor is based on the current facing and current relative position.
        #
        cosine      = np.dot(dirn0, self.dirn)
        vis_factor  = 0.5 + 0.25 * (1 + cosine)
        I_soc      *= vis_factor

        return I_soc

    #
    # Devised to try to get the best of both worlds:
    #   proactive elliptical repulsion over shorter timescales,
    #   proactive circular repulsion over longer timescales, attenuated by the timescale; and
    # with a continuous transition between the two.
    #
    # @todo: Consider a cubic blending function, e.g.:
    #   [(1-t)^3 + 3t(1-t)]I1 + [3t^2(1-t) + t^3]I2
    # would yield a C^1 function of t
    #
    def calc_attenuated_semiproactive_interpedestrian_repulsion(self, ctragent):
        # print("Pedestrian::calc_interpedestrian_repulsion".format(self.id, ctragent.id))

        dx     = ctragent.posn - self.posn
        dx_sqr = np.dot(dx, dx)
        s_max = 2.
        sum_radii = self.radius + ctragent.radius
        if dx_sqr > (s_max + sum_radii)**2:
            return np.zeros(2)

        d0, dirn0 = smnd.calc_polar_decomp(dx)
        # print("".format())

        dv    = ctragent.v - self.v
        delta = dx + self.model.T_react * dv
        d1, dirn1 = smnd.calc_polar_decomp(delta)
        # sum_radii = self.radius + ctragent.radius

        T_react = self.model.T_react
        b = 0.5 * np.sqrt((d0 + d1)**2 - T_react**2 * smnd.calc_norm_sqr(dv))

        #
        # Blended circular and elliptical SFMs
        #
        t_min = -np.dot(dx, dv) / smnd.calc_norm_sqr(dv)
        if t_min <= 0: # self.model.T_react:
            # I_soc -= smf.calc_repulsion(b, 0.04, 3.22) * ((d0 + d1)/ (2 * b)) * 0.5 * (dirn0 + dirn1)
            I_soc = smf.calc_repulsion(b, 0.04, 3.22) * ((d0 + d1)/ (2 * b)) * 0.5 * -(dirn0 + dirn1)
        elif t_min >= T_react: # else
            I_soc = smf.calc_interagent_repulsion(d1 - sum_radii, self.agent_type, ctragent.agent_type) * -dirn1

            #
            # Attenuation for t_min > T_react.
            #
            # @todo: Consider subtracting 2 ((t_min - T_react)/ t_min**2) dv from I_soc, i.e.
            #   I_soc *= attenuation_factor**2 - 2 * ((t_min - T_react)/ t_min**2) * dv
            #
            if t_min >= 2 * T_react:
                attenuation_factor = T_react / t_min
                I_soc *= attenuation_factor**2
        else:
            I1 = smf.calc_repulsion(b, 0.04, 3.22) * ((d0 + d1)/ 2 * b ) * 0.5 * -(dirn0 + dirn1)
            I2 = smf.calc_interagent_repulsion(d1 - sum_radii, self.agent_type, ctragent.agent_type) * -dirn1
            
            blending_factor = t_min / T_react
            I_soc           = I1 + blending_factor * (I2 - I1)

        #
        # Note that the visibility factor is based on the current facing and current relative position.
        #
        cosine      = np.dot(dirn0, self.dirn)
        vis_factor  = 0.5 + 0.25 * (1 + cosine)
        I_soc      *= vis_factor

        return I_soc

    def calc_interpedestrian_repulsion(self, ctragent):
        return self.calc_elliptical_interpedestrian_repulsion(ctragent)
    
    def calc_contravehicular_repulsion(self, ctrvhcl):
        # print("Pedestrian::calc_contravehicular_repulsion".format(self.id, ctrvhcl.id))

        l_car = ctrvhcl.get_length_axis() # 0.5 * ctrvhcl.length
        w_car = ctrvhcl.get_width_axis() # 0.5 * ctrvhcl.width
        assert(l_car >= w_car)
        e_car = np.sqrt(l_car * l_car - w_car * w_car) / l_car

        #
        # @todo: Use these if they result in cleaner or more optimal code.
        #
        # wsqr  = w_car**2
        # lsqr  = l_car**2


        #
        # @todo: Introduce a max_s into the Model for pedestrian-car interactions?
        #
        ds        = ctrvhcl.posn - self.posn
        d0, dirn0 = smnd.calc_polar_decomp(ds)

        c0 = np.dot(ctrvhcl.dirn, -dirn0)
        r0 = w_car / np.sqrt(1 - (c0 * e_car)**2)
        assert(w_car <= r0)
        assert(r0 <= l_car)

        sum0_radii = self.radius + r0


        dv = ctrvhcl.v - self.v
        delta       = ds + self.model.T_react * dv
        d1, dirn1 = smnd.calc_polar_decomp(delta)
        
        c1 = np.dot(ctrvhcl.dirn, -dirn1)
        r1 = w_car / np.sqrt(1 - (c1 * e_car)**2)
        assert(w_car <= r1)
        assert(r1 <= l_car)

        sum1_radii = self.radius + r1

        I_soc  = np.zeros(2)
        # I_soc += smf.calc_interagent_repulsion(d0 - sum0_radii, self.agent_type, ctrvhcl.agent_type) * -dirn0
        I_soc += smf.calc_interagent_repulsion(d1 - sum1_radii, self.agent_type, ctrvhcl.agent_type) * -dirn1

        #
        # Attenuate by visibility:
        #     Note that c0, the initial cosine angle, is used rather than c1, the future cosine angle.
        #
        lambda_pc   = smf.lambda_mu[smad.AgentType.PEDESTRIAN, ctrvhcl.agent_type]
        # vis_factor  = lambda_pc + (1 - lambda_pc) * 0.5 * (1 + c0)
        vis_factor  = lambda_pc + (1 - lambda_pc) * 0.5 * (1 + np.dot(self.dirn, dirn0))
        I_soc *= vis_factor
        return I_soc

    def calc_interagent_influence(self, ctragent):
        # print("Pedestrian::calc_interagent_influence".format(self.id, ctragent.id))
        I_soc = np.zeros(2)
        ctrtype = ctragent.agent_type
        if ctrtype == smad.AgentType.PEDESTRIAN:
           I_soc = self.calc_interpedestrian_repulsion(ctragent)
        elif ctrtype == smad.AgentType.CAR:
            I_soc = self.calc_contravehicular_repulsion(ctragent)
        return I_soc

    # def clip_velocity(self):
        # v_norm, unit_v     = smnd.calc_polar_decomp(self.v)
        # self.dirn  = unit_v
        # if v_norm > self.v_max:
            # self.v = self.v_max * self.dirn
        # return

    def init_best_possible_stop(self, ctragent, interaction, t0):
        # assert(False)
        
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

        # T_brk = self.T_relax
            
        a = 2 * np.dot(x_ctragent_future - x_self_future, self.dirn) / (T_curve - self.model.T_react)**2
        if a * self.T_relax < -self.v_max:
            #
            # Cannot stop in time but can decelerate as much as possible to reduce injury and improve chance of
            # cooperation sufficing to avoid incident.
            #
            
            a = -self.v_ped / self.T_relax
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
        v = self.v + I * smc.T_step
        # print("Ptn::clip_intrinsic_force - {} has velocity {} and wants velocity {}".format(self.id, self.v, v))

        # max_angular_velocity = 0.25 * np.pi
        # cos_max_delta_angle = np.cos(smc.T_step * max_angular_velocity)
        # sin_max_delta_angle = np.sin(smc.T_step * max_angular_velocity)

        cos_max_delta_angle = smad.cos_theta_max[smad.AgentType.PEDESTRIAN]
        AV_is_excessive = np.dot(v, self.dirn) ** 2 < smnd.calc_norm_sqr(v) * cos_max_delta_angle ** 2
        LV_is_excessive = smnd.calc_norm_sqr(v) > self.v_ped ** 2
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
            print("Ptn::clip_intrinsic_force - {} has excessive desired AV whose cos is {} < {}".format(self.id,
                                                                                                        np.dot(new_dirn,
                                                                                                               self.dirn),
                                                                                                        cos_max_delta_angle))
            perp_dirn = smnd.rotate_90ACW(self.dirn)
            if np.dot(perp_dirn, new_dirn) < 0:
                perp_dirn = -perp_dirn
            sin_max_delta_angle = smad.sin_theta_max[smad.AgentType.PEDESTRIAN]
            new_dirn = cos_max_delta_angle * self.dirn + sin_max_delta_angle * perp_dirn
            print("Ptn::clip_intrinsic_force - {} has new cos {}, sin {}, dirn {} != {}".format(self.id,
                                                                                                cos_max_delta_angle,
                                                                                                sin_max_delta_angle,
                                                                                                new_dirn, self.dirn))
            # assert(False)

        if LV_is_excessive:
            # print("Ptn::clip_intrinsic_force - {} has excessive desired LV {}".format(self.id, new_spd))

            newer_spd = smnd.clip_magnitude_asymptotic(new_spd, self.v_ped, self.v_max)
            assert (newer_spd <= new_spd)
            # newer_spd = smnd.clip_magnitude_softmax(new_spd, self.v_ped, self.v_max)

            # print("Ptn::clip_intrinsic_force - {} has clipped desired LV (spd) from {} to {}".format(self.id, new_spd, newer_spd))
            new_spd = newer_spd
            # assert(new_spd >= self.v_ped)
            # assert(new_spd <= self.v_max)

        # print("Ptn::clip_intrinsic_force - {} is clipping new velocity from {} to {}".format(self.id, v, new_spd * new_dirn))

        v = new_spd * new_dirn
        I = (v - self.v) / smc.T_step
        return I

    def DEPRECATED_clip_intrinsic_force(self, I):
        v = self.v + I * smc.T_step
        # print("Ptn::clip_intrinsic_force - {} has velocity {} and wants velocity {}".format(self.id, self.v, v))
        
        
        # max_angular_velocity = 0.25 * np.pi
        # cos_max_delta_angle = np.cos(smc.T_step * max_angular_velocity)
        # sin_max_delta_angle = np.sin(smc.T_step * max_angular_velocity)


        cos_max_delta_angle = smad.cos_theta_max[smad.AgentType.PEDESTRIAN]
        AV_is_excessive = np.dot(v, self.dirn)**2 < smnd.calc_norm_sqr(v) * cos_max_delta_angle**2
        LV_is_excessive = smnd.calc_norm_sqr(v) > self.v_ped**2
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
            print("Ptn::clip_intrinsic_force - {} has excessive desired AV whose cos is {} < {}".format(self.id, np.dot(new_dirn, self.dirn), cos_max_delta_angle))
            perp_dirn = smnd.rotate_90ACW(self.dirn)
            if np.dot(perp_dirn, new_dirn) < 0:
                perp_dirn = -perp_dirn
            sin_max_delta_angle = smad.sin_theta_max[smad.AgentType.PEDESTRIAN]
            new_dirn = cos_max_delta_angle * self.dirn + sin_max_delta_angle * perp_dirn
            print("Ptn::clip_intrinsic_force - {} has new cos {}, sin {}, dirn {} != {}".format(self.id, cos_max_delta_angle, sin_max_delta_angle, new_dirn, self.dirn))
            # assert(False)

        if LV_is_excessive:
            # print("Ptn::clip_intrinsic_force - {} has excessive desired LV {}".format(self.id, new_spd))

            newer_spd = smnd.clip_magnitude_asymptotic(new_spd, self.v_ped, self.v_max)
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
            
    def draw_social_aura(self, view):
        side_dirn  = np.array([-self.dirn[1], self.dirn[0]])
        major_axis = 0.5 * (smc.d_safe_IA_before + smc.d_safe_IA_behind) * self.dirn
        minor_axis = smc.d_safe_IA_beside * side_dirn
        metacentre = self.posn + 0.5 * (smc.d_safe_IA_before - smc.d_safe_IA_behind) * self.dirn

        clr = svd.white
        if self.goal_state.state_id == smas.GoalStateId.WAITING:
            clr = svd.light_green
        elif self.goal_state.state_id == smas.GoalStateId.CONVERGING:
            clr = svd.light_blue

        if (smnd.calc_norm_sqr(self.dirn) - 1)**2 >= smnd.EPS_m_sqr:
            view.fill_circle(clr, self.posn, smc.d_safe_IA_beside)
        else:
            view.fill_ellipse(clr, metacentre, major_axis, minor_axis)
        ENHANCEMENT_2025_11_20 = True
        if ENHANCEMENT_2025_11_20:
            if self.conflict_state.state_id == smas.ConflictStateId.VEERING:
                # spd_des, dirn_des = smnd.calc_polar_decomp(self.conflict_state.v_des)
                pts = [self.posn, self.posn + view.pedestrian_upscale_factor * self.v, self.posn + view.pedestrian_upscale_factor * self.conflict_state.v_des]
                view.fill_polygon(svd.red, pts)
            else:
                view.draw_line(svd.blue, self.posn, self.posn + view.pedestrian_upscale_factor * self.v, 0)

        text = "{}".format(self.id)
        # view.draw_text(text, svd.black, self.posn - 0.25 * major_axis - np.array([1, 0]))
        view.draw_text(text, svd.black, metacentre + 0.125 * major_axis + np.array([-0.375, 0.25]), False)

        return

    def draw_physical_area(self, view):
        spd = smnd.calc_norm(self.v)

        clr = svd.dark_blue
        if self.conflict_state.state_id == smas.ConflictStateId.VEERING:
            clr = svd.dark_green
        elif self.conflict_state.state_id == smas.ConflictStateId.YIELDING:
            clr = svd.light_red
        elif self.conflict_state.state_id == smas.ConflictStateId.AWAITING:
            clr = svd.red
        view.fill_circle(clr, self.posn, self.radius)
        # view.fill_circle(self.src_colour, self.posn, self.radius)

        ENHANCEMENT_2025_11_20 = True
        if not ENHANCEMENT_2025_11_20:
            view.draw_line(svd.white, self.posn, self.posn + view.pedestrian_upscale_factor * spd * self.radius * self.dirn, 0)
            if self.conflict_state.state_id == smas.ConflictStateId.VEERING:
                spd_des, dirn_des = smnd.calc_polar_decomp(self.conflict_state.v_des)
                view.draw_line(svd.yellow, self.posn, self.posn + view.pedestrian_upscale_factor * spd_des * self.radius * dirn_des, 0)

        return
