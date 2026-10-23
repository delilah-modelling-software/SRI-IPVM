import numpy as np
# import scipy as sp
# import scipy.stats as ss

import view.defs as svd
import view.view as svv
# import model.defs as smd
import model.configuration as smc
import model.topology.edge as smte
import model.topology.graph as smtg
import model.topology.approachnode as smtApp
import model.topology.activitynode as smtAct
import model.topology.topology as smtt
import model.geometry.grid as smgg
import model.geometry.shape as smgs
import model.agency.defs as smad
import model.agency.state as smas
import model.agency.reaction as smar
import model.agency.agent as smaa
import model.zone.zone as smzz
import model.interaction.codec as smic
import model.interaction.prioritising as smip
import model.interaction.interaction as smii
import model.interaction.assessing as smia
import model.force as smf
import model.numerical.defs as smnd
import model.logger as sml

class SafetyRecord:
    def __init__(self, agent, ctragent):
        self.agent_id    = agent.id
        self.ctragent_id = ctragent.id

        self.agent_src_id = agent.true_src_id
        self.agent_dst_id = agent.true_dst_id
        self.ctragent_src_id = ctragent.true_src_id
        self.ctragent_dst_id = ctragent.true_dst_id

        self.collided = False

        relposn = ctragent.posn - agent.posn
        relv = ctragent.v - agent.v

        combined_id = smic.interagent_codec.encode(self.agent_id, self.ctragent_id)
        relposn, d, e, r_total, s, delta_shape = smia.assess_IA_position(agent, ctragent, combined_id, agent.model.cur_timestep_idx)
        relv = ctragent.v - agent.v
        if s < 0.:
            self.collided = True
            self.TTC = self.PET1 = self.PET2 = 0.
        else:
            perp_dirn_alpha = smnd.rotate_90ACW(agent.dirn)
            perp_dirn_beta = smnd.rotate_90ACW(ctragent.dirn)

            y_alpha = np.dot(relposn, perp_dirn_beta)
            y_beta = np.dot(relposn, perp_dirn_alpha)
            v_alpha = np.dot(relv, perp_dirn_beta)
            v_beta = np.dot(relv, perp_dirn_alpha)

            l_sum = agent.get_length_axis() + ctragent.get_length_axis()

            self.TTC = None
            self.PET1 = None
            self.PET2 = None
            if v_alpha ** 2 < smnd.EPS_m_sqr or v_beta ** 2 < smnd.EPS_m_sqr:
                #
                # Parallel
                #
                w_sum = agent.get_width_axis() + ctragent.get_width_axis()
                if y_alpha ** 2 < w_sum ** 2 or y_beta ** 2 < w_sum ** 2:

                    T_pass = smip.compute_passing_interval(agent, ctragent)

                    if T_pass != None and T_pass[0] >= 0.:
                        self.TTC = T_pass[0]

                    if np.dot(relposn, agent.dirn) >= 0.:
                        self.PET1 = (np.dot(relposn, agent.dirn) - l_sum) / np.dot(agent.v, agent.dirn)
                    if np.dot(-relposn, ctragent.dirn) >= l_sum:
                        self.PET2 = (np.dot(-relposn, ctragent.dirn) - l_sum) / np.dot(ctragent.v, ctragent.dirn)
            else:
                #
                # Non-Parallel
                #
                l_alpha_cross = agent.get_length_axis() + ctragent.get_width_axis() + 2 * smc.d_safe_IA_beside
                l_beta_cross = ctragent.get_length_axis() + agent.get_width_axis() + 2 * smc.d_safe_IA_beside

                T_cross_alpha = smip.compute_crossing_interval(y_alpha, v_alpha, l_alpha_cross)
                T_cross_beta = smip.compute_crossing_interval(y_beta, v_beta, l_beta_cross)
                T_conflict = [max(T_cross_alpha[0], T_cross_beta[0]), min(T_cross_alpha[1], T_cross_beta[1])]

                if T_conflict[0] <= T_conflict[1] and T_conflict[0] >= 0.:
                    self.TTC = T_conflict[0]

                PET1 = T_cross_beta[1] - T_cross_alpha[0]
                if PET1 >= 0.:
                    self.PET1 = PET1
                PET2 = T_cross_alpha[1] - T_cross_beta[0]
                if PET2 >= 0.:
                    self.PET2 = PET2

        self.TCE = None
        self.DCE = None
        tce = -np.dot(relv, relposn) / smnd.calc_norm_sqr(relv)
        if tce >= 0.:
            self.TCE = tce

            relposn1 = relposn + self.TCE * relv
            self.DCE, e1 = smnd.calc_polar_decomp(relposn1)

        self.Spd1     = None
        self.Spd2     = None
        self.DSpd     = None
        self.CrashDv1 = None
        self.CrashDv2 = None
        if self.TTC != None and self.TTC >= 0. or self.PET1 != None and self.PET1 >= 0. or self.PET2 != None and self.PET2 >= 0.:
            self.Spd1 = smnd.calc_norm(agent.v)
            self.Spd2 = smnd.calc_norm(ctragent.v)
            self.DSpd = smnd.calc_norm(ctragent.v - agent.v)

            v_postcrash = (agent.mass * agent.v + ctragent.mass * ctragent.v) / (agent.mass + ctragent.mass)
            self.CrashDv1 = smnd.calc_norm(v_postcrash - agent.v)
            self.CrashDv2 = smnd.calc_norm(v_postcrash - ctragent.v)

        return
    def is_nontrivial(self):
        if self.TCE != None or self.DCE != None:
            return True

        if self.TTC != None or self.PET1 != None or self.PET2 != None:
            return True

        if self.Spd1 != None or self.Spd2 != None or self.DSpd != None:
            return True

        if self.collided == True or self.CrashDv1 != None or self.CrashDv2 != None:
            return True

        return False

    def get_desc_list(self):
        return [self.agent_id, self.agent_src_id, self.agent_dst_id, self.ctragent_id, self.ctragent_src_id, self.ctragent_dst_id, self.collided, self.TCE, self.DCE, self.TTC, self.PET1, self.PET2, self.Spd1, self.Spd2, self.DSpd, self.CrashDv1, self.CrashDv2]


class SafetyHistory:
    def __init__(self):
        self.registry = {}
        return

    def register_safety_record(self, safety_record):
        combined_id = smic.interagent_codec.encode(safety_record.agent_id, safety_record.ctragent_id)

        if not safety_record.is_nontrivial():
            return

        if combined_id in self.registry:
            existing_record = self.registry[combined_id]

            if not existing_record.is_nontrivial():
                self.registry[combined_id] = safety_record
                return

            if existing_record.TCE == None or safety_record.TCE != None and safety_record.TCE < existing_record.TCE:
                existing_record.TCE = safety_record.TCE
            if existing_record.DCE == None or safety_record.DCE != None and safety_record.DCE < existing_record.DCE:
                existing_record.DCE = safety_record.DCE

            if existing_record.TTC == None or safety_record.TTC != None and safety_record.TTC < existing_record.TTC:
                existing_record.TTC = safety_record.TTC
            if existing_record.PET1 == None or safety_record.PET1 != None and safety_record.PET1 < existing_record.PET1:
                existing_record.PET1 = safety_record.PET1
            if existing_record.PET2 == None or safety_record.PET2 != None and safety_record.PET2 < existing_record.PET2:
                existing_record.PET2 = safety_record.PET2

            if existing_record.Spd1 == None or safety_record.Spd1 != None and safety_record.Spd1 > existing_record.Spd1:
                existing_record.Spd1 = safety_record.Spd1
            if existing_record.Spd2 == None or safety_record.Spd2 != None and safety_record.Spd2 > existing_record.Spd2:
                existing_record.Spd2 = safety_record.Spd2
            if existing_record.DSpd == None or safety_record.DSpd != None and safety_record.DSpd > existing_record.DSpd:
                existing_record.DSpd = safety_record.DSpd
            if existing_record.CrashDv1 == None or safety_record.CrashDv1 != None and safety_record.CrashDv1 > existing_record.CrashDv1:
                existing_record.CrashDv1 = safety_record.CrashDv1
            if existing_record.CrashDv2 == None or safety_record.CrashDv2 != None and safety_record.CrashDv2 > existing_record.CrashDv2:
                existing_record.CrashDv2 = safety_record.CrashDv2

            self.registry[combined_id] = existing_record
        else:
            self.registry[combined_id] = safety_record
