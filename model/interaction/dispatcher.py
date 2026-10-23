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
import model.interaction.history as smih
import model.interaction.interaction as smii
import model.interaction.assessing as smia
import model.force as smf
import model.numerical.defs as smnd
import model.logger as sml

class Dispatcher:
    def __init__(self, model):
        self.model                 = model
        self.agent_interaction_set = set()
        self.zone_interaction_set  = set()
        self.safety_history        = smih.SafetyHistory()
        return

    def draw_interagent_interaction(self, agent, ctragent, view):
        assert(np.dot(agent.posn, agent.posn) > smnd.EPS_m_sqr)
        assert(np.dot(ctragent.posn, ctragent.posn) > smnd.EPS_m_sqr)
        
        thickness = 0.06125

        if agent.agent_type != ctragent.agent_type:
            colour = svd.red
        else:
            colour = svd.white
            if agent.agent_type == smad.AgentType.CAR:
                u = smnd.calc_unit_vector(ctragent.posn - agent.posn)
                c_self  = np.dot(agent.dirn, u)
                c_other = np.dot(ctragent.dirn, u)
                p = c_self >= smc.cos_IAI_QP and c_other >= smc.cos_IAI_QP
                if p:
                    colour = svd.blue

        centre         = 0.5 * (agent.posn + ctragent.posn)
        bar_dirn       = smnd.rotate_90ACW(smnd.calc_unit_vector(ctragent.posn - agent.posn))
        bar_halflength = 2 * thickness
        bar_vector     = bar_halflength * bar_dirn

        dot_radius = bar_halflength
        dot1_posn = 0.5 * (agent.posn + centre)
        dot2_posn = 0.5 * (ctragent.posn + centre)
        
        view.draw_line(colour, agent.posn, ctragent.posn, thickness)
        view.draw_line(colour, centre - bar_vector, centre + bar_vector, thickness)
        if ctragent.id in agent.interagent_interactions:
            view.fill_circle(colour, dot1_posn, dot_radius)
            
            iai, reaction = agent.interagent_interactions[ctragent.id]
            if iai.relv_cone != None:
                assert(iai.relv_cone != None)
                assert(np.dot(iai.relv_cone.apex, iai.relv_cone.apex) <= smnd.EPS_m_sqr)
                iai.relv_cone.draw_parts(agent.posn, svd.cyan, svd.red, view)

        if agent.id in ctragent.interagent_interactions:
            view.fill_circle(colour, dot2_posn, dot_radius)
        return

    def draw_contrazone_interaction(self, agent, zone, view):
        thickness = 0.06125
        x_omega, d, nv, tv = zone.shape.get_nearest_point_info(agent.posn)
        view.draw_line(svv.silver, agent.posn, x_omega, thickness)

        assert(zone.id in agent.contrazone_interactions)
        czi = agent.contrazone_interactions[zone.id]
        if czi.relv_cone != None:
            assert(czi.relv_cone != None)
            assert(np.dot(czi.relv_cone.apex, czi.relv_cone.apex) <= smnd.EPS_m_sqr)
            czi.relv_cone.draw(SVD.magenta, SVD.green, view)
        
        return

    def draw(self, active_agents, zone_config, view):
        if view.layer_flags & svd.LayerFlagset.INTERACTION:
            for IAI_key in self.agent_interaction_set:
                id1, id2 = smic.interagent_codec.decode(IAI_key)
                agent    = active_agents[id1]
                ctragent = active_agents[id2]
                if ctragent.id in agent.interagent_interactions or agent.id in ctragent.interagent_interactions:
                    if view.layer_flags & svd.LayerFlagset.INTRAMODAL and agent.agent_type == ctragent.agent_type:
                        self.draw_interagent_interaction(agent, ctragent, view)
                    elif view.layer_flags & svd.LayerFlagset.INTERMODAL and agent.agent_type != ctragent.agent_type:
                        self.draw_interagent_interaction(agent, ctragent, view)
            if view.layer_flags & svd.LayerFlagset.CONTRAZONE:
                for CZI_key in self.zone_interaction_set:
                    agent_id, zone_id = smic.contrazone_codec.decode(CZI_key)
                    agent = active_agents[agent_id]
                    zone  = zone_config.zone_tbl[zone_id]
                    if zone.id in agent.contrazone_interactions:
                        self.draw_contrazone_interaction(agent, zone, view)

        return

    def register_all_mboxen(self, agent_metaset, active_agents, grid):
        # print("Model::register_all_mboxen: begin")
        for type in agent_metaset:
            agent_idset = agent_metaset[type]

            for id in agent_idset:
                agent = active_agents[id]

                # assert(agent.goal_state.state_id != smas.GoalStateId.TURNING)

                #
                # Clear agent's interaction-registries.
                #
                agent.interagent_interactions = {}
                agent.contrazone_interactions = {} # set()

                
                #
                # @todo: If and when Agent::truncate_horizon() is implemented before interaction-determination,
                # calc the personal horizon instead
                #
                # horizon = self.horizon
                # horizon = agent.calc_personal_horizon(self.exclusion_zones[type])

                mboxen = agent.calc_motion_bboxen(smc.T_master_horizon)
                print("Dispatcher::register_all_mboxen agent {} w/ posn {}, vel {}".format(agent.id, agent.posn, agent.v))
                for mbox in mboxen:
                    qbox = grid.quantise_bbox(mbox, expansion = 1)
                    grid.register_agent_coverage(id, qbox)
        return        

    def prepare_interactions(self, agent_metaset, active_agents, grid):
        # print("Model::determine_all_interactions: begin")
        self.register_all_mboxen(agent_metaset, active_agents, grid)

        #
        # Note: I considered adding a set of older interactions and excluding its members
        # from consideration, but concluded that putting that logic, along with
        # horizon-truncation, into the Agent class would be better (probably not performance-wise)!
        #
        # Each interaction that passes
        #    the Proximity+Convergence (), Classification, and Awareness+Urgency tests in this method should be reported to aware affected agents
        #    with a timestep and the results of these tests. Each agent will then be able to apply private information:
        #        e.g. whether the agent will be rerouting/veering before the conflict time and whether a conflict occurs in an
        #            access zone that excludes either or both agents or prioritises one of the agents.
        #    These access zones could be considered earlier as a performance optimisation.
        #
        #    Access Zones? Physical obstacles (e.g. walls, bollards, etc.) can probably be used to
        #    improve prediction or truncate agent horizons during one of the shared/common knowledge tests and could also
        #    be used for sociolegal obstacles, but only the agent can truly know whether they will respect a particular sociolegal
        #    obstacle in a given situation.
        #
        
        self.agent_interaction_set, self.zone_interaction_set = grid.construct_possible_interaction_sets()
        return

    def dispatch_contrazone_interaction(self, agent, zone, CZI_key, cur_timestep_idx, logger):
        #
        # @todo: Add restrictions to zones.
        #   Pseudocode for method would be:
        #       if not zone.excludes(agent) and not zone.restricts(agent):
        #           return
        #
        #       ... = zone.shape.get_nearest_point_info(agent)
        #       if s < 0:
        #           if zone.is_physical():
        #               agent.register_collision(zone, s, nrml_dirn, tan_dirn)
        #           [elif zone.excludes(agent):
        #               agent.register_conflict(zone, s, nrml_dirn, tan_dirn)]
        #           elif zone.restricts(agent):
        #               agent.register_restriction(zone)
        #           return
        #
        #       #
        #       # Assume unfamiliarity, but subdividing road users into those familiar and unfamiliar with the area
        #       # and assuming awareness only for those already familiar with the area would be better.
        #       #
        #
        #
        #       ... = zone.shape.get_nearest_approch_info(agent)
        #       if s_min < d_safe_CZ and zone.excludes(agent):
        #           agent.register_conflict(zone, s, nrml_dirn, tan_dirn)
        #           return
        #
        #       if s_min < 0 and zone.restricts(agent):
        #           agent.register_restriction(zone)
        #           return
        #
        #       return
        #
        
        #
        # @todo: If the zone excludes other types but not the agent's type,
        # consider adding to a set of potential safe-zones for agent to allow
        # safe-zone seeking [Johora+Mueller 2018, 2020].
        #
        if not zone.excludes(agent):
            #
            # Since zone does not exclude agent, agent does not need to avoid it.
            #
            return

        #
        # Positional assessment
        #
        print("Dispatcher::dispatch_contrazone_interaction - beginning positional assessment of agent {} vs zone {}".format(agent.id, zone.id))
        
        l_alpha     = agent.get_length_axis()
        # r_zeta      = zone.shape.radius
        # d_safe_CZ   = smc.d_safe_CZ

        x0_zeta, d0, nv0, tv0 = zone.shape.get_nearest_point_info(agent.posn)
        rel_posn0 = x0_zeta - agent.posn
        e_CZ   = -nv0

        r_alpha_now = agent.get_directional_radius(e_CZ)
        r_zeta      = zone.shape.radius

        r_total_now = r_alpha_now + r_zeta
        s0 = d0 - r_total_now
        if s0 < 0 and zone.is_physical():
            #
            # Register as impact/collision, ignore social interactions.
            #
            agent.register_impact(zone, s0, e_CZ)
            return

        near = (s0 < smc.d_safe_CZ)

        print("Dispatcher::dispatch_contrazone_interaction - beginning perceptual assessment of agent {} vs zone {}".format(agent.id, zone.id))

        #
        # 
        #
        agent_circle = smgs.Circle(agent.posn, agent.radius)
        # posn_cone    = smgs.TruncatedCone(zone.shape, agent_circle, 1., True)
        print("Dispatcher::dispatch_contrazone_interaction {}".format(zone.shape))
        vis_cone = smgs.construct_tcone(agent_circle, zone.shape, 1., True)
        if vis_cone != None:
            print("Dispatcher::dispatch_contrazone_interaction - cone for {} v {} has apex {} bounded by {} and {}".format(agent.id, zone.id, vis_cone.apex, vis_cone.e_left, vis_cone.e_right))
            print("Dispatcher::dispatch_contrazone_interaction - agent {} has posn {}, velocity {}, and facing {}".format(agent.id, agent.posn, agent.v, agent.dirn))   

        #
        # Perceptual assessment - visibility, [audibility, familiarity]
        #

        #
        # Visibility
        #

        # cos_gma = agent.get_cos_gma(smnd.calc_norm(agent.v))
        cos_gma = 0

        vis_alpha = zone.shape.contains(agent.posn)
        if not vis_alpha and vis_cone != None:
            agent_dirn_perp = smnd.rotate_90ACW(agent.dirn)
            focused   = np.dot(agent_dirn_perp, vis_cone.e_left) >= 0 and np.dot(agent_dirn_perp, vis_cone.e_right) <= 0
            # focused   = np.dot(agent.dirn, -smnd.rotate_90ACW(posn_cone.e_left)) > 0 and np.dot(agent.dirn, smnd.rotate_90ACW(posn_cone.e_right)) > 0
            peripheral = False
            if not focused:
                peripheral = np.dot(agent.dirn, vis_cone.e_left) >= cos_gma or np.dot(agent.dirn, vis_cone.e_right) >= cos_gma

            vis_alpha = focused or peripheral
        
        # cos_left  = np.dot(agent.dirn, posn_cone.e_left)
        # cos_right = np.dot(agent.dirn, posn_cone.e_right)
        # cos_min   = min(cos_left, cos_right)
        # cos_max   = max(cos_left, cos_right)
        # cos_CC = np.dot(CC.left_dirn, CC.right_dirn)
        
        # vis_alpha   = (cos_max >= cos_gma or cos_min >= cos_CC)
        aware_alpha = vis_alpha
        if not aware_alpha:
            return
        

        #
        # Velocicentric assessment
        #

        rel_v    = -agent.v

        #
        # @todo Consider potential convergence too, e.g.
        #   cur_dst_idx = self.route[self.routing_idx + 1]
        #   cur_dst     = self.nodegraph.nodes[cur_dst_idx]
        #   x_omega, dist, nv, tv = cur_dst.approach_shape.get_nearest_point_info(self.posn)
        #   v_des           = self.get_target_speed() * smnd.calc_unit_vector(x_omega - self.posn)
        #   potentially_converging = np.dot(rel_posn0, -v_des) < 0
        #
        converging = (np.dot(rel_posn0, rel_v) < 0)
        if not near and not converging:
            #
            # Ignore due to irrelevance
            #
            return        


        print("Dispatcher::dispatch_contrazone_interaction - beginning velocicentric assessment of agent {} vs zone {}".format(agent.id, zone.id))

        x1_zeta, x1_alpha, d1, nv1, tv1, t_min = zone.shape.get_nearest_approach_info(agent.posn, agent.v, smc.T_master_horizon)

        e_CZ_ftr   = -nv1

        r_alpha_ftr = agent.get_directional_radius(e_CZ_ftr)
        r_total_ftr = r_alpha_ftr + r_zeta
        s1 = d1 - r_total_ftr

        if s1 > smc.d_safe_CZ:
            #
            # Ignore due to lack of concern.
            #
            return
        elif s1 >= 0:
            posn_cone = vis_cone # smgs.construct_tcone(agent_circle, zone.shape, 1., True)
            czi = smii.ContrazoneInteraction(CZI_key, agent, zone, posn_cone, t_min)
            agent.register_contrazone_interaction(zone.id, czi)
            return

        assert(s1 < 0)

        #
        # Compute speed of convergence, use it to determine a better, safer deadline for the agent's conflict avoidance to be executed.
        #
        # spd_cnv     = np.dot(agent.v, e_CZ_ftr)
        # t_alpha_min = t_min + s1 / spd_cnv
        relv_cone   = vis_cone
        # czi         = smii.ContrazoneInteraction(CZI_key, agent, zone, relv_cone, t_alpha_min)
        # agent.register_contrazone_conflict(zone.id, czi, t_alpha_min)
        czi         = smii.ContrazoneInteraction(CZI_key, agent, zone, relv_cone, t_min)
        agent.register_contrazone_conflict(zone.id, czi, t_min)

        return
    
    def dispatch_interagent_interaction(self, agent, ctragent, IAI_key, cur_timestep_idx, logger):
        #
        # @todo: Perform a three-stage analysis:
        #    Proximity+Convergence:
        #        NEAR/FAR (norm(rel_posn) - (l_\alpha + l_\beta) < 2.)
        #        CONVERGING/NONCONVERGING (rel_posn \dot rel_v < 0)
        #    Classification:
        #        ANTIPARALLEL (v_alpha \dot v_beta \approx -1)
        #        PARALLEL (v_alpha \dot v_beta \approx 1)
        #            Will need to use d(t) = norm(rel_posn) - (l_\alpha + l_beta)
        #            Antiparallel will need lateral deviation support.
        #            Parallel will need CFM-like and possibly LCM-like (lateral deviation) support.
        #        CROSSING (-1 << v_alpha \dot v_beta << 1)
        #            Will need to use an Airplane/RVO-like criterion.
        #    Awareness+Urgency
        #        AWARE/UNAWARE (rel_posn \dot v_\alpha, rel_posn \dot v_\alpha)
        #            MUTUALLY UNAWARE
        #            SOLELY AWARE
        #            MUTUALLY AWARE
        #        SRCA(<= 0.5s)/MRCA (<= 2s)/LRCA(<= 4s)/ULRCA(> 4s)
        #           SRCA                 => SFM
        #           MRCA                 => Deviation/Deceleration
        #           LRCA+!MUTUALLY_AWARE => Deviation+Deceleration
        #           LRCA+MUTUALLY_AWARE  => GSFM => Deviation+Acceleration/Deceleration
        #           ULRCA                => Nothing (yet!)
        #

        # print("Dispatcher::dispatch_interagent_interation - beginning positional assessment of agent {} vs ctragent {}".format(agent.id, ctragent.id))

        #
        # Positional assessment
        #
        relposn, d, e, r_total, s, delta_shape = smia.assess_IA_position(agent, ctragent, IAI_key, cur_timestep_idx)
        relv = ctragent.v - agent.v
        if s < 0.:
            if True:
                momentum_factor_alpha = momentum_factor_beta = 1
                if agent.mass != ctragent.mass:
                    #
                    # rcp_dnm   Reciprocal of the common denominator of the momentum factors
                    #
                    # Note: mf_alpha + mf_beta = 2
                    #
                    rcp_dnm = 1 / (agent.mass + ctragent.mass)
                    momentum_factor_alpha = 2 * ctragent.mass * rcp_dnm
                    momentum_factor_beta  = 2 * agent.mass    * rcp_dnm
                tngnt = smnd.rotate_90ACW(e)
                agent.register_interagent_collision(ctragent, s, relv, -e, -tngnt, momentum_factor_alpha)
                ctragent.register_interagent_collision(agent, s, -relv, e, tngnt, momentum_factor_beta)

            if False: # logger != None:
                types       = [agent.agent_type, ctragent.agent_type]
                importance  = (True, None, True) # (near, converging?, impacted)
                
                interaction = smii.InteragentInteraction(IAI_key, types, importance, 0, relposn, None, None, None, None, None, None, None)
                logger.log_interagent_interaction(cur_timestep_idx, interaction)
            return

        # print("Dispatcher::dispatch_interagent_interation - beginning perceptual assessment of agent {} vs ctragent {}".format(agent.id, ctragent.id))

        #
        # SSAM (Surrogate Safety Measurement Computation)
        #   Deceleration Rates are not currently computed.
        #
        ENHANCEMENT_SSAM_COMPUTATION = True
        if ENHANCEMENT_SSAM_COMPUTATION:
            safety_record = smih.SafetyRecord(agent, ctragent)

            #
            #
            #
            self.safety_history.register_safety_record(safety_record)

        #
        # Perceptual assessment
        #
        proactive_alpha = ctragent.id in agent.interagent_interactions
        proactive_beta  = agent.id    in ctragent.interagent_interactions

        aud_alpha           = s < smad.d_audio[ctragent.agent_type]
        aud_beta            = s < smad.d_audio[agent.agent_type]
        vis_alpha, vis_beta = smia.assess_IA_visibility(agent, ctragent, s)

        aware_alpha = True
        aware_beta  = True
        if not smc.OMNISCIENCE_ENABLED:
            aware_alpha = aud_alpha or vis_alpha or proactive_alpha
            aware_beta  = aud_beta  or vis_beta  or proactive_beta

        #
        # Optimised mutual unawareness. No awareness, no reaction.
        #
        if not aware_alpha and not aware_beta:
            return

        #
        # Reflex case: s < smd.d_reflex_IA => agents react without analysing velocity.
        #
        # @todo: Consider excluding previously known counteragents from this case.
        #

        # if s < smc.T_reflex_IA * np.dot(relv, e):
        if smc.REFLEXIVE_REPULSION_ENABLED and s < smc.T_reflex_IA * np.dot(relv, e):
            #
            # @todo: Consider removing legacy initialisation code if the conical scheme works significantly better!
            #
            # types       = [agent.agent_type, ctragent.agent_type]
            # importance  = (near, converging, impacted)
            # types       = [agent.agent_type, ctragent.agent_type]
            importance  = (True, None, False)

            interaction = smii.construct_IAI_sharing(IAI_key, agent, ctragent, importance, 0, relposn, None, None, None, None, aware_alpha, aware_beta, None)
            interaction.set_relevance(smii.Relevance.URGENT)

            if aware_alpha:
                reaction = smar.RepelReaction(s, relposn, relv)
                agent.register_interagent_interaction(interaction, reaction)
            if aware_beta:
                reaction = smar.RepelReaction(s, -relposn, -relv)
                ctragent.register_interagent_interaction(interaction, reaction)

            if logger != None:
                logger.log_interagent_interaction(cur_timestep_idx, interaction)

            return

        #
        # @todo: Implement special cases for car-following. Harking back to Anvari+ (2012), the criterion for
        # [quasi-]parallelism has been chosen to be theta <= pi/18 (10 degrees)
        #
        cos_QP = np.cos(np.pi / 18)
        if agent.agent_type == smad.AgentType.CAR and ctragent.agent_type == smad.AgentType.CAR:
            if np.dot(agent.dirn, ctragent.dirn) >= cos_QP:
                # cos_IA = np.dot(agent.dirn, e)
                if np.dot(ctragent.dirn, -e) >= cos_QP:
                    ctragent.register_leader(agent, s)
                elif np.dot(agent.dirn, e) >= cos_QP:
                    agent.register_leader(ctragent, s)

        # print("Dispatcher::dispatch_interagent_interaction - beginning velocicentric assessment of agent {} vs ctragent {}".format(agent.id, ctragent.id))

        #
        # Velocicentric assessment to determine how close the agents will get and construct a truncated cone:
        #   the truncated cone will need to be translated if a velocity obstacle is desired:
        #       VO:         apex -> ctragent.v
        #       RVO:        apex -> 0.5 * (agent.v + ctragent.v)
        #       GRVO(w):    apex -> (1 - w) * agent.v + w * ctragent.v [w being the weight of responsibility]
        #   depending on whether a
        #       velocity obstacle VO (Fiorini+Shiller 1993) or
        #       [generalised] reciprocal velocity obstacle [G]RVO (van den Berg, Lin, Manocha 2008)
        #   is desired.
        # 
        #   @todo: Snape and Manocha (2010) determined variable reciprocity weights with the formula:
        #       w_alpha = Integral({possible v_alpha}) / [Integral({possible v_alpha}) + Integral({possible v_beta})]
        #       w_beta  = Integral({possible v_beta}) / [Integral({possible v_alpha}) + Integral({possible v_beta})]
        #   where the Integrals represent areas or volumes in velocity-space (they intended to target aircraft but boats, ground vehicles, and pedestrians are also viable candidates).
        #
        #   @todo: Compute areas in velocity space A = Integral(0, t_min; D(0, a)) which represents ... maneuverability (?),
        #   offering usable criteria for fairly distributing the weights of responsibility.
        #

        relv    = ctragent.v - agent.v
        results = smia.assess_IA_velocity(agent, ctragent, relposn, relv, s, delta_shape)
        if results == None:
            return
        reaction1, reaction2 = results
        if reaction1 == None and reaction2 == None:
            return

        if reaction2 != None and reaction2.reaction_type == smar.ReactionType.FOLLOW:
            assert(reaction1 == None)
            ctragent.register_leader(agent, reaction2.s)
            return
        elif reaction1 != None and reaction1.reaction_type == smar.ReactionType.FOLLOW:
            assert(reaction2 == None)
            agent.register_leader(ctragent, reaction1.s)
            return

        assert(reaction1 == None or reaction1.reaction_type != smar.ReactionType.FOLLOW)
        assert(reaction2 == None or reaction2.reaction_type != smar.ReactionType.FOLLOW)

        if reaction2 != None and reaction2.reaction_type == smar.ReactionType.AWAIT:
            # assert(reaction1 == None)
            iai = smii.InteragentInteraction(IAI_key, ctragent, agent, relposn, relv, None, None)
            iai.assign_awarenesses(aware_alpha, aware_beta)
            # iai.assign_current_importance(importance)
            # iai.assign_linear_approach_info(quasiparallel, approach, impactful)
            # iai.assign_required_conflict_states(smas.ConflictStateId.MODULATING, smas.ConflictStateId.MODULATING)
            iai.assign_required_conflict_states(reaction1, reaction2)

            if reaction1 != None:
                assert reaction1.reaction_type == smar.ReactionType.VEER
                iai.t_conflict = reaction1.t_conflict
                agent.register_interagent_interaction(iai, reaction1)
            ctragent.register_interagent_interaction(iai, reaction2)
            return
        elif reaction1 != None and reaction1.reaction_type == smar.ReactionType.AWAIT:
            assert(reaction2 == None)
            iai = smii.InteragentInteraction(IAI_key, agent, ctragent, relposn, relv, None, None)
            iai.assign_awarenesses(aware_alpha, aware_beta)
            # iai.assign_current_importance(importance)
            # iai.assign_linear_approach_info(quasiparallel, approach, impactful)
            # iai.assign_required_conflict_states(smas.ConflictStateId.MODULATING, smas.ConflictStateId.MODULATING)
            iai.assign_required_conflict_states(reaction1, reaction2)

            agent.register_interagent_interaction(iai, reaction1)
            if reaction2 != None:
                assert reaction2.reaction_type == smar.ReactionType.VEER
                iai.t_conflict = reaction2.t_conflict
                ctragent.register_interagent_interaction(iai, reaction2)
            return


        RAI = smia.assess_relative_approach(agent, ctragent, delta_shape)
        if RAI != None:
            relposn1, d1, e1, r_total, s1, t_conflict, relv_cone = RAI
            iai = smii.InteragentInteraction(IAI_key, agent, ctragent, relposn, relv, t_conflict, relv_cone)
        else:
            iai = smii.InteragentInteraction(IAI_key, agent, ctragent, relposn, relv, 0., None)

        iai.assign_awarenesses(aware_alpha, aware_beta)
        # iai.assign_current_importance(importance)
        # iai.assign_linear_approach_info(quasiparallel, approach, impactful)
        # iai.assign_required_conflict_states(smas.ConflictStateId.MODULATING, smas.ConflictStateId.MODULATING)
        iai.assign_required_conflict_states(reaction1, reaction2)

        #
        # Register the interaction with one or both agents!
        #
        if reaction1 != None and aware_alpha:
            agent.register_interagent_interaction(iai, reaction1)

        if reaction2 != None and aware_beta:
            ctragent.register_interagent_interaction(iai, reaction2)

        return

        #
        # DEPRECATED
        #
        assert(reaction1 != None or reaction2 != None)
        edge_relationship, next_edge_alpha, next_edge_beta = smia.classify_edge_relationships(agent, ctragent)

        #
        # Special cases:
        #   MERGING and SHARING
        #   DIVERGING
        #

        if edge_relationship == smte.Relationship.INDEPENDENT:
            assert False
            assert(edge_relationship != smte.Relationship.INDEPENDENT)
            return

        if agent.agent_type == smad.AgentType.PEDESTRIAN and ctragent.agent_type == smad.AgentType.PEDESTRIAN:
            edge_relationship = smte.Relationship.SHARING

        if edge_relationship in {smte.Relationship.MERGING, smte.Relationship.SHARING}:
            assert(s != None)
            assert(s >= 0)

            # assert(priority != None)
            # assert(priority in {0, 1})
            # if priority == 1:
            if reaction2 != None and reaction2.reaction_type == smar.ReactionType.FOLLOW:
                assert(reaction1 == None)
                ctragent.register_leader(agent, reaction2.s)
            elif reaction1 != None and reaction1.reaction_type == smar.ReactionType.FOLLOW:
                assert(reaction2 == None)
                agent.register_leader(ctragent, reaction1.s)
            else:
                RAI = smia.assess_relative_approach(agent, ctragent, delta_shape)
                if RAI == None:
                    return
                relposn1, d1, e1, r_total, s1, t_conflict, relv_cone = RAI

                iai = smii.InteragentInteraction(IAI_key, agent, ctragent, relposn, relv, t_conflict, relv_cone)
                iai.assign_awarenesses(aware_alpha, aware_beta)
                # iai.assign_current_importance(importance)
                # iai.assign_linear_approach_info(quasiparallel, approach, impactful)
                # iai.assign_required_conflict_states(smas.ConflictStateId.MODULATING, smas.ConflictStateId.MODULATING)
                iai.assign_required_conflict_states(reaction1, reaction2)

                if reaction1.reaction_type != smar.ReactionType.IGNORE and aware_alpha:
                    agent.register_interagent_interaction(iai, reaction1)
                if reaction2.reaction_type != smar.ReactionType.IGNORE and aware_beta:
                    ctragent.register_interagent_interaction(iai, reaction2)

            return

        elif edge_relationship == smte.Relationship.DIVERGING:
            # assert(s_CFM == None)
            # assert(hulls == None)
            # assert(priority != None)
            # assert(conflict_start_times != None)
            # approach_info = smia.assess_relative_approach(agent, ctragent, delta_shape)
            # relposn1, d1, e1, r_total, s1, t_min, relv_cone = approach_info

            XAI = smia.assess_extrapolated_approach(agent, ctragent)
            if XAI == None:
                return
            priority, T_conflict, yield_times = XAI

            # if conflict_start_times == None:
                # assert(False)
                # return

            # assert(False)
            assert(priority != None)
            assert(priority in {0, 1})
            l_sum = agent.get_length_axis() + ctragent.get_length_axis()
            if priority == 1:
                s_CFM = np.dot(ctragent.posn - agent.posn, agent.dirn) - l_sum
                if s_CFM >= 0:
                    agent.register_leader(ctragent, s_CFM)
                    return
            elif priority == 0:
                s_CFM = np.dot(agent.posn - ctragent.posn, ctragent.dirn) - l_sum
                # assert(s >= 0)
                if s_CFM >= 0:
                    ctragent.register_leader(agent, s_CFM)
                    return
            else:
                return
        else:
            assert(edge_relationship == smte.Relationship.CROSSING)
            RAI = smia.assess_relative_approach(agent, ctragent, delta_shape)
            if RAI != None:
                relposn1, d1, e1, r_total, s1, t_conflict, relv_cone = RAI
            else:
                turning_alpha = agent.goal_state.state_id == smas.GoalStateId.NAVIGATING and \
                                agent.goal_state.turning_info != None
                turning_beta  = ctragent.goal_state.state_id == smas.GoalStateId.NAVIGATING and \
                                ctragent.goal_state.turning_info != None
                if turning_alpha or turning_beta:
                    t_conflict = 0.
                    relv_cone = None
                else:
                    return
            iai = smii.InteragentInteraction(IAI_key, agent, ctragent, relposn, relv, t_conflict, relv_cone)
            iai.assign_awarenesses(aware_alpha, aware_beta)
            # iai.assign_current_importance(importance)
            # iai.assign_linear_approach_info(quasiparallel, approach, impactful)
            iai.assign_required_conflict_states(reaction1, reaction2)

            if reaction1 != None and reaction1.reaction_type != smar.ReactionType.IGNORE and aware_alpha:
                agent.register_interagent_interaction(iai, reaction1)
            if reaction2 != None and reaction2.reaction_type != smar.ReactionType.IGNORE and aware_beta:
                ctragent.register_interagent_interaction(iai, reaction2)
            return

        assert False
        RAI = smia.assess_relative_approach(agent, ctragent, delta_shape)
        if RAI == None:
            return
        relposn1, d1, e1, r_total, s1, t_conflict, relv_cone = RAI

        iai = smii.InteragentInteraction(IAI_key, agent, ctragent, relposn, relv, t_conflict, relv_cone)
        iai.assign_awarenesses(aware_alpha, aware_beta)
        # iai.assign_current_importance(importance)
        # iai.assign_linear_approach_info(quasiparallel, approach, impactful)
        iai.assign_required_conflict_states(reaction1, reaction2)
        if reaction1 != smar.ReactionType.IGNORE and aware_alpha:
            agent.register_interagent_interaction(iai, reaction1)
        if reaction2 != smar.ReactionType.IGNORE and aware_beta:
            ctragent.register_interagent_interaction(iai, reaction2)
        return

        assert(False)

        #
        #
        #

        interaction = smii.InteragentInteraction(IAI_key, agent, ctragent, relposn, relv, t_conflict, relv_cone)
        iai.assign_awarenesses(aware1, aware2)
        # iai.assign_current_importance(importance)
        # iai.assign_linear_approach_info(quasiparallel, approach, impactful)
        iai.assign_required_conflict_states(reaction1, reaction2)

        return

        #
        # Deprecate everything beyond!
        #

        # approachinfo, priority, conflict_start_times = results
        # relposn1, d1, e1, r_total, s1, t_min, relv_cone = approach_info
        edge_relationship, approach_info, priority, s_CFM, hulls, conflict_start_times = results
        if edge_relationship in {smte.Relationship.MERGING, smte.Relationship.SHARING}:
            assert(s_CFM != None)
            assert(approach_info == None)

            assert(priority != None)
            assert(priority in {0, 1})
            if priority == 1:
                agent.register_leader(ctragent, s_CFM)
            else:
                ctragent.register_leader(agent, s_CFM)
            return

        elif edge_relationship == smte.Relationship.DIVERGING:
            assert(False)
            assert(edge_relationship != smte.Relationship.DIVERGING)
            assert(s_CFM == None)
            assert(hulls == None)
            assert(priority != None)
            assert(conflict_start_times != None)
            assert(approach_info != None)
            relposn1, d1, e1, r_total, s1, t_min, relv_cone = approach_info

            if conflict_start_times == None:
                assert(False)
                return

            assert(False)
            assert(priority != None)
            assert(priority in {0, 1})
            if priority == 1:
                s_longitudinal = np.dot(ctragent.posn - agent.posn, agent.dirn) - l_sum
                assert (s_longitudinal >= 0)
                agent.register_leader(ctragent, s_longitudinal)
                return
            else:
                s_longitudinal = np.dot(agent.posn - ctragent.posn, ctragent.dirn) - l_sum
                assert(s_longitudinal >= 0)
                ctragent.register_leader(agent, s_longitudinal)
                return

        elif edge_relationship == smte.Relationship.INDEPENDENT:
            assert(False)
            assert(edge_relationship != smte.Relationship.INDEPENDENT)
            return

        #
        # Temporarily (?) deprecated.
        #

        assert(edge_relationship == smte.Relationship.CROSSING)

        # assert(s_CFM == None)
        assert((priority != None) == (conflict_start_times != None))
        if approach_info == None:
            return
        relposn1, d1, e1, r_total, s1, t_min, relv_cone = approach_info

        if conflict_start_times != None:
            assert(False)
            assert (conflict_start_times[0] <= t_min)
            assert (conflict_start_times[1] <= t_min)

        # assert(False)

        #
        # NB: Hulls may or may not be None
        #

        if hulls != None:
            MH_alpha, MH_beta = hulls
        
        assert(approach_info == None or relv_cone == None or np.array_equal(relv_cone.apex, np.zeros(2)))

        # print("Dispatcher::dispatch_IAI v-centric: approach dx_ftr {}, d_ftr {}, e_ftr {}, r_sum {}, s_ftr {}, t_min {}".format(relposn1, d1, e1, r_total, s1, t_min))
        # print("Dispatcher::dispatch_IAI v-centric: priority {} and conflict start times {}".format(priority, conflict_start_times))

        if s1 < smc.d_safe_IA_maximum:
            # assert(t_min > 0)
            t_min -= (smc.d_safe_IA_maximum - s1)  / np.dot(relv, e1)
            if t_min < 0:
                t_min = 0

            relposn1 = relposn + t_min * relv

            #
            # Legacy initialisation
            #
            types       = [agent.agent_type, ctragent.agent_type]
            importance  = (False, True, False)
            quasiparallel = np.dot(agent.dirn, ctragent.dirn)**2 > (np.cos(np.pi / 18)**2)
            
            approach = smii.Approach.OBLIQUE
            c = np.dot(agent.dirn, ctragent.dirn)
            if c**2 > 0.75 and 16 * (c**2 - 0.5)**2 > 3:
                approach = smii.Approach.LONGITUDINAL
            if c**2 < 0.25 and 16 * (c**2 - 0.5)**2 > 3:
                approach = smii.Approach.TRANSVERSE

            impactful = s1 < 0

            #
            # @todo Priority (precedence and otherwise), motion hulls
            #
            if priority != None:
                assert(False)

                if priority == 1:
                    assert(False)
                    assert(aware_alpha)

                    t_conflict = conflict_start_times[0]
                    if agent.agent_type == smad.AgentType.PEDESTRIAN and ctragent.goal_state.state_id != smas.GoalStateId.TURNING:
                        interaction = \
                            smii.construct_IAI_evading(IAI_key, agent, ctragent, importance, t_conflict, relposn1,
                                                        relv, quasiparallel, approach, impactful, aware_alpha,
                                                        aware_beta, relv_cone)
                    else:
                        interaction = \
                            smii.construct_IAI_yielding(IAI_key, agent, ctragent, importance, t_conflict, relposn1,
                                relv, quasiparallel, approach, impactful, aware_alpha, aware_beta, relv_cone)
                    interaction.set_relevance(smii.Relevance.IMPORTANT)

                    agent.register_interagent_interaction(ctragent, interaction, False)
                    return
                elif priority == 0:
                    assert(False)
                    assert(aware_beta)

                    negative_relv_cone            = relv_cone
                    negative_relv_cone.apex       = -negative_relv_cone.apex
                    negative_relv_cone.left_dirn  = -negative_relv_cone.left_dirn
                    negative_relv_cone.right_dirn = -negative_relv_cone.right_dirn

                    t_conflict = conflict_start_times[1]
                    if ctragent.agent_type == smad.AgentType.PEDESTRIAN:
                        interaction = \
                            smii.construct_IAI_evading(IAI_key, ctragent, agent, importance, t_conflict, -relposn1,
                                                        -relv, quasiparallel, approach, impactful, aware_beta,
                                                        aware_alpha, negative_relv_cone)
                    else:
                        interaction = \
                            smii.construct_IAI_yielding(IAI_key, ctragent, agent, importance, t_conflict, -relposn1,
                                                        -relv, quasiparallel, approach, impactful, aware_beta,
                                                        aware_alpha, negative_relv_cone)
                    interaction.set_relevance(smii.Relevance.IMPORTANT)

                    ctragent.register_interagent_interaction(agent, interaction, False)

                    return
            else:
                assert(agent.agent_type    == smad.AgentType.PEDESTRIAN)
                assert(ctragent.agent_type == smad.AgentType.PEDESTRIAN)

                interaction = smii.construct_IAI_sharing(IAI_key, agent, ctragent, importance, t_min, relposn1, relv, quasiparallel, approach, impactful, aware_alpha, aware_beta, relv_cone)
                agent.register_interagent_interaction(ctragent, interaction, False)
                return

                if relv_cone != None:
                    interaction = smii.construct_IAI_evading(IAI_key, agent, ctragent, importance, t_min, relposn1, relv, quasiparallel, approach, impactful, aware_alpha, aware_beta, relv_cone)
                    interaction.set_relevance(smii.Relevance.IMPORTANT)
                    if aware_alpha:
                        assert(False)
                        agent.register_interagent_interaction_evade(ctragent, interaction, False)
                    if aware_beta:
                        assert(False)
                        ctragent.register_interagent_interaction_evade(agent, interaction, True)
                else:
                    interaction = smii.construct_IAI_yielding(IAI_key, agent, ctragent, importance, t_min, relposn1, relv, quasiparallel, approach, impactful, aware_alpha, aware_beta, relv_cone)
                    interaction.set_relevance(smii.Relevance.IMPORTANT)
                    if aware_alpha:
                        assert(False)
                        agent.register_interagent_interaction_yield(ctragent, interaction, False)
                    if aware_beta:
                        assert(False)
                        ctragent.register_interagent_interaction(agent, interaction, True)

            if logger != None:
                logger.log_interagent_interaction(cur_timestep_idx, interaction)

            return
        
        return

        #
        # Consider potential convergence too, e.g.
        #
        if relv_cone == None:
            return

        cur_dst_idx = agent.route[agent.routing_idx + 1]
        cur_dst     = agent.nodegraph.nodes[cur_dst_idx]
        x_omega, dist, nv, tv = cur_dst.approach_shape.get_nearest_point_info(agent.posn)
        v_des           = agent.get_target_speed() * smnd.calc_unit_vector(x_omega - agent.posn)
        relv_des        = ctragent.v - v_des

        e1      = smnd.calc_unit_vector(-relv)
        e2      = smnd.calc_unit_vector(-relv_des)
        e1_perp = smnd.rotate_90ACW(e1)
        if np.dot(e1_perp, e2) >= 0:
            descone_left  = e2
            descone_right = e1
        else:
            descone_left  = e2
            descone_right = e1

        #
        # If descone_left is not to the right of fearcone_right and descone_right is not to the left of fearcone_left,
        # then the desired velocity cone (the cone between the agent's actual velocity and their desired velocity) and
        # the feared cone (i.e. defined by the velocity obstacle) intersect.
        #
        fearcone_left_perp  = smnd.rotate_90ACW(relv_cone.e_left)
        fearcone_right_perp = smnd.rotate_90ACW(relv_cone.e_right)
        if np.dot(descone_left, fearcone_right_perp) >= 0 and np.dot(descone_right, fearcone_left_perp) <= 0:
            interaction = smii.InteragentInteraction(IAI_key, types, importance, t_min, relposn1, relv, quasiparallel, approach, impactful, aware_alpha, aware_beta, relv_cone)
            interaction.set_relevance(smii.Relevance.RELEVANT)
            if aware_alpha:
                agent.register_interagent_interaction(ctragent, interaction, False)
            if aware_beta:
                ctragent.register_interagent_interaction(agent, interaction, True)
            return

        return

    def dispatch_interactions(self, agent_metaset, active_agents, zone_config, grid, cur_timestep_idx, logger):
        self.prepare_interactions(agent_metaset, active_agents, grid)
        
        #
        # We now have two sets of interactions to mine for potential conflicts:
        #     one registers interagent interactions; and
        #     the other one records agent vs zone interactions
        #
        # print("Model::determine_all_interactions: combined interaction set has {} elements".format(len(self.interaction_set)))

        for CZI_key in self.zone_interaction_set:
            agent_id, zone_id = smic.contrazone_codec.decode(CZI_key)
            if agent_id not in active_agents or zone_id not in zone_config.zone_tbl:
                continue

            # if zone_id not in self.zone_config.zone_tbl:
                # continue

            agent = active_agents[agent_id]
            zone  = zone_config.zone_tbl[zone_id]

            self.dispatch_contrazone_interaction(agent, zone, CZI_key, cur_timestep_idx, logger)

        for IAI_key in self.agent_interaction_set:
            agent_id, ctragent_id = smic.interagent_codec.decode(IAI_key)
            # print("Model::determine_all_interactions: interaction between {} and {}".format(agent_id, ctragent_id))
            assert(agent_id != ctragent_id)

            agent    = active_agents[agent_id]
            ctragent = active_agents[ctragent_id]

            if agent_id not in active_agents:
                if agent.agent_type != smad.AgentType.PEDESTRIAN and ctragent.agent_type != smad.AgentType.PEDESTRIAN:
                    print("Model::determine_all_interactions - vhcl {} and ctrvhcl {}, vhcl {} is inactive".format(agent.id, ctragent.id, agent.id))
                continue
            if ctragent_id not in active_agents:
                if agent.agent_type != smad.AgentType.PEDESTRIAN and ctragent.agent_type != smad.AgentType.PEDESTRIAN:
                    print("Model::determine_all_interactions - vhcl {} and ctrvhcl {}, ctrvhcl {} is inactive".format(agent.id, ctragent.id, ctragent.id))
                continue

            self.dispatch_interagent_interaction(agent, ctragent, IAI_key, cur_timestep_idx, logger)

        # grid.clear_agent_coverage()
        return
