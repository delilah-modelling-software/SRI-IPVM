from enum import Enum

# import numpy as np
# import scipy as sp
# import scipy.stats as ss

import model.defs as smd
import model.agency.defs as smad
import model.agency.state as smas
import model.numerical.defs as smnd
import model.zone.defs as smzd
import model.zone.zone as smzz

import model.interaction.codec as smic


class Approach(Enum):
    LONGITUDINAL = 1
    TRANSVERSE   = 2
    OBLIQUE      = 3

#
# Each agent will need to determine this (probably more than once) for each
# interactor (counteragent or zone). Should probably refer back to [@todo: find the paper]
# and explain the more selective nature of the conical superenvelope.
#
# Relevance:
#   UNKNOWN     not yet analysed
#   IRRELEVANT  analysed, non-NEAR and non-CONVERGING (relv \dot relx < 0) and convergence is not desired
#   RELEVANT    analysed, non-NEAR and non-CONVERGING but convergence is desired.
#   IMPORTANT   analysed, non-NEAR and CONVERGING
#   URGENT      analysed, NEAR and CONVERGING
#
# @todo: This is a personal perceptual estimation which could be parametrised (e.g. Likert)?
#
class Relevance(Enum):
    UNKNOWN    = -1
    IRRELEVANT =  0 # NB: IRRELEVANT and RELEVANT may be difficult to distinguish in online video processing,
    RELEVANT   =  1 # but easier in offline processing afterwards. What visible signals are detectable?
    IMPORTANT  =  2
    URGENT     =  3

class Interaction:
    def __init__(self, combined_id):
        self.combined_id = combined_id
        self.relevance   = Relevance.UNKNOWN
        return

    def set_relevance(self, relevance):
        self.relevance = relevance

class ContrazoneInteraction(Interaction):
    def __init__(self, combined_id, agent, zone, relv_cone, t_min):
        super().__init__(combined_id)
        self.agent       = agent
        self.zone        = zone
        self.relv_cone   = relv_cone
        self.t_min       = t_min

        return

class InteragentInteraction(Interaction):
    def __init__(self, combined_id, agent, ctragent, rel_posn, rel_v, t_min, relv_cone):
        super().__init__(combined_id)
        self.agent       = agent
        self.ctragent    = ctragent

        self.rel_posn = rel_posn
        self.rel_v    = rel_v
        self.t_conflict  = t_min

        self.relv_cone   = relv_cone
        return

    #
    # @todo: Will these ever need to be swapped?
    #
    def assign_awarenesses(self, aware1, aware2):
        self.awarenesses = [aware1, aware2]
        return

    def assign_current_importance(self, importance):
        self.current_importance = importance  # triple of booleans (near, converging, impacted)
        return

    def assign_linear_approach_info(self, quasiparallel, approach, impactful):
        self.quasiparallel = quasiparallel
        self.approach      = approach
        self.impactful     = impactful
        return

    def assign_required_conflict_states(self, rcs1, rcs2):
        self.required_conflict_states = (rcs1, rcs2)
        return

    def get_desc_list(self):
        near_str = "FAR"
        near, converging, impacted = self.current_importance
        if near:
            near_str = "NEAR"
        converging_str = "NONCONVERGING"
        if converging:
            converging_str = "CONVERGING"
        impacted_str = "UNIMPACTED"
        if impacted:
            impacted_str = "IMPACTED"

        qp_str = "NONPARALLEL"
        if self.quasiparallel:
            qp_str = "QUASIPARALLEL"

        apprch_str = "OBLIQUE"
        if self.approach == Approach.TRANSVERSE:
            apprch_str = "TRANSVERSE"
        elif self.approach == Approach.LONGITUDINAL:
            apprch_str = "LONGITUDINAL"
            
        impactful_str = "NONIMPACTFUL"
        if self.impactful:
            impactful_str = "IMPACTFUL"
            
        reference_sublist      = [self.combined_id, self.agent.agent_type, self.ctragent.agent_type]
        importance_sublist     = [near_str, converging_str, impacted_str]
        spatiotemporal_sublist = [self.t_conflict, self.rel_posn, self.rel_v]
        analysis_sublist       = [qp_str, apprch_str, impactful_str, self.awarenesses]
        # analysis_sublist       =[qp_str, apprch_str, impactful_str, self.awarenesses[0], self.awarenesses[1]]

        return reference_sublist + importance_sublist + spatiotemporal_sublist + analysis_sublist

#
# Factory functions for
#

# def legacy_IAI_construction(combined_id, agent, ctragent, importance, t_min, rel_posn, rel_v, quasiparallel,
    # approach, impactful, aware1, aware2, relv_cone):
def construct_IAI_sharing(combined_id, agent, ctragent, importance, t_conflict, rel_posn, rel_v, quasiparallel, \
        approach, impactful, aware1, aware2, relv_cone):

    assert(aware1 or aware2)
    # assert(agent.agent_type == ctragent.agent_type)

    iai = InteragentInteraction(combined_id, agent, ctragent, rel_posn, rel_v, t_conflict, relv_cone)
    iai.assign_awarenesses(aware1, aware2)
    iai.assign_current_importance(importance)
    iai.assign_linear_approach_info(quasiparallel, approach, impactful)
    iai.assign_required_conflict_states(smas.ConflictStateId.MODULATING, smas.ConflictStateId.MODULATING)

    return iai

def construct_IAI_evading(combined_id, agent, ctragent, importance, t_conflict, rel_posn, rel_v, quasiparallel, \
        approach, impactful, aware1, aware2, relv_cone):
    assert(relv_cone != None)

    assert(aware1 or aware2)
    assert(agent.agent_type == smad.AgentType.CAR or ctragent.agent_type == smad.AgentType.CAR)

    iai = InteragentInteraction(combined_id, agent, ctragent, rel_posn, rel_v, t_conflict, relv_cone)
    iai.assign_awarenesses(aware1, aware2)
    iai.assign_current_importance(importance)
    iai.assign_linear_approach_info(quasiparallel, approach, impactful)
    iai.assign_required_conflict_states(smas.ConflictState.VEERING, None)

    return iai

def construct_IAI_yielding(combined_id, agent, ctragent, importance, t_conflict, rel_posn, rel_v, quasiparallel, \
        approach, impactful, aware1, aware2, relv_cone):
    assert(relv_cone == None)

    assert(aware1 or aware2)
    assert(agent.agent_type == smad.AgentType.CAR or ctragent.agent_type == smad.AgentType.CAR)

    iai = InteragentInteraction(combined_id, agent, ctragent, rel_posn, rel_v, t_conflict, relv_cone)
    iai.assign_awarenesses(aware1, aware2)
    iai.assign_current_importance(importance)
    iai.assign_linear_approach_info(quasiparallel, approach, impactful)
    iai.assign_required_conflict_states(smas.ConflictState.YIELDING, None)

    return iai
