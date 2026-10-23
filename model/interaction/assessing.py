import copy as cp
import numpy as np
# import scipy as sp
# import scipy.stats as ss
import scipy.optimize as so

import model.agency.defs as smad
import model.agency.reaction as smar
import model.agency.state as smas
import model.logger as sml
import model.numerical.defs as smnd
import model.geometry.shape as smgs
# import model.defs as smd
import model.configuration as smc
# import view.defs as svd
# import view.view as svv
import model.topology.edge as smte
# import model.topology.graph as smtg
# import model.topology.approachnode as smtApp
# import model.topology.activitynode as smtAct
# import model.topology.topology as smtt
# import model.geometry.grid as smgg
# import model.agency.agent as smaa
# import model.zone.zone as smzz
# import model.interaction.codec as smic
import model.interaction.prioritising as smip
# import model.interaction.interaction as smii
# import model.force as smf


#
# A set of functions assessing the (relative positions) of different interactees:
#   assess_interagent_position delegates 
#

def assess_IA_position_PtnVsPtn(agent, ctragent, IAI_key, cur_timestep_idx):
    relposn = ctragent.posn - agent.posn
    d, e    = smnd.calc_polar_decomp(relposn)
    r_total = agent.radius + ctragent.radius
    s       = d - r_total

    delta_shape = smgs.Circle(relposn, r_total)

    return relposn, d, e, r_total, s, delta_shape

# def assess_IA_position_PtnVsBcl():
    # ctragent_shape = smgs.Segment(ctragent.posn, ctragent.dirn, halflength, radius)

def assess_IA_position_PtnVsCar(agent, ctragent, IAI_key, cur_timestep_idx):
    halflength = 0.5 * ctragent.rect_length
    halfwidth  = 0.5 * ctragent.rect_width
    radius     = ctragent.radius

    ctragent_shape = smgs.Rectangle(ctragent.posn, ctragent.dirn, halflength, halfwidth, radius)

    x_beta, d, nv, tv = ctragent_shape.get_nearest_point_info(agent.posn)
    r_total = agent.radius + ctragent.radius

    # s = d - r_total
    inside = ctragent_shape.contains(agent.posn)
    if inside:
        s = -d - r_total
        nv = -nv
    else:
        s = d - r_total

    # relposn = x_beta - agent.posn
    relposn = ctragent.posn - agent.posn
    e       = -nv

    delta_shape = smgs.Rectangle(relposn, ctragent.dirn, halflength, halfwidth, r_total)

    # @todo: Let Python tell us what belongs here!
    return relposn, d, e, r_total, s, delta_shape

# def assess_IA_position_BclVsBcl():
    # agent_shape = smgs.Segment(agent.posn, agent.dirn, halflength, agent.radius)
    # ctragent_shape = smgs.Segment(ctragent.posn, ctragent.dirn, halflength, ctragent.radius)

# def assess_IA_position_BclVsCar():
    # halflength = 0.5 * ctragent.rect_length
    # halfwidth  = 0.5 * ctragent.rect_width
    # radius     = smd.d_safe_IA
    # ctragent_shape = smgs.Rectangle(ctragent.posn, ctragent.dirn, halflength, halfwidth, radius)

    # agent_shape = smgs.Segment(agent.posn, agent.dirn, halflength, agent.radius)
    # ctragent_shape = smgs.Segment(ctragent.posn, ctragent.dirn, halflength, ctragent.radius)

def assess_IA_position_CarVsCar(agent, ctragent, IAI_key, cur_timestep_idx):
    hl_alpha     = 0.5 * agent.rect_length
    hw_alpha     = 0.5 * agent.rect_width
    radius_alpha = agent.radius
    agent_shape  = smgs.Rectangle(agent.posn, agent.dirn, hl_alpha, hw_alpha, radius_alpha)

    hl_beta        = 0.5 * ctragent.rect_length
    hw_beta        = 0.5 * ctragent.rect_width
    radius_beta    = ctragent.radius
    ctragent_shape = smgs.Rectangle(ctragent.posn, ctragent.dirn, hl_beta, hw_beta, radius_beta)

    delta_shape = smgs.construct_minkowski_difference_rect_rect(agent_shape, ctragent_shape)
    r_total     = delta_shape.radius
    assert (agent_shape.radius + ctragent_shape.radius - delta_shape.radius)**2 < smnd.EPS_m_sqr

    relposn, d, nv, tv = delta_shape.get_nearest_point_info(np.zeros(2))
    inside             = delta_shape.contains(np.zeros(2))
    if inside:
        s = -d - delta_shape.radius
        e = nv
    else:
        s = d - delta_shape.radius
        e = -nv

    return relposn, d, e, r_total, s, delta_shape

def assess_IA_position(agent, ctragent, IAI_key, cur_timestep_idx):
    #
    # Positional assessment:
    #   Determine the relative position, Euclidean distance, direction, sum of radii, and hence the netto distance.
    #

    # agent_circle = smgs.Circle(agent.posn, agent.radius)
    if agent.agent_type == smad.AgentType.PEDESTRIAN:
        if ctragent.agent_type == smad.AgentType.PEDESTRIAN:
            relposn, d, e, r_total, s, dshape = assess_IA_position_PtnVsPtn(agent, ctragent, IAI_key, cur_timestep_idx)

        # elif ctragent.agent_type == smad.AgentType.BICYCLE:
            # return assess_IA_position_PtnVsBcl(agent, ctragent, IAI_key, cur_timestep_idx, logger)
        else:
            assert(ctragent.agent_type == smad.AgentType.CAR)
            relposn, d, e, r_total, s, dshape = assess_IA_position_PtnVsCar(agent, ctragent, IAI_key, cur_timestep_idx)

    # elif agent.agent_type == smad.AgentType.BICYCLE:
        # if ctragent.agent_type == smad.AgentType.BICYCLE:
            # relposn, d, e, r_total, s, dshape = assess_IA_position_BclVsBcl(agent, ctragent, IAI_key, cur_timestep_idx)
        # else:
            # assert(ctragent.agent_type == smad.AgentType.CAR)
            # relposn, d, e, r_total, s, dshape = assess_IA_position_BclVsCar(agent, ctragent, IAI_key, cur_timestep_idx)

    else:
        assert(agent.agent_type == smad.AgentType.CAR)
        assert(ctragent.agent_type == smad.AgentType.CAR)
        relposn, d, e, r_total, s, dshape = assess_IA_position_CarVsCar(agent, ctragent, IAI_key, cur_timestep_idx)

    return relposn, d, e, r_total, s, dshape

#
# The next set of functions are used to perform perceptual assessment.
#


def assess_IA_visibility_cone(agent, viewable_cone):
    visible_alpha     = False
    if viewable_cone != None:
        if agent.goal_state.state_id == smas.GoalStateId.NAVIGATING and agent.goal_state.turning_info != None:
            dst      = agent.nodegraph.nodes[agent.goal_state.node_id]
            alt_dirn = smnd.calc_unit_vector(dst.approach_shape.posn - agent.posn)

            perp_dirn = smnd.rotate_90ACW(agent.dirn)
            if np.dot(perp_dirn, alt_dirn) >= 0:
                left_dirn      = alt_dirn
                left_perp_dirn = smnd.rotate_90ACW(alt_dirn)

                right_dirn      = agent.dirn
                right_perp_dirn = perp_dirn
            else:
                left_dirn      = agent.dirn
                left_perp_dirn = perp_dirn

                right_dirn      = alt_dirn
                right_perp_dirn = smnd.rotate_90ACW(alt_dirn)

            # focused_alpha = np.dot(left_perp_dirn, viewable_cone.e_right) >= 0 and np.dot(right_perp_dirn, viewable_cone.e_right) <= 0
            # peripheral_alpha = np.dot(left_dirn, viewable_cone.e_left) >= 0 or np.dot(right_dirn, viewable_cone.e_right) >= 0
            # visible_alpha     = focused_alpha or peripheral_alpha
            focal_overlap      = np.dot(viewable_cone.e_right, left_perp_dirn) <= 0 and np.dot(viewable_cone.e_left, right_perp_dirn) >= 0
            peripheral_overlap = np.dot(viewable_cone.e_right, left_dirn) >= 0 or np.dot(viewable_cone.e_left, right_dirn) >= 0
            visible_alpha      = focal_overlap or peripheral_overlap
        else:
            agent_dirn_perp    = smnd.rotate_90ACW(agent.dirn)
            focused_alpha = np.dot(agent_dirn_perp, viewable_cone.e_left) >= 0 and np.dot(agent_dirn_perp, viewable_cone.e_right) <= 0

            #
            # Consider using [ctr]agent.cos_gma != 0
            #
            peripheral_alpha = np.dot(agent.dirn, viewable_cone.e_left) >= 0 or np.dot(agent.dirn, viewable_cone.e_right) >= 0
            visible_alpha     = focused_alpha or peripheral_alpha
    return visible_alpha

def assess_IA_visibility_PtnVsPtn(agent, ctragent, s):
    # agent_circle    = smgs.Circle(agent.posn, 0.2)
    agent_circle      = smgs.Circle(agent.posn, 0.)
    ctragent_shape    = smgs.Circle(ctragent.posn, ctragent.radius)

    visibility_cone_beta = smgs.construct_tcone(agent_circle, ctragent_shape, 1., True)
    if visibility_cone_beta == None:
        assert(False)
        return True, True

    visibility_cone_alpha         = cp.deepcopy(visibility_cone_beta)
    visibility_cone_alpha.apex    = ctragent.posn
    visibility_cone_alpha.e_left  = -visibility_cone_beta.e_left
    visibility_cone_alpha.e_right = -visibility_cone_beta.e_right

    visible_beta = assess_IA_visibility_cone(agent, visibility_cone_beta)
    visible_alpha = assess_IA_visibility_cone(ctragent, visibility_cone_alpha)

    return visible_beta, visible_alpha

def assess_IA_visibility_PtnVsCar(agent, ctragent, s):
    hl_beta = 0.5 * ctragent.rect_length
    hw_beta = 0.5 * ctragent.rect_width

    # agent_circle       = smgs.Circle(agent.posn, 0.2)
    agent_circle         = smgs.Circle(agent.posn, 0.)
    ctragent_circle      = smgs.Circle(ctragent.posn, 0.)
    agent_shape          = smgs.Circle(agent.posn, agent.radius)
    ctragent_shape       = smgs.Rectangle(ctragent.posn, ctragent.dirn, hl_beta, hw_beta, ctragent.radius)

    visibility_cone_beta  = smgs.construct_tcone(agent_circle, ctragent_shape, 1., True)
    visibility_cone_alpha = smgs.construct_tcone(ctragent_circle, agent_shape, 1., True)

    # vis_alpha, vis_beta = assess_IA_visibility(agent, ctragent, vis_cone1, vis_cone2)

    visible_beta  = assess_IA_visibility_cone(agent, visibility_cone_beta)
    visible_alpha = assess_IA_visibility_cone(ctragent, visibility_cone_alpha)

    #
    # @todo: Once the velocicentric code appears to be working, change the code back and see if it actually works.
    #
    return True, True

    #
    # Temporarily deprecated.
    #
    return visible_beta, visible_alpha

def assess_IA_visibility_CarVsCar(agent, ctragent, s):
    #
    # If these assumptions are wrong, two view cones should be calculated
    # instead of negating the agent cone for the counteragent to use.
    #
    assert((agent.rect_length - ctragent.rect_length)**2 < smnd.EPS_m_sqr)
    assert((agent.rect_width - ctragent.rect_width)**2 < smnd.EPS_m_sqr)

    hl_alpha = 0.5 * agent.rect_length
    hw_alpha = 0.5 * agent.rect_width
    hl_beta  = 0.5 * ctragent.rect_length
    hw_beta  = 0.5 * ctragent.rect_width

    hl_sum = hl_alpha + hl_beta
    hw_sum = hw_alpha + hw_beta

    #
    # Note: it is assumed that the driver's position is [approximately] in the centre of the car,
    # and that the viewer's shape does not significantly affect the visibility of other agents
    #

    # agent_circle       = smgs.Circle(agent.posn, 0.2)
    agent_circle         = smgs.Circle(agent.posn, 0.)
    ctragent_circle      = smgs.Circle(ctragent.posn, 0.)
    # radius             = agent_circle.radius + ctragent.radius
    # ctragent_viewshape = smgs.Rectangle(ctragent.posn, ctragent.dirn, hl_beta, hw_beta, agent_circle.radius)

    #
    # Note: The visibility cones are *not* reflections of each other.
    #
    agent_viewshape      = smgs.Rectangle(agent.posn, agent.dirn, hl_alpha, hw_alpha, ctragent.radius)
    ctragent_viewshape   = smgs.Rectangle(ctragent.posn, ctragent.dirn, hl_beta, hw_beta, ctragent.radius)


    #
    # @todo: To support vehicles crossing at other angles, test the reflection code to verify that
    # it still works correctly (at least for vehicles with the same length x width). To support
    # heterogeneous vehicles, calculate the visibility cones independently.
    #
    visibility_cone_beta  = smgs.construct_tcone(agent_circle, ctragent_viewshape, 1., True)
    visibility_cone_alpha = smgs.construct_tcone(ctragent_circle, agent_viewshape, 1., True)

    visible_beta  = assess_IA_visibility_cone(agent, visibility_cone_beta)
    visible_alpha = assess_IA_visibility_cone(ctragent, visibility_cone_alpha)

    #
    # @todo: Once the velocicentric code appears to be working, change the code back and see if it actually works.
    #
    return True, True

    #
    # Temporarily deprecated.
    #
    return visible_beta, visible_alpha

def assess_IA_visibility(agent, ctragent, s):
    if agent.agent_type == smad.AgentType.PEDESTRIAN:
        if ctragent.agent_type == smad.AgentType.PEDESTRIAN:
            return assess_IA_visibility_PtnVsPtn(agent, ctragent, s)

        # elif ctragent.agent_type == smad.AgentType.BICYCLE:
            # return assess_IA_visibility_PtnVsBcl(self, agent, ctragent, s)
        else:
            assert(ctragent.agent_type == smad.AgentType.CAR)
            return assess_IA_visibility_PtnVsCar(agent, ctragent, s)

    # elif agent.agent_type == smad.AgentType.BICYCLE:
        # if ctragent.agent_type == smad.AgentType.BICYCLE:
            # return assess_IA_visibility_BclVsBcl(agent, ctragent)
        # else:
            # assert(ctragent.agent_type == smad.AgentType.CAR)
            # return assess_IA_visibility_BclVsCar(agent, ctragent)

    else:
        assert(agent.agent_type == smad.AgentType.CAR)
        assert(ctragent.agent_type == smad.AgentType.CAR)
        return assess_IA_visibility_CarVsCar(agent, ctragent, s)

#
# Velocicentric assessment functions.
#

def classify_edge_relationships(agent, ctragent):
    alpha_src_idx = agent.route[agent.routing_idx]
    beta_src_idx = ctragent.route[ctragent.routing_idx]

    alpha_src = agent.nodegraph.nodes[alpha_src_idx]
    beta_src  = ctragent.nodegraph.nodes[beta_src_idx]

    alpha_dst = agent.nodegraph.nodes[agent.goal_state.node_id]
    beta_dst  = ctragent.nodegraph.nodes[ctragent.goal_state.node_id]

    # alpha_edge = smte.Edge(alpha_src, alpha_dst)
    # beta_edge  = smte.Edge(beta_src,  beta_dst)
    alpha_edge = alpha_src.get_edge(alpha_dst)
    beta_edge  = beta_src.get_edge(beta_dst)

    cur_rltn = alpha_edge.classify_relationship(beta_edge)
    if cur_rltn != smte.Relationship.INDEPENDENT:
        return cur_rltn, False, False

    alpha_next_exists = agent.routing_idx    + 2 < len(agent.route)
    beta_next_exists  = ctragent.routing_idx + 2 < len(ctragent.route)

    alpha_next_edge = None
    if alpha_next_exists:
        alpha_nextdst_idx = agent.route[agent.routing_idx + 2]
        alpha_nextdst     = agent.nodegraph.nodes[alpha_nextdst_idx]
        # alpha_next_edge   = smte.Edge(alpha_dst, alpha_nextdst)
        alpha_next_edge   = alpha_dst.get_edge(alpha_nextdst)
        alpha_next_rltn   = alpha_next_edge.classify_relationship(beta_edge)
        if alpha_next_rltn != smte.Relationship.INDEPENDENT:
            return alpha_next_rltn, True, False

    beta_next_edge = None
    if beta_next_exists:
        beta_nextdst_idx = ctragent.route[ctragent.routing_idx + 2]
        beta_nextdst   = ctragent.nodegraph.nodes[beta_nextdst_idx]
        # beta_next_edge = smte.Edge(beta_dst, beta_nextdst)
        beta_next_edge = beta_dst.get_edge(beta_nextdst)
        beta_next_rltn = alpha_edge.classify_relationship(beta_next_edge)
        if beta_next_rltn != smte.Relationship.INDEPENDENT:
            return beta_next_rltn, False, True

    if alpha_next_exists and beta_next_exists:
        both_next_rltn = alpha_next_edge.classify_relationship(beta_next_edge)
        if both_next_rltn != smte.Relationship.INDEPENDENT:
            return both_next_rltn, True, True

    #
    # If a conflict is more than one graph-edge from now for either participant, it should not need to be handled now.
    #
    return smte.Relationship.INDEPENDENT, None, None

def assess_IA_velocity_PtnVsPtn(agent, ctragent, relposn, relv, s, dshape):
    # assert (dshape == None)

    edge_relationship, is_next_edge_alpha, is_next_edge_beta = classify_edge_relationships(agent, ctragent)
    if edge_relationship == smte.Relationship.INDEPENDENT:
        return None, None
    edge_relationship = smte.Relationship.CROSSING

    t_min = -np.dot(relv, relposn) / smnd.calc_norm_sqr(relv)
    # assert(t_min > 0)

    relposn1 = relposn + t_min * relv

    d1, e1 = smnd.calc_polar_decomp(relposn1)
    r_total = agent.radius + ctragent.radius
    s1 = d1 - r_total

    vel_cone = None
    if s1 < smc.d_safe_IA_maximum and t_min > smc.T_SRCA_horizon:
        T_react = 0

        anticipated_posn_alpha = agent.posn
        anticipated_posn_beta = ctragent.posn
        agent_circle = smgs.Circle(anticipated_posn_alpha, agent.radius)
        ctragent_shape = smgs.Circle(anticipated_posn_beta, ctragent.radius)

        vel_cone = smgs.construct_tcone(agent_circle, ctragent_shape, t_min - T_react, True)
        if vel_cone == None:
            return None

        # print("vel. cone apex {} vs agent posn {}".format(vel_cone.apex, agent_circle.posn))
        assert (smnd.calc_norm_sqr(vel_cone.apex - agent_circle.posn) <= smnd.EPS_m_sqr)
        vel_cone.translate(-agent_circle.posn)

    approach_info = relposn1, d1, e1, r_total, s1, t_min, vel_cone
    # return approach_info, None, None
    assert(edge_relationship == smte.Relationship.CROSSING)

    # return edge_relationship, approach_info, None, None, None, None
    return smar.RepelReaction(s, relposn, relv), smar.RepelReaction(s, -relposn, -relv)

# def assess_IA_velocity_PtnVsBcl(self, agent, ctragent, relposn, relv, s, dshape):

def DEPRECATED_assess_approach(agent, ctragent, alpha_posn, alpha_dirn, beta_posn, beta_dirn, relposn, relv):
    #
    # BEGIN
    #
    hl_beta  = 0.5 * ctragent.rect_length
    hw_beta  = 0.5 * ctragent.rect_width

    agent_circle   = smgs.Circle(alpha_posn, agent.radius)
    ctragent_shape = smgs.Rectangle(beta_posn, beta_dirn, hl_beta, hw_beta, ctragent.radius)

    approach_info = ctragent_shape.get_nearest_approach_info(agent.posn, relv, smc.T_master_horizon)
    if approach_info == None:
        return None
    x_beta_ftr, x_alpha_ftr, d1, nv1, tv1, t_min = approach_info
    r_total = agent.radius + ctragent.radius

    relposn1 = x_beta_ftr - x_alpha_ftr
    FIX_2025_11_26 = True
    if FIX_2025_11_26:
        inside = ctragent_shape.contains(agent.posn + relposn1)
    else:
        inside = ctragent_shape.contains(agent.posn - (relposn1 - relposn))
    if inside:
        e1       = nv1
        s1       = -d1 - r_total
    else:
        e1       = -nv1
        s1       = d1 - r_total

    vel_cone = None
    if s1 < smc.d_safe_IA_maximum: # and t_min > smc.T_SRCA_horizon:
        if True: # False: # s1 < smd.d_safe_IA:
            cnv_spd = np.dot(relv, e1)
            t_min -= (smc.d_safe_IA_maximum - s1) / cnv_spd
            relposn1 = relposn + t_min * relv

        # assert(t_min >= 0)
        print("assess_IA_velocity_PtnVsCar ptn {} vs car {}, future relposn {}, relv {}, future s {}".format(agent.id, ctragent.id, relposn1, relv, s1))

        if t_min >= 0:
            vel_cone = smgs.construct_tcone(agent_circle, ctragent_shape, t_min, True)
            if vel_cone != None:
                print("vel. cone apex {} vs agent posn {}".format(vel_cone.apex, agent.posn))
                assert(smnd.calc_norm_sqr(vel_cone.apex - agent.posn) <= smnd.EPS_m_sqr)
                vel_cone.translate(-agent.posn)
        else:
                return None

        return relposn1, d1, e1, r_total, s1, t_min, vel_cone

    return None



# def assess_IA_velocity_BclVsBcl(agent, ctragent, relposn, relv):

# def assess_IA_velocity_BclVsCar(agent, ctragent, relposn, relv):

def assess_extrapolated_approach(agent, ctragent):
    relposn = ctragent.posn - agent.posn
    relv    = ctragent.v    - agent.v

    # y_alpha = np.dot(relposn, ctragent.dirn)
    # y_beta  = np.dot(relposn, agent.dirn)
    # v_alpha = np.dot(relv, ctragent.dirn)
    # v_beta  = np.dot(relv, agent.dirn)

    perp_dirn_alpha = smnd.rotate_90ACW(agent.dirn)
    perp_dirn_beta  = smnd.rotate_90ACW(ctragent.dirn)

    y_alpha = np.dot(relposn, perp_dirn_beta)
    y_beta  = np.dot(relposn, perp_dirn_alpha)
    v_alpha = np.dot(relv, perp_dirn_beta)
    v_beta  = np.dot(relv, perp_dirn_alpha)

    # y_alpha, v_alpha = agent.get_perpendicular_position_and_motion(ctragent.dirn)
    # y_beta,  v_beta  = ctragent.get_perpendicular_position_and_motion(agent.dirn)

    T_conflict = None
    if v_alpha ** 2 < smnd.EPS_m_sqr or v_beta ** 2 < smnd.EPS_m_sqr:
        # assert((v_alpha + v_beta)**2 < smnd.EPS_m_sqr)
        l_sum = agent.get_length_axis() + ctragent.get_length_axis()

        #
        # Parallel travel => no conflict
        #
        w_sum = agent.get_width_axis() + ctragent.get_width_axis()
        if y_alpha ** 2 > w_sum ** 2 and y_beta ** 2 > w_sum ** 2:
            return None

        priority = smip.prioritise_by_longitudinal_position(agent, ctragent)
        T_pass = smip.compute_passing_interval(agent, ctragent)
        if T_pass == None:
            return None

        T_conflict = T_pass
        yield_times = [T_pass[0], T_pass[0]]
        yield_times[priority] = None
    else:
        l_alpha_cross = agent.get_length_axis()    + ctragent.get_width_axis() + 2 * smc.d_safe_IA_beside
        l_beta_cross  = ctragent.get_length_axis() + agent.get_width_axis() + 2 * smc.d_safe_IA_beside

        if smc.PRUDENT_ACCELERATION:
            T_cross_alpha = smip.compute_crossing_interval(y_alpha, v_alpha, l_alpha_cross)
            T_cross_beta  = smip.compute_crossing_interval(y_beta, v_beta, l_beta_cross)

            if v_alpha < agent.get_target_speed():
                T_cross_alpha_ideal = smip.compute_crossing_interval(y_alpha, v_alpha, l_alpha_cross)
                T_cross_alpha = [T_cross_alpha_ideal[0], T_cross_alpha[1]]
            if v_beta < ctragent.get_target_speed():
                T_cross_beta_ideal = smip.compute_crossing_interval(y_alpha, v_alpha, l_alpha_cross)
                T_cross_beta = [T_cross_beta_ideal[0], T_cross_beta[1]]
        else:
            T_cross_alpha = smip.compute_crossing_interval(y_alpha, v_alpha, l_alpha_cross)
            T_cross_beta  = smip.compute_crossing_interval(y_beta, v_beta, l_beta_cross)
            if not smip.temporal_overlap_exists(T_cross_alpha, T_cross_beta):
                return None

        T_conflict = [max(T_cross_alpha[0], T_cross_beta[0]), min(T_cross_alpha[1], T_cross_beta[1])]
        priority = smip.prioritise_by_precedence(T_cross_alpha[0], T_cross_beta[0])
        yield_times = [T_cross_alpha[0], T_cross_beta[0]]

    return priority, T_conflict, yield_times

def assess_relative_approach(agent, ctragent, dshape):
    relposn = ctragent.posn - agent.posn
    relv    = ctragent.v    - agent.v
    r_total = dshape.radius

    FIX_2025_11_26 = True
    if FIX_2025_11_26:
        approach_info = dshape.get_nearest_approach_info(np.zeros(2), -relv, smc.T_master_horizon)
    else:
        approach_info = dshape.get_nearest_approach_info(np.zeros(2), -relv, smc.T_master_horizon)
    if approach_info == None:
        return None
    x_beta_ftr, x_alpha_ftr, d1, nv1, tv1, t_min = approach_info
    # assert(t_min > 0)

    FIX_2025_11_26 = True
    if FIX_2025_11_26:
        e1 = nv1
        s1 = d1 - r_total
        relposn1 = x_beta_ftr - x_alpha_ftr
        if smnd.calc_norm_sqr(relposn1) < smnd.EPS_m_sqr or dshape.contains(relposn - relposn1):
            inside = True
            if inside:
                e1 = -nv1
                s1 = -d1 - r_total
    else:
        e1 = -nv1
        s1 = d1 - r_total
        relposn1 = x_beta_ftr - x_alpha_ftr
        if smnd.calc_norm_sqr(relposn1) >= smnd.EPS_m_sqr:
            inside = dshape.contains(relposn - relposn1)
            if inside:
                # e1 = nv1
                s1 = -d1 - r_total

    if False:
        inside = dshape.contains(relposn - relposn1)
        if inside:
            e1 = nv1
            s1 = -d1 - r_total
        else:
            e1 = -nv1
            s1 = d1 - r_total

    vel_cone = None
    if t_min > smc.T_master_horizon:
        return None

    # t_min    = smc.T_master_horizon
    if s1 < smc.d_safe_IA_maximum:  # and t_min > smc.T_SRCA_horizon:
        if True:  # s1 < smc.d_safe_IA:
            cnv_spd = np.dot(relv, -e1)
            if cnv_spd ** 2 < smnd.EPS_m_sqr:
                t_min = 0
                relposn1 = relposn
            else:
                t_min -= (smc.d_safe_IA_maximum - s1) / cnv_spd
                relposn1 = relposn + t_min * relv

    #
    # @todo: Should use [ctr]agent.T_react
    #   NB: If (agent.T_react - ctragent.T_react)**2 >> 0, cones will need to be generated independently.
    #
    anticipated_posn_alpha = agent.posn
    anticipated_posn_beta = ctragent.posn
    anticipated_relposn = anticipated_posn_beta - anticipated_posn_alpha
    agent_circle = smgs.Circle(anticipated_posn_alpha, 0)
    # ctragent_shape = dshape.translated(anticipated_relposn - relposn)
    ctragent_shape = dshape.translated(anticipated_posn_beta)

    if s1 >= smc.d_safe_IA_maximum or t_min <= 0.:
        return None

    FIX_2025_11_06_TCONE_CONSTRUCTION = True
    if FIX_2025_11_06_TCONE_CONSTRUCTION:
        null_circle = smgs.Circle(np.zeros(2), 0)
        vel_cone     = smgs.construct_tcone(null_circle, dshape, t_min, True)
    else:
        vel_cone = smgs.construct_tcone(agent_circle, ctragent_shape, t_min, True)

    if vel_cone == None:
        return None

    # print("vel. cone apex {} vs agent posn {}".format(vel_cone.apex, agent_circle.posn))
    if FIX_2025_11_06_TCONE_CONSTRUCTION:
        assert (smnd.calc_norm_sqr(vel_cone.apex) <= smnd.EPS_m_sqr)
        vel_cone.translate(ctragent.v)
    else:
        assert (smnd.calc_norm_sqr(vel_cone.apex - agent_circle.posn) <= smnd.EPS_m_sqr)
        #
        # @todo: Consider alternative apex-candidates.
        #
        vel_cone.translate(ctragent.v - agent_circle.posn)



    approach_info = relposn1, d1, e1, r_total, s1, t_min, vel_cone
    return approach_info


def assess_IA_velocity_PtnVsCar_nonturning(agent, ctragent, edge_relationships, relposn, relv, s, dshape):
    DAI = assess_extrapolated_approach(agent, ctragent)
    if DAI == None:
        return None
    priority, T_conflict, yield_times = DAI

    edge_relationship, is_next_edge_alpha, is_next_edge_beta = edge_relationships
    if edge_relationship == smte.Relationship.INDEPENDENT:
        return None

    if is_next_edge_alpha and is_next_edge_beta and smc.YIELDING_ENABLED:
        if priority == 1:
            assert(yield_times[1] == None or yield_times[1] < yield_times[0])
            return smar.AwaitReaction(), None
        else:
            assert(priority == 0)
            return None, smar.AwaitReaction()
    elif is_next_edge_alpha and smc.YIELDING_ENABLED:
        priority = 1
        assert (priority == 1)
        return smar.AwaitReaction(), None
    elif is_next_edge_beta and smc.YIELDING_ENABLED:
        priority = 0
        assert (priority == 0)
        return None, smar.AwaitReaction()
    else:
        if yield_times[1] == None and yield_times[0] == None:
            assert(False)
            return None

        if priority == 1 and smc.VEERING_ENABLED:
            s = yield_times[0] * agent.get_target_speed()

            RAI = assess_relative_approach(agent, ctragent, dshape)
            if RAI == None:
                return None
            relposn1, d1, e1, r_total, s1, t_conflict, relv_cone = RAI
            # if relv_cone == None:
                # return None

            t_min = min(yield_times[0], t_conflict)


            return smar.VeerReaction(s, t_min, relv_cone), None
        elif priority == 0 and smc.YIELDING_ENABLED:
            # s = yield_times[1] * ctragent.get_target_speed()
            s_min = agent.radius + ctragent.rect_length + smc.d_safe_IA_maximum
            t_min = -np.dot(relposn, relv) / smnd.calc_norm_sqr(relv)
            stop_posn = ctragent.posn + t_min * ctragent.v - s_min * ctragent.dirn
            stop_shape = smgs.Circle(stop_posn, ctragent.rect_width)
            return None, smar.YieldReaction(stop_shape, np.dot(stop_posn - ctragent.posn, ctragent.dirn))
        else:
            # assert(False)
            return None

def DEPRECATED_assess_IA_velocity_PtnVsCar_turning(agent, ctragent, edge_relationships, relposn, relv, s, dshape):
    l_alpha = agent.get_length_axis()
    w_alpha = agent.get_width_axis()

    l_beta = ctragent.get_length_axis()
    w_beta = ctragent.get_width_axis()

    L_alpha_arc, alpha_chord_dirn, MH_alpha = agent.predict_navigation_trajectory(w_alpha + l_beta, True)
    L_beta_arc,  beta_chord_dirn,  MH_beta  = ctragent.predict_navigation_trajectory(w_beta + l_alpha, True)

    if MH_beta.contains(agent.posn):
        priority = 0

        # approach_info = assess_approach(agent, ctragent, agent.posn, agent.dirn, ctragent.posn, ctragent.dirn, relposn, relv)
        # if approach_info == None:
            # assert(False)
            # return None
        # yield_times = (None, approach_info[5])

        #
        #
        #
        assert(MH_alpha != None)
        x_omega, d, nv, tv = MH_alpha.get_nearest_point_info(ctragent.posn)
        yield_times = (None, d / ctragent.get_target_speed())

        # return approach_info, priority, yield_times
        # return edge_relationship, approach_info, priority, None, (MH_alpha, MH_beta), yield_times
        return edge_relationship, None, priority, None, (MH_alpha, MH_beta), yield_times
    else:
        approach_info = MH_beta.get_nearest_approach_info(agent.posn, relv, smc.T_master_horizon)
        if approach_info == None:
            # assert(False)
            return None
        x_hull_ftr, x_alpha_ftr, d1, nv1, tv1, t_min = approach_info

        entering = MH_beta.contains(x_alpha_ftr)
        if not entering:
            return None

        t_alpha_arrival = t_min

        y_beta, v_beta = ctragent.get_perpendicular_position_and_motion(agent.dirn)
        L_beta_cross = ctragent.get_width_axis() + agent.radius
        T_beta_cross = smip.compute_crossing_interval(y_beta, v_beta, L_beta_cross)
        t_beta_arrival = T_beta_cross[0]

        priority = smip.prioritise_by_precedence(t_alpha_arrival, t_beta_arrival)

        approach_info = assess_approach(agent, ctragent, agent.posn, agent.dirn, ctragent.posn, ctragent.dirn, relposn,
                                        relv)
        if approach_info == None:
            return None

        # return approach_info, priority, (t_alpha_arrival, t_beta_arrival)
        return edge_relationship, approach_info, priority, None, (MH_alpha, MH_beta), yield_times

#
# This function takes a legal rather than social approach to right-of-way:
#   agents will give way to someone already in an intersection;
#   turning vehicles will give way to pedestrians;
#
def assess_IA_velocity_PtnVsCar_turning(agent, ctragent, edge_relationships, relposn, relv, s, dshape):
    edge_relationship, is_next_edge_alpha, is_next_edge_beta = edge_relationships
    if edge_relationship == smte.Relationship.INDEPENDENT:
        return None

    if is_next_edge_alpha and is_next_edge_beta:
        L_arc_alpha = agent.estimate_arclength_to_curdst()
        L_arc_beta = ctragent.estimate_arclength_to_curdst()

        spd1 = smnd.calc_norm(agent.v)
        spd2 = smnd.calc_norm(ctragent.v)

        t_entry_alpha = L_arc_alpha / spd1
        t_entry_beta  = L_arc_beta  / spd2
        if t_entry_alpha <= t_entry_beta and smc.YIELDING_ENABLED:
            return None, smar.AwaitReaction()
        elif smc.YIELDING_ENABLED:
            return smar.AwaitReaction(), None
        else:
            return None
    elif is_next_edge_alpha:
        assert(not is_next_edge_beta)
        if smc.YIELDING_ENABLED:
            return smar.AwaitReaction(), None
        else:
            return None
    elif is_next_edge_beta:
        assert(not is_next_edge_alpha)

        DAI = assess_extrapolated_approach(agent, ctragent)
        if DAI == None:
            return None
        priority, T_conflict, yield_times = DAI
        FIX_2020_12_05 = True
        if FIX_2020_12_05:
            if priority == 0 and (yield_times[0] == None or yield_times[0] <= smc.T_master_horizon) and yield_times[1] <= smc.T_master_horizon and smc.YIELDING_ENABLED:
                return None, smar.AwaitReaction()
        else:
            if priority == 0 and (yield_times[0] == None or yield_times[0] >= smc.T_master_horizon) and yield_times[1] < smc.T_master_horizon and smc.YIELDING_ENABLED:
                return None, smar.AwaitReaction()

        if yield_times[0] != None:
            s = yield_times[0] * agent.get_target_speed()
        elif yield_times[1] != None:
            s = yield_times[1] * ctragent.get_target_speed()

        # reaction1 = None
        reaction1 = reaction2 = None
        RAI = assess_relative_approach(agent, ctragent, dshape)
        if RAI != None:
            # if False:
            assert(yield_times[0] != None)
            if yield_times[0] <= smc.T_master_horizon and smc.VEERING_ENABLED:
                relposn1, d1, e1, r_total, s1, t_conflict, relv_cone = RAI
                t_min = min(yield_times[0], t_conflict)
                ENHANCEMENT_2025_11_24 = True
                if not ENHANCEMENT_2025_11_24:
                    if s1 <= smc.d_safe_IA_maximum:
                        reaction1 = smar.VeerReaction(s, t_min, relv_cone)
            if yield_times[1] != None and yield_times[1] <= smc.T_master_horizon and smc.YIELDING_ENABLED:
                reaction2 = smar.AwaitReaction()

        return reaction1, reaction2


        RAI = assess_relative_approach(agent, ctragent, dshape)
        if RAI != None and priority == 0 and smc.VEERING_ENABLED:
            relposn1, d1, e1, r_total, s1, t_conflict, relv_cone = RAI
            t_min = min(yield_times[0], t_conflict)
            return smar.VeerReaction(s, t_min, relv_cone), smar.AwaitReaction()

        if smc.YIELDING_ENABLED:
            return None, smar.AwaitReaction()
        else:
            return None

    else:
        assert (not is_next_edge_alpha and not is_next_edge_beta)

        pts_alpha = agent.construct_cubic_spline()
        pts_beta  = ctragent.construct_cubic_spline()

        l1 = agent.radius
        l2 = 0.5 * ctragent.rect_length
        spline_crossing_info = smnd.find_first_spline_intersection(pts_alpha, pts_beta, l1, l2)
        if spline_crossing_info == None:
            return None  # , None
        x_conflict, s1, s2, e1, e2 = spline_crossing_info

        spln1a, spln1b = smnd.subdivide_cubic_spline(pts_alpha, s1)
        L1 = smnd.calc_spline_arclength(spln1a)
        spd1 = smnd.calc_norm(agent.v)
        t1_cross = L1 / spd1
        if s1 < 0:
            t1_cross = -t1_cross
        # dt1 = agent.radius / spd1
        dt1 = (agent.radius + 0.5 * ctragent.rect_width) / spd1
        t1_min = t1_cross - dt1

        spln2a, spln2b = smnd.subdivide_cubic_spline(pts_beta, s2)
        L2 = smnd.calc_spline_arclength(spln2a)
        spd2 = smnd.calc_norm(ctragent.v)
        t2_cross = L2 / spd2
        if s2 < 0:
            t2_cross = -t2_cross
        # dt2 = 0.5 * ctragent.rect_length / spd2
        dt2 = (0.5 * ctragent.rect_length + agent.radius) / spd2
        t2_min = t2_cross - dt2

        if max(t1_min, t2_min) > smc.T_master_horizon:
            return None

        if t1_min < t2_min and smc.YIELDING_ENABLED:
            # stop_shape = smgs.Circle(x - ctragent.rect_length * e2, ctragent.rect_width)
            e2_perp = smnd.rotate_90ACW(e2)
            # stop_posn = spln2a[3] - 0.5 * ctragent.rect_length * e2
            s_stop = L2 - 0.5 * ctragent.rect_length
            stop_posn = smnd.evaluate_cubic_spline(spln2a, s_stop / L2)
            # stop_shape = smgs.create_segment(stop_posn, e2_perp, agent.radius, ctragent.rect_width)
            stop_shape = smgs.Circle(stop_posn, ctragent.rect_width)
            return None, smar.YieldReaction(stop_shape, s_stop)
        elif t1_min > t2_min and smc.YIELDING_ENABLED:
            stop_shape = smgs.Circle(spln1a[3] - agent.radius * e1, agent.radius)
            s = L1 - agent.radius
            return smar.YieldReaction(stop_shape, s), None

    # assert(False)

    #
    # In the case of turning vehicles, one of the agents is expected to yield before entering an intersection.
    # @todo: Consider using estimated hull entry times to prioritise
    #
    return None, None

def assess_IA_velocity_PtnVsCar(agent, ctragent, relposn, relv, s, dshape):
    priority = None

    edge_relationships = classify_edge_relationships(agent, ctragent)
    edge_relationship, is_next_edge_alpha, is_next_edge_beta = edge_relationships
    assert(edge_relationship not in {smte.Relationship.SHARING, smte.Relationship.MERGING})
    if edge_relationship == smte.Relationship.INDEPENDENT:
        return None

    assert(edge_relationship == smte.Relationship.CROSSING)
    assert (ctragent.conflict_state.state_id != smas.ConflictStateId.VEERING)

    turning_beta = None
    if is_next_edge_beta:
        cur_src = ctragent.get_current_src_node()

        # cur_dst_id = self.route[self.routing_idx + 1]
        # cur_dst     = self.nodegraph.nodes[cur_dst_id]
        cur_dst = ctragent.get_current_dst_node()

        # next_dst_id = self.route[self.routing_idx + 2]
        # next_dst     = self.nodegraph.nodes[next_dst_id]
        next_dst = ctragent.get_next_dst_node()

        dx1 = cur_dst.approach_shape.posn - cur_src.approach_shape.posn
        dx2 = next_dst.approach_shape.posn - cur_dst.approach_shape.posn

        L1, e1 = smnd.calc_polar_decomp(dx1)
        L2, e2 = smnd.calc_polar_decomp(dx2)

        turning_beta = np.dot(e1, e2) < smad.cos_theta_max[smad.AgentType.CAR]
    else:
        turning_beta = ctragent.goal_state.state_id == smas.GoalStateId.NAVIGATING and \
                  ctragent.goal_state.turning_info != None

    if not turning_beta: # ctragent.goal_state.state_id != smas.GoalStateId.TURNING and :
        return assess_IA_velocity_PtnVsCar_nonturning(agent, ctragent, edge_relationships, relposn, relv, s, dshape)
    else:
        return assess_IA_velocity_PtnVsCar_turning(agent, ctragent, edge_relationships, relposn, relv, s, dshape)

#
# Neither car is TURNING
#
def DEPRECATED_assess_IA_velocity_CarVsCar_straight(agent, ctragent, edge_relationships, relposn, relv, s, dshape):
    edge_relationship, is_next_edge_alpha, is_next_edge_beta = edge_relationships
    if is_next_edge_alpha and is_next_edge_beta:
        l_alpha = agent.get_length_axis()
        w_alpha = agent.get_width_axis()

        l_beta = ctragent.get_length_axis()
        w_beta = ctragent.get_width_axis()

        L_alpha_arc, alpha_chord_dirn, MH_alpha = agent.predict_navigation_trajectory(w_alpha + l_beta, True)
        L_beta_arc, beta_chord_dirn, MH_beta = ctragent.predict_navigation_trajectory(w_beta + l_alpha, True)

        t_edge_entry_alpha = L_alpha_arc / smnd.calc_norm(agent.v)
        t_edge_entry_beta  = L_beta_arc  / smnd.calc_norm(ctragent.v)

        if t_edge_entry_alpha < t_edge_entry_beta:
            return None, smar.AwaitReaction(ctragent, agent)
        elif t_edge_entry_beta < t_edge_entry_alpha:
            return smar.AwaitReaction(agent, ctragent), None
        else:
            priority = prioritise_by_lateral_position(agent, ctragent)
            if priority == 0:
                return None, smar.AwaitReaction(ctragent, agent)
            else:
                return smar.AwaitReaction(ctragent, agent), None
    elif is_next_edge_alpha:
        #
        # @todo: Consider the case where beta will exit its conflicting edge before alpha enters its conflicting edge.
        #
        return smar.AwaitReaction(), None
    elif is_next_edge_beta:
        #
        # @todo: Consider the case where alpha will exit its conflicting edge before beta enters its conflicting edge.
        #
        return None, smar.AwaitReaction()



    assert(edge_relationship == smte.Relationship.CROSSING)
    assert (dshape != None)

    XAI = assess_extrapolated_approach(agent, ctragent)
    if XAI == None:
        return None
    priority, T_conflict, yield_times = XAI

    if priority == 0:
        dx = yield_times[1] * ctragent.v
        s         = np.dot(dx, ctragent.dirn)

        stop_posn = ctragent.posn + ds
        stop_shape = smgs.Circle(stop_posn, ctragent.rect_width)

        return None, smar.YieldReaction(stop_shape, s)
    else:
        dx = yield_times[0] * agent.v
        s         = np.dot(dx, agent.dirn)

        stop_posn = agent.posn + dx
        stop_shape = smgs.Circle(stop_posn, agent.rect_width)

        return smar.YieldReaction(stop_shape, s), None

def DEPRECATED_assess_IA_velocity_CarVsCar_turning(agent, ctragent, edge_relationships, relposn, relv, s, dshape):
    edge_relationship, is_next_edge_alpha, is_next_edge_beta = edge_relationships
    if is_next_edge_alpha and is_next_edge_beta:
        l_alpha = agent.get_length_axis()
        w_alpha = agent.get_width_axis()

        l_beta = ctragent.get_length_axis()
        w_beta = ctragent.get_width_axis()

        L_alpha_arc, alpha_chord_dirn, MH_alpha = agent.predict_navigation_trajectory(w_alpha + l_beta,    True)
        L_beta_arc,  beta_chord_dirn,  MH_beta  = ctragent.predict_navigation_trajectory(w_beta + l_alpha, True)

        t_edge_entry_alpha = L_alpha_arc / smnd.calc_norm(agent.v)
        t_edge_entry_beta  = L_beta_arc  / smnd.calc_norm(ctragent.v)

        if t_edge_entry_alpha < t_edge_entry_beta:
            return None, smar.AwaitReaction(ctragent, agent)
        elif t_edge_entry_beta < t_edge_entry_alpha:
            return smar.AwaitReaction(agent, ctragent), None
        else:
            priority = prioritise_by_lateral_position(agent, ctragent)
            if priority == 0:
                return None, smar.AwaitReaction(ctragent, agent)
            else:
                return smar.AwaitReaction(ctragent, agent), None
    elif is_next_edge_alpha:
        #
        # @todo: Consider the case where beta will exit its conflicting edge before alpha enters its conflicting edge.
        #
        return smar.AwaitReaction(), None
    elif is_next_edge_beta:
        #
        # @todo: Consider the case where alpha will exit its conflicting edge before beta enters its conflicting edge.
        #
        return None, smar.AwaitReaction()

    #
    # In the case of turning vehicles, one of the agents is expected to yield before entering an intersection.
    # @todo: Consider using estimated hull entry times to prioritise
    #
    return None, None

def assess_IA_velocity_CarVsCar(agent, ctragent, relposn, relv, s, dshape):
    assert(agent.conflict_state.state_id != smas.ConflictStateId.VEERING)
    assert(ctragent.conflict_state.state_id != smas.ConflictStateId.VEERING)

    priority = None

    edge_relationships = classify_edge_relationships(agent, ctragent)
    edge_relationship, is_next_edge_alpha, is_next_edge_beta = edge_relationships
    if edge_relationship == smte.Relationship.INDEPENDENT:
        return None
    elif edge_relationship == smte.Relationship.SHARING:
        # l_sum = agent.get_length_axis() + ctragent.get_length_axis()
        l_sum = 0.5 * (agent.rect_length + ctragent.rect_length)
        L_arc_alpha = agent.estimate_arclength_to_curdst()
        L_arc_beta  = ctragent.estimate_arclength_to_curdst()

        # L_edge_alpha = agent.estimate_arclength_of_current_edge()
        # L_edge_beta  = ctragent.estimate_arclength_of_current_edge()

        L_exit_alpha = L_exit_beta = None
        if is_next_edge_alpha and not is_next_edge_beta:
            # dL = -(L_arc_alpha + (L_edge_beta - L_arc_beta))
            # dL = L_arc_beta - (L_arc_alpha - L_edge_alpha)
            L_edge_alpha = agent.estimate_arclength_of_next_edge()

            L_exit_alpha = L_arc_alpha + L_edge_alpha
            L_exit_beta  = L_arc_beta
            # dL = L_arc_beta - (L_arc_alpha + L_edge_beta)
        elif is_next_edge_beta and not is_next_edge_alpha:
            L_edge_beta = ctragent.estimate_arclength_of_next_edge()

            L_exit_alpha = L_arc_alpha
            L_exit_beta  = L_arc_beta + L_edge_beta
            # dL = (L_arc_beta + L_edge_alpha) - L_arc_alpha
        else:
            L_exit_alpha = L_arc_alpha
            L_exit_beta  = L_arc_beta
            # dL = L_arc_beta - L_arc_alpha

        dL = L_exit_beta - L_exit_alpha

        s = np.abs(dL) - l_sum
        if dL >= 0:
            return None, smar.FollowReaction(s)
        else:
            return smar.FollowReaction(s), None

        assert(False)
        # return edge_relationship, None, priority, s, (MH_alpha, MH_beta), None
    elif edge_relationship == smte.Relationship.MERGING:
        l_sum = agent.get_length_axis() + ctragent.get_length_axis()
        L_arc_alpha = agent.estimate_arclength_to_curdst()
        L_arc_beta = ctragent.estimate_arclength_to_curdst()

        L_edge_alpha = agent.estimate_arclength_of_current_edge()
        L_edge_beta = ctragent.estimate_arclength_of_current_edge()

        if is_next_edge_alpha and is_next_edge_beta and smc.YIELDING_ENABLED:
            cd1 = agent.get_current_dst_node()
            nd1 = agent.get_next_dst_node()
            edge1 = cd1.get_edge(nd1)

            cd2 = ctragent.get_current_dst_node()
            nd2 = ctragent.get_next_dst_node()
            edge2 = cd2.get_edge(nd2)

            assert (not edge1.priority is None)
            assert (not edge2.priority is None)
            assert (edge1.priority != edge2.priority)
            if edge1.priority > edge2.priority:
                return None, smar.AwaitReaction()
            else:
                assert (edge1.priority < edge2.priority)
                return smar.AwaitReaction(), None

        elif (is_next_edge_alpha or is_next_edge_beta) and smc.YIELDING_ENABLED:
            if is_next_edge_alpha:
                return smar.AwaitReaction(), None
            else:
                assert (is_next_edge_beta)
                return None, smar.AwaitReaction()
        else:
            dL = L_arc_beta - L_arc_alpha

            s = np.abs(dL) - l_sum
            if dL >= 0:
                if s <= 0.:
                    return None, smar.AwaitReaction()

                return None, smar.FollowReaction(s)
            else:
                if s <= 0.:
                    return smar.AwaitReaction(), None

                return smar.FollowReaction(s), None

        assert (False)
        # return edge_relationship, None, priority, s, (MH_alpha, MH_beta), None

    elif edge_relationship == smte.Relationship.DIVERGING:
        #
        # If the vehicles are likely to collide, one of them should already be following the other.
        #
        return None

        collision_info = assess_extrapolated_approach(agent, ctragent)
        if collision_info == None:
            return None
        priority, T_conflict, yield_times = collision_info
        if priority == 1:
            s = np.dot(agent.v, agent.dirn) * yield_times[0]
            return smar.FollowReaction(s), None
        else:
            s = np.dot(ctragent.v, ctragent.dirn) * yield_times[1]
            return None, smar.FollowReaction(s)

    assert(edge_relationship == smte.Relationship.CROSSING)

    #
    # if both edges are next, use edge-priority to choose one AWAITING agent
    # if one edge is next, that agent must yield by AWAITING
    # if neither edge is next:
    #   find the intersection's (x, s1, s2, e1, e2)-info
    #   use de Casteljau to subdivide splines at the intersection-point;
    #   use Gravesen to compute arc-lengths until intersection-point
    #   t_i  = L_i / v_i
    #   dt_i = l_i / v_i
    #   T_i  = [t_i - dt_i, t_i + dt_i]
    #   if T_1 T_2 != Nullset:
    #       if min T1 <= min T2:
    #           t_yield = min T2
    #           stop_shape = Circle(x - l2 * e2, w2) # consider perpendicular segment
    #           agent2 YIELDING at stop_shape
    #       else:
    #           t_yield = min T1
    #           stop_shape = Circle(x - l1 * e1, w1) # consider perpendicular segment
    #           agent1 YIELDING at stop_shape
    #
    if is_next_edge_alpha and is_next_edge_beta:
        if not smc.YIELDING_ENABLED:
            return None

        cd1   = agent.get_current_dst_node()
        nd1   = agent.get_next_dst_node()
        edge1 = cd1.get_edge(nd1)

        cd2   = ctragent.get_current_dst_node()
        nd2   = ctragent.get_next_dst_node()
        edge2 = cd2.get_edge(nd2)

        assert(not edge1.priority is None)
        assert(not edge2.priority is None)
        assert(edge1.priority != edge2.priority)
        if edge1.priority > edge2.priority:
            return None, smar.AwaitReaction()
        else:
            assert(edge1.priority < edge2.priority)
            return smar.AwaitReaction(), None

    elif (is_next_edge_alpha or is_next_edge_beta):
        if not smc.YIELDING_ENABLED:
            return None

        if is_next_edge_alpha:
            return smar.AwaitReaction(), None
        else:
            assert(is_next_edge_beta)
            return None, smar.AwaitReaction()
    else:
        assert(not is_next_edge_alpha and not is_next_edge_beta)

        cd1 = agent.get_current_dst_node()
        cd2 = ctragent.get_current_dst_node()

        # assert(agent.goal_state == smas.GoalStateId.NAVIGATING)
        pts_alpha = agent.construct_cubic_spline()

        # assert(ctragent.goal_state == smas.GoalStateId.NAVIGATING)
        pts_beta = ctragent.construct_cubic_spline()

        l_alpha = 0.5 * agent.rect_length
        l_beta = 0.5 * ctragent.rect_length
        spline_crossing_info = smnd.find_first_spline_intersection(pts_alpha, pts_beta, l_alpha, l_beta)
        if spline_crossing_info == None:
            return None # , None
        x, s1, s2, e1, e2 = spline_crossing_info

        spln1a, spln1b = smnd.subdivide_cubic_spline(pts_alpha, s1)
        L1 = smnd.calc_spline_arclength(spln1a)
        spd1 = smnd.calc_norm(agent.v)
        t1_cross = L1 / spd1
        dt1      = agent.rect_length    / spd1
        t1_min   = t1_cross - dt1

        spln2a, spln2b = smnd.subdivide_cubic_spline(pts_beta, s2)
        L2 = smnd.calc_spline_arclength(spln2a)
        spd2 = smnd.calc_norm(ctragent.v)
        t2_cross = L2 / spd2
        dt2      = ctragent.rect_length / spd2
        t2_min   = t2_cross - dt2

        if t1_min < t2_min and smc.YIELDING_ENABLED:
            l_safety   = 0.5 * (ctragent.rect_length + agent.rect_width)
            x2         = x - l_safety * e2
            stop_shape = smgs.Circle(x2, ctragent.rect_width)
            # e2_perp = smnd.rotate_90ACW(e2)
            # stop_shape = smgs.create_segment(x2, e2_perp, ctragent.rect_width, agent.rect_width)
            s          = L2 - 0.5 * (ctragent.rect_length + agent.rect_width)
            return None, smar.YieldReaction(stop_shape, s)
        elif t1_min > t2_min and smc.YIELDING_ENABLED:
            l_safety   = 0.5 * (agent.rect_length + ctragent.rect_width)
            x1         = x - l_safety * e1
            stop_shape = smgs.Circle(x - agent.rect_length * e1, agent.rect_width)
            # e1_perp = smnd.rotate_90ACW(e1)
            # stop_shape = smgs.create_segment(x1, e1_perp, agent.rect_width, ctragent.rect_width)
            s          = L1 - 0.5 * (agent.rect_length + ctragent.rect_width)
            return smar.YieldReaction(stop_shape, s), None
        else:
            cs1 = agent.get_current_src_node()
            cs2 = ctragent.get_current_src_node()

            edge1 = cs1.get_edge(cd1)
            edge2 = cs2.get_edge(cd2)

            assert(edge1.priority != edge2.priority)
            if edge1.priority > edge2.priority and smc.YIELDING_ENABLED:
                l_safety = 0.5 * (ctragent.rect_length + agent.rect_width)
                x2       = x - l_safety * e2

                stop_shape = smgs.Circle(x2, ctragent.rect_width)
                s          = L2 - ctragent.rect_length
                return None, smar.YieldReaction(stop_shape, s)
            elif smc.YIELDING_ENABLED:
                l_safety = 0.5 * (agent.rect_length + ctragent.rect_width)
                x1       = x - l_safety * e1

                stop_shape = smgs.Circle(x1, agent.rect_width)
                s          = L1 - agent.rect_length
                return smar.YieldReaction(stop_shape, s), None

        # assert(False)
        return None, None
    assert(False)
    return None




    #
    # The cars' paths are known to cross.
    #
    # if agent.goal_state.state_id != smas.GoalStateId.TURNING and ctragent.goal_state.state_id != smas.GoalStateId.TURNING:
        #
        # Neither car is turning
        #
        # return assess_IA_velocity_CarVsCar_straight(agent, ctragent, edge_relationships, relposn, relv, s, dshape)
    # else:
        #
        # At least one car is turning.
        #
        # return assess_IA_velocity_CarVsCar_turning(agent, ctragent, edge_relationships, relposn, relv, s, dshape)

def assess_IA_velocity(agent, ctragent, relposn, relv, s, dshape):
    #
    # If an agent is assessing the counteragent's velocity, we assume that the agents are not so near as
    # to justify further consideration of non-converging counteragents.
    #
    # assert(s >= smd.d_reflex_IA)


    converging = np.dot(relposn, relv) < 0.
    # if not converging:
    if False:
        #
        # Interaction can be ignored. Note: the interaction will not be logged!
        #
        return None

    if agent.agent_type == smad.AgentType.PEDESTRIAN:
        if ctragent.agent_type == smad.AgentType.PEDESTRIAN:
            return assess_IA_velocity_PtnVsPtn(agent, ctragent, relposn, relv, s, dshape)

        # elif ctragent.agent_type == smad.AgentType.BICYCLE:
            # return assess_IA_velocity_PtnVsBcl(agent, ctragent, relposn, relv, s, dshape)
        else:
            assert(ctragent.agent_type == smad.AgentType.CAR)
            return assess_IA_velocity_PtnVsCar(agent, ctragent, relposn, relv, s, dshape)

    # elif agent.agent_type == smad.AgentType.BICYCLE:
        # if ctragent.agent_type == smad.AgentType.BICYCLE:
            # return assess_IA_velocity_BclVsBcl(agent, ctragent, relposn, relv, s, dshape)
        # else:
            # assert(ctragent.agent_type == smad.AgentType.CAR)
            # return assess_IA_velocity_BclVsCar(agent, ctragent, relposn, relv, s, dshape)

    else:
        assert(agent.agent_type == smad.AgentType.CAR)
        assert(ctragent.agent_type == smad.AgentType.CAR)

        return assess_IA_velocity_CarVsCar(agent, ctragent, relposn, relv, s, dshape)


