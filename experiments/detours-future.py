#
# Detour class, one simple subclasses, ArcDetour
#
# @todo: Add a composite subclass, PolyarcDetour, an open shape comprising a set of small circles and intermediate segments.
# @todo: Add an annular arc.
#

import numpy as np
import view.view as svv
# import model.force as smf
import model.defs as smd
import model.numerical.defs as smnd


    def calc_detour_properties(agent, ctragent, x_alphastart, u_alpha, x_detour, u_detour):
        r_sqr =
        theta = 
        T = theta / self.max_angular_velocity
        return (T, r_sqr)

    def determine_viable_detours(agent, ctragent, interaction, agent_tbl, zone_cfg):
        detours = set()

        if interaction.t_min > horizon:
            continue

        if interaction.t_min < 0:
            continue

        id1, id2     = intrctn_codec.decode_interaction(interaction.combined_id)
        assert(self.id == id1 or  self.id == id2)
        id_ctragent  = id1 ^ id2 ^ self.id
        assert(id_ctragent == id1 or id_ctragent == id2)
        assert(self.id != id_ctragent)

        ctragent_idx       = id_ctragent == id2
        ctragent_type      = interaction.types[ctragent_idx]
        ctragent_awareness = interaction.awarenesses[ctragent_idx]

        ctragent     = agent_tbl[id_ctragent]
        # ctragent     = agent_tbl[ctragent_type][id_ctragent]
        rel_priority = self.assess_zone_priority(self, ctragent, zone_cfg)
            
        print("Agent::calc_external_influence: agent {} has relative priority {} vs counteragent {}".format(self.id, rel_priority, id_ctragent))

        #
        # @todo Consider using a more subtle scheme than absolute priority.
        #
        if rel_priority > 0:
            continue

        x_alpha = self.posn
        v_alpha = self.v
        x_beta = ctragent.posn
        v_beta = ctragent.v

        rel_posn = x_beta - x_alpha
        spd_alpha, dirn_alpha = smnd.calc_polar_decomp(v_alpha)
        spd_beta, dirn_beta = smnd.calc_polar_decomp(v_beta)

        x_alphastart = x_alpha + self.T_react * v_alpha
        x_betafinish = x_beta + interaction.t_min * v_beta

        angvel = np.pi / 4
        if self.agent_type == smad.BICYCLE:
            angvel = np.pi / 6
        r_min_sqr = smnd.calc_norm_sqr(v_alpha) / angvel**2
        if r_sqr > r_min_sqr and T < t_min:
            detours.add(Detour(...))

        if interaction.approach == Approach.LONGITUDINAL:
            #
            # Check both lateral detours.
            #

            #
            # Right (counteragent's left)
            #
            u_alpha  = -smnd.rotate_90ACW(dirn_alpha)
            u_detour = smnd.rotate_90ACW(dirn_beta)
            x_detour = x_betafinish + d_safety_before * u_detour

            T, r_sqr = calc_detour_properties(x_alphastart, u_alpha, x_detour, u_detour)
            if r_sqr > r_min_sqr and T < t_min:
                detours.add(Detour(...))

            #
            # Left (counteragent's right)
            #
            assert(smnd.calc_det2x2(rel_posn, v_alpha))
            u_alpha  = smnd.rotate_90ACW(dirn_alpha)
            u_detour = -smnd.rotate_90ACW(dirn_beta)
            x_detour = x_betafinish + d_safety_before * u_detour

            T, r_sqr = calc_detour_properties(x_alphastart, u_alpha, x_detour, u_detour)
            if r_sqr > r_min_sqr and T < t_min:
                detours.add(Detour(...))

        elif interaction.approach == Approach.TRANSVERSE:
            #
            # Check detours before and behind counteragent.
            #

            #
            # Before
            #
            T_cross         = ctragent.get_width() / self.get_desired_speed()
            d_safety_before = d_safety + 0.5 * ctragent.length + 0.5 * self.width
            
            u_detour = dirn_beta
            x_detour = x_betafinish + d_safety_before * u_detour

            if smnd.calc_det2x2(v_alpha, x_detour - x_alpha) > 0:
                #
                # Rightwards
                #
                u_alpha  = smnd.rotate_90ACW(dirn_alpha)
                T, r_sqr = calc_detour_properties(x_alphastart, u_alpha, x_detour, u_detour)
                if r_sqr > r_min_sqr and T < t_min:
                    detours.add(Detour(...))
            else:
                #
                # Leftwards
                #
                u_alpha  = -smnd.rotate_90ACW(dirn_alpha)
                T, r_sqr = calc_detour_properties(x_alphastart, u_alpha, x_detour, u_detour)
                if r_sqr > r_min_sqr and T < t_min:
                    detours.add(Detour(...))

            #
            # Behind
            #
            u_detour = -dirn_beta
            x_detour = x_betafinish + d_safety_behind * u_detour
            if smnd.calc_det2x2(v_alpha, x_detour - x_alpha) > 0:
                #
                # Rightwards
                #
                u_alpha  = smnd.rotate_90ACW(dirn_alpha)
                T, r_sqr = calc_detour_properties(x_alphastart, u_alpha, x_detour, u_detour)
                if r_sqr > r_min_sqr and T < t_min:
                    detours.add(Detour(...))
            else:
                #
                # Leftwards
                #
                u_alpha  = -smnd.rotate_90ACW(dirn_alpha)
                T, r_sqr = calc_detour_properties(x_alphastart, u_alpha, x_detour, u_detour)
                if r_sqr > r_min_sqr and T < t_min:
                    detours.add(Detour(...))
        else:
            if smnd.calc_det2x2(rel_posn, v_alpha) > 0:
                #
                # Right (counteragent's left) lateral
                #
                u_alpha  = -smnd.rotate_90ACW(dirn_alpha)
                u_detour = smnd.rotate_90ACW(dirn_beta)
                x_detour = x_betafinish + d_safety_before * u_detour
                T, r_sqr = calc_detour_properties(x_alphastart, u_alpha, x_detour, u_detour)
                if r_sqr > r_min_sqr and T < t_min:
                    detours.add(Detour(...))

                #
                # [Left] before
                #
                u_alpha  = smnd.rotate_90ACW(dirn_alpha)
                u_detour = dirn_beta
                x_detour = x_betafinish + d_safety_before * u_detour
                T, r_sqr = calc_detour_properties(x_alphastart, u_alpha, x_detour, u_detour)
                if r_sqr > r_min_sqr and T < t_min:
                    detours.add(Detour(...))
            else:
                assert(smnd.calc_det2x2(rel_posn, v_alpha))
                u_alpha  = smnd.rotate_90ACW(dirn_alpha)
                u_detour = -smnd.rotate_90ACW(dirn_beta)
                x_detour = x_betafinish + d_safety_before * u_detour
                T, r_sqr = calc_detour_properties(x_alphastart, u_alpha, x_detour, u_detour)
                if r_sqr > r_min_sqr and T < t_min:
                    detours.add(Detour(...))

                #
                # [Right] before
                #
                u_alpha  = -smnd.rotate_90ACW(dirn_alpha)
                u_detour = dirn_beta
                x_detour = x_betafinish + d_safety_before * u_detour                
                T, r_sqr = calc_detour_properties(x_alphastart, u_alpha, x_detour, u_detour)
                if r_sqr > r_min_sqr and T < t_min:
                    detours.add(Detour(...))
        
        return detours


class Detour:
    def __init__(self):

        return

# e.g. Repulsive: tree/fountain/sculpture, pool, pillar
# e.g. Attractive: tree/fountain/sculpture, pool, public transport stops, seats
# e.g. Obscuring: tree/fountain/sculpture, pillar
class ArcDetour(Detour):
    def __init__(self, posn, r):
        super(Detour, self).__init__()

        return

#
# Note: Only two successive arcs are currently supported.
#
class PolyarcDetour(Detour):
    def __init__(self, pts, radius):
        super(Detour, self).__init__()

        return
