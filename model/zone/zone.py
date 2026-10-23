import numpy as np
# import scipy as sp
# import scipy.stats as ss
# import pygame as pg

# from collections import deque

from model.interaction.interactor import Interactor

import view.defs as svd
import view.view as svv
# import model.model as smm
import model.defs as smd
import model.configuration as smc
import model.agency.defs as smad
# import model.numerical.defs as smnd
import model.agency.agent as smaa
import model.force as smf
import model.geometry.shape as smgs
import model.zone.defs as smzd
import model.zone.zone as smzz

ptn_exclusion  = (smad.AgentType.PEDESTRIAN, 10., 10., 0.2)
bike_exclusion = (smad.AgentType.BICYCLE, 10., 10., 0.2)
car_exclusion  = (smad.AgentType.CAR, 10., 10., 0.2)

ptn_priority     = {smad.AgentType.PEDESTRIAN:1, smad.AgentType.BICYCLE:0, smad.AgentType.CAR:0}
bike_priority    = {smad.AgentType.PEDESTRIAN:0, smad.AgentType.BICYCLE:1, smad.AgentType.CAR:0}
car_priority     = {smad.AgentType.PEDESTRIAN:0, smad.AgentType.BICYCLE:0, smad.AgentType.CAR:1}
active_priority  = {smad.AgentType.PEDESTRIAN:1, smad.AgentType.BICYCLE:1, smad.AgentType.CAR:0}
vehicle_priority = {smad.AgentType.PEDESTRIAN:0, smad.AgentType.BICYCLE:1, smad.AgentType.CAR:1}
shared_priority  = {smad.AgentType.PEDESTRIAN:0, smad.AgentType.BICYCLE:0, smad.AgentType.CAR:0}

class Regime:
    def __init__(self, exclusions, priorities, directionality = None):
        self.exclusions = {}
        for exclsn in exclusions:
            key   = exclsn[0]
            value = exclsn[1:]
            self.exclusions[key] = value
        self.priorities     = priorities
        self.directionality = directionality
        return

#
# @todo: access zones could be augmented with attractive forces too.
#
# If there is a nearby AccessZone which offers an agent more exclusivity or priority
# than they currently enjoy, then apply an attractive force to the agent.
#
# Possible examples:
#     ptn_attraction  = (0, A)
#     bike_attraction = (1, A)
#     car_attraction  = (2, A)
#
# More details:
#     We need a definition of "nearby": how about:
#         np.dot(x_omega_new - x_omega_old, x_Alpha - x_omega_old) > 0
#     as part of this definition?
#     If there is an agent near an AccessZone with A = A_possible and the agent
#     is in a Zone with A = A_now, set dA = A_possible - A_now.
#         if dA > 0:
#             x_omega = zone.get_nearest_point_info()[0]
#             dist, dx = SMND.calc_polar_decomp(x_omega - x_agent)
#             w \exists [0, 1], e.g.
#                 w = dA / (A_possible + A_now)
#             agent.I_soc += w * agent.calc_driving_force(x_omega)
#
class Zone(Interactor):
    def __init__(self, id, shape):
        super(Zone, self).__init__(id, False)

        # self.id     = id
        self.shape  = shape
        # self.colour = colour

        return
    
    def is_physical(self):
        return False

    def register_model(self, model):
        self.model = model
        return

    def calc_repulsion(self, agent):
        return np.zeros(2)

    def excludes(self, agent):
        return False

    def draw(self, view):
        return

class SimpleZone(Zone):
    def __init__(self, id, regime, shape, colour):
        super(SimpleZone, self).__init__(id, shape)

        self.regime = regime
        self.colour = colour
        
        return

    def get_desc_list(self):
        exclusion_bitset = 0
        for key in self.regime.exclusions.keys():
            exclsn            = self.regime.exclusions[key]
            exclusion_bitset |= 1 << key

        return [self.id, self.shape.posn, exclusion_bitset, exclusion_bitset]
        # return [self.id, self.shape.posn[0], self.shape.posn[1], exclusion_bitset, exclusion_bitset]
    
    def excludes(self, agent):
        return agent.agent_type in self.regime.exclusions

    def calc_repulsion(self, agent):
        # print("Zone::calc_repulsion - zone {} against agent {}({})".format(self.id, agent.id, agent.agent_type))

        if not self.excludes(agent):
            return np.zeros(2)

        #
        # Report that agent is in a zone's exclusion-table
        #
        # print("Zone {} is excluding agent {} of type {}".format(self.id, agent.id, agent.agent_type))
        
        A, B, D = self.regime.exclusions[agent.agent_type]

        #
        # Each shape
        #
        info = self.shape.get_nearest_point_info(agent.posn + agent.model.T_react * agent.v)
        if info == None:
            return np.zeros(2)

        #
        # @todo: Grand unified obstacle repulsion
        #
        x_omega, d, nv, tv = info
        if agent.agent_type in {smad.AgentType.PEDESTRIAN, smad.AgentType.CAR}:
            agent_directional_radius = agent.get_directional_radius(nv)
            
            max_s = 2. + (self.shape.radius + agent_directional_radius)
            if d > max_s:
                return np.zeros(2)
            I_rep = smf.calc_repulsion(d - (self.shape.radius + agent_directional_radius), A, D) * nv
            
        return I_rep

    def draw(self, view):
        # print("Zone::draw", self.id)
        self.shape.draw(self.colour, view)
        return

def create_footpath(id, shape):
    access_limit = Regime([car_exclusion], ptn_priority)
    return SimpleZone(id, access_limit, shape, svd.colour(0, 128, 0))

def create_bikepath(id, shape):
    access_limit = Regime([car_exclusion], bike_priority)
    return SimpleZone(id, access_limit, shape, svd.colour(0, 128, 0))

def create_bikelane(id, shape):
    access_limit = Regime([ptn_exclusion], bike_priority)
    return SimpleZone(id, access_limit, shape, svd.colour(64, 64, 128))

def create_roadway(id, shape):
    access_limit = Regime([ptn_exclusion], car_priority)
    return SimpleZone(id, access_limit, shape, svd.colour(128, 128, 128))

def create_zebra_crossing(id, shape):
    access_limit = Regime([], ptn_priority)
    return SimpleZone(id, access_limit, shape, svd.colour(0, 64, 0))

def create_streetscape(id, shape):
    access_limit = Regime([], car_priority)
    return SimpleZone(id, access_limit, shape, svd.colour(64, 64, 64))

def create_obstacle(id, shape):
    access_limit = Regime([ptn_exclusion, bike_exclusion, car_exclusion], {})
    return SimpleZone(id, access_limit, shape, svd.white)

class ZoneConfig:
    def __init__(self, zones, model):
        self.zone_tbl = {}
        self.definite_exclusion_zones = {}
        self.possible_exclusion_zones = {}
        for i in [smad.AgentType.PEDESTRIAN, smad.AgentType.BICYCLE, smad.AgentType.CAR]:
            self.definite_exclusion_zones[i] = set()
            self.possible_exclusion_zones[i] = set()

        print("ZoneConfig constructor should be adding {} zones".format(len(zones)))
        for zone in zones:
            print("ZoneConfig constructor will add zone {}".format(zone.id))
            self.zone_tbl[zone.id] = zone
            zone.register_model(model)
            if isinstance(zone, SimpleZone):
                for type in zone.regime.exclusions.keys():
                    self.definite_exclusion_zones[type].add(zone.id)
                    self.possible_exclusion_zones[type].add(zone.id)
            elif isinstance(zone, SwitchingZone):
                for type in zone.definite_exclusion_set:
                    self.definite_exclusion_zones[type].add(zone.id)
                for type in zone.possible_exclusion_set:
                    self.possible_exclusion_zones[type].add(zone.id)


class SwitchingInfo:
    def __init__(self, regime_table, regime_schedule):
        self.regime_table    = regime_table
        self.regime_schedule = regime_schedule

#
# Note: may actually want multiple colours:
#     a mean colour and a set of colours associated with states.
# Whenever drawing the zone, the mean colour could be used as a
# zone background with the current state colour as foreground.
# By drawing a smaller state-based shape within a larger zone-based shape,
# this could be made clear.
#
class SwitchingZone(Zone):
    def __init__(self, id, switching_info, shape, base_colour, regime_colour_table):
        super(SwitchingZone, self).__init__(id, shape)

        self.regime_tbl      = switching_info.regime_table
        self.regime_schedule = switching_info.regime_schedule

        self.base_colour         = base_colour
        self.regime_colour_table = regime_colour_table
        
        # print("SwitchingZone constructor {}".format(id))

        assert(len(self.regime_schedule) >= 2)
        name0 = self.regime_schedule[0][0]
        name1 = self.regime_schedule[1][0]
        # print("SwitchingZone constructor {} - first two states are named {} and {}".format(id, name0, name1))
        regime0 = self.regime_tbl[name0]
        regime1 = self.regime_tbl[name1]
        exclusions0 = regime0.exclusions
        exclusions1 = regime1.exclusions
        # print("SwitchingZone constructor {} - first two regimes' exclusions are {} and {}".format(id, exclusions0, exclusions1))

        self.definite_exclusion_set = self.regime_tbl[name0].exclusions.keys() & self.regime_tbl[name1].exclusions.keys()
        self.possible_exclusion_set = self.regime_tbl[name0].exclusions.keys() | self.regime_tbl[name1].exclusions.keys()
        for i in range(2, len(self.regime_schedule)):
            name = self.regime_schedule[i][0]
            exclusions = self.regime_tbl[name].exclusions
            self.definite_exclusion_set &= exclusions.keys()
            self.possible_exclusion_set |= exclusions.keys()
            # print("SwitchingZone constructor {} - {}th regimes' exclusions are {}".format(id, i, exclusions))

        self.period = 0
        for item in self.regime_schedule:
            self.period += item[1]

        return

    def get_regime_name(self, t):
        # print("SwitchingZone::get_regime_name: zone {} with period {} at time {}".format(self.id, self.period, t))
        
        t_rel = t % self.period
        # print("SwitchingZone::get_regime_name: zone {}; {} % {} = {}".format(self.id, t, self.period, t_rel))
        cur_state_end_time = 0
        for i in range(len(self.regime_schedule) - 1):
            cur_state_end_time += self.regime_schedule[i][1]
            # print("SwitchingZone::get_regime_name: zone {}; {} > {}?".format(self.id, cur_state_end_time, t_rel))
            if cur_state_end_time > t_rel:
                smd.dbgprint("SwitchingZone::get_regime_name: zone {} with period {} at time {} (eqv {}) returning {}".format(self.id, self.period, t, t_rel, self.regime_schedule[i][0]))
                return self.regime_schedule[i][0]
        smd.dbgprint("SwitchingZone::get_regime_name: zone {} with period {} at time {} (eqv {}) returning {}".format(self.id, self.period, t, t_rel, self.regime_schedule[-1][0]))
        return self.regime_schedule[-1][0]

    def get_desc_list(self):
        definite_exclusion_bitset = 0
        for key in self.definite_exclusion_set:
            definite_exclusion_bitset |= 1 << key
        possible_exclusion_bitset = 0
        for key in self.possible_exclusion_set:
            possible_exclusion_bitset |= 1 << key

        return [self.id, self.shape.posn, definite_exclusion_bitset, possible_exclusion_bitset]
        # return [self.id, self.shape.posn[0], self.shape.posn[1], exclusion_bitset, exclusion_bitset]
    
    def get_current_regime(self):
        t = int(self.model.cur_timestep_idx * smc.T_step)
        regime_name = self.get_regime_name(t)
        regime = self.regime_tbl[regime_name]
        # print("SwitchingZone::get_current_regime: zone {} in regime {} has exclusions {} and priorities {}".format(self.id, regime_name, regime.exclusions, regime.priorities))
        return regime

    def excludes(self, agent):
        regime = self.get_current_regime()
        return agent.agent_type in regime.exclusions

    def calc_repulsion(self, agent):
        print("SwitchingZone::calc_repulsion - zone {} against agent {}({})".format(self.id, agent.id, agent.agent_type))

        # I_rep = np.zeros(2)

        regime         = self.get_current_regime()
        if agent.agent_type not in regime.exclusions and self.directionality == None:
            return np.zeros(2)

        A, B, D = 0., 0., 0.
        if agent.agent_type in regime.exclusions:
            A, B, D = regime.exclusions[agent.agent_type]
        else:
            assert(self.directionality != None)
            if agent.agent_type not in self.directionality:
                return np.zeros(2)
            
            preferred_dirn, A, B, D = self.directionality[agent.agent_type]
            if np.dot(preferred_dirn, agent.v) >= 0:
                return np.zeros(2)
                

        #
        # Report that agent is in a zone's exclusion-table
        #
        print("Zone {} is excluding agent {} of type {}".format(self.id, agent.id, agent.agent_type))
        
        # A, B, D = regime.exclusions[agent.agent_type]

        #
        # Each shape
        #
        info = self.shape.get_nearest_point_info(agent.posn + agent.model.T_react * agent.v)
        if info == None:
            return np.zeros(2)

        #
        # @todo: Grand unified obstacle repulsion
        #
        x_omega, d, nv, tv = info
        if agent.agent_type in {smad.AgentType.PEDESTRIAN, smad.AgentType.CAR}:
            agent_directional_radius = agent.get_directional_radius(nv)
            
            max_s = 2. + (self.shape.radius + agent_directional_radius)
            if d > max_s:
                return np.zeros(2)
            I_rep = smf.calc_repulsion(d - (self.shape.radius + agent_directional_radius), A, D) * nv
            
        print("Zone {} is excluding agent {} of type {} with I = {}".format(self.id, agent.id, agent.agent_type, I_rep))
        assert(np.dot(I_rep, x_omega - (agent.posn + agent.model.T_react * agent.v)) <= 0.)
        return I_rep

    def draw(self, view):
        # print("SwitchingZone::draw", self.id)
        t           = int(self.model.cur_timestep_idx * smc.T_step)
        regime_name = self.get_regime_name(t)
        regime      = self.regime_tbl[regime_name]
        
        #
        # @todo: Start using the base colour and phase colours. This may require the redesign of
        # the Shape::draw interface so that a shape can be drawn with a base (outer)
        # colour and then redrawn with a top (inner) colour and a smaller
        # radius/halfwidth. Simplest way might be to add a second draw function to shape?
        #
        self.shape.draw(self.regime_colour_table[regime_name], view)
        return


def create_pelican_crossing(id, shape):
    # ptn_regime    = Regime([], ptn_priority)
    # car_regime    = Regime([], car_priority)

    ptn_regime      = Regime([car_exclusion], ptn_priority)
    car_regime      = Regime([ptn_exclusion], car_priority)
    inactive_regime = Regime([], {})
    ptn_flush_regime = Regime([ptn_exclusion, car_exclusion], ptn_priority)
    car_flush_regime = Regime([ptn_exclusion, car_exclusion], car_priority)

    #
    # @todo: Pass the four durations in as a list. Pass the same list in to create_counterpelican_turning and then we are in a position to do something even better.
    #
    regime_tbl      = {"PTN_Start": ptn_regime, "PTN_Flush": ptn_flush_regime, "CAR_Start": car_regime, "CAR_Flush": car_flush_regime}
    # regime_tbl      = {"PTN_Start": ptn_regime, "PTN_Flush": ptn_regime, "CAR_Start": car_regime, "CAR_Flush": car_regime}
    regime_schedule = [("PTN_Start", 5), ("PTN_Flush", 5), ("CAR_Start", 15), ("CAR_Flush", 5)]

    switching_info = SwitchingInfo(regime_tbl, regime_schedule)

    #
    # Select inner and outer colours. Since the PTN phase occupies 1/6 of the
    # total time and the CAR phase occupies 5/6 of the time, I chose a 1:5 ratio
    # to blend the colours to create the outer colour.
    #
    phase_colours = {"PTN_Start": svd.colour(0, 128, 0), "PTN_Flush": svd.colour(0, 96, 0), "CAR_Start": svd.colour(96, 96, 96), "CAR_Flush": svd.colour(64, 96, 64)}
    base_colour  = svd.colour(70, 96, 70)
    
    return SwitchingZone(id, switching_info, shape, base_colour, phase_colours)

def create_counterpelican_turning(id, shape, ptn_flush_direction, car_flush_direction):
    inactive_regime = Regime([], {})
    flush_ptn_regime = Regime([], {}, {smad.AgentType.PEDESTRIAN: (ptn_flush_direction, 10., 10., 0.2)})
    car_regime = Regime([ptn_exclusion], car_priority)
    # flush_car_regime = Regime([ptn_exclusion], {smad.AgentType.CAR: (car_flush_direction, 10., 10., 0.2)})

    A_PvC = smf.A_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.CAR]
    D_PvC = smf.D_mu[smad.AgentType.PEDESTRIAN, smad.AgentType.CAR]

    # ptn_superexclusion  = (smad.AgentType.PEDESTRIAN, A_PvC, A_PvC, D_PvC)
    # inactive_regime = Regime([], {})
    # flush_ptn_regime = Regime([], {}, {smad.AgentType.PEDESTRIAN: (ptn_flush_direction, A_PvC, A_PvC, D_PvC)})
    # car_regime = Regime([ptn_superexclusion], car_priority)
    # flush_car_regime = Regime([ptn_superexclusion], car_priority)

    regime_tbl      = {"PTN_Start": inactive_regime, "PTN_Flush": flush_ptn_regime, "CAR_Start": car_regime, "CAR_Flush": car_regime}
    regime_schedule = [("PTN_Start", 15), ("PTN_Flush", 5), ("CAR_Start", 30), ("CAR_Flush", 10)]

    # regime_tbl      = {"PTN_Start": inactive_regime, "PTN_Flush": flush_ptn_regime, "CAR_Start": car_regime, "CAR_Flush": car_regime}
    # regime_schedule = [("PTN_Start", 5), ("PTN_Flush", 5), ("CAR_Start", 15), ("CAR_Flush", 5)]

    # regime_tbl      = {"PTN_Start": inactive_regime, "PTN_Flush": flush_ptn_regime, "CAR_Start": car_regime, "CAR_Flush": flush_car_regime}
    # regime_schedule = [("PTN_Start", 5), ("PTN_Flush", 5), ("CAR_Start", 15), ("CAR_Flush", 5)]
        
    switching_info = SwitchingInfo(regime_tbl, regime_schedule)

    phase_colours = {"PTN_Start": svd.colour(0, 96, 0), "PTN_Flush": svd.colour(0, 64, 0), "CAR_Start": svd.colour(128, 128, 128), "CAR_Flush": svd.colour(96, 96, 96)}
    base_colour  = svd.colour(70, 96, 70)

    return SwitchingZone(id, switching_info, shape, base_colour, phase_colours)

def create_counterpelican_virtual_barrier(id, shape, ptn_flush_direction):
    inactive_regime = Regime([], {})
    flush_ptn_regime = Regime([], {}, {smad.AgentType.PEDESTRIAN: (ptn_flush_direction, 10., 10., 0.2)})
    # car_regime = Regime([ptn_exclusion], car_priority)
    # flush_car_regime = Regime([ptn_exclusion], {smad.CAR: (car_flush_direction, 10., 10., 0.2)})

    regime_tbl      = {"PTN_Start": inactive_regime, "PTN_Flush": flush_ptn_regime, "CAR_Start": flush_ptn_regime, "CAR_Flush": flush_ptn_regime}
    regime_schedule = [("PTN_Start", 15), ("PTN_Flush", 5), ("CAR_Start", 30), ("CAR_Flush", 10)]

    # regime_tbl      = {"PTN_Start": inactive_regime, "PTN_Flush": flush_ptn_regime, "CAR_Start": flush_ptn_regime, "CAR_Flush": flush_ptn_regime}
    # regime_schedule = [("PTN_Start", 5), ("PTN_Flush", 5), ("CAR_Start", 15), ("CAR_Flush", 5)]

    switching_info = SwitchingInfo(regime_tbl, regime_schedule)

    phase_colours = {"PTN_Start": svd.colour(0, 96, 0), "PTN_Flush": svd.colour(0, 64, 0), "CAR_Start": svd.colour(128, 128, 128), "CAR_Flush": svd.colour(96, 96, 96)}
    base_colour  = svd.colour(70, 96, 70)

    return SwitchingZone(id, switching_info, shape, base_colour, phase_colours)
