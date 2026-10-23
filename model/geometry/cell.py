import numpy as np
# import scipy as sp
# import scipy.stats as ss

# from collections import deque

import view.defs as svd
# import view.view as svv
import model.agency.defs as smad
import model.interaction.codec as smic
import model.interaction.interaction as smii
# import model.agency.agent as smaa
# import model.force as smf
# import model.logger as sml

class Cell:
    def __init__(self, x, y, half_w, half_h):
        self.posn   = np.array([x, y])
        self.half_w = half_w
        self.half_h = half_h
        self.radius = min(half_w, half_h)
        self.area   = 4 * half_w * half_h
        
        self.registered_agent_idsets = {}
        self.registered_agent_idsets[smad.AgentType.PEDESTRIAN] = set()

        self.zone_coverage_idset  = set()
        self.agent_coverage_idset = set()

        self.mean_density  = 0
        self.mean_spd  = 0.
        self.mean_flow = 0.

        return

    def register_zone_coverage(self, zone_id):
        # print("Cell::register_zone_coverage zone {}".format(zone_id))
        self.zone_coverage_idset.add(zone_id)
        return

    def clear_agent_coverage(self):
        self.agent_coverage_idset = set()
        return
    
    def register_agent_coverage(self, agent_id):
        self.agent_coverage_idset.add(agent_id)
        return

    def clear_agent_location_registry(self):
        self.registered_agent_idsets[smad.AgentType.PEDESTRIAN] = set()
        return
    
    def register_agent_location(self, agent):
        # print("Cell::add_pedestrian - start: cell at {} adding agent {} at posn {}".format(self.posn, agent.id, agent.posn))

        #
        # @todo: Extend to loop over all registrable agent types?
        #
        self.registered_agent_idsets[agent.agent_type].add(agent.id)
        
        delta = agent.posn - self.posn
        # print("Cell::add_pedestrian is adding pedestrian {} at {} to the cell centred at {} with delta {}".format(p.id, p.posn, self.posn, delta))
        if np.abs(delta[0]) > self.half_w or np.abs(delta[1]) > self.half_h:
            print("Cell::register_agent_location agent {} is positioned at {}, outside the cell at {}".format(agent.id, agent.posn, self.posn))
        # assert(np.abs(delta[0]) <= self.half_w)
        # assert(np.abs(delta[1]) <= self.half_h)
        
        # print("Cell::add_pedestrian - finish: id, posn, new n_ped", p.id, p.posn, self.n_ped)
        return

    def construct_possible_interaction_sets(self):
        agent_intrctn_set = set()
        zone_intrctn_set = set()
        agent_keys = sorted(self.agent_coverage_idset)
        for i in range(len(agent_keys)):
            agent_id = agent_keys[i]

            #
            # Register all potential interactions with counteragents.
            #
            for j in range(i + 1, len(agent_keys)):
                ctragent_id = agent_keys[j]
                assert(agent_id < ctragent_id)
                if agent_id >= ctragent_id:
                    continue
                        
                # print("Model::determine_all_interactions: interaction between agents {} and {}".format(agent_id, ctragent_id))
                assert(agent_id != ctragent_id)
                assert(agent_id < ctragent_id)
                agent_intrctn_key = smic.interagent_codec.encode(agent_keys[i], agent_keys[j])
                # print("Model::determine_all_interactions: interaction key {}".format(intrctn_key))
                agent_intrctn_set.add(agent_intrctn_key)

            #
            # Register all potential interactions with counterzones.
            #
            for ctrzone_id in self.zone_coverage_idset:
                zone_intrctn_key = smic.contrazone_codec.encode(agent_id, ctrzone_id)
                zone_intrctn_set.add(zone_intrctn_key)
        return agent_intrctn_set, zone_intrctn_set
        
    def update_aggregate_properties(self, active_agents):
        self.mean_predicted_density = 0
        coverage_idset = self.agent_coverage_idset
        if len(coverage_idset) != 0:
            self.mean_predicted_density = len(coverage_idset) / self.area

        registered_pedestrian_idset = self.registered_agent_idsets[smad.AgentType.PEDESTRIAN]

        n_pdstrns = len(registered_pedestrian_idset)
        if n_pdstrns == 0:
            self.mean_density  = 0
            self.mean_spd  = 0.
            self.mean_flow = 0.
            return

        #
        # 
        #
        # agent_tbl = active_agents[smad.AgentType.PEDESTRIAN]
        n_pdstrns = 0
        sum_spd = 0.
        for id in registered_pedestrian_idset:
            if id not in active_agents:
                continue
            
            n_pdstrns += 1
            pdstrn = active_agents[id]
            sum_spd += np.sqrt(np.dot(pdstrn.v, pdstrn.v))
            
        if n_pdstrns == 0:
            self.mean_density  = 0
            self.mean_spd  = 0.
            self.mean_flow = 0.
            return

        self.mean_density = n_pdstrns / self.area
        self.mean_spd  = sum_spd / n_pdstrns

        #
        # @todo: Multiply by cell_width to get people per second?
        #
        self.mean_flow = self.mean_density * self.mean_spd

    def draw_background(self, active_agents, view): # optional dashed gridlines + cell's pedestrians.draw() calls
        if self.mean_density == 0 and self.mean_predicted_density == 0:
            return
        
        #
        # @todo: Count pedestrians, calculate density (#ped / m^2), density -> red component
        #
        # registered_pedestrian_idset = self.registered_agent_idsets[smad.PEDESTRIAN]
        # if len(self.active_pedestrians.keys()) != self.n_ped:
            # print("Cell::draw_background - mismatch between table-size {} and pedestrian-count {} could be explained by #emigrants {}".format(len(self.active_pedestrians.keys()), self.n_ped, len(self.emigrants)))

        #
        # Scale the cell's mean density and mean speed to generate colour-components <= 255
        #     max safe density = 5.4 pedestrians/m^2
        #     max shared space speed = 30km/h ~= 8.33m/s
        # so I chose 1/6 and 1/9 as scaling factors.
        #
        scaled_density     = self.mean_density / 6
        scaled_speed       = self.mean_spd / 9
        scaled_conflict_density = self.mean_predicted_density / 6

        greenness   = 51 + min(int(204 * scaled_density), 204)
        blueness    = 51 + min(int(204 * scaled_speed), 204)
        redness     = 51 + min(int(204 * scaled_conflict_density), 204)

        cell_colour = svd.colour(redness, greenness, blueness)

        # p1 = self.posn - np.array([self.half_w, self.half_h])
        # p2 = self.posn + np.array([self.half_w, -self.half_h])
        # p3 = self.posn + np.array([-self.half_w, self.half_h])
        # p4 = self.posn + np.array([self.half_w, self.half_h])
        # view.draw_rect(cell_colour, p3, p2)

        view.fill_circle(cell_colour, self.posn, self.radius)

        return

    def draw_foreground(self, active_agents, view): # optional dashed gridlines + cell's pedestrians.draw() calls
        if self.mean_density >= 4:
            p1 = self.posn - 0.5 * np.array([self.half_w, self.half_h])
            p2 = self.posn + 0.5 * np.array([self.half_w, -self.half_h])
            p3 = self.posn + 0.5 * np.array([-self.half_w, self.half_h])
            p4 = self.posn + 0.5 * np.array([self.half_w, self.half_h])
            if self.mean_density >= 6:
                view.draw_thin_line(svd.red, p1, p4)
                view.draw_thin_line(svd.red, p2, p3)
            else:
                view.draw_thin_line(svd.yellow, p1, p4)
                view.draw_thin_line(svd.yellow, p2, p3)
            
        #
        # @todo: Remove deprecated code
        #
        for agent_type in {}: # {smad.AgentType.PEDESTRIAN}:
            agent_idset = self.registered_agent_idsets[agent_type]
            # agent_tbl   = active_agents[agent_type]
            for id in agent_idset:
                if id not in active_agents:
                    continue
                
                agent = active_agents[id]
                agent.draw_social_aura(view)
                agent.draw_physical_area(view)

        #
        # @todo: Remove deprecated code
        #
        if False: # self.mean_predicted_density * self.area > 1:
            for id in self.agent_coverage_idset:
                agent = active_agents[id]
                view.draw_thin_line(svd.blue, self.posn, agent.posn)
                
        return
    
    def get_props_desc_list(self):
        # return [self.posn, self.mean_density, self.mean_spd, self.mean_flow]
        return [self.posn[0], self.posn[1], self.mean_density, self.mean_spd, self.mean_flow]
