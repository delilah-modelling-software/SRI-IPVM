# import numpy as np
# import scipy as sp
# import scipy.stats as ss

import model.configuration as smc
# import model.node as smn
# import model.geometry as smg
import model.agency.defs as smad
import model.agency.state as smas
# import model.agency.agent as smaa
# import model.zone as smz
# import model.force as smf

import csv as csv

dlm = ','
# delimiter = ' '

def log_static_components(centroids, waypoints, activity_nodes, zone_cfg, seed):
    #
    # Geometric and topological aspects expected to remain constant should not
    # need an output-file held open.
    #
    fname = 'zones.csv'
    with open(fname, 'w', newline='') as fp:
        zone_writer = \
            csv.writer(fp, delimiter=dlm, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for id in zone_cfg.zone_tbl:
            zone = zone_cfg.zone_tbl[id]
            zone_writer.writerow(zone.get_desc_list())

    fname = 'centroids.csv'
    with open(fname, 'w', newline='') as fp:
        cntrd_writer = \
            csv.writer(fp, delimiter=dlm, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for centroid in centroids:
            cntrd_writer.writerow(centroid.get_desc_list())

    fname = 'waypoints.csv'
    with open(fname, 'w', newline='') as fp:
        wpnt_writer = \
            csv.writer(fp, delimiter=dlm, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for waypoint in waypoints:
            wpnt_writer.writerow(waypoint.get_desc_list())

    fname = 'activity_nodes.csv'
    with open(fname, 'w', newline='') as fp:
        wpnt_writer = \
            csv.writer(fp, delimiter=dlm, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for activity_node in activity_nodes:
            wpnt_writer.writerow(activity_node.get_desc_list())
    return

def log_stream_statistics(model, seed):
    switch_str = ''
    if smc.OMNISCIENCE_ENABLED:
        switch_str += '1'
    else:
        switch_str += '0'
    if smc.VEERING_ENABLED:
        switch_str += '1'
    else:
        switch_str += '0'
    if smc.YIELDING_ENABLED:
        switch_str += '1'
    else:
        switch_str += '0'
    if smc.PRUDENT_ACCELERATION:
        switch_str += '1'
    else:
        switch_str += '0'
    if smc.STABILISED_YIELDING:
        switch_str += '1'
    else:
        switch_str += '0'
    if smc.REFLEXIVE_REPULSION_ENABLED:
        switch_str += '1'
    else:
        switch_str += '0'
    seed_str = str(seed)

    fname = 'stream-stats-' + switch_str + '-' + seed_str + '.csv'
    fp = open(fname, 'w', newline='')

    stream_stats_writer = csv.writer(fp, delimiter=dlm, quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for src in model.centroids:
        for dst_id in src.stream_table:
            stream = src.stream_table[dst_id]
            stream_stats_writer.writerow(stream.get_desc_list())
    return

def log_safety_measures(model, seed):
    switch_str = ''
    if smc.OMNISCIENCE_ENABLED:
        switch_str += '1'
    else:
        switch_str += '0'
    if smc.VEERING_ENABLED:
        switch_str += '1'
    else:
        switch_str += '0'
    if smc.YIELDING_ENABLED:
        switch_str += '1'
    else:
        switch_str += '0'
    if smc.PRUDENT_ACCELERATION:
        switch_str += '1'
    else:
        switch_str += '0'
    if smc.STABILISED_YIELDING:
        switch_str += '1'
    else:
        switch_str += '0'
    if smc.REFLEXIVE_REPULSION_ENABLED:
        switch_str += '1'
    else:
        switch_str += '0'
    seed_str = str(seed)

    fname = 'safety-measures-' + switch_str + '-' + seed_str + '.csv'
    fp = open(fname, 'w', newline='')

    safety_measures_writer = csv.writer(fp, delimiter=dlm, quotechar='|', quoting=csv.QUOTE_MINIMAL)
    safety_history = model.dispatcher.safety_history
    for combo_id in safety_history.registry:
        safety_record = safety_history.registry[combo_id]
        safety_measures_writer.writerow(safety_record.get_desc_list())
    return

class Logger:
    def __init__(self, seed):
        self.seed = seed
        #
        # The first three files will store information in a Lagrangian (moving) frame of
        # reference.
        #

        switch_str = ''
        if smc.OMNISCIENCE_ENABLED:
            switch_str += '1'
        else:
            switch_str += '0'
        if smc.VEERING_ENABLED:
            switch_str += '1'
        else:
            switch_str += '0'
        if smc.YIELDING_ENABLED:
            switch_str += '1'
        else:
            switch_str += '0'
        if smc.PRUDENT_ACCELERATION:
            switch_str += '1'
        else:
            switch_str += '0'
        if smc.STABILISED_YIELDING:
            switch_str += '1'
        else:
            switch_str += '0'
        if smc.REFLEXIVE_REPULSION_ENABLED:
            switch_str += '1'
        else:
            switch_str += '0'
        seed_str = str(seed)

        fname = 'agent-properties-' + switch_str + '-' + seed_str + '.csv'
        fp = open(fname, 'w', newline='')
        self.agent_writer = \
            csv.writer(fp, delimiter=dlm, quotechar='|', quoting=csv.QUOTE_MINIMAL)

        fname = 'interagent-interactions-' + switch_str + '-' + seed_str + '.csv'
        fp = open(fname, 'w', newline='')
        self.interagent_interaction_writer = \
            csv.writer(fp, delimiter=dlm, quotechar='|', quoting=csv.QUOTE_MINIMAL)

        fname = 'major-conflicts-' + switch_str + '-' + seed_str + '.csv'
        fp = open(fname, 'w', newline='')
        self.major_conflict_writer = \
            csv.writer(fp, delimiter=dlm, quotechar='|', quoting=csv.QUOTE_MINIMAL)

        fname = 'spline-candidates-' + switch_str + '-' + seed_str + '.csv'
        fp = open(fname, 'w', newline='')
        self.spline_candidate_writer = \
            csv.writer(fp, delimiter=dlm, quotechar='|', quoting=csv.QUOTE_MINIMAL)

        #
        # The last file  will store information in an Eulerian (stationary) frame of
        # reference.
        #
        fname = 'cell-properties-' + switch_str + '-' + seed_str + '.csv'
        fp = open(fname, 'w', newline='')
        self.cellprops_writer = \
            csv.writer(fp, delimiter=dlm, quotechar='|', quoting=csv.QUOTE_MINIMAL)

    def log_agent_properties(self, cur_step_idx, agent_idset, agent_tbl):
        for id in agent_idset:
            agent = agent_tbl[id]
            self.agent_writer.writerow([cur_step_idx] + agent.get_desc_list())
        return

    def log_interagent_interaction(self, cur_step_idx, interaction):
        self.interagent_interaction_writer.writerow([cur_step_idx] + interaction.get_desc_list())

    def log_major_conflicts(self, cur_step_idx, agent_idset, agent_tbl):
        for id in agent_idset:
            agent = agent_tbl[id]
            if agent.conflict_state.state_id != smas.ConflictStateId.MODULATING:
                self.major_conflict_writer.writerow([cur_step_idx, agent.id, agent.posn, agent.v] + agent.conflict_state.get_desc_list() + [agent.goal_state.state_id])
        return

    def log_candidate_splines(self, cur_step_idx, candidate_splines):
        for candidate in candidate_splines:
            self.spline_candidate_writer.writerow([cur_step_idx] + candidate)

    def DEPRECATED_log_agent_interactions(self, cur_step_idx, interactions):
        for interaction in interactions:
            self.interagent_interaction_writer.writerow([cur_step_idx] + interaction.get_kinematic_desc_list())
        return

    def log_cell_props(self, cur_step_idx, cell_mtrx):
        for row in cell_mtrx:
            for cell in row:
                if len(cell.registered_agent_idsets[smad.AgentType.PEDESTRIAN]) != 0:
                    self.cellprops_writer.writerow([cur_step_idx] + cell.get_props_desc_list())
        return
