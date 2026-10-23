import numpy as np
import scipy as sp
import scipy.stats as ss

from collections import deque

import view.view as svv
import model.distributor as smd
import model.numerical.defs as smnd
import model.agency.defs as smad
# import model.agency.agent as smaa
import model.agency.pedestrian as smap
import model.agency.vehicle as smav
import model.topology.centroid as smtc


#
# @todo: Consider moving Stream class between topology and agency modules
#

class Stream:
    def DEPRECATED__init__(self, id, src_id, dst_id, type, T_shift):
        self.id      = id
        self.src_id  = src_id
        self.dst_id  = dst_id
        self.type    = type
        self.T_shift = T_shift
        return

    def __init__(self, src_id, dst_id, graph, distributor, T_simulation, T_step, n_agents):
        # print("Centroid::register_stream from {} to {} with {} agents".format(self.id, dst_id, n_agents))

        self.src_id  = src_id
        self.dst_id  = dst_id

        self.route = graph.find_route(src_id, dst_id)
        self.times = distributor.generate_values(T_simulation, T_step, n_agents)
        self.graph = graph
        # print("Centroid::register_stream registered {}, {}".format(route, times))
        assert(self.route != None or n_agents == 0)

        self.T_simulation = T_simulation
        self.total_delays = 0.
        self.n_agents     = n_agents
        self.n_spawned    = 0
        self.n_despawned  = 0

        self.next_id = self.base_id = smad.reserve_id_block(n_agents)

        return

    def reinit(self):
        self.total_delays = 0.

        self.n_spawned    = 0
        self.n_despawned  = 0

        self.next_id = self.base_id
        return

    def get_next_id(self):
        #
        # Initialise id to be returned
        #
        id = self.next_id
        assert(id < self.base_id + self.n_agents)
        assert(id == self.base_id + self.n_spawned)

        #
        # Increment self.next_id so that ids are not duplicated.
        #
        self.next_id += 1

        return id

    def record_spawning(self):
        self.n_spawned += 1
        return

    def record_despawning(self, T):
        self.n_despawned  += 1
        self.total_delays += T
        return

    #
    # Nota bene! If agents have activities scheduled, they should not be added
    # to the streams count, because they would yield wrong results. Since agents
    # without scheduled activities are likely to prioritise their time
    # significantly more than those with them, this should yield sufficiently
    # good results for now.
    #
    # @todo: Consider finding ways to calculate sensible delay times for
    # activity-focused agents.
    #

    def calc_mean_delay_time(self):
        print("Stream::calc_mean_delay_time - stream between {} and {} has despawned {} agents".format(self.src_id, self.dst_id, self.n_despawned))
        if self.n_despawned == 0:
            return None
        return self.total_delays / self.n_despawned

    def calc_trip_completion_rate(self):
        return self.n_despawned / self.n_spawned

    def estimate_GEH(self):
        tsf = 3600 / self.T_simulation
        delta = self.n_despawned - self.n_agents
        sum   = self.n_despawned + self.n_agents

        return np.sqrt(2 * tsf * delta * delta / sum)

    def get_desc_list(self):
        src = self.graph.nodes[self.src_id]
        dst = self.graph.nodes[self.dst_id]
        assert(isinstance(src, smtc.Centroid))
        assert(isinstance(dst, smtc.Centroid))

        if self.n_spawned == 0:
            return [self.src_id, self.dst_id, 0., None, None, None]

        n_unspawned = len(self.times)

        spawning_rate = float(self.n_spawned) / self.n_agents

        d = smnd.calc_norm(src.approach_shape.posn - dst.approach_shape.posn)
        route = self.route
        s_route = 0.
        for i in range(len(self.route) - 1):
            src_id = self.route[i]
            dst_id = self.route[i + 1]

            src = self.graph.nodes[src_id]
            dst = self.graph.nodes[dst_id]

            delta = dst.approach_shape.posn - src.approach_shape.posn
            h     = np.sqrt(np.dot(delta, delta))
            s_route += h


        T = self.calc_mean_delay_time()
        if T != None:
            avg_effective_spd = s_route/T
        else:
            avg_effective_spd = None
                
        return [self.src_id, self.dst_id, self.n_agents, self.times, n_unspawned, self.n_spawned, spawning_rate, self.calc_trip_completion_rate(), T, avg_effective_spd, self.estimate_GEH()]
        
