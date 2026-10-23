import numpy as np
import scipy as sp
import scipy.stats as ss

from collections import deque

import view.defs as svd
import view.view as svv
import model.configuration as smc
import model.distributor as smd
import model.numerical.defs as smnd
import model.agency.defs as smad
import model.agency.reaction as smar
import model.agency.agent as smaa
import model.agency.pedestrian as smap
import model.agency.vehicle as smav

import model.geometry.shape as smgs

import model.topology.node as smtn
import model.topology.stream as smns

class Centroid(smtn.Node): # gates, doorways, elevators, transporters, stopped [public transport] vehicles
    def __init__(self, id, approach_shape, colour, agent_colour, distributor, agent_type = smad.AgentType.PEDESTRIAN):
        super(Centroid, self).__init__(id, approach_shape, colour, agent_type)

        self.agent_colour = agent_colour
        # self.streamspawner = streamspawner
        self.distributor = distributor

        self.stream_table = {}
        
        if self.agent_type != smad.AgentType.PEDESTRIAN:
            assert(self.agent_type in {smad.AgentType.CAR})
            self.queue_length  = 0
            self.head_agent_id = -1

        return

    def is_possible_terminus(self):
        return True
    
    def register_stream(self, dst_id, graph, T_simulation, T_step, n_agents):
        # print("Centroid::register_stream from {} to {} with {} agents".format(self.id, dst_id, n_agents))

        stream = smns.Stream(self.id, dst_id, graph, self.distributor, T_simulation, T_step, n_agents)
        self.stream_table[dst_id] = stream
        # print("Centroid::register_stream registered {}, {}".format(route, times))
        return

    def has_queue(self):
        return self.agent_type in {smad.AgentType.CAR}
    
    def spawn_queued_agents(self, model):
        print("Centroid::spawn_queued_agents - {} will look for a spawnable agent, etc.".format(self.id))
        assert(self.has_queue())

        print("Centroid::spawn_queued_agents - step ", model.cur_timestep_idx, ": start")

        sentinel = 0
        stream  = None
        for dst_id in self.stream_table.keys():
            stream        = self.stream_table[dst_id]
            # route         = stream.route
            spawning_list = stream.times
            # route, spawning_list = self.stream_table[dst_id]
            # print("Centroid::spawn_pedestrians - from {} to {} along {}".format(self.id, dst_id, route))

            print("Centroid[{}]::spawn_vehicles - spawning_table[{}]".format(self.id, dst_id))
            # spawning_list = self.stream_table[dst_id]

            #
            # @todo: Simplify, e.g.:
            #    sentinel = len(spawning_list) > 0 and spawning_list[0] <= model.cur_timestep_idx
            #
            if sentinel < len(spawning_list) and spawning_list[sentinel] <= model.cur_timestep_idx:
                sentinel += 1
                break
            

        if sentinel == 0:
            #
            # Nothing to enqueue!
            #
            return


        #
        # @todo: sentinel > 1, should be == 1. Verify that the spawning_list is truncated only if the queue is empty!
        #
        assert(sentinel == 1)
        print("Centroid::spawn_queued_agents - {} has an agent ready to spawn".format(self.id))
        assert(stream != None)
        print("Centroid::spawn_queued_agents - step {} - src {} dst {}".format(model.cur_timestep_idx, self.id, stream.route[-1]))

        #
        # If the last spawned vehicle is at least the safety distance from the next vehicle,
        # then the 'mini-queue' can be eliminated.
        #
        prev_agent = None
        if self.head_agent_id != -1 and self.head_agent_id in model.active_agents:
            # agent_tbl  = model.active_agents[self.agent_type]
            # prev_agent = agent_tbl[self.head_agent_id]
            prev_agent = model.active_agents[self.head_agent_id]
            
            dx         = prev_agent.posn - self.approach_shape.posn

            #
            # @todo Support heterogeneous vehicle length:
            #     d_safe      = smaa.get_safety_distance(self.agent_type)
            #     L, W        = smaa.generate_randomised_agent_dimensions(self.agent_type)
            #     d_min = 0.5 * (prev_agent.length + L) + intercar_distance
            #
            L = 4.8 # 9.6
            W = 1.8 # 3.6
            # L = 4.8
            # W = 1.8

            # d_safe = 1.8 # can be 0.4m rather than 1.8m for stopped cars
            d_safe = smc.CFM_s_safe + smc.CFM_v_max * smc.CFM_T_headway
            d_min  = 0.5 * (prev_agent.rect_length + L) + d_safe

            #
            # @todo: Replace vhcl.length by 0.5 * (prev vhcl length + cur vhcl length)
            #
            if smnd.calc_norm_sqr(dx) < d_min**2:
                print("Centroid::spawn_queued_agents - since norm({}) < {}, nothing will spawn".format(dx, d_min))
                return

            # if smnd.calc_norm_sqr(dx) < d_min**2:
                # self.head_agent_id = -1

        # if self.head_agent_id != -1:
            # return
            
        # self.stream_table = route, spawning_list[sentinel:]


        print("Centroid::spawn_queued_agents - the stream between {} and {} has unlinked a spawnable agent, must enqueue it".format(self.id, stream.route[-1]))

        graph = stream.graph
            

        cur_src = self
        route   = stream.route
        cur_dst = graph.nodes[route[1]] # model.centroids[route[1]] # An improvement on using model.centroids[dst_id]

        delta = cur_dst.approach_shape.posn - cur_src.approach_shape.posn
        dirn = smnd.calc_unit_vector(delta)
        perp = smnd.rotate_90ACW(dirn)

        #
        # @todo: Randomise transverse position of vehicles too, e.g.
        #     deviant_magnitude = smaa.generate_randomised_deviation_magnitude(self.radius, W, self.agent_type)
        #
        deviant_magnitude = 0.
        if self.agent_type == smad.AgentType.PEDESTRIAN:
            deviant_magnitude = self.radius * (ss.beta.rvs(2, 0.5) * 2 - 1)
        posn = self.approach_shape.posn + deviant_magnitude * perp

        if not self.approach_shape.contains(posn):
            print("Centroid::spawn_queued_agents {} whose shape is {} at {} somehow does not contain {}".format(self.id, self.approach_shape, self.approach_shape.posn, posn))
        assert self.approach_shape.contains(self.approach_shape.posn)
        assert self.approach_shape.contains(posn)

        agent_id            = stream.get_next_id() # self.next_agent_id
        # agent_id            = self.next_agent_id
        # self.next_agent_id += 1
                
        if self.agent_type == smad.AgentType.PEDESTRIAN:
            dst   = graph.nodes[route[-1]]
            agent = smap.Pedestrian(agent_id, model.topology[smad.PEDESTRIAN], route, posn, dirn, self.agent_colour, dst.agent_colour)
        else:
            assert(self.agent_type == smad.AgentType.CAR)
            #
            # @todo: Change smav.Vehicle() to have a prototype more similar to smap.Pedestrian()
            #
            # dst = model.centroids[route[-1]]
            dst   = graph.nodes[route[-1]]
            # agent = smav.Vehicle(agent_id, self.agent_type, model.topology.graph_map[self.agent_type], route, posn, dirn, 4.8, 1.8, self.agent_colour)
            # agent = smav.Vehicle(agent_id, self.agent_type, model.topology.graph_map[self.agent_type], route, posn, dirn, 9.6, 3.6, self.agent_colour)
            agent = smav.Vehicle(agent_id, self.agent_type, model.topology.graph_map[self.agent_type], route, posn, dirn, 4.4, 1.7, 0.1, self.agent_colour)

            if self.head_agent_id != -1 and self.head_agent_id in model.active_agents: # prev_agent != None:
                prev_agent = model.active_agents[self.head_agent_id]
                # assert(False)
                d_sqr = smnd.calc_norm_sqr(prev_agent.posn - agent.posn)
                print("Centroid::spawn_queued_agents - step {}: src {}, dst {}, new agent {} and prev agent {} are {}m apart".format(model.cur_timestep_idx, self.id, dst.id, agent.id, prev_agent.id, np.sqrt(d_sqr)))
                assert(d_sqr >= d_min**2)
            else:
                self.head_agent_id = -1
                prev_agent         = None
                # print("Centroid::spawn_queued_agents - step {}: src {}, dst {}, new agent {} has no prev agent".format(model.cur_timestep_idx, self.id, dst.id, agent.id))

        agent_spd = agent.get_target_speed()
        agent.v = agent_spd * dirn
        
        # assert(False)
        print("Centroid::spawn_queued_agents - {} is adding agent {} w/ posn {} and  vel {}".format(self.id, agent.id, agent.posn, agent.v))
        model.add_agent(agent)
        self.head_agent_id = agent.id
        agent.register_model(model)
        if prev_agent != None:
            # assert(False)
            l_sum = 0.5 * (prev_agent.rect_length + agent.rect_length)
            s = np.dot(prev_agent.posn - self.approach_shape.posn, dirn) - l_sum
            agent.register_leader(prev_agent, s)

        #
        # Since the agent is mobility- rather than activity-focussed, record their spawning.
        #
        print("Centroid::spawn_queued_agents - stream is recording spawning for {} ".format(agent.id))
        stream.times = stream.times[sentinel:]
        stream.record_spawning()
        agent.t_spawn = model.cur_timestep_idx * smc.T_step

        return
    
    def generate_new_agent(self, model, cur_dst, stream, route):
        graph = stream.graph

        cur_src = self

        
        delta = cur_dst.approach_shape.posn - cur_src.approach_shape.posn
        dirn = smnd.calc_unit_vector(delta)
        perp = smnd.rotate_90ACW(dirn)

        #
        # @todo: Randomise transverse position of vehicles too, e.g.
        #     deviant_magnitude = smaa.generate_randomised_deviation_magnitude(self.radius, W, self.agent_type)
        #
        deviant_magnitude = 0.
        if self.agent_type != smad.AgentType.CAR:
            # deviant_magnitude = self.approach_shape.radius * (ss.beta.rvs(2, 0.5) * 2 - 1)
            if isinstance(self.approach_shape, smgs.Circle):
                deviant_magnitude = self.approach_shape.radius * (ss.beta.rvs(2, 0.5) * 2 - 1)
                posn = self.approach_shape.posn + deviant_magnitude * perp
            elif isinstance(self.approach_shape, smgs.Segment):
                deviant_magnitude = 0.5 * self.approach_shape.length * (ss.beta.rvs(2, 0.5) * 2 - 1)
                posn = self.approach_shape.posn + deviant_magnitude * perp

        agent_id            = stream.get_next_id() # self.next_agent_id
        # agent_id            = self.next_agent_id
        # self.next_agent_id += 1
                
        if self.agent_type == smad.AgentType.PEDESTRIAN:
            dst   = graph.nodes[route[-1]]
            agent = smap.Pedestrian(agent_id, model.topology.graph_map[smad.AgentType.PEDESTRIAN], route, posn, dirn, self.agent_colour, dst.agent_colour)
        else:
            # print("Centroid::spawn_agent will spawn agent {}, an instance of type {}".format(agent_id, self.agent_type))
            assert(self.agent_type == smad.AgentType.CAR)
            #
            # @todo: Change smaa.Vehicle() to have a prototype more similar to smaa.Pedestrian()
            #
            # dst = model.centroids[route[-1]]
            dst   = graph.nodes[route[-1]]
            # agent = smav.Vehicle(agent_id, self.agent_type, model.topology.graph_map[self.agent_type], route, posn, dirn, 4.8, 1.8, self.agent_colour)
            agent = smav.Vehicle(agent_id, self.agent_type, model.topology.graph_map[self.agent_type], route, posn, dirn, 4.5, 1.8, self.agent_colour)

        return agent
    
    # def GESTATING_spawn_unqueued_agents(self, model):
    def spawn_unqueued_agents(self, model):
        assert(not self.has_queue())

        # print("Centroid::spawn_unqueued_agents - step ", model.cur_timestep_idx, ": start")

        for dst_id in self.stream_table.keys():
            stream        = self.stream_table[dst_id]
            
            if dst_id != self.id:
                #
                # @todo: Prepare default routing for agents without activities.
                #
                route         = stream.route
                spawning_list = stream.times
                # route, spawning_list = self.stream_table[dst_id]
                # print("Centroid::spawn_pedestrians - from {} to {} along {}".format(self.id, dst_id, route))

                # print("Centroid[{}]::spawn_vehicles - spawning_table[{}]".format(self.id, dst_id))
                # spawning_list = self.stream_table[dst_id]

                sentinel = 0
                while sentinel < len(spawning_list) and spawning_list[sentinel] == model.cur_timestep_idx:
                    sentinel += 1

                assert(sentinel >= 0)
                if sentinel == 0:
                    continue

                # self.stream_table = route, spawning_list[sentinel:]
                stream.times = stream.times[sentinel:]

                graph = stream.graph

                cur_src = self
                # print("Centroid::spawn_unqueued_agents from C {} to C {} along route {}".format(self.id, dst_id, route))
                cur_dst_node_idx = route[1]
                # print("Centroid::spawn_unqueued_agents from C {} to C {} along route {} via N {}".format(self.id, dst_id, route, cur_dst_node_idx))
                cur_dst = graph.nodes[cur_dst_node_idx] # model.centroids[route[1]] # An improvement on using model.centroids[dst_id]

                if model.activity_demand_info != None:
                    p_activity = model.activity_demand_info.activity_probability
                    activity_probability_list = model.activity_demand_info.activity_popularities

                    for i in range(sentinel):
                        # cur_src = self
                        # cur_dst = graph.nodes[route[1]] # model.centroids[route[1]] # An improvement on using model.centroids[dst_id]
                        
                        #
                        # @todo: Need a per-agent probability of an intermediate activity, a per-activity probability,
                        #   generation of a random number between 0 and 1 (uniform distribution), and route-selection.
                        #
                        # p_activity = 0.1
                        val = ss.uniform.rvs(loc = 0, scale = 1)
                        if val > p_activity:
                            agent = self.generate_new_agent(model, cur_dst, stream, route)
                            model.add_agent(agent)
                            agent.register_model(model)

                            #
                            # Since the agent is mobility- rather than activity-focussed, record their spawning.
                            #
                            stream.record_spawning()
                            agent.t_spawn = model.cur_timestep_idx * smc.T_step
                        else:
                            val   = ss.uniform.rvs(loc = 0, scale = 1)
                            p_sum = 0
                            activity_name = None
                            for i in range(len(activity_probability_list) - 1):
                                name, prob = activity_probability_list[i]
                                p_sum += prob
                                if val <= p_sum:
                                    activity_name = name
                                    break
                            if activity_name == None:
                                activity_name = activity_probability_list[-1][0]
                            assert(activity_name != None)

                            #
                            # Choose the intermediate activity node closet to the src node, then splice src->ian->dst
                            # together.
                            #
                            # @todo: Consider improving the algorithm by comparing complete routes.
                            # For all possible intermediate activity nodes, construct a composite route:
                            #   src->ian
                            #   ian->dst
                            #   total_pathlength
                            # and append it to a list
                            # Sort the list.
                            # Select shortest composite path, splice into a route.
                            #
                            possible_routes = []
                            for ian_id in model.topology.activity_map[activity_name]:
                                graph = model.topology.graph_map[self.agent_type]
                                start_route = graph.find_route(self.id, ian_id)

                                gd = 0.
                                for j in range(len(start_route) - 1):
                                    delta = start_route[j + 1] - start_route[j]
                                    gd += np.sqrt(np.dot(delta, delta))
                                possible_routes.append((start_route, ian_id, gd))
                            possible_routes.sort(key = lambda pr: pr[2])
                            start_route, ian_id, gd = possible_routes[0]
                            finish_route = graph.find_route(ian_id, dst_id)
                            total_route = start_route + finish_route[1:]

                            agent = self.generate_new_agent(model, cur_dst, stream, total_route)
                            model.add_agent(agent)
                            agent.register_model(model)

                            stream.record_spawning()
                            agent.t_spawn = model.cur_timestep_idx * smc.T_step

                            #
                            # And register the agent's chosen activity.
                            #
                            agent.register_activity(activity_name)

                        # agent.t_spawn = model.cur_timestep_idx * smnd.T_step
                    
                    #
                    # @todo: Allocate some agents activities and route them appropriately.
                    #
                    # return
                else:
                    for i in range(sentinel):
                        # cur_src = self
                        # cur_dst = graph.nodes[route[1]] # model.centroids[route[1]] # An improvement on using model.centroids[dst_id]
                        
                        #
                        # @todo: Need a per-agent probability of an intermediate activity, a per-activity probability,
                        #   generation of a random number between 0 and 1 (uniform distribution), and route-selection.
                        #
                        agent = self.generate_new_agent(model, cur_dst, stream, route)
                        model.add_agent(agent)
                        agent.register_model(model)

                        stream.record_spawning()
                        agent.t_spawn = model.cur_timestep_idx * smc.T_step

            else:
                route         = stream.route
                spawning_list = stream.times
                # route, spawning_list = self.stream_table[dst_id]
                # print("Centroid::spawn_pedestrians - from {} to {} along {}".format(self.id, dst_id, route))

                # print("Centroid[{}]::spawn_vehicles - spawning_table[{}]".format(self.id, dst_id))
                # spawning_list = self.stream_table[dst_id]

                sentinel = 0
                while sentinel < len(spawning_list) and spawning_list[sentinel] == model.cur_timestep_idx:
                    sentinel += 1

                assert(sentinel >= 0)
                if sentinel == 0:
                    continue

                # self.stream_table = route, spawning_list[sentinel:]
                stream.times = stream.times[sentinel:]

                graph = stream.graph

                cur_src = self
                # print("Centroid::spawn_unqueued_agents from C {} to C {} along route {}".format(self.id, dst_id, route))
                # cur_dst_node_idx = route[1]
                # print("Centroid::spawn_unqueued_agents from C {} to C {} along route {} via N {}".format(self.id, dst_id, route, cur_dst_node_idx))
                # cur_dst = graph.nodes[cur_dst_node_idx] # model.centroids[route[1]] # An improvement on using model.centroids[dst_id]

                #
                # @todo: Allocate every agent an activity at random.
                #
                # @todo: Need a per-activity probability, generation of a random
                #   number between 0 and 1 (uniform distribution), and route-selection.
                #
                assert(model.activity_demand_info != None)
                activity_probability_list = model.activity_demand_info.activity_popularities

                for i in range(sentinel):
                    # cur_src = self
                    # cur_dst = graph.nodes[route[1]] # model.centroids[route[1]] # An improvement on using model.centroids[dst_id]
                    
                    #
                    # @todo: Need a per-agent probability of an intermediate activity, a per-activity probability,
                    #   generation of a random number between 0 and 1 (uniform distribution), and route-selection.
                    #
                    if 1:
                        val   = ss.uniform.rvs(loc = 0, scale = 1)
                        p_sum = 0
                        activity_name = None
                        for i in range(len(activity_probability_list) - 1):
                            name, prob = activity_probability_list[i]
                            p_sum += prob
                            if val <= p_sum:
                                activity_name = name
                                break
                        if activity_name == None:
                            activity_name = activity_probability_list[-1][0]
                        assert(activity_name != None)

                        #
                        # Choose the intermediate activity node closet to the src node, then splice src->ian->dst
                        # together.
                        #
                        # @todo: Consider improving the algorithm by comparing complete routes.
                        # For all possible intermediate activity nodes, construct a composite route:
                        #   src->ian
                        #   ian->dst
                        #   total_pathlength
                        # and append it to a list
                        # Sort the list.
                        # Select shortest composite path, splice into a route.
                        #
                        possible_routes = []
                        for ian_id in model.topology.activity_map[activity_name]:
                            graph = model.topology.graph_map[self.agent_type]
                            start_route = graph.find_route(self.id, ian_id)

                            gd = 0.
                            for j in range(len(start_route) - 1):
                                delta = start_route[j + 1] - start_route[j]
                                gd += np.sqrt(np.dot(delta, delta))
                            possible_routes.append((start_route, ian_id, gd))
                        possible_routes.sort(key = lambda pr: pr[2])
                        start_route, ian_id, gd = possible_routes[0]
                        finish_route = graph.find_route(ian_id, dst_id)
                        total_route = start_route + finish_route[1:]

                        cur_dst_node_idx = total_route[1]
                        cur_dst = graph.nodes[cur_dst_node_idx]
                        
                        agent = self.generate_new_agent(model, cur_dst, stream, total_route)
                        model.add_agent(agent)
                        agent.register_model(model)

                        stream.record_spawning()
                        agent.t_spawn = model.cur_timestep_idx * smc.T_step


                        agent.register_activity(activity_name)

                #
                # @todo: Allocate some agents activities and route them appropriately.
                #
                # return

        return

    def DEPRECATED_spawn_unqueued_agents(self, model):
    # def spawn_unqueued_agents(self, model):
        assert(not self.has_queue())

        # print("Centroid::spawn_unqueued_agents - step ", model.cur_timestep_idx, ": start")

        for dst_id in self.stream_table.keys():
            stream        = self.stream_table[dst_id]
            
            #
            # @todo: Allocate every agent an activity at random.
            #
            if dst_id == self.id:
                continue

            route         = stream.route
            spawning_list = stream.times
            # route, spawning_list = self.stream_table[dst_id]
            # print("Centroid::spawn_pedestrians - from {} to {} along {}".format(self.id, dst_id, route))

            # print("Centroid[{}]::spawn_vehicles - spawning_table[{}]".format(self.id, dst_id))
            # spawning_list = self.stream_table[dst_id]

            sentinel = 0
            while sentinel < len(spawning_list) and spawning_list[sentinel] == model.cur_timestep_idx:
                sentinel += 1

            assert(sentinel >= 0)
            if sentinel == 0:
                continue

            # self.stream_table = route, spawning_list[sentinel:]
            stream.times = stream.times[sentinel:]

            graph = stream.graph
            
            cur_src = self
            # print("Centroid::spawn_unqueued_agents from C {} to C {} along route {}".format(self.id, dst_id, route))
            cur_dst_node_idx = route[1]
            # print("Centroid::spawn_unqueued_agents from C {} to C {} along route {} via N {}".format(self.id, dst_id, route, cur_dst_node_idx))
            cur_dst = graph.nodes[cur_dst_node_idx] # model.centroids[route[1]] # An improvement on using model.centroids[dst_id]
            for i in range(sentinel):
                # cur_src = self
                # cur_dst = graph.nodes[route[1]] # model.centroids[route[1]] # An improvement on using model.centroids[dst_id]

                agent = self.generate_new_agent(model, cur_dst, stream, route)
                model.add_agent(agent)
                agent.register_model(model)

                stream.record_spawning()
                agent.t_spawn = model.cur_timestep_idx * smc.T_step

        return

    def spawn_agents(self, model):
        if self.has_queue():
            self.spawn_queued_agents(model)
        else:
            self.spawn_unqueued_agents(model)
        return


    def DEPRECATED_spawn_agents(self, model):
        # print("Centroid::spawn_agents - step ", model.cur_timestep_idx, ": start")

        for dst_id in self.stream_table.keys():
            stream        = self.stream_table[dst_id]
            route         = stream.route
            spawning_list = stream.times
            # route, spawning_list = self.stream_table[dst_id]
            # print("Centroid::spawn_pedestrians - from {} to {} along {}".format(self.id, dst_id, route))

            # print("Centroid[{}]::spawn_vehicles - spawning_table[{}]".format(self.id, dst_id))
            # spawning_list = self.stream_table[dst_id]

            sentinel = 0
            while sentinel < len(spawning_list) and spawning_list[sentinel] == model.cur_timestep_idx:
                sentinel += 1

            if self.has_queue():
                sentinel = min(sentinel, 1)

            #
            # If the last spawned vehicle is at least the safety distance from the centroid,
            # then the queue can be eliminated.
            #
            if self.has_queue() and self.head_agent_id != -1:
                agent_tbl  = model.active_agents[self.agent_type]
                prev_agent = agent_tbl[self.head_agent_id]
                ds         = prev_agent.posn - self.posn

                #
                # @todo Support heterogeneous vehicle length:
                #     d_safe      = smaa.get_safety_distance(self.agent_type)
                #     L, W        = smaa.generate_randomised_agent_dimensions(self.agent_type)
                #     d_min = 0.5 * (prev_agent.length + L) + intercar_distance
                #
                L = 9.6
                W = 3.6
                # L = 4.8
                # W = 1.8
                d_safe = 1.8 # can be 0.4m rather than 1.8m for stopped cars
                d_min  = 0.5 * (prev_agent.length + L) + d_safe

                #
                # @todo: Replace vhcl.length by 0.5 * (prev vhcl length + cur vhcl length)
                #
                if np.dot(ds, ds) >= d_min**2:
                    self.head_agent_id = -1

            assert(sentinel >= 0)
            if sentinel == 0:
                continue

            # self.stream_table = route, spawning_list[sentinel:]
            stream.times = stream.times[sentinel:]

            graph = stream.graph
            for i in range(sentinel):
                cur_src = self
                cur_dst = graph.nodes[route[1]] # model.centroids[route[1]] # An improvement on using model.centroids[dst_id]

                delta = cur_dst.posn - cur_src.posn
                dirn = smnd.calc_unit_vector(delta)
                perp = smnd.rotate_90ACW(dirn)

                #
                # @todo: Randomise transverse position of vehicles too, e.g.
                #     deviant_magnitude = smaa.generate_randomised_deviation_magnitude(self.radius, W, self.agent_type)
                #
                deviant_magnitude = 0.
                if self.agent_type == smad.AgentType.PEDESTRIAN:
                    deviant_magnitude = self.radius * (ss.beta.rvs(2, 0.5) * 2 - 1)
                    assert(deviant_magnitude >= -1)
                    assert(deviant_magnitude <=  1)
                posn = self.posn + deviant_magnitude * perp

                agent_id            = stream.get_next_id() # self.next_agent_id
                # agent_id            = self.next_agent_id
                # self.next_agent_id += 1
                
                if self.agent_type == smad.AgentType.PEDESTRIAN:
                    dst   = graph.nodes[route[-1]]
                    agent = smap.Pedestrian(agent_id, model.topology[smad.AgentType.PEDESTRIAN], route, posn, dirn, self.agent_colour, dst.agent_colour, self.view)
                else:
                    # print("Centroid::spawn_agent will spawn agent {}, an instance of type {}".format(agent_id, self.agent_type))
                    assert(self.agent_type == smad.CAR)
                    #
                    # @todo: Change smaa.Vehicle() to have a prototype more similar to smaa.Pedestrian()
                    #
                    # dst = model.centroids[route[-1]]
                    dst   = graph.nodes[route[-1]]
                    agent = smav.Vehicle(agent_id, self.agent_type, model.topology[self.agent_type], route, posn, dirn, 4.8, 1.8, self.agent_colour, self.view)

                model.add_agent(agent)
                agent.register_model(model)

        return
    
    def draw(self, view):
        self.approach_shape.draw(self.colour, view)
        text = "C {}".format(self.id)
        view.draw_text(text, svd.white, self.approach_shape.posn, True)
        # view.fill_circle(self.colour, self.shape.posn, self.shape.radius)
        return
#
# Note: Centroid could be extended to act as a Portal between Models, i.e. each Portal within a Model
# would be paired with a Portal within another Model. Exiting a Model via a Portal would also mean
# entering the Model associated with the counter-Portal. This could be especially useful if the
# Portals were spatiotemporally separated by a mascroscopically modelled region, preferably with
# a minimum delay >= the communication time between the computer systems running the Models.
#


#
# Macro to define virtual gates, encompassing physical gates, doorways, and all other model entrances/exits.
#
def create_virtual_gate(id, approach_shape, colour, agent_colour, distributor, agent_type):
    return Centroid(id, approach_shape, colour, agent_colour, distributor, agent_type)

def create_oblong_gate(id, centre, halfwidth_vector, halfdepth, gate_colour, agent_colour, distributor, agent_type = smad.AgentType.PEDESTRIAN):
    x0 = centre + halfwidth_vector
    x1 = centre - halfwidth_vector
    oblong = smgs.Segment(x0, x1, halfdepth)
    return create_virtual_gate(id, oblong, gate_colour, agent_colour, distributor, agent_type)

def create_round_gate(id, centre, radius, gate_colour, agent_colour, distributor, agent_type = smad.AgentType.PEDESTRIAN):
    circle = smgs.Circle(centre, radius)
    return create_virtual_gate(id, circle, gate_colour, agent_colour, distributor, agent_type)
