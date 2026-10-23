import numpy as np
# import scipy as sp
# import scipy.stats as ss

import view.defs as svd
import view.view as svv
import model.configuration as smc
import model.defs as smd
import model.topology.graph as smtg
import model.topology.approachnode as smtApp
import model.topology.activitynode as smtAct
import model.topology.topology as smtt
import model.geometry.grid as smgg
import model.agency.defs as smad
import model.agency.state as smas
import model.agency.agent as smaa
import model.zone.zone as smzz
# import model.interaction.interaction as smii
import model.interaction.dispatcher as smid
import model.force as smf
import model.numerical.defs as smnd
import model.logger as sml

class Model:
    def __init__(self, width, height, T_simulation, reaction_factor, partial_topologies, OD_matrices, zones = []):
        # self.cell_map = {}

        self.w            = width
        self.h            = height
        self.T_simulation = T_simulation
        self.T_react      = reaction_factor * smc.T_step

        #
        # @todo: Model and Agent both hardcode the same horizon, but this needs to be formalised with a common source of the horizon.
        #
        # self.T_SRCA_horizon = 0.5
        # self.T_MRCA_horizon = 2.
        # self.T_LRCA_horizon = 4.
        # self.horizon = self.T_MRCA_horizon

        self.topology       = smtt.Topology(partial_topologies, OD_matrices, self.T_simulation, smc.T_step)
        self.centroids      = self.topology.centroids
        self.waypoints      = self.topology.waypoints
        self.activity_nodes = self.topology.activity_nodes

        self.activity_demand_info = None

        #
        # Copy the list of zones.
        #
        # @todo: Extend this functionality.
        #   create a table of all zones (id -> zone)
        #   create a list (ordered by agent-type) of sets (of zone-ids) of exclusion-zones (incl. obstacles);
        #   create a set of priority-zones and register these zones with intersecting grid-cells.
        #
        self.zone_config = smzz.ZoneConfig(zones, self)
        
        #
        # @todo: Consider putting a master table of interactions in Model as well as or instead of the tables each agent maintains:
        #     pedestrians, cyclists, and cars
        # can be registered and logged, e.g.:
        #     self.interactions = {}
        #     self.interactions[smad.PEDESTRIAN] = {}
        #     self.interactions[smad.BICYCLE] = {}
        #     self.interactions[smad.CAR] = {}
        #
        # @todo: Determine whether 2^10 == 1024 agents is too many or too few.
        #
        self.dispatcher = smid.Dispatcher(self)

        # self.DEPRECATED_interaction_codec = smii.InteractionCodec(smad.AGENT_ID_NBITS)
        
        #
        # Note: self.obstacles is DEPRECATED, self.{attractors|relationships} are believed redundant. 
        #
        # self.obstacles     = obstacles
        # self.attractors    = attractors
        # self.relationships = relationships

        self.active_agents = {}

        self.agent_metaset = {}
        self.agent_metaset[smad.AgentType.PEDESTRIAN] = set()
        self.agent_metaset[smad.AgentType.BICYCLE]    = set()
        self.agent_metaset[smad.AgentType.CAR]        = set()

        #
        # @todo: Rename inactive_agents -> retiring_agents for clarity, with the
        # side-benefit that the interpreter/compiler will find most of the live code
        # referencing it.
        #
        self.inactive_agent_metaset    = {}
        self.inactive_agent_metaset[smad.AgentType.PEDESTRIAN] = set()
        self.inactive_agent_metaset[smad.AgentType.BICYCLE]    = set()
        self.inactive_agent_metaset[smad.AgentType.CAR]        = set()

        self.cur_timestep_idx             = 0

        #
        # @todo: Decompose model into cells.
        #     Cut model area into tessellating cells (squares, but regular hexagons and triangles could also work).
        #     Construct 2D array of cell-ids.
        #     Move centroids, small obstacles (and attractors, obscurers), and pedestrians into cells.
        #     As pedestrians move, it will be necessary to move them between cells.
        #
        self.grid = smgg.Grid(width, height, 2., 2.)
        self.grid.register_all_zones_coverage(self.zone_config)

        self.candidate_splines = []

        self.logger = None

    def clear_activity_map(self):
        self.activity_map = {}
        return
    
    def register_activity_node(self, activity_node):
        for name in activity_node.activity_tbl:
            if name in self.activity_map:
                tmp_list = self.activity_map[name]
                tmp_list.append(activity_node.id)
            else:
                self.activity_map[name] = [activity_node.id]
        return

    def register_activity_nodes(self, node_list):
        self.clear_activity_map()
        for node in node_list:
            self.register_activity_node(node)
        return

    def register_activity_demand_info(self, activity_demand_info):
        self.activity_demand_info = activity_demand_info
        return
    
    def register_logger(self, logger):
        self.logger = logger
        return

    def register_timer_posn(self, posn):
        self.timer_posn = posn
        return

    def spawn_agents(self):
        for centroid in self.centroids:
            centroid.spawn_agents(self)        
        return

    def despawn_agents(self):
        #
        # @todo: Store agent-despawning info, so that avg delay (between O&D) and spd can be calculated and logged later. 
        #
        
        for agent_type in self.inactive_agent_metaset:
            assert(agent_type in self.inactive_agent_metaset)
            assert(agent_type in self.agent_metaset)

            inactive_idset = self.inactive_agent_metaset[agent_type]
            agent_idset    = self.agent_metaset[agent_type]
            
            for agent_id in inactive_idset:
                assert(agent_id in self.active_agents)
                agent = self.active_agents[agent_id]
                if not agent.activity_focussed:
                    true_src_id = agent.route[0]
                    true_dst_id = agent.route[-1]

                    graph    = self.topology.graph_map[agent.agent_type]
                    true_src = graph.nodes[true_src_id]
                    stream   = true_src.stream_table[true_dst_id]

                    t        = self.cur_timestep_idx * smc.T_step
                    assert(agent.t_spawn != None)
                    T_travel = t - agent.t_spawn 
                    stream.record_despawning(T_travel)
                    
                assert(agent_id in self.active_agents)
                del(self.active_agents[agent_id])
                agent_idset.remove(agent_id)
            assert(agent_type in self.inactive_agent_metaset)
            self.inactive_agent_metaset[agent_type] = set()
            
        return

    def add_agent(self, agent):
        assert(agent.id not in self.active_agents)
        self.active_agents[agent.id] = agent

        assert(agent.agent_type in self.agent_metaset)
        self.agent_metaset[agent.agent_type].add(agent.id)
        return

    def transfer_migrants(self):
        grid = self.grid
        grid.clear_agent_location_registry()
        grid.register_agent_locations(self.active_agents, self.inactive_agent_metaset)
        grid.clear_agent_coverage()
        return
        
    def DEPRECATED_update_pedestrian_kinematics(self, p):
        self.grid.update_pedestrian_kinematics(p)
        return

    def remove_vehicle(self, id):
        assert(False)
        assert(id in self.active_vehicles)
        del self.active_vehicles[id]
        return
   
    def draw_nodal_attraction(self, agent, view):
        thickness = 0.06125 # * (1 + agent.goal_state.state_id)
        cur_dst_idx = agent.route[agent.routing_idx + 1]
        cur_dst = agent.nodegraph.nodes[cur_dst_idx]
        
        x_omega, dist, nv, tv = cur_dst.approach_shape.get_nearest_point_info(agent.posn)
        if agent.goal_state.state_id == smas.GoalStateId.NAVIGATING:
            if agent.goal_state.turning_info != None:
                if len(agent.goal_state.turning_info.stages) > 0:
                    cur_stage             = agent.goal_state.turning_info.stages[0]
                    x_omega, dist, nv, tv = cur_stage.shape.get_nearest_point_info(agent.posn)
            else:
                if isinstance(cur_dst, smtApp.ApproachNode):
                    x_omega, dist, nv, tv = cur_dst.focal_shape.get_nearest_point_info(agent.posn)
                else:
                    assert(isinstance(cur_dst, smtAct.ActivityNode))
                    assert(agent.activity_schedule != None)
                    cur_focal_shape = cur_dst.focal_shape_tbl[agent.activity_schedule[0]][0]
                    x_omega, dist, nv, tv = cur_focal_shape.get_nearest_point_info(agent.posn)

        view.draw_line(svd.gold, agent.posn, x_omega, thickness)
        return

    def draw_conflict_trajectory(self, agent, view):
        if agent.conflict_state.state_id == smas.ConflictStateId.MODULATING:
            return

        thickness   = 0.06125 * (1 + 0.5 * agent.conflict_state.state_id)

        if agent.conflict_state.state_id == smas.ConflictStateId.VEERING:
        # if agent.conflict_state.state_id == smas.ConflictStateId.SWERVING:
            x_curve_pts = agent.conflict_state.x_curve_pts
            # view.draw_line(svv.white, self.conflict_state.ctragent_x0, self.conflict_state.ctragent_x1, 2)
            view.draw_line(svv.blue, x_curve_pts[0], x_curve_pts[1], thickness)
            view.draw_line(svv.blue,  x_curve_pts[1], x_curve_pts[2], thickness)
            view.draw_line(svv.blue, x_curve_pts[2], x_curve_pts[3], thickness)

            #
            # Draw arrowhead
            #
            
        else:
            assert(agent.conflict_state.state_id == smas.ConflictStateId.YIELDING)
            # view.draw_line(svv.white, agent.conflict_state.ctragent_x0, agent.conflict_state.ctragent_x1, 2)
            view.draw_line(svv.yellow, agent.posn, agent.conflict_state.x_des, thickness)
            
            #
            # Draw stopline
            #
            bar_dirn       = smnd.rotate_90ACW(smnd.calc_unit_vector(agent.conflict_state.x_des - agent.posn))
            bar_halflength = 2 * thickness
            bar_vector     = bar_halflength * bar_dirn
            view.draw_line(svv.yellow, agent.conflict_state.x_des - bar_vector, agent.conflict_state.x_des + bar_vector, thickness)
            
        return

    def draw_timer(self, view):
        t = self.cur_timestep_idx * smc.T_step
        text = "{}s".format(t)
        view.fill_circle(svd.white, self.timer_posn, 3)
        view.draw_text(text, svd.black, self.timer_posn - np.array([1, 0]), True)


    def draw(self, view):
        view.draw_viewport_rect(view.background_colour)

        if view.layer_flags & svd.LayerFlagset.GEOMETRY:
            zone_cfg = self.zone_config
            for id in zone_cfg.zone_tbl:
                zone = zone_cfg.zone_tbl[id]
                zone.draw(view) # [White fill + Black stroke] shapes for obstacles
            for centroid in self.centroids:
                centroid.draw(view) # Dull-coloured circles
            for waypoint in self.waypoints:
                waypoint.draw(view) # Dark grey circles
            for activity_node in self.activity_nodes:
                activity_node.draw(view) # Dark grey circles

        if view.layer_flags & svd.LayerFlagset.MACROSCOPIC:
            self.grid.draw_background(self.active_agents, view)
            
        if view.layer_flags & svd.LayerFlagset.TOPOLOGY:
            self.topology.draw(view)
            
        if view.layer_flags & svd.LayerFlagset.MACROSCOPIC:
            self.grid.draw_foreground(self.active_agents, view)

        if view.layer_flags & svd.LayerFlagset.MICROSCOPIC:
            ptnset = self.agent_metaset[smad.AgentType.PEDESTRIAN]
            carset = self.agent_metaset[smad.AgentType.CAR]

            for vhcl_id in carset:
                assert(vhcl_id in self.active_agents)
                vhcl = self.active_agents[vhcl_id]
                vhcl.draw_social_aura(view)
            for ptn_id in ptnset:
                assert(ptn_id in self.active_agents)
                ptn = self.active_agents[ptn_id]
                ptn.draw_social_aura(view)
            # self.grid.draw_foreground(self.active_agents, view)
            for vhcl_id in carset:
                assert(vhcl_id in self.active_agents)
                vhcl = self.active_agents[vhcl_id]
                vhcl.draw_physical_area(view)
            for ptn_id in ptnset:
                assert(ptn_id in self.active_agents)
                ptn = self.active_agents[ptn_id]
                ptn.draw_physical_area(view)


        if view.layer_flags & svd.LayerFlagset.ATTRACTION:
            for agent_id in self.active_agents:
                assert(agent_id in self.active_agents)
                agent = self.active_agents[agent_id]
                self.draw_nodal_attraction(agent, view)
                
        self.dispatcher.draw(self.active_agents, self.zone_config, view)

        if view.layer_flags & svd.LayerFlagset.TRAJECTORY:
            for agent_id in self.active_agents:
                assert(agent_id in self.active_agents)
                agent = self.active_agents[agent_id]
                # self.draw_conflict_trajectory(agent, view)
                agent.draw_conflict_trajectory(view)

        self.draw_timer(view)
                
        #
        # @todo: Explore inclusion of transient and persistent groups.
        #
        # for relationship in self.relationships:
            # relationship.draw(view) # [Gold] connecting lines around groups, connecting agents who know one another

    
    def update_forces(self):
        smd.dbgprint("Model::update_forces - step {} - start".format(self.cur_timestep_idx))
        t = self.cur_timestep_idx * smc.T_step
        for type in self.agent_metaset:
            smd.dbgprint("Model::update_forces - step {} - iterating over all type-{} agents".format(self.cur_timestep_idx, type))
            agent_idset = self.agent_metaset[type]
            for agent_id in agent_idset:
                smd.dbgprint("Model::update_forces - step {} - ready to execute Agent::calc_total_force() for agent[type={}][id={}]".format(self.cur_timestep_idx, type, id))
                assert(agent_id in self.active_agents)
                agent = self.active_agents[agent_id]
                agent.calc_total_force(self.active_agents, self.zone_config, t)
        return
    
    def update_kinematics(self):
        t = self.cur_timestep_idx * smc.T_step
        for type in self.agent_metaset:
            agent_idset = self.agent_metaset[type]
            for agent_id in agent_idset:
                assert(agent_id in self.active_agents)
                agent = self.active_agents[agent_id]
                inactive_agent_idset = self.inactive_agent_metaset[type]
                agent.update_kinematics(t, smc.T_step, inactive_agent_idset)
        self.transfer_migrants()
        return

    def register_candidate_spline(self, candidate_spline):
        self.candidate_splines.append(candidate_spline)
        return
    
    def advance_timestep(self):
        #
        # @todo: After extending Model::determine_interactions below, use it
        # as the basis for reimplementing Model::update_forces
        #
        smd.dbgprint("Model::advance_timestep - step {} - ready to execute self.determine_all_interactions()".format(self.cur_timestep_idx))
        self.dispatcher.dispatch_interactions(self.agent_metaset, self.active_agents, self.zone_config, self.grid, self.cur_timestep_idx, self.logger)

        
        smd.dbgprint("Model::advance_timestep - step {} - ready to execute self.update_forces()".format(self.cur_timestep_idx))
        self.update_forces()
        if self.logger != None:
            self.logger.log_cell_props(self.cur_timestep_idx, self.grid.cells)

            for type, agent_idset in self.agent_metaset.items():
                # smd.dbgprint(agent_tbl)
                self.logger.log_agent_properties(self.cur_timestep_idx, agent_idset, self.active_agents)
                self.logger.log_major_conflicts(self.cur_timestep_idx, agent_idset, self.active_agents)

            self.logger.log_candidate_splines(self.cur_timestep_idx, self.candidate_splines)
            self.candidate_splines = []

        self.cur_timestep_idx += 1

        smd.dbgprint("Model::advance_timestep - step {} - ready to execute self.grid.update_aggregate_properties()".format(self.cur_timestep_idx))
        self.grid.update_aggregate_properties(self.active_agents)

        smd.dbgprint("Model::advance_timestep - step {} - ready to execute self.update_kinematics()".format(self.cur_timestep_idx))
        self.update_kinematics()
        
        # if self.logger != None:
            # self.logger.log_cell_props(self.cur_timestep_idx, self.grid.cells)

            # for type, agent_idset in self.agent_metaset.items():
                # smd.dbgprint(agent_tbl)
                # self.logger.log_agent_properties(self.cur_timestep_idx, agent_idset, self.active_agents)

        # smd.dbgprint("Model::advance_timestep - step ", self.cur_timestep_idx, ": finish")
        return
