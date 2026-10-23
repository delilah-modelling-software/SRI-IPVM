import numpy as np
# import scipy as sp
# import scipy.stats as ss

# from collections import deque

import view.defs as svd
import view.view as svv
# import model.distributor as smd
# import model.numerical.defs as smnd
# import model.agency.defs as smad
# import model.agency.agent as smaa
# import model.agency.pedestrian as smap
# import model.agency.vehicle as smav
import model.topology.graph as smtg

class AgentNodeConfig:
    def __init__(self, centroids, waypoints, activity_nodes, agent_type):
        self.centroids       = centroids
        self.waypoints       = waypoints
        self.activity_nodes  = activity_nodes
        self.agent_type      = agent_type
        return

class Topology:
    #
    # Construct the topology for a model from a dictionary:
    #   @params
    #       self        Model topology being constructed.
    #       anc_dict    List of AgentNodeConfig objects
    #
    def __init__(self, anc_list, OD_matrices, T_simulation, T_step):
        centroids      = []
        waypoints      = []
        activity_nodes = []

        self.centroid_idx_offsets = {}

        self.graph_map = {}
        self.activity_map = {}
        for anc in anc_list:
            centroids      += anc.centroids
            waypoints      += anc.waypoints
            activity_nodes += anc.activity_nodes

            self.centroid_idx_offsets[anc.agent_type] = len(centroids)
            
            graph = smtg.Graph()
            graph.add_nodelist(anc.centroids + anc.waypoints + anc.activity_nodes)
            graph.construct_all_routes()

            # smd.dbgprint("Topology::__init__ agent-type {} has {} centroids".format(pt.agent_type, len(centroids)))
            odm = OD_matrices[anc.agent_type]
            # smd.dbgprint("Topology::__init__ agent-type {} has a {}x{} OD-matrix".format(pt.agent_type, odm.shape[0], odm.shape[1]))
            for i in range(len(anc.centroids)):
                src = anc.centroids[i]
                for j in range(len(anc.centroids)):
                    # smd.dbgprint("Model::__init__ agent-type {} i = {}, j = {}".format(pt.agent_type, i, j))
                    dst      = anc.centroids[j]
                    n_agents = odm[i, j]
                    if n_agents != 0:
                        # smd.dbgprint("Model::__init__ agent-type {}[{}, {}]: {} agents travel from {} to {}".format(pt.agent_type, i, j, n_agents, src.id, dst.id))
                        src.register_stream(dst.id, graph, T_simulation, T_step, n_agents)

            for i in range(len(anc.activity_nodes)):
                activity_node = anc.activity_nodes[i]
                for name in activity_node.activity_tbl:
                    if name in self.activity_map:
                        node_list = self.activity_map[name]
                        node_list.append(activity_node.id)
                        self.activity_map[name] = node_list
                    else:
                        self.activity_map[name] = [activity_node.id]
                        
            self.graph_map[anc.agent_type] = graph
        self.centroids      = centroids
        self.waypoints      = waypoints
        self.activity_nodes = activity_nodes
        
        return

    def draw(self, view):
        for agent_type, graph in self.graph_map.items():
            for node_id, node in graph.nodes.items():
                for ctrnode_id in node.adjacency_tbl:
                    # smd.dbgprint("Model::draw_topology - edge between {} and {}".format(node_id, ctrnode_id))
                    counternode = graph.nodes[ctrnode_id]
                    # view.draw_thin_line(svv.black, node.posn, counternode.posn)
                    view.draw_line(svd.black, node.approach_shape.posn, counternode.approach_shape.posn, 0.06125 * (1 + agent_type))
        return

