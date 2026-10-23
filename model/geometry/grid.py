import numpy as np
# import scipy as sp
# import scipy.stats as ss

# from collections import deque

# import view.view as svv
import model.defs as smd
import model.agency.defs as smad
# import model.force as smf
# import model.logger as sml

import model.geometry.cell as smgc

class BoundingBox:
    def __init__(self, pt1, pt2):
        print("BoundingBox {} {}".format(pt1, pt2))
        self.posn1 = np.array([min(pt1[0], pt2[0]), min(pt1[1], pt2[1])])
        self.posn2 = np.array([max(pt1[0], pt2[0]), max(pt1[1], pt2[1])])

class Grid:
    def __init__(self, grid_w, grid_h, cell_w = 2, cell_h = 2):

        self.cell_width  = cell_w
        self.cell_height = cell_h
        # self.cell_width = self.cell_height = 4.

        self.horizontal_padding  = 2
        self.vertical_padding    = 2

        x = 0. - 0.5 * self.horizontal_padding  * self.cell_width
        y = 0. - 0.5 * self.vertical_padding * self.cell_height

        self.origin = np.array([x, y])
        
        #
        # @todo: To allow arbitrary orientation of the grid, add two extra parameters:
        #       g0  grid origin
        #       e1  primary direction
        # and then calculate e2 = R_pi/2 * e1 = [-e1[1], e1[0]]
        #
        # g0 = -np.array([horizontal_padding, vertical_padding])
        # x, y = 1, 0
        # e1 = np.array([x, y])
        # e2 = np.array([-y, x])
        #
        # Then forall i, j in range(m) x range(n):
        #       cell.posn(i, j) = g0 + (j + 0.5) * e2 + (i + 0.5) * e1
        # and forall posn u in R^2:
        #       v = u - g0; j = floor(v . e1)/ grid_w, i = floor(v . e2) / grid_h
        #
        
        self.m = int(np.ceil(grid_h / self.cell_height)) + self.vertical_padding
        self.n = int(np.ceil(grid_w / self.cell_width))  + self.horizontal_padding
        self.cells = []

        half_w = 0.5 * self.cell_width
        half_h = 0.5 * self.cell_height
        for i in range(self.m):
            # cell_y = i * self.cell_height + 0.5 * self.cell_height
            # cell_y = (i - 0.5) * self.cell_height
            cell_y = y + i * self.cell_height + half_h
            self.cells.append([])
            
            for j in range(self.n):
                # self.ped_cells.append(Cell(j * self.cell_width + half_w, cell_y, half_w, half_h))
                # self.ped_cells.append(Cell(j * self.cell_width, cell_y, half_w, half_h))
                # cell_x = j * self.cell_width - half_w
                cell_x = x + j * self.cell_width + half_w

                self.cells[i].append(smgc.Cell(cell_x, cell_y, half_w, half_h))
                
        return
    
    def get_cell_coordinates(self, posn):
        rel_posn = posn - self.origin
        i = int(rel_posn[1] / self.cell_height)
        j = int(rel_posn[0] / self.cell_width)

        if False:
            m = self.m
            n = self.n
            print("Grid::get_cell_coordinates for rel_posn {} -> ({}, {}) < ({}, {})".format(rel_posn, i, j, m, n))
            assert (i >= 0)
            assert (i < self.m)
            assert (j >= 0)
            assert (j < self.n)

            cell = self.cells[i][j]
            dx   = posn - cell.posn
            dx0  = dx[0]
            dx1  = dx[1]
            assert(dx0 ** 2 <= 1.)
            assert(dx1 ** 2 <= 1.)
        return (i, j)

    def quantise_bbox(self, bbox, expansion = 0):
        #
        # Note: Code hinges on bbox's posn1 being bottom left and posn2 being top right.
        #

        # rel_posn1 = bbox.posn1 - self.origin
        # rel_posn2 = bbox.posn2 - self.origin

        print("Grid::quantise_bbox - unquantised {} and {}".format(bbox.posn1, bbox.posn2))
        # print("Grid::quantise_bbox - unquantised {} and {}".format(rel_posn1, rel_posn2))

        # q_posn1   = self.get_cell_coordinates(rel_posn1)
        # q_posn2   = self.get_cell_coordinates(rel_posn2)
        q_posn1   = self.get_cell_coordinates(bbox.posn1)
        q_posn2   = self.get_cell_coordinates(bbox.posn2)

        q_posn1 -= np.array([expansion, expansion])
        q_posn2 += np.array([expansion, expansion])

        print("Grid::quantise_bbox - expanded {} and {}".format(q_posn1, q_posn2))

        # max_posn = np.maximum(float(q_posn1), np.zeros(2))
        # min_posn = np.minimum(float(q_posn2), np.array([self.m - 1, self.n - 1]), dtype='float')
        # q_posn1 = np.rint(max_posn)
        # q_posn2 = np.rint(min_posn)

        assert(q_posn2[0] >= q_posn1[0])
        assert(q_posn2[1] >= q_posn1[1])
        if q_posn2[0] > self.m - 1:
            q_posn2[0] = self.m - 1
        if q_posn2[1] > self.n - 1:
            q_posn2[1] = self.n - 1
        if q_posn1[0] < 0:
            q_posn1[0] = 0
        if q_posn1[1] < 0:
            q_posn1[1] = 0
        print("Grid::quantise_bbox - rounded {} and {}".format(q_posn1, q_posn2))

        print("Grid::quantise_bbox started with bbox [{}, {}] origin {}, and quantised it to {}{}".format(bbox.posn1, bbox.posn2, self.origin, q_posn1, q_posn2))

        return [q_posn1, q_posn2]
    
    def register_zone_coverage(self, zone_id, qbox):
        for i in range(int(qbox[0][0]), int(qbox[1][0])):
            for j in range(int(qbox[0][1]), int(qbox[1][1])):
                cell = self.cells[i][j]
                cell.register_zone_coverage(zone_id)
        return
        
    def register_all_zones_coverage(self, zone_config):
        for zone_id in zone_config.zone_tbl:
            zone = zone_config.zone_tbl[zone_id]

            #
            # Generate bbox. Quantise with quantise_bbox.
            # For each cell indicated by the qbox, call the
            # Cell::register_zone_coverage method.
            #
            # @todo: Move to a Bresenham approach when the agent-coverage does.
            #
            zbox = zone.shape.calc_bbox()
            qbox = self.quantise_bbox(zbox, expansion = 1)
            self.register_zone_coverage(zone_id, qbox)
        return

    def agent_type_is_registrable(self, agent_type):
        return agent_type in {smad.AgentType.PEDESTRIAN}
    
    def clear_agent_location_registry(self):
        # smd.dbgprint("Grid::clear_agent_location_registry")
        for row in self.cells:
            for cell in row:
                cell.clear_agent_location_registry()
        return
    
    def is_registrable(self, agent):
        return agent.agent_type in {smad.AgentType.PEDESTRIAN}

    def register_agent_location(self, agent):
    # def register_agent(self, agent):
        if not self.is_registrable(agent):
            return

        i, j = self.get_cell_coordinates(agent.posn)
        if i < 0 or i >= self.m or j < 0 or j >= self.n:
            print("Grid::register_agent_location {} at {} registered at ({}, {})".format(agent.id, agent.posn, i, j))
            print("Grid::register_agent_location 0 <= {} < {} and 0 <= {} < {} ?!".format(i, self.m, j, self.n))

            assert(False)
            
        print("Grid::register_agent_location: agent {} at {} map to cell[{}, {}]".format(agent.id, agent.posn, i, j))
        cell = self.cells[i][j]
        cell.register_agent_location(agent)

        i_tmp, j_tmp = self.get_cell_coordinates(agent.posn)
        assert(i_tmp == i)
        assert(j_tmp == j)
        return

    def register_agent_locations(self, active_agents, inactive_agent_metaset):
        for id, agent in active_agents.items():
            if not self.agent_type_is_registrable(agent.agent_type):
                continue
            if id in inactive_agent_metaset[agent.agent_type]:
                continue
            
            self.register_agent_location(agent)
        return


    def clear_agent_coverage(self):
        smd.dbgprint("Grid::clear_agent_coverage")
        for row in self.cells:
            for cell in row:
                cell.clear_agent_coverage()
        return

    def register_agent_coverage(self, id, qbox):
        #
        # @todo: Can the Model's functionality be moved into Grid, similar to the
        # register_zone_coverage method?
        #
        for i in range(int(qbox[0][0]), int(qbox[1][0]) + 1):
            for j in range(int(qbox[0][1]), int(qbox[1][1]) + 1):
                cell = self.cells[i][j]
                cell.register_agent_coverage(id)
        return

    #
    # Construct all potential interactions from all cells and collect them into a single pair of sets.
    #
    # @todo: Multithread (e.g. map-reduce) to ...
    #     intrctn_idsets = map(generate_interaction_set, self.grid.cells)
    #
    def construct_possible_interaction_sets(self):
        agent_interaction_set = set()
        zone_interaction_set = set()
        for row in self.cells:
            for cell in row:
                #
                # agent_interaction_set, zone_interaction_set = Cell::construct_possible_interaction_sets(cell)
                #
                tmp_agent_interaction_set, tmp_zone_interaction_set = cell.construct_possible_interaction_sets()
                agent_interaction_set |= tmp_agent_interaction_set
                zone_interaction_set  |= tmp_zone_interaction_set
        return agent_interaction_set, zone_interaction_set

    def update_aggregate_properties(self, active_agents):
        for row in self.cells:
            for cell in row:
                cell.update_aggregate_properties(active_agents)
        return

    def draw_background(self, active_agents, view):
        for row in self.cells:
            for cell in row:
                cell.draw_background(active_agents, view) # density-coloured rectangles + optional diagonal lines
        return

    def draw_foreground(self, active_agents, view):
        for row in self.cells:
            for cell in row:
                cell.draw_foreground(active_agents, view) # optional dashed gridlines + cell's pedestrians.draw() calls
        return
    
    def log_aggregate_properties(self, logger, cur_step_idx):
        # for row in self.cells:
            # for cell in row:
                # cell.log_aggregate_properties(logger, cur_step_idx) # optional dashed gridlines + cell's pedestrians.draw() calls
        logger.log_cell_props(cur_step_idx, self.cells)
        return
