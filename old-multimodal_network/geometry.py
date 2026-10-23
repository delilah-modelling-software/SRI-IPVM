import numpy as np
import scipy as sp
import scipy.stats as ss

from collections import deque

import multimodal_network.view as mnv
# import multimodal_network.node as mnn
import multimodal_network.agent as mna

class Cell:
    def __init__(self, x, y, half_w, half_h):
        self.posn   = np.array([x, y])
        self.half_w = half_w
        self.half_h = half_h
        self.area   = 4 * half_w * half_h
        
        self.active_pedestrians   = {}
        self.n_ped                = 0
        self.inactive_pedestrians = []
        self.emigrants            = []

        return

    def add_pedestrian(self, p):
        # print("Cell::add_pedestrian - start: id, posn, old n_ped", p.id, p.posn, self.n_ped)
        self.active_pedestrians[p.id] = p
        
        delta = p.posn - self.posn
        # print("Cell::add_pedestrian is adding pedestrian {} at {} to the cell centred at {} with delta {}".format(p.id, p.posn, self.posn, delta))
        assert(np.abs(delta[0]) <= self.half_w)
        assert(np.abs(delta[1]) <= self.half_h)
        
        self.n_ped += 1
        # print("Cell::add_pedestrian - finish: id, posn, new n_ped", p.id, p.posn, self.n_ped)
        return

    def register_emigrant(self, id, i, j):
        p = self.active_pedestrians[id]
        # print("Cell::remove_pedestrian - start: id, posn, old n_ped", id, p.posn, self.n_ped)
        self.emigrants.append((id, (i, j)))
        self.n_ped -= 1
        # print("Cell::preremove_pedestrian - finish: id, old n_ped", id, self.n_ped)
        return

    def despawn_pedestrians(self):
        # print("Cell::despawn_pedestrians - start: posn, old n_ped", self.posn, self.n_ped)
        for ped_id in self.inactive_pedestrians:
            if ped_id not in self.active_pedestrians:
                continue
            del(self.active_pedestrians[ped_id])
            self.n_ped -= 1
        self.inactive_pedestrians = []
        # print("Cell::despawn_pedestrians - finish: posn, new n_ped", self.posn, self.n_ped)
        return

    def draw_background(self, view): # optional dashed gridlines + cell's pedestrians.draw() calls
        #
        # @todo: Count pedestrians, calculate density (#ped / m^2), density -> red component
        #
        if len(self.active_pedestrians.keys()) != self.n_ped:
            print("Cell::draw_background - mismatch between table-size {} and pedestrian-count {} could be explained by #emigrants {}".format(len(self.active_pedestrians.keys()), self.n_ped, len(self.emigrants)))
            
        density   = self.n_ped / self.area
        alpha     = density / 6
        greyness  = int(np.minimum(64 + int(alpha * 128), 192))
        cell_colour = mnv.colour(greyness, greyness, greyness)
        
        p1 = self.posn - np.array([self.half_w, self.half_h])
        p2 = self.posn + np.array([self.half_w, -self.half_h])
        p3 = self.posn + np.array([-self.half_w, self.half_h])
        p4 = self.posn + np.array([self.half_w, self.half_h])
        # if self.posn[0] < 55 and self.posn[1] < 55:
            # view.draw_rect(white, p3, p2)
        # view.draw_rect(white, p3, p2)
        view.draw_rect(cell_colour, p3, p2)
        return

    def draw_foreground(self, view): # optional dashed gridlines + cell's pedestrians.draw() calls
        density   = self.n_ped / self.area
        if density >= 4:
            p1 = self.posn - 0.5 * np.array([self.half_w, self.half_h])
            p2 = self.posn + 0.5 * np.array([self.half_w, -self.half_h])
            p3 = self.posn + 0.5 * np.array([-self.half_w, self.half_h])
            p4 = self.posn + 0.5 * np.array([self.half_w, self.half_h])
            if density >= 6:
                view.draw_thin_line(mnv.red, p1, p4)
                view.draw_thin_line(mnv.red, p2, p3)
            else:
                view.draw_thin_line(mnv.yellow, p1, p4)
                view.draw_thin_line(mnv.yellow, p2, p3)
            
        for p_id in self.active_pedestrians:
            p = self.active_pedestrians[p_id]
            p.draw(view)
        return
    
class DEPRECATED_Grid:
    def __init__(self, grid_w, grid_h, cell_w, cell_h):

        self.cell_width = self.cell_height = 2.
        # self.cell_width = self.cell_height = 4.

        extra_width  = 2
        extra_height = 2

        x = 0. - 0.5 * extra_width  * self.cell_width
        y = 0. - 0.5 * extra_height * self.cell_height

        self.origin = np.array([x, y])
        
        #
        # @todo: To allow arbitrary orientation of the grid, add two extra parameters:
        #       g0  grid origin
        #       e1  primary direction
        # and then calculate e2 = R_pi/2 * e1 = [-e1[1], e1[0]]
        #
        # g0 = -np.array([extra_width, extra_height])
        # x, y = 1, 0
        # e1 = np.array([x, y])
        # e2 = np.array([-y, x])
        #
        # Then forall i, j in range(m) x range(n):
        #       cell.posn(i, j) = g0 + (j + 0.5) * e2 + (i + 0.5) * e1
        # and forall posn u in R^2:
        #       v = u - g0; j = floor(v . e1)/ grid_w, i = floor(v . e2) / grid_h
        #
        
        self.m = int(np.ceil(grid_h / self.cell_height)) + extra_height
        self.n = int(np.ceil(grid_w / self.cell_width))  + extra_width
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

                self.cells[i].append(Cell(cell_x, cell_y, half_w, half_h))
                
        # print("Grid({}, {}, {}, {}) has {} x {} cells from {}".format(grid_w, grid_h, cell_w, cell_h, self.m, self.n, self.origin))
        if 0:
            for i in range(self.m):
                print("Grid.cells[{}]??".format(i))
                print(self.cells[i])
                for j in range(self.n):
                    # print("Grid.cells[{}][{}]??".format(i, j))
                    # print("Grid.cells[{}][{}] has posn {}".format(i, j, self.cells[i][j].posn))
                    cell = self.cells[i][j]
                    i_tmp, j_tmp = self.get_cell_coordinates(cell.posn)
                    assert(i_tmp == i)
                    assert(j_tmp == j)

        return
    
    def get_cell_coordinates(self, posn):
        rel_posn = posn - self.origin
        i = int(rel_posn[1] / self.cell_height)
        j = int(rel_posn[0] / self.cell_width)
        return (i, j)

    def add_pedestrian(self, p):
        i, j = self.get_cell_coordinates(p.posn)
        cell = self.cells[i][j]
        # print("Model::add_pedestrian - pre-cell-distribution: ", p.id, p.posn, i, j)
        cell.add_pedestrian(p)

        i_tmp, j_tmp = self.get_cell_coordinates(p.posn)
        assert(i_tmp == i)
        assert(j_tmp == j)
        return

    def despawn_pedestrians(self):
        for row in self.cells:
            for cell in row:
                cell.despawn_pedestrians()
        return

    def transfer_migrants(self):
        # print("Model::transfer_migrants - start")
        for row in self.cells:
            for old_cell in row:
                # print("Transferring {} pedestrians from current cell".format(len(old_cell.emigrants)))
                for emigrant in old_cell.emigrants:
                    id, index_pair = emigrant
                    
                    i, j     = index_pair
                    # print("Transferring pedestrian {} to cell ({}, {})".format(id, i, j))
                    p        = old_cell.active_pedestrians[id]

                    i_tmp, j_tmp = self.get_cell_coordinates(p.posn)
                    if i_tmp != i or j_tmp != j:
                        print("Transferring pedestrian {} to cell ({}, {}) instead of ({}, {})".format(id, i, j, i_tmp, j_tmp))
                    assert(i_tmp == i)
                    assert(j_tmp == j)
                    
                    # new_cell = self.cells[i][j]
                    new_cell = self.cells[i_tmp][j_tmp]

                    del old_cell.active_pedestrians[id]
                    new_cell.add_pedestrian(p)
                    
                old_cell.emigrants = []
        # print("Model::transfer_migrants - finish")
        return
    
    def update_pedestrian_forces(self, obstacles, active_vehicles):
        for i in range(self.m):
            for j in range(self.n):
                cell = self.cells[i][j]
                i_tmp, j_tmp = self.get_cell_coordinates(cell.posn)
                assert(i_tmp == i)
                assert(j_tmp == j)
                for p_id in cell.active_pedestrians.keys():
                    p = cell.active_pedestrians[p_id]

                    i_tmp, j_tmp = self.get_cell_coordinates(p.posn)
                    if i_tmp != i or j_tmp != j:
                        print("Grid::update_pedestrian_forces - cell[{}, {}] contains pedestrian {} at {} expected to be in cell[{}, {}]".format(i, j, p.id, p.posn, i_tmp, j_tmp))
                    assert(i_tmp == i)
                    assert(j_tmp == j)

                    #
                    # @todo: Compute resultant force
                    #
                    p.f_total = np.zeros(2)
                    p.f_total += p.calculate_driving_force()
                    assert(np.dot(p.f_total, p.f_total) >= 0.)
                    # for k in range(i - 4, i + 5):
                    for k in range(i - 1, i + 2):
                        if k < 0 or k >= self.m:
                            continue
                        # for l in range(j - 4, j + 5):
                        for l in range(j - 1, j + 2):
                            if l < 0 or l >= self.n:
                                continue
                            countercell = self.cells[k][l]
                            for q_id in countercell.active_pedestrians.keys():
                                q = countercell.active_pedestrians[q_id] 
                                k_tmp, l_tmp = self.get_cell_coordinates(q.posn)
                                if k_tmp != k or l_tmp != l:
                                    print("Grid::update_pedestrian_forces - countercell[{}, {}] contains pedestrian {} at {} expected to be in cell[{}, {}]".format(k, l, q.id, q.posn, k_tmp, l_tmp))
                                assert(k_tmp == k)
                                assert(l_tmp == l)
                                if q.id == p.id:
                                    continue
                                f_pq = p.calculate_interpedestrian_repulsion(q, False)
                                # print("Model::advance_timestep - interpedestrian repulsion force (p.id, q.id, f_R): ", p.id, q.id, p.f_total)
                                p.f_total += f_pq
                                assert(np.dot(p.f_total, p.f_total) >= 0.)
                            # print("Grid::advance_timestep - total interpedestrian repulsion force (id, f_R): ", p.id, p.f_total)

                    for b in obstacles:
                        # print("Model::advance_timestep - pedestrian {} vs obstacle {}".format(p.id, b.id))
                        f_pb = b.calc_repulsive_force(p)
                        # f_pb = p.calculate_obstacle_repulsion(b)
                        p.f_total += f_pb
                        assert(np.dot(p.f_total, p.f_total) >= 0.)

                    for vhcl_id in active_vehicles:
                        #
                        # @todo: Compute resultant force
                        #
                        vhcl       = active_vehicles[vhcl_id]
                        f_pv       = p.calculate_vehicle_avoidance_force(vhcl)
                        # print("Grid::update_pedestrian_forces - pedestrian {} at {} travelling at vel. {} under total force {} ...".format(p.id, p.posn, p.v, p.f_total))
                        # print("... is encountering vehicle {} at {} travelling at vel. {}, adding new force {} ...".format(vhcl.id, vhcl.posn, vhcl.v, f_pv))
                        p.f_total += f_pv
                        # print("... resulting in new total force {}".format(p.f_total))
                        assert(np.dot(p.f_total, p.f_total) >= 0.)

    def update_pedestrian_kinematics(self, delta_t):
        for i in range(self.m):
            for j in range(self.n):
                cell = self.cells[i][j]

                # print("Grid::update_pedestrian_kinematics - cell at {}".format(cell.posn))
                for p_id in cell.active_pedestrians.keys():
                    p = cell.active_pedestrians[p_id] 

                    #
                    # @todo: Compute new position and add to position history.
                    #

                    old_posn = p.posn
                    
                    ptn_v  = p.v
                    v_norm = mna.calc_norm(ptn_v)
                    if v_norm > p.v_max:
                        ptn_v = p.v_max * p.dirn
                    new_posn = old_posn + ptn_v * delta_t
                    
                    # print("Pedestrian {} moving at {}m/s for {}s".format(p.id, p.v, delta_t))
                    # print("Pedestrian {} moving from {} to {}".format(p.id, old_posn, new_posn))
                    assert(0 <= np.dot(ptn_v, ptn_v) and np.dot(ptn_v, ptn_v) <= 4.)

                    # if np.dot(delta_s, delta_s) > 4.:
                        # print("Model::update_pedestrian_kinematics ped#{} is moving from {} to {}".format(p.id, old_posn, new_posn))
                    i_old, j_old = self.get_cell_coordinates(old_posn)
                    if i_old != i or j_old != j:
                        print("Grid::update_pedestrian_kinematics has a pedestrian moving from the wrong cell")
                        print("Pedestrian {} at {} is starting from ({}, {}) instead of ({}, {})".format(p.id, p.posn, i_old, j_old, i, j))

                    old_cell = self.cells[i][j]
                    # print("Grid::update_pedestrian_kinematics - old cell at {}".format(old_cell.posn))
                    assert(cell == old_cell)

                    i_new, j_new = self.get_cell_coordinates(new_posn)
                    new_cell = self.cells[i_new][j_new]
                    
                    if i_new != i_old or j_new != j_old:
                        old_cell.register_emigrant(p.id, i_new, j_new)

                    p.update_kinematics(delta_t)

                    cur_dst_idx = p.route[p.routing_idx + 1]
                    cur_dst     = p.nodegraph.nodes[cur_dst_idx]
                    
                    delta_finish = cur_dst.posn - p.posn 
                    d_sqr_dst    = np.dot(delta_finish, delta_finish)
                    if d_sqr_dst < cur_dst.radius * cur_dst.radius:
                        p.routing_idx += 1
                        if p.routing_idx + 1 == len(p.route):
                            new_cell.inactive_pedestrians.append(p.id)

        return

    def draw_background(self, view):
        for row in self.cells:
            for cell in row:
                cell.draw_background(view) # density-coloured rectangles + optional diagonal lines
        return

    def draw_foreground(self, view):
        for row in self.cells:
            for cell in row:
                cell.draw_foreground(view) # optional dashed gridlines + cell's pedestrians.draw() calls
        return
    


    
