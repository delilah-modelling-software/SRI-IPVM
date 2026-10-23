import numpy as np
import scipy as sp
import scipy.stats as ss
import pygame as pg

from collections import deque

#
# The following values for A, D are taken from Anvari (2015)
#     PvP   0.7 +- 0.25      2.25 +- 1
#     PvC   5   +- 1.        3 +- 1
#     CvP   6   +- 1.        5 +- 1
#     CvC   8   +- 1.       12 +- 1
# @todo: Organise these parameters into 2D tables for an arbitrary number of agent classes, e.g.:
#     0 -> pedestrian
#     1 -> bicycle
#     2 -> motorcycle
#     3 -> car
#     4 -> truck
#     5 -> bus
#     6 -> tram
#     A[0, 0] = A_PvP
#     A[0, 3] = A_PvC
#     etc.
# @todo: Replace parameters by normal-ish distributions: A -> N(A_mu, A_sigma); etc.
#
A_PvP =  0.7
D_PvP =  2.25
A_PvC =  5.
D_PvC =  3.
A_CvP =  6.
D_CvP =  5.
A_CvC =  8.
D_CvC = 12.

A       = np.zeros((6, 6))
A[0, 0] =  0.7
A[0, 3] =  5.
A[3, 0] =  6.
A[3, 3] =  8.

D       = np.zeros((6, 6))
D[0, 0] =  2.25
D[0, 3] =  3.
D[3, 0] =  5.
D[3, 3] = 12.

#
# d = min distance between centre[-line]s - sum(radii | halfstrokewidths)
# A, D are properties of the repulsor
# Note: Passing 1/D instead may allow reduction of #divisions
#
def calc_repulsion(d, A, D): # distance, amplitude, characteristic distance
    return A * np.exp(-d / D)

#
# d = min distance between centre[-line]s - sum(radii | halfstrokewidths)
# t = time since pedestrian (or driver) perceived attractor
# A, D, T are properties of the attractor
# Note: Passing 1/D, 1/T instead may allow reduction of #divisions
#
def calc_attraction(d, t, A, D, T): # distance, time, amplitude, characteristic distance, characteristic duration
    #
    # @todo: Example: callers should consider making T a random function of the pedestrians or pedestrian+attractor concerned.
    #
    # @todo: Is attraction exponentially decaying in both space and time? Or should the below formula be more complicated?
    # @todo: Consider more complicated formulae, e.g. a generic separable formula f(d) * g(t)
    #       hence, def calc_attraction(d, t, f, g): return f(d) * g(t)
    # @todo: Consider truly generic formulae, e.g f(d, t)
    #
    return A * np.exp(-(d / D + t / T))

class View:
    def __init__(self, width, height, screen_size, scale_factor, pedestrian_upscale_factor = 2.5):
        self.width                     = width
        self.height                    = height
        self.screen_size               = screen_size
        self.scale_factor              = scale_factor
        self.pedestrian_upscale_factor = pedestrian_upscale_factor
        return
    
    def fill_circle(self, colour, posn, radius):
        inverted_posn = np.array([posn[0], self.height - posn[1]])
        pg.draw.circle(pg.display.get_surface(), colour, self.scale_factor * inverted_posn, self.scale_factor * radius)
        return

    def stroke_circle(self, colour, posn, radius):
        inverted_posn = np.array([posn[0], self.height - posn[1]])
        pg.draw.circle(pg.display.get_surface(), colour, self.scale_factor * inverted_posn, self.scale_factor * radius, 1)
        return

    def draw_line(self, colour, posn1, posn2, radius):
        scaled_radius = int(np.maximum(2 * self.scale_factor * radius, 1))
        scaled_inverted_posn1 = self.scale_factor * np.array([posn1[0], self.height - posn1[1]])
        scaled_inverted_posn2 = self.scale_factor * np.array([posn2[0], self.height - posn2[1]])
        pg.draw.line(pg.display.get_surface(), colour,scaled_inverted_posn1, scaled_inverted_posn2, scaled_radius)
        return

    def draw_thin_line(self, colour, posn1, posn2):
        radius = 1
        inverted_posn1 = np.array([posn1[0], self.height - posn1[1]])
        inverted_posn2 = np.array([posn2[0], self.height - posn2[1]])
        pg.draw.line(pg.display.get_surface(), colour, self.scale_factor * inverted_posn1, self.scale_factor * inverted_posn2, radius)
        return

    def draw_rect(self, colour, posn1, posn2):
        scaled_inverted_posn1 = self.scale_factor * np.array([posn1[0], self.height - posn1[1]])
        scaled_inverted_posn2 = self.scale_factor * np.array([posn2[0], self.height - posn2[1]])
        pg.draw.rect(pg.display.get_surface(), colour, pg.Rect(scaled_inverted_posn1, scaled_inverted_posn2 - scaled_inverted_posn1))
        return

#
# Useful Colours
#
white          = pg.Color(255, 255, 255)
black          = pg.Color(0, 0, 0)
red            = pg.Color(255, 0, 0)
yellow         = pg.Color(255, 255, 0)
grey           = pg.Color(128, 128, 128)
dark_grey      = pg.Color(64, 64, 64)
gold           = grey.lerp(white, 0.5)
black_and_gold = black.lerp(gold, 0.5)


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
        self.n_ped += 1
        # print("Cell::add_pedestrian - finish: id, posn, new n_ped", p.id, p.posn, self.n_ped)
        return

    def preremove_pedestrian(self, id):
        p = self.active_pedestrians[id]
        # print("Cell::remove_pedestrian - start: id, posn, old n_ped", id, p.posn, self.n_ped)
        self.emigrants.append(self.active_pedestrians[id])
        self.n_ped -= 1
        # print("Cell::preremove_pedestrian - finish: id, old n_ped", id, self.n_ped)
        return

    def postremove_pedestrians(self):
        # print("Cell::postremove_pedestrians - start")
        for emigrant in self.emigrants:
            del self.active_pedestrians[emigrant.id]
        self.emigrants = []
        # print("Cell::postremove_pedestrians - finish")
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
        cell_colour = pg.Color(greyness, greyness, greyness)
        
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
        if density >= 6:
            p1 = self.posn - 0.5 * np.array([self.half_w, self.half_h])
            p2 = self.posn + 0.5 * np.array([self.half_w, -self.half_h])
            p3 = self.posn + 0.5 * np.array([-self.half_w, self.half_h])
            p4 = self.posn + 0.5 * np.array([self.half_w, self.half_h])

            view.draw_thin_line(red, p1, p4)
            view.draw_thin_line(red, p2, p3)
            
        for p_id in self.active_pedestrians:
            p = self.active_pedestrians[p_id]
            p.draw(view)
        return
    
class Model:
    def __init__(self, width, height, screen_size, scale_factor, T_step, reaction_factor, centroids, obstacles = [], attractors = [], obscurers = [], relationships = [], background_colour = pg.Color(128, 128, 128)):
        # self.cell_map = {}

        self.w          = width
        self.h          = height
        self.T_step     = T_step
        self.T_reaction = reaction_factor * T_step
        
        self.centroids     = centroids
        self.obstacles     = obstacles
        self.attractors    = attractors
        self.relationships = relationships

        self.MRCA_flag = False

        # self.active_pedestrians   = {}
        # self.inactive_pedestrians = deque()
        self.active_vehicles      = {}
        self.n_vehicles           = 0
        self.inactive_vehicles    = deque()
        self.next_pedestrian_id   = 0
        self.next_vehicle_id      = 0
        self.cur_step             = 0


        #
        # @todo: Decompose model into cells.
        #     Cut model area into tessellating cells (squares, but regular hexagons and triangles could also work).
        #     Construct 2D array of cell-ids.
        #     Move centroids, small obstacles (and attractors, obscurers), and pedestrians into cells.
        #     As pedestrians move, it will be necessary to move them between cells.
        #
        
        self.cell_width = self.cell_height = 2.
        # self.cell_width = self.cell_height = 4.
        self.m_ped_cells = int(np.ceil(height / self.cell_height) + 2.)
        self.n_ped_cells = int(np.ceil(width / self.cell_width)   + 2.)
        self.ped_cells   = []
        half_w = 0.5 * self.cell_width
        half_h = 0.5 * self.cell_height
        for i in range(self.m_ped_cells):
            # cell_y = i * self.cell_height + 0.5 * self.cell_height
            cell_y = (i - 0.5) * self.cell_height
            for j in range(self.n_ped_cells):
                # self.ped_cells.append(Cell(j * self.cell_width + half_w, cell_y, half_w, half_h))
                # self.ped_cells.append(Cell(j * self.cell_width, cell_y, half_w, half_h))
                self.ped_cells.append(Cell(j * self.cell_width - half_w, cell_y, half_w, half_h))

        self.colour = background_colour

    def spawn_pedestrians(self):
        # print("Model::spawn_pedestrians - step ", self.cur_step, ": start")
        for centroid in self.centroids:
            centroid.spawn_pedestrians(self)
        # print("Model::spawn_pedestrians - step ", self.cur_step, ": finish")
        
    def despawn_pedestrians(self):
        for cell in self.ped_cells:
            cell.despawn_pedestrians()
        return

    def spawn_vehicles(self):
        # print("Model::spawn_vehicles - step ", self.cur_step, ": start")
        for centroid in self.centroids:
            centroid.spawn_vehicles(self)
        # print("Model::spawn_vehicles - step ", self.cur_step, ": finish")
        
    def despawn_vehicles(self):
        # print("Model::despawn_vehicles - start: old n_vhcl", self.n_vehicles)
        for vhcl_id in self.inactive_vehicles:
            if vhcl_id not in self.active_vehicles:
                continue
            del(self.active_vehicles[vhcl_id])
            self.n_vehicles -= 1
        self.inactive_vehicles = []
        # print("Model::despawn_vehicles - finish: new n_vhcl", self.n_vehicles)
        return

    def spawn_agents(self):
        self.spawn_pedestrians()
        self.spawn_vehicles()
        return

    def despawn_agents(self):
        self.despawn_pedestrians()
        self.despawn_vehicles()
        return

    def get_cell_index(self, i, j):
        return i * self.n_ped_cells + j

    def add_pedestrian(self, p):
        # print("Model::add_pedestrian - start: ", self.next_pedestrian_id, p.posn)
        self.next_pedestrian_id += 1

        # i = int(p.posn[1] / self.cell_height)
        # j = int(p.posn[0] / self.cell_width)
        i = int(p.posn[1] / self.cell_height + 1.)
        j = int(p.posn[0] / self.cell_width  + 1.)
        cell = self.ped_cells[self.get_cell_index(i, j)]
        # print("Model::add_pedestrian - pre-cell-distribution: ", p.id, p.posn, i, j)
        cell.add_pedestrian(p)

    def update_pedestrian_kinematics(self, p, delta_s):
        old_posn = p.posn
        new_posn = old_posn + delta_s

        # i_old = int(old_posn[1] / self.cell_height)  
        # j_old = int(old_posn[0] / self.cell_width)
        # i_new = int(new_posn[1] / self.cell_height)  
        # j_new = int(new_posn[0] / self.cell_width)

        # i_old = int(old_posn[1] / self.cell_height + 0.5)  
        # j_old = int(old_posn[0] / self.cell_width  + 0.5)
        # i_new = int(new_posn[1] / self.cell_height + 0.5)  
        # j_new = int(new_posn[0] / self.cell_width  + 0.5)

        #
        # 
        #
        print("Pedestrian::update_pedestrian_kinematics ped#{} is moving from {} to {}".format(p.id, old_posn, new_posn))
        i_old = int(old_posn[1] / self.cell_height + 1.)
        j_old = int(old_posn[0] / self.cell_width  + 1.)
        i_new = int(new_posn[1] / self.cell_height + 1.)
        j_new = int(new_posn[0] / self.cell_width  + 1.)

        p.posn = new_posn

        old_cell = self.ped_cells[self.get_cell_index(i_old, j_old)]
        # new_cell = self.ped_cells[self.get_cell_index(i_new, j_new)]
        nci      = self.get_cell_index(i_new, j_new)
        if nci < 0 or nci >= self.m_ped_cells * self.n_ped_cells:
            print("Cell index out-of-range ({}, {}) not in [0, {})x[0, {})".format(i_new, j_new, self.m_ped_cells, self.n_ped_cells))
        new_cell = self.ped_cells[nci]
        if i_new != i_old or j_new != j_old:
            new_cell.add_pedestrian(p)
            old_cell.preremove_pedestrian(p.id)

        p.v    += self.T_step * p.f_total
        v_sqr   = np.dot(p.v, p.v)
        p.dirn  = p.v / np.sqrt(v_sqr)
        if v_sqr > p.v_max * p.v_max:
            p.v = p.v_max * p.dirn
            
        delta_finish = p.dst.posn - p.posn 
        d_sqr_dst    = np.dot(delta_finish, delta_finish)
        if d_sqr_dst < p.dst.radius * p.dst.radius:
            new_cell.inactive_pedestrians.append(p.id)

    def add_vehicle(self, v):
        self.next_vehicle_id += 1
        self.active_vehicles[v.id] = v
        self.n_vehicles += 1
        return

    def remove_vehicle(self, id):
        del self.active_vehicles[id]
        return

    def draw(self, view):
        # pg.draw.rect(pg.display.get_surface(), self.colour, pg.Rect((0., 0.), (view.scale_factor * self.w, view.height - view.scale_factor * self.h)))
        pg.draw.rect(pg.display.get_surface(), self.colour, pg.Rect((0., 0.), (view.scale_factor * self.w, view.scale_factor * self.h)))
        for cell in self.ped_cells:
            cell.draw_background(view) # density-coloured rectangles + optional diagonal lines
        for centroid in self.centroids:
            centroid.draw(view) # Dull circles
        for obstacle in self.obstacles:
            print("Model::draw obstacle", obstacle.id)
            obstacle.draw(view) # [White fill + Black stroke] shapes
        for attractor in self.attractors:
            attractor.draw(view)    # [Gold fill + Black stroke] shapes
        for cell in self.ped_cells:
            cell.draw_foreground(view) # optional dashed gridlines + cell's pedestrians.draw() calls

        #
        # @todo: Add support for obscurers, cells, and/or relationships.
        #
        
        # for obscurer in self.obscurers:
            # obscurer.draw(view)     # [Black polka dot pattern fill] shapes
        # for cell in self.ped_cells:
            # cell.draw(view) # optional dashed gridlines + cell's pedestrians.draw() calls
        for k, v in self.active_vehicles.items():
            v.draw(view)
        # for relationship in self.relationships:
            # relationship.draw(view) # [Gold] connecting lines between pedestrian-pairs

    def update_vehicle_kinematics(self, vhcl, delta_s):
        old_posn = vhcl.posn
        new_posn = old_posn + delta_s

        vhcl.posn += delta_s

        vhcl.v    += self.T_step * vhcl.f_total
        v_sqr   = np.dot(vhcl.v, vhcl.v)
        vhcl.dirn  = vhcl.v / np.sqrt(v_sqr)
        if v_sqr > vhcl.v_max * vhcl.v_max:
            vhcl.v = vhcl.v_max * vhcl.dirn
            
        delta_finish = vhcl.dst.posn - vhcl.posn 
        d_sqr_dst    = np.dot(delta_finish, delta_finish)
        is_near_dst  = d_sqr_dst < vhcl.dst.radius * vhcl.dst.radius
        
        travel_goal  = vhcl.dst.posn - vhcl.src.posn
        travel       = vhcl.posn - vhcl.src.posn
        has_travelled_past_dst = np.dot(travel, travel_goal) > np.dot(travel_goal, travel_goal)
        if is_near_dst or has_travelled_past_dst:
            self.inactive_vehicles.append(vhcl.id)
        return

    def advance_timestep(self):
        # print("Model::advance_timestep - step ", self.cur_step, ": start")
        for i in range(self.m_ped_cells):
            for j in range(self.n_ped_cells):
                cell = self.ped_cells[self.get_cell_index(i, j)]
                for p_id in cell.active_pedestrians.keys():
                    p = cell.active_pedestrians[p_id] 

                    #
                    # @todo: Compute resultant force
                    #
                    p.f_total = np.zeros(2)
                    p.f_total += p.calculate_driving_force()
                    # for k in range(i - 4, i + 5):
                    for k in range(i - 1, i + 2):
                        if k < 0 or k >= self.m_ped_cells:
                            continue
                        # for l in range(j - 4, j + 5):
                        for l in range(j - 1, j + 2):
                            if l < 0 or l >= self.n_ped_cells:
                                continue
                            countercell = self.ped_cells[self.get_cell_index(k, l)]
                            for q_id in countercell.active_pedestrians.keys():
                                q = countercell.active_pedestrians[q_id] 
                                if q.id == p.id:
                                    continue
                                f_pq = p.calculate_interpedestrian_repulsion(q, self.MRCA_flag)
                                # print("Model::advance_timestep - interpedestrian repulsion force (p.id, q.id, f_R): ", p.id, q.id, p.f_total)
                                p.f_total += f_pq
                            # print("Model::advance_timestep - total interpedestrian repulsion force (id, f_R): ", p.id, p.f_total)

                    for b in self.obstacles:
                        f_pb = p.calculate_obstacle_repulsion(b, self.MRCA_flag)
                        p.f_total += f_pb

                    for v in self.active_vehicles:
                        #
                        # @todo: Compute resultant force
                        #
                        f_pv = p.calculate_vehicle_avoidance_force(v)
                        p.f_total += f_pv

        for k, v in self.active_vehicles.items():
            #
            # @todo: Compute resultant force
            #
            v.f_total = np.zeros(2)
            v.f_total += v.calculate_driving_force()
            for l, w in self.active_vehicles.items():
                if w.id == v.id:
                    continue

                f_vw       = v.calculate_intercar_repulsion(w, self.MRCA_flag)
                # v.f_total += f_vw
            # v.f_total += v.calculate_total_pedestrian_avoidance_force()
            
            for b in self.obstacles:
                f_vb       = v.calculate_obstacle_repulsion(b, self.MRCA_flag)
                # v.f_total += f_vb

        self.cur_step += 1

        #
        # Movement stage
        #
        for cell in self.ped_cells:
            for p_id in cell.active_pedestrians.keys():
                p = cell.active_pedestrians[p_id] 

                #
                # @todo: Compute new position and add to position history.
                # @todo: Move between cells if necessary.
                #

                # print("Model::advance_timestep - pre - pedestrian (id, posn, v, f_total)", p.id, p.posn, p.v, p.f_total)
                # p.posn += self.T_step * p.v
                self.update_pedestrian_kinematics(p, self.T_step * p.v)
                
                # print("Model::advance_timestep - post - pedestrian (id, posn, v, f_total)", p.id, p.posn, p.v, p.f_total)

        for cell in self.ped_cells:
            cell.postremove_pedestrians()

        for k, v in self.active_vehicles.items():
            #
            # @todo: Compute new position and add to position history.
            # @todo: Move between cells if necessary.
            #
            # v.posn += self.T_step * v.v
            # v.v    += self.T_step * v.f_total
            self.update_vehicle_kinematics(v, self.T_step * v.v)
        # print("Model::advance_timestep - step ", self.cur_step, ": finish")
        return

#
# @todo: Move into Model, etc.
#
A_pedestrian = 2.1
D_pedestrian = 0.3

epsilon_v_sqr = 0.0625 * 0.0625

class Agent:
    def __init__(self, id, A, D, src, dst, posn, dirn, view):
        self.id   = id
        self.A    = A
        self.D    = D
        self.src  = src
        self.dst  = dst
        self.posn = posn
        self.dirn = dirn
        self.view = view

        return

class Pedestrian(Agent):
    def __init__(self, id, A, D, src, dst, posn, dirn, src_colour, dst_colour, view):
        # super(Agent, self).__init__()
        super(Pedestrian, self).__init__(id, A, D, src, dst, posn, dirn, view)

        self.src_colour = src_colour
        self.dst_colour = dst_colour
        
        self.f_total = np.zeros(2)
        self.v       = np.zeros(2)

        self.v_ped          = 1.33
        self.v_max          = 1.3 * self.v_ped # (Helbing and Moln{Ctrl+', a}r 1995)
        self.T_relax        = 0.3 # 0.3s, matches Anvari et al. (2015) rather than the 0.5 from (Helbing and Moln{Ctrl+', a}r 1995)

        self.T_SRCA_horizon = 0.5
        self.T_MRCA_horizon = 2.
        self.T_LRCA_horizon = 4.
        self.radius         = 0.2
        return

    def calculate_driving_force(self):
        v   = self.dst.posn - self.posn
        u   = v / np.sqrt(np.dot(v, v))
        f_D = (self.v_ped * u - self.v) / self.T_relax
        # print("Pedestrian:calculate_driving_force (id, v_desired, v, f_D): ", self.id, self.v_ped * u, self.v, f_D)
        return f_D

    def calculate_interpedestrian_repulsion(self, q, MRCA_flag):
        self.T_horizon = self.T_SRCA_horizon
        if MRCA_flag:
            self.T_horizon = self.T_MRCA_horizon

        #
        # @todo: Moving max_s into the Model and refining it may be a good idea.
        #
        ds = q.posn - self.posn
        max_s = 2. # 1.4 * 1.33 * self.T_horizon**2
        if np.dot(ds, ds) > max_s * max_s:
            return np.zeros(2)

        dv = q.v - self.v

        t_min     = 0.
        if 0: # np.dot(dv, dv) >= epsilon_v_sqr:
            t_min = -np.dot(ds, dv) / np.dot(dv, dv)
            if t_min < 0:
                t_min = 0
            elif t_min > self.T_horizon:
                t_min = self.T_horizon

        delta    = ds + t_min * dv
        distance = np.sqrt(np.dot(delta, delta)) 
        u        = delta / distance
        if distance < self.radius + q.radius:
            print("Conflict (expected impact) between {} and {} at {}".format(self.id, q.id, 0.5 * (self.posn + q.posn)))
        f_R      = calc_repulsion(distance - (self.radius + q.radius), self.A, self.D) * -u

        dv_sqr  = np.dot(dv, dv)
        dv_norm = np.sqrt(dv_sqr)
        DV_EPSILON = 0.25 * 0.25
        if dv_norm > DV_EPSILON: # MRCA_flag and t_min > self.T_SRCA_horizon:
            # f_R /= t_min / self.T_SRCA_horizon

            #
            # @todo: Reject component of f_R parallel to dv
            #
            unit_dv = dv / dv_norm
            # f_R -= np.dot(f_R, unit_dv) * unit_dv
            f_R = np.dot(f_R, unit_dv) * unit_dv

        # print("Pedestrian:calculate_repulsion (id, q.id, f_R): ", self.id, q.id, f_R)
        return f_R

    def calculate_vehicle_avoidance_force(self, v):
        f_R = np.zeros(2)
        return f_R

    def calculate_circular_obstacle_repulsion(self, b, MRCA_flag):
        A_b = b.A
        D_b = b.D

        ds = b.shape.posn - self.posn
        dv = -self.v

        t_min     = 0.
        T_horizon = self.T_SRCA_horizon
        if MRCA_flag:
            T_horizon = self.T_MRCA_horizon

        if np.dot(dv, dv) >= epsilon_v_sqr:
            t_min = -np.dot(ds, dv) / np.dot(dv, dv)
            if t_min < 0:
                t_min = 0
            elif t_min > T_horizon:
                t_min = T_horizon

        delta    = ds + t_min * dv
        distance = np.sqrt(np.dot(delta, delta)) 
        u        = delta / distance
        if distance < self.radius + b.shape.radius:
            print("Collision between pedestrian {} and obstacle {}", self.id, b.id)
        f_R      = calc_repulsion(distance - (self.radius + b.shape.radius), A_b, D_b) * -u
        if MRCA_flag and t_min > self.T_SRCA_horizon:
            f_R /= t_min / self.T_SRCA_horizon0
            #
            # @todo: Reject component of f_R parallel to dv
            #
            unit_dv = dv / np.sqrt(np.dot(dv, dv))
            f_R -= np.dot(f_R, unit_dv) * unit_dv

        # f_R = np.zeros(2)
        return f_R

    def calculate_linear_obstacle_repulsion(self, b, MRCA_flag):
        #
        # @todo: Redesign and recode this functionality?
        # @todo: Add a composite obstacle type. Find the nearest component of the obstacle during the
        # near-future and use its repulsion rule. By combining linear and circular obstacles, arbitrary
        # shapes can be supported.
        #
        
        A_b = b.A
        D_b = b.D

        v = self.v
        w = b.shape.posn2 - b.shape.posn1

        x0 = b.shape.posn1
        x1 = b.shape.posn2
        if np.dot(v, w) <= 0:
            w = -w
            x0 = b.shape.posn2
            x1 = b.shape.posn1
        u = self.posn - b.shape.posn1
            
        unit_w = w / np.sqrt(np.dot(w, w))
        
        #
        # Calculate the projection and rejection of u, v w.r.t. w
        #
        u_par  = np.dot(u, unit_w) * unit_w
        u_perp = u - u_par
        u_perp_sqr = np.dot(u_perp, u_perp)
        v_par  = np.dot(v, unit_w) * unit_w
        v_perp = v - v_par
        v_perp_sqr = np.dot(v_perp, v_perp)

        t_min     = 0.
        T_horizon = self.T_SRCA_horizon
        if MRCA_flag:
            T_horizon = self.T_MRCA_horizon

        if np.dot(u_perp, v_perp) <= 0 and v_perp_sqr >= epsilon_v_sqr:
            v_sqr = np.dot(v, v)
            t_min = np.sqrt(u_perp_sqr / v_perp_sqr) - 0.5
            if t_min < 0.:
                t_min = 0.
            elif t_min > T_horizon:
                t_min = T_horizon
        delta_par  = u_par + t_min * v_par
        delta_perp = u_perp + t_min * v_perp
        delta      = delta_par + delta_perp

        distance   = np.sqrt(np.dot(delta, delta)) 
        unit_delta = delta_perp / distance
        if distance < self.radius + b.shape.radius:
            print("Collision between pedestrian {} and obstacle {}", self.id, b.id)
        f_R      = calc_repulsion(distance - (self.radius + b.shape.radius), A_b, D_b) * unit_delta
        if MRCA_flag and t_min > self.T_SRCA_horizon:
            f_R /= t_min / self.T_SRCA_horizon
            #
            # @todo: Reject component of f_R parallel to dv
            #
            # unit_dv = dv / np.sqrt(np.dot(dv, dv))
            #f_R -= np.dot(f_R, unit_dv) * unit_dv

        # f_R = np.zeros(2)
        return f_R

    def calculate_obstacle_repulsion(self, b, MRCA_flag):
        if isinstance(b.shape, Circle):
            return self.calculate_circular_obstacle_repulsion(b, MRCA_flag)
        elif isinstance(b.shape, Segment):
            return self.calculate_linear_obstacle_repulsion(b, MRCA_flag)
        else:
            return np.zeros(2)


    def calculate_total_force(self):
        return

    def draw(self, view):
        # print("Pedestrian::draw (id, posn, r)", self.id, self.posn, self.radius)
        # view.draw_circle(self.src_colour, self.posn, self.radius)
        # view.draw_line(white, self.posn, self.posn + self.radius * self.dirn, 0)
        spd = np.sqrt(np.dot(self.v, self.v))

        view = self.view
        view.fill_circle(self.src_colour, self.posn, view.pedestrian_upscale_factor * self.radius)
        # view.stroke_circle(black, self.posn, 2.5 * self.radius)
        view.draw_line(white, self.posn, self.posn + view.pedestrian_upscale_factor * spd * self.radius * self.dirn, 0)
        return

class Vehicle(Agent):
    def __init__(self, id, A, D, src, dst, posn, dirn, length, width, colour, view):
        super(Vehicle, self).__init__(id, A, D, src, dst, posn, dirn, view)
        
        self.length = length # 9.6m
        self.width  = width  # 3.6m

        self.colour = colour

        #
        # @todo: A starting velocity should probably be passed in.
        #
        self.f_total = np.zeros(2)
        self.v       = np.zeros(2)

        self.v_legal        = 2.78 # 10 km/h = 25/9 m/s ~ 2.78 m/s
        self.v_max          = 8.33 # 30 km/h = 25/3 m/s, matches Anvari (2015), good enough for now.
        self.T_relax        = 2.4  # 2.4s, matches Anvari et al. (2015)
        self.T_SRCA_horizon = 0.5
        self.T_MRCA_horizon = 2.
        self.T_LRCA_horizon = 4.

        self.f_total = np.zeros(2)
        return

    def calculate_driving_force(self):
        ds      = self.dst.posn - self.posn
        unit_ds = ds / np.sqrt(np.dot(ds, ds))

        #
        # @todo: Experiment with using v_legal instead.
        #
        f_D = (self.v_max * unit_ds - self.v) / self.T_relax
        print("Vehicle:calculate_driving_force (id, v_desired, v, f_D): ", self.id, self.v_max * unit_ds, self.v, f_D)
        return f_D

    #
    # @todo: This function probably needs to be specialised into a pair of functions, e.g.:
    #     calculate_intercar_repulsion_section; and
    #     calculate_intercar_repulsion_node,
    # where the first function would simplify most of the calculations significantly. 
    #
    def calculate_intercar_repulsion(self, car, MRCA_flag):
        ds = car.posn - self.posn
        dv = car.v    - self.v
        
        distance = np.sqrt(np.dot(ds, ds))
        u        = ds / distance


        f_R = np.zeros(2)
        # return f_R

        #
        # Calculate a social force based purely on current distance and accumulate.
        #
        max_s = 2 * (self.v_legal + car.v_legal)
        if np.dot(ds, ds) > max_s * max_s:
            c_self = np.dot(self.dirn, ds) / distance
            # s_self = np.sqrt(1 - c_self * c_self)

            #
            # @todo: After copying the following code to a Vehicle::calculate_pedestrian_repulsion
            # function, we should be able to simplify to r_self = 0.5 * self.length, r_car = 0.5 * car.length
            #

            c_other = -np.dot(car.dirn, ds) / distance
            # s_other = np.sqrt(1 - c_other * c_other)

            l_self = 0.5 * self.length
            w_self = 0.5 * self.width
            assert(l_self >= w_self)
            e_self = np.sqrt(l_self * l_self - w_self * w_self) / l_self
            r_self = w_self / np.sqrt(1 - (e_self * c_self)**2)
            
            l_car = 0.5 * car.length
            w_car = 0.5 * car.width
            assert(l_car >= w_car)
            e_car = np.sqrt(l_car * l_car - w_car * w_car) / l_car
            r_car = w_car / np.sqrt(1 - (e_car * c_self)**2)

            #
            # @todo: Initialise lambda to something sensible from one of Anvari's papers if at all possible.
            #
            lambda_self       = 1.
            perception_factor = lambda_self + (1 - lambda_self) * 0.5 * (1 + c_self)
            
            
            f_R += perception_factor * calc_repulsion(distance - (r_self + r_car), self.A, self.D) * -u

        #
        # Calculate a car-following force based on differential velocity and accumulate.
        #
        
        # Theta = np.dot(dv, ds) > 0
        # f_R += Theta * calc_repulsion(distance - (r_self + r_car), A, D) * -dv / T_relaxation

        return f_R

    def calculate_pedestrian_repulsion(self, pedestrian, MRCA_flag):
        ds = car.posn - self.posn
        dv = car.v    - self.v
        
        distance = np.sqrt(np.dot(ds, ds))


        f_R = np.zeros(2)
        return f_R

        #
        # Calculate a social force based purely on current distance and accumulate.
        #
        max_s = 2 * (self.v_legal + car.v_legal)
        if np.dot(ds, ds) > max_s * max_s:
            c_self = np.dot(self.dirn, ds) / distance
            # s_self = np.sqrt(1 - c_self * c_self)

            #
            # @todo: After copying the following code to a Vehicle::calculate_pedestrian_repulsion
            # function, we should be able to simplify to r_self = 0.5 * self.length, r_car = 0.5 * car.length
            #

            l_self = 0.5 * self.length
            w_self = 0.5 * self.width
            assert(l_self >= w_self)
            e_self = np.sqrt(l_self * l_self - w_self * w_self) / l_self
            r_self = w_self / np.sqrt(1 - (e_self * c_self)**2)
            
            #
            # @todo: Initialise lambda to something sensible from one of Anvari's papers if at all possible.
            #
            lambda_self       = 0.2
            perception_factor = lambda_self + (1 - lambda_self) * 0.5 * (1 + c_self)
            
            
            f_R += perception_factor * calc_repulsion(distance - (r_self + ped.radius), self.A, self.D) * -u
        return f_R
        

    def draw(self, view):
        print("Vehicle::draw (id, posn, v)", self.id, self.posn, self.v)
        spd = np.sqrt(np.dot(self.v, self.v))

        view.draw_line(self.colour, self.posn - 0.5 * self.length * self.dirn, self.posn + 0.5 * self.length * self.dirn, 0.5 * self.width)
        view.draw_line(white, self.posn, self.posn + 0.5 * (spd / self.v_max) * self.length * self.dirn, 0)
        return

class Streamspawner:
    def __init__(self):
        return

    def generate_spawning_times(self, T_simulation, T_step, n_agents):
        return []

#
# @todo: Add a T_period to Clusterspawner and Betaspawner, after which the spawning times will be calculated
# in two stages: determine (with uniform distribution) within which period of the simulation each spawning event
# will occur and then determine (using appropriate GPDF) when within that period the spawning even will occur.
#

#
# Generalised PDF (probability distribution function) is the Dirac delta function. All pedestrians spawn at once.
# Suitable for people who have arrived in an enclosed box, e.g. elevators, buses, etc.
#
class Clusterspawner(Streamspawner):
    def __init__(self, timeshift):
        super(Clusterspawner, self).__init__(self)
        self.T_shift = timeshift
        return

    def generate_spawning_times(self, T_simulation, T_step, n_agents):
        for k in range(int(n_agents)):
            t_sample = np.fmod(sample_times[k] + self.T_shift, T_simulation)
            step = int(sample_times[k] / T_step)
            spawning_list.append(step)
            # print("Spawning pedestrians for ({}, {}) @ time-step {}".format(i, j, step0))
        spawning_list.sort()
        return spawning_list

#
# PDF (probability distribution function) is the uniform distribution. Pedestrians
# spawn continuously at a uniform rate. Suitable for people who have arrived without
# passing through traffic signalised crossings, e.g. footbridges and pedestrian tunnels.
#
class Uniformspawner(Streamspawner):
    def __init__(self):
        super(Uniformspawner, self).__init__()
        return
    
    def generate_spawning_times(self, T_simulation, T_step, n_agents):
        sample_times = ss.uniform.rvs(loc = 0, scale = T_simulation, size = n_agents)
        spawning_list = []
        for k in range(int(n_agents)):
            t_sample = sample_times[k]
            step     = int(t_sample / T_step)
            spawning_list.append(step)
            # print("Spawning agents for ({}, {}) @ time-step {}".format(i, j, step0))
        spawning_list.sort()
        return spawning_list

#
# PDF (probability distribution function) is a beta distribution. Pedestrians
# spawn continuously with one or two modal times. Suitable for people who have arrived by
# passing through timed bottlenecks, e.g. signalised crossings.
#
class Betaspawner(Streamspawner):
    def __init__(self, timeshift, a, b):
        super(Betaspawner, self).__init__()
        self.T_shift = timeshift
        self.a = a
        self.b = b
        return

    def generate_spawning_times(self, T_simulation, T_step, n_agents):
        sample_times  = ss.beta.rvs(self.a, self.b, loc = 0, scale = T_simulation, size = n_agents)
        spawning_list = []
        for k in range(int(n_agents)):
            t_sample = np.fmod(sample_times[k] + self.T_shift, T_simulation)
            step     = int(t_sample / T_step)
            spawning_list.append(step)
            # print("Spawning pedestrians for ({}, {}) @ time-step {}".format(i, j, step0))
        spawning_list.sort()
        return spawning_list

class Stream:
    def __init__(self, id, src_id, dst_id, type, T_shift):
        self.id      = id
        self.src_id  = src_id
        self.dst_id  = dst_id
        self.type    = type
        self.T_shift = T_shift


class Centroid: # gates, doorways, elevators, transporters, stopped [public transport] vehicles
    def __init__(self, id, posn, r, colour, agent_colour, view, streamspawner, agent_type = Pedestrian):
        self.id = id
        
        self.posn         = posn
        self.radius       = r
        self.colour       = colour
        self.agent_colour = agent_colour
        
        self.view          = view
        self.streamspawner = streamspawner
        self.agent_type    = agent_type

        self.spawning_table = {}
        
        return

    def register_spawning_list(self, dst, spawning_list):
        self.spawning_table[dst.id] = spawning_list
        return
    
    def spawn_pedestrians(self, model):
        # print("Centroid::spawn_pedestrians - step ", model.cur_step, ": start")
        if self.agent_type == Pedestrian:
            # print("Centroid[{}]::spawn_pedestrians - step ".format(self.id), model.cur_step, ": start")
            for dst_id in self.spawning_table.keys():
                # print("Centroid::spawn_pedestrians - spawning_table[", dst_id, "]")
                spawning_list = self.spawning_table[dst_id]

                spawning_sentinel = 0
                while spawning_sentinel < len(spawning_list) and spawning_list[spawning_sentinel] == model.cur_step:
                    spawning_sentinel += 1
                if spawning_sentinel <= 0:
                    continue
                self.spawning_table[dst_id] = spawning_list[spawning_sentinel:]
                for i in range(spawning_sentinel):
                    src   = self
                    dst   = model.centroids[dst_id]
                    delta = dst.posn - self.posn
                    dirn  = delta / np.sqrt(np.dot(delta, delta))

                    perp              = np.array([-dirn[1], dirn[0]])
                    deviant_magnitude = self.radius * (ss.beta.rvs(2, 0.5) * 2 - 1)
                    posn              = self.posn + deviant_magnitude * perp

                    p = Pedestrian(model.next_pedestrian_id, A_pedestrian, D_pedestrian, self, dst, posn, dirn, self.agent_colour, dst.agent_colour, self.view)
                    # print("Model::spawn_pedestrians - new pedestrian ((id, posn, dirn))", p.id, p.posn, p.dirn)

                    model.add_pedestrian(p)
                    # model.next_pedestrian_id += 1
                    # model.active_pedestrians[p.id] = p
            # print("Centroid::spawn_pedestrians - step ", model.cur_step, ": finish")
        # print("Centroid::spawn_pedestrians - step ", model.cur_step, ": finish")
        return

    def spawn_vehicles(self, model):
        if self.agent_type == Vehicle:
            # print("Centroid[{}]::spawn_vehicles - step ".format(self.id), model.cur_step, ": start")
            for dst_id in self.spawning_table.keys():
                # print("Centroid[{}]::spawn_vehicles - spawning_table[".format(self.id), dst_id, "]")
                spawning_list = self.spawning_table[dst_id]

                spawning_sentinel = 0
                while spawning_sentinel < len(spawning_list) and spawning_list[spawning_sentinel] == model.cur_step:
                    spawning_sentinel += 1
                print("Centroid[{}]::spawn_vehicles - spawning_table[{}] should spawn {} vehicles".format(self.id, dst_id, spawning_sentinel))
                if spawning_sentinel <= 0:
                    continue

                self.spawning_table[dst_id] = spawning_list[spawning_sentinel:]
                for i in range(spawning_sentinel):
                    src   = self
                    dst   = model.centroids[dst_id]
                    delta = dst.posn - self.posn
                    dirn  = delta / np.sqrt(np.dot(delta, delta))

                    perp              = np.array([-dirn[1], dirn[0]])
                    deviant_magnitude = 0. # self.radius * (ss.beta.rvs(2, 0.5) * 2 - 1)
                    posn              = self.posn + deviant_magnitude * perp

                    v = Vehicle(model.next_vehicle_id, A_CvC, D_CvC, self, dst, posn, dirn, 4.8, 1.8, self.agent_colour, self.view)
                    print("Centroid::spawn_vehicles - new vehicle ((id, posn, dirn))", v.id, v.posn, v.dirn)

                    model.add_vehicle(v)
                    # model.next_pedestrian_id += 1
                    # model.active_pedestrians[p.id] = p
            print("Centroid::spawn_vehicles - step ", model.cur_step, ": finish")
        # print("Centroid::spawn_vehicles - step ", model.cur_step, ": finish")
        return

    def spawn_agents(self, model):
        if self.agent_type == Vehicle:
            self.spawn_vehicles(model)
        else:
            self.spawn_pedestrians(model)            
        return
    
    def draw(self, view):
        view.fill_circle(self.colour, self.posn, self.radius)
        return
#
# Note: Centroid could be extended to act as a Portal between Models, i.e. each Portal within a Model
# would be paired with a Portal within another Model. Exiting a Model via a Portal would also mean
# entering the Model associated with the counter-Portal. This could be especially useful if the
# Portals were spatiotemporally separated by a mascroscopically modelled region, preferably with
# a minimum delay >= the communication time between the computer systems running the Models.
#

class Shape:
    def __init__(self):
        return

    def draw(self, colour, view):
        print("Shape::draw")
        return

# e.g. Repulsive: tree/fountain/sculpture, pool, pillar
# e.g. Attractive: tree/fountain/sculpture, pool, public transport stops, seats
# e.g. Obscuring: tree/fountain/sculpture, pillar
class Circle(Shape):
    def __init__(self, posn, r):
        super(Circle, self).__init__()

        self.posn   = posn
        self.radius = r
        
        return

    def draw(self, colour, view):
        print("Circle::draw", self.posn, self.radius)
        view.fill_circle(colour, self.posn, self.radius)
        return

# e.g. Repulsive: wall, railing, road, train/tram tracks
# e.g. Attractive: pool, shop window, flower beds, sports TV screen, bar
# e.g. Obscuring: building, wall, fog/mist/haze
class Segment(Shape):
    def __init__(self, posn1, posn2, radius):
        super(Segment, self).__init__()

        self.posn1  = posn1
        self.posn2  = posn2
        self.radius = radius

        return

    def draw(self, colour, view):
        print("Segment::draw", self.posn1, self.posn2, self.radius)
        view.draw_line(colour, self.posn1, self.posn2, self.radius)
        return

#
# Repulsors repel people away from them.
#
class Repulsor:
    def __init__(self, id, A, D, shape):
        self.id    = id
        self.A     = A
        self.D     = D
        self.shape = shape
        return

    def draw(self, view):
        print("Repulsor::draw", self.id)
        self.shape.draw(white, view)
        return
#
# Attractors attract people (esp. pedestrians) to them. The most attractive thing for
# most vehicles is likely to be an empty lane or a parking space, dependent on the driver.
# The most attractive thing for pedestrians is quite variable, e.g. shop windows, trees, fountains,
# statuary, pools.
#
class Attractor:
    def __init__(self, id, A, D, T, shape):
        self.id    = id
        self.A     = A
        self.D     = D
        self.T     = T
        self.shape = shape
        return

    def draw(self, view):
        self.shape.draw(black_and_gold, view)
        return

#
# Obscurers prevent visual perception of other pedestrians, vehicles, etc.
# This will impair the effect of Attractors and Repulsors. If an object is to
# serve multiple-duty, the code would be simplest if the radius of the Obstruction
# were smaller than that of the coincident Attractor/Repulsor.
#
# @todo: Obscurers should also introduce uncertainty, esp. in pedestrian v vehicle scenarios.
# @todo: A more advanced implementation may require some vehicles (esp. trucks) to obscure
#     visibility too. If these vehicles are non-stationary, then efficient implementation may
#     be significantly more difficult.
#
class Obscurer:
    def __init__(self, id, shape):
        self.id    = id
        self.shape = shape
        return

