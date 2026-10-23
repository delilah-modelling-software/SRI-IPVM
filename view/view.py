import os

import numpy as np
import pygame as pg
import pygame.freetype as pgft

import screeninfo as si

import model.numerical.defs as smnd
import view.defs as svd

class View:
    def __init__(self, width, height, screen_size, scale_factor, layer_flags = -1, background_colour = svd.colour(64, 64, 64), pedestrian_upscale_factor = 1.):
        self.width                     = width
        self.height                    = height
        self.screen_size               = screen_size
        self.scale_factor              = scale_factor
        self.background_colour         = background_colour
        self.layer_flags               = layer_flags
        self.pedestrian_upscale_factor = pedestrian_upscale_factor

        n_monitors = 0
        min_screen_width = 1 << 16
        for monitor in si.get_monitors():
            min_screen_width = min(min_screen_width, monitor.width)
            n_monitors += 1
        if n_monitors > 1:
            x = min_screen_width
            y = 0
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"

        pg.init()
        self.display = pg.display
        self.display.set_mode(self.screen_size)

        # self.font = pgft.SysFont("times", 12)
        self.largefontsize = 5 * np.ceil(scale_factor / 6)
        self.largefont = pgft.SysFont("times", self.largefontsize)

        self.smallfontsize = 3 * np.ceil(scale_factor / 6)
        self.smallfont = pgft.SysFont("times", self.smallfontsize)
        
        return
    
    def flip(self):
        self.display.flip()
        # self.display.update()
        return
        
    def get_viewport_rect(self):
        return pg.Rect((0., 0.), (self.scale_factor * self.width, self.scale_factor * self.height))

    def fill_circle(self, colour, posn, radius):
        inverted_posn = np.array([posn[0], self.height - posn[1]])
        pg.draw.circle(pg.display.get_surface(), colour, self.scale_factor * inverted_posn, self.scale_factor * radius)
        return

    def draw_text(self, text, colour, posn, large):
        inverted_posn = np.array([posn[0], self.height - posn[1]])
        # inverted_posn = (posn[0], self.height - posn[1])
        font = self.smallfont
        if large:
            font = self.largefont

        font.render_to(pg.display.get_surface(), self.scale_factor * inverted_posn, text, colour)
        return

    def stroke_circle(self, colour, posn, radius, stroke_width = 1):
        inverted_posn = np.array([posn[0], self.height - posn[1]])
        pg.draw.circle(pg.display.get_surface(), colour, self.scale_factor * inverted_posn, self.scale_factor * radius, stroke_width)
        return

    def draw_line(self, colour, posn1, posn2, radius):
        assert(posn1[0] != None)
        assert(posn1[1] != None)
        assert(posn2[0] != None)
        assert(posn2[1] != None)
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

    def stroke_polygon(self, colour, pts, radius):
        scaled_radius = int(np.maximum(2 * self.scale_factor * radius, 1))

        tmp_pts = []
        for posn in pts:
            tmp_pts.append(self.scale_factor * np.array([posn[0], self.height - posn[1]]))

        pg.draw.polygon(pg.display.get_surface(), colour, tmp_pts, scaled_radius)
        return

    def stroke_polysegment(self, colour, pts, radius):
        scaled_radius = int(np.maximum(2 * self.scale_factor * radius, 1))

        tmp_pts = []
        for posn in pts:
            tmp_pts.append(self.scale_factor * np.array([posn[0], self.height - posn[1]]))

        pg.draw.lines(pg.display.get_surface(), colour, False, tmp_pts, scaled_radius)
        return

    def fill_polygon(self, colour, pts):
        tmp_pts = []
        for posn in pts:
            tmp_pts.append(self.scale_factor * np.array([posn[0], self.height - posn[1]]))

        pg.draw.polygon(pg.display.get_surface(), colour, tmp_pts)
        return

    def fill_ellipse(self, colour, centre, major_axis, minor_axis):
        tmp_pts = svd.generate_elliptical_polygon(centre, major_axis, minor_axis)
        self.fill_polygon(colour, tmp_pts)
        return
    
    def draw_cubic_spline(self, colour, pts, epsilon_sqr, radius):
        x0 = pts[0]
        x1 = pts[1]
        x2 = pts[2]
        x3 = pts[3]

        dx_chord = x3 - x0
        u1 = smnd.calc_unit_vector(dx_chord)
        u2 = smnd.rotate_90ACW(u1)

        z1 = np.dot(x1 - x0, u2)
        z2 = np.dot(x2 - x0, u2)

        #
        # Are you sure you want to do this recursively?!
        #

        if z1**2 < epsilon_sqr and z2**2 < epsilon_sqr:
            scaled_radius = int(np.maximum(2 * self.scale_factor * radius, 1))
            self.draw_line(colour, x0, x3, radius)
            return

        y, z = smnd.subdivide_cubic_spline(pts, 0.5)
        # print("View::draw_cubic_spline has subdivided {} into {} and {}".format(pts, y, z))
        self.draw_cubic_spline(colour, y, epsilon_sqr, radius)
        self.draw_cubic_spline(colour, z, epsilon_sqr, radius)
        return

        #
        # @todo: Why 2 * scl_fctr * r rather than scl_fctr * r?!
        #
        #
        x0 = pts[0]
        x1 = pts[1]
        x2 = pts[2]
        x3 = pts[3]
        self.draw_line(colour, x0, x1, radius)
        self.draw_line(colour, x1, x2, radius)
        self.draw_line(colour, x2, x3, radius)

        return

    def draw_viewport_rect(self, colour):
        # print("Drawing viewport rectangle")
        self.draw_rect(colour, np.array([0, self.height]), np.array([self.width, 0]))
        # self.draw_rect(colour, np.zeros(2), np.array([self.width, self.height]))
        return
