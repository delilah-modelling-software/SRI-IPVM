import numpy as np
import scipy as sp
import scipy.stats as ss
import pygame as pg

from collections import deque

def colour(r, g, b):
    return pg.Color(r, g, b)

def delay(n_ms):
    pg.time.delay(n_ms)
    return

cosines = []
cosines.append(1.)
cosines.append(0.25 * np.sqrt(2.) * (np.sqrt(3) + 1))
cosines.append(0.5 * np.sqrt(3.))
cosines.append(0.5 * np.sqrt(2.))
cosines.append(0.5)
cosines.append(0.25 * np.sqrt(2.) * (np.sqrt(3) - 1))
cosines.append(0.)

def generate_elliptical_polygon(centre, major_axis, minor_axis):
    au = major_axis
    bv = minor_axis

    pts = []
    for j in range(6):
        pts.append(centre + cosines[j]     * au + cosines[6 - j] * bv)
    for j in range(6):
        pts.append(centre - cosines[6 - j] * au + cosines[j]     * bv)
    for j in range(6):
        pts.append(centre - cosines[j]     * au - cosines[6 - j] * bv)
    for j in range(6):
        pts.append(centre + cosines[6 - j] * au - cosines[j]     * bv)

    return pts

class View:
    def __init__(self, width, height, screen_size, scale_factor, pedestrian_upscale_factor = 2.5):
        self.width                     = width
        self.height                    = height
        self.screen_size               = screen_size
        self.scale_factor              = scale_factor
        self.pedestrian_upscale_factor = pedestrian_upscale_factor

        pg.init()
        self.display = pg.display
        self.display.set_mode(self.screen_size)
        return
    
    def flip(self):
        self.display.flip()
        self.display.update()
        return
        
    def get_viewport_rect(self):
        return pg.Rect((0., 0.), (self.scale_factor * self.width, self.scale_factor * self.height))

    def fill_circle(self, colour, posn, radius):
        inverted_posn = np.array([posn[0], self.height - posn[1]])
        pg.draw.circle(pg.display.get_surface(), colour, self.scale_factor * inverted_posn, self.scale_factor * radius)
        return

    def stroke_circle(self, colour, posn, radius, stroke_width = 1):
        inverted_posn = np.array([posn[0], self.height - posn[1]])
        pg.draw.circle(pg.display.get_surface(), colour, self.scale_factor * inverted_posn, self.scale_factor * radius, stroke_width)
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

    def stroke_polygon(self, colour, pts, radius):
        scaled_radius = int(np.maximum(2 * self.scale_factor * radius, 1))

        tmp_pts = []
        for posn in pts:
            tmp_pts.append(self.scale_factor * np.array([posn[0], self.height - posn[1]]))

        pg.draw.polygon(pg.display.get_surface(), colour, tmp_pts, scaled_radius)
        return

    def fill_polygon(self, colour, pts):
        tmp_pts = []
        for posn in pts:
            tmp_pts.append(self.scale_factor * np.array([posn[0], self.height - posn[1]]))

        pg.draw.polygon(pg.display.get_surface(), colour, tmp_pts)
        return

    def fill_ellipse(self, colour, centre, major_axis, minor_axis):
        tmp_pts = generate_elliptical_polygon(centre, major_axis, minor_axis)
        self.fill_polygon(colour, tmp_pts)

    def draw_viewport_rect(self, colour):
        self.draw_rect(colour, np.zeros(2), np.array([self.width, self.height]))
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
darker_grey    = pg.Color(32, 32, 32)
gold           = grey.lerp(white, 0.5)
black_and_gold = black.lerp(gold, 0.5)
