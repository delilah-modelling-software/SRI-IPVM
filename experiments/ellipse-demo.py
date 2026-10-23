import sys
# import shared_spaces_types as sst
import numpy as np
import scipy as sp
import scipy.stats as ss
import pygame as pg

# import shared_spaces_types as sst

width  = 600
height = 600

scale_factor   = 1
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width, screen_height)

centre_true = np.array([0.5 * width, 0.5 * height])

cosines = []
cosines.append(1.)
cosines.append(0.25 * np.sqrt(2.) * (np.sqrt(3) + 1))
cosines.append(0.5 * np.sqrt(3.))
cosines.append(0.5 * np.sqrt(2.))
cosines.append(0.5)
cosines.append(0.25 * np.sqrt(2.) * (np.sqrt(3) - 1))
cosines.append(0.)

for i in range(6):
    assert(cosines[i] > cosines[i + 1])

if  __name__ == "__main__":
    display = pg.display
    display.set_mode(screen_size)

    a = 50 * 4.8
    b = 50 * 1.8

    for i in range(720):
        pg.draw.rect(display.get_surface(), (0, 0, 0), pg.Rect(np.zeros(2), np.array([width, height])))

        theta = i * np.pi / 36
        u = np.array([np.cos(theta), np.sin(theta)])
        v = np.array([-u[1], u[0]])

        #
        # @todo: Remove after fixing immediate problem
        #
        # centre_true = np.zeros(2)

        pts = []
        au = a * u
        bv = b * v
        for j in range(6):
            pts.append(centre_true + cosines[j]     * au + cosines[6 - j] * bv)
        for j in range(6):
            pts.append(centre_true - cosines[6 - j] * au + cosines[j]     * bv)
        for j in range(6):
            pts.append(centre_true - cosines[j]     * au - cosines[6 - j] * bv)
        for j in range(6):
            pts.append(centre_true + cosines[6 - j] * au - cosines[j]     * bv)

        pg.draw.polygon(display.get_surface(), (128, 128, 128), pts, 0)
        pg.draw.polygon(display.get_surface(), (255,   0,   0), pts, 4)
        pg.draw.line(display.get_surface(), (255, 255, 255), centre_true, pts[0], 8)
        pg.draw.line(display.get_surface(), (  0,   0, 255), pts[0], pts[12], 4)
        pg.draw.line(display.get_surface(), (  0,   0, 255), pts[6], pts[18], 2)

        # pg.draw.line(pg.display.get_surface(), colour, self.scale_factor * inverted_posn1, self.scale_factor * inverted_posn2, radius)

        pg.display.flip()
        pg.time.wait(200)
