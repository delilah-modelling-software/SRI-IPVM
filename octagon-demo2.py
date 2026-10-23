# import sys
import numpy as np
# import scipy as sp
# import scipy.stats as ss
# import pygame as pg

import model.geometry.shape as SMGS
import model.defs as SMD
import view.defs as SVD
import view.view as SVV
import controller.controller as SCC
import model.numerical.defs as SMND

width  = 600
height = 600

scale_factor   = 1
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width, screen_height)

layer_flags = SVD.LayerFlagset.GEOMETRY | SVD.LayerFlagset.TOPOLOGY | SVD.LayerFlagset.MACROSCOPIC | SVD.LayerFlagset.MICROSCOPIC
view = SVV.View(width, height, screen_size, scale_factor, layer_flags)

centre_true = np.array([0.5 * width, 0.5 * height])
e1 = np.array([1, 0])
e2 = np.array([0, 1])

corner_NW = np.array([0, height])
corner_SW = np.array([0, 0])
corner_SE = np.array([width, 0])
corner_NE = np.array([width, height])

centre_NW = 0.5 * (centre_true + corner_NW)
centre_SW = 0.5 * (centre_true + corner_SW)
centre_SE = 0.5 * (centre_true + corner_SE)
centre_NE = 0.5 * (centre_true + corner_NE)

centre_true = 0.5 * (centre_NE + centre_SW)

l = 60
w = 24
rect_VO = SMGS.Rectangle(centre_true, e1, l, w, 3)

u1 = l * rect_VO.e_major
u2 = w * rect_VO.e_minor


# rect_SE_pts = [centre_SE + l_vec + w_vec, centre_SE - l_vec + w_vec, centre_SE - l_vec - w_vec, centre_SE + l_vec - w_vec]
# rect_SE = SMGS.Polygon(rect_SE_pts, 6)

anticipation = 1
wait_time    = 400
spacestep    = 0.25
half_n_steps = 900
n_steps      = 2 * half_n_steps

if __name__ == "__main__":
    for j in range(720):
        view.draw_viewport_rect(SVD.grey)

        theta = j * np.pi / 36
        dirn = np.array([np.cos(theta), np.sin(theta)])
        # dirn_perp = np.array([-dirn[1], dirn[0]])

        rect_SW = SMGS.Rectangle(centre_SW, dirn, l, w, 3)

        rect_VO.draw(SVD.blue, view)
        rect_SW.draw(SVD.yellow, view)

        polycentre = rect_VO.posn # - rect_SW.posn
        c = np.dot(rect_SW.e_major, rect_VO.e_major)
        s = np.dot(rect_SW.e_major, rect_VO.e_minor)
        # print("Angle between SW-rect and NE-rect defined by cosine {} and sine{}".format(c, s))

        if c**2 < SMND.EPS_m_sqr:
            l_res = rect_VO.halflength + rect_SW.halfwidth
            w_res = rect_VO.halfwidth + rect_SW.halflength
            rect_relposn = SMGS.Rectangle(polycentre, rect_VO.e_major, l_res, w_res, 3)
            rect_relposn.draw(SVD.red, view)

            kyklos = SMGS.Circle(rect_SW.posn, SMD.d_safe_IA)

            cone = SMGS.construct_tcone(kyklos, rect_relposn, anticipation, True)

        elif s**2 < SMND.EPS_m_sqr:
            l_res = rect_VO.halflength + rect_SW.halflength
            w_res = rect_VO.halfwidth + rect_SW.halfwidth
            rect_relposn = SMGS.Rectangle(polycentre, rect_VO.e_major, l_res, w_res, 3)
            rect_relposn.draw(SVD.green, view)

            kyklos = SMGS.Circle(rect_SW.posn, SMD.d_safe_IA)

            cone = SMGS.construct_tcone(kyklos, rect_relposn, anticipation, True)
        else:
            pts = []
            if c > 0 and s >= 0:
                v1 = l * rect_SW.e_major
                v2 = w * rect_SW.e_minor
            elif c <= 0 and s > 0:
                v1 = -w * rect_SW.e_minor
                v2 = l * rect_SW.e_major
            elif c < 0 and s <= 0:
                v1 = -l * rect_SW.e_major
                v2 = -w * rect_SW.e_minor
            elif c >= 0 and s < 0:
                v1 = w * rect_SW.e_minor
                v2 = -l * rect_SW.e_major

            #
            #
            #
            centre = polycentre + u1 + u2
            pt1 = centre + v1 - v2
            pt2 = centre + v1 + v2
            pts.append(pt1)
            pts.append(pt2)

            centre = polycentre - u1 + u2
            pt1 = centre + v2 + v1
            pt2 = centre + v2 - v1
            pts.append(pt1)
            pts.append(pt2)

            centre = polycentre - u1 - u2
            pt1 = centre - v1 + v2
            pt2 = centre - v1 - v2
            pts.append(pt1)
            pts.append(pt2)

            centre = polycentre + u1 - u2
            pt1 = centre - v2 - v1
            pt2 = centre - v2 + v1
            pts.append(pt1)
            pts.append(pt2)

            print("Octagon? {}".format(pts))

            octagon = SMGS.Polygon(pts, rect_SW.radius + rect_VO.radius)
            octagon.draw(SVD.white, view)

            kyklos = SMGS.Circle(rect_SW.posn, SMD.d_safe_IA)

            cone = SMGS.construct_tcone(kyklos, octagon, anticipation, True)
            
        cone.draw_parts(SVD.yellow, SVD.blue, view)
        
        view.flip()
        SCC.wait(wait_time)
