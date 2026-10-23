# import sys
import numpy as np
# import scipy as sp
# import scipy.stats as ss
# import pygame as pg

import model.geometry.shape as SMGS
import view.defs as SVD
import view.view as SVV
import controller.controller as SCC

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

circle_NW  = SMGS.Circle(centre_NW, 30)

l = 30
w = 12

segment_SW = SMGS.Segment(centre_SW - l * e1, centre_SW + l * e1, 6)

# l_vec   = l * e1
# w_vec   = w * e2
rect_SE = SMGS.Rectangle(centre_SE, e1, l, w, 6)
# rect_SE_pts = [centre_SE + l_vec + w_vec, centre_SE - l_vec + w_vec, centre_SE - l_vec - w_vec, centre_SE + l_vec - w_vec]
# rect_SE = SMGS.Polygon(rect_SE_pts, 6)

anticipation = 1
wait_time    = 2
spacestep    = 0.25
half_n_steps = 900
n_steps      = 2 * half_n_steps

def generate_cones(circle, segment, rect, kyklos, anticipation):
    cone_C = SMGS.construct_tcone(kyklos, circle, anticipation, True)
    cone_S = SMGS.construct_tcone(kyklos, segment, anticipation, True)
    cone_R = SMGS.construct_tcone(kyklos, rect, anticipation, True)

    # cone_C = SMGS.TruncatedCone(circle, kyklos, anticipation, True)
    # cone_S = SMGS.TruncatedCone(segment, kyklos, anticipation, True)
    # cone_R = SMGS.TruncatedCone(rect, kyklos, anticipation, True)
    return cone_C, cone_S, cone_R

def draw_cones(cone_C, cone_S, cone_P):
    if cone_C != None:
        cone_C.draw_parts(SVD.cyan, SVD.red, view)

    if cone_S != None:
        cone_S.draw_parts(SVD.magenta, SVD.green, view)
        
    if cone_P != None:
        cone_P.draw_parts(SVD.yellow, SVD.blue, view)

    return


if __name__ == "__main__":
    posn = 0.5 * (centre_NE + corner_NE)
    kyklos = SMGS.Circle(posn, 15)
    for i in range(n_steps):
        view.draw_viewport_rect(SVD.grey)
        
        kyklos.posn -= spacestep * e1
        kyklos.draw(SVD.black, view)

        cone_C, cone_S, cone_P = generate_cones(circle_NW, segment_SW, rect_SE, kyklos, anticipation)
        draw_cones(cone_C, cone_S, cone_P)

        circle_NW.draw(SVD.white, view)
        segment_SW.draw(SVD.white, view)
        rect_SE.draw(SVD.white, view)

        view.flip()
        SCC.wait(wait_time)

    for i in range(half_n_steps):
        view.draw_viewport_rect(SVD.grey)
        
        kyklos.posn -= spacestep * e2
        kyklos.draw(SVD.black, view)

        cone_C, cone_S, cone_P = generate_cones(circle_NW, segment_SW, rect_SE, kyklos, anticipation)
        draw_cones(cone_C, cone_S, cone_P)

        circle_NW.draw(SVD.white, view)
        segment_SW.draw(SVD.white, view)
        rect_SE.draw(SVD.white, view)

        view.flip()
        SCC.wait(wait_time)

    for i in range(n_steps):
        view.draw_viewport_rect(SVD.grey)
        
        kyklos.posn += spacestep * e1
        kyklos.draw(SVD.black, view)

        cone_C, cone_S, cone_P = generate_cones(circle_NW, segment_SW, rect_SE, kyklos, anticipation)
        draw_cones(cone_C, cone_S, cone_P)

        circle_NW.draw(SVD.white, view)
        segment_SW.draw(SVD.white, view)
        rect_SE.draw(SVD.white, view)

        view.flip()
        SCC.wait(wait_time)

    for i in range(half_n_steps):
        view.draw_viewport_rect(SVD.grey)
        
        kyklos.posn -= spacestep * e2
        kyklos.draw(SVD.black, view)

        cone_C, cone_S, cone_P = generate_cones(circle_NW, segment_SW, rect_SE, kyklos, anticipation)
        draw_cones(cone_C, cone_S, cone_P)

        circle_NW.draw(SVD.white, view)
        segment_SW.draw(SVD.white, view)
        rect_SE.draw(SVD.white, view)

        view.flip()
        SCC.wait(wait_time)

    for i in range(n_steps):
        view.draw_viewport_rect(SVD.grey)
        
        kyklos.posn -= spacestep * e1
        kyklos.draw(SVD.black, view)

        cone_C, cone_S, cone_P = generate_cones(circle_NW, segment_SW, rect_SE, kyklos, anticipation)
        draw_cones(cone_C, cone_S, cone_P)

        circle_NW.draw(SVD.white, view)
        segment_SW.draw(SVD.white, view)
        rect_SE.draw(SVD.white, view)

        view.flip()
        SCC.wait(wait_time)

    for i in range(n_steps):
        view.draw_viewport_rect(SVD.grey)
        
        kyklos.posn += spacestep * e2
        kyklos.draw(SVD.black, view)

        cone_C, cone_S, cone_P = generate_cones(circle_NW, segment_SW, rect_SE, kyklos, anticipation)
        draw_cones(cone_C, cone_S, cone_P)

        circle_NW.draw(SVD.white, view)
        segment_SW.draw(SVD.white, view)
        rect_SE.draw(SVD.white, view)

        view.flip()
        SCC.wait(wait_time)

    for i in range(half_n_steps):
        view.draw_viewport_rect(SVD.grey)
        
        kyklos.posn += spacestep * e1
        kyklos.draw(SVD.black, view)

        cone_C, cone_S, cone_P = generate_cones(circle_NW, segment_SW, rect_SE, kyklos, anticipation)
        draw_cones(cone_C, cone_S, cone_P)

        circle_NW.draw(SVD.white, view)
        segment_SW.draw(SVD.white, view)
        rect_SE.draw(SVD.white, view)

        view.flip()
        SCC.wait(wait_time)

    for i in range(n_steps):
        view.draw_viewport_rect(SVD.grey)
        
        kyklos.posn -= spacestep * e2
        kyklos.draw(SVD.black, view)

        cone_C, cone_S, cone_P = generate_cones(circle_NW, segment_SW, rect_SE, kyklos, anticipation)
        draw_cones(cone_C, cone_S, cone_P)

        circle_NW.draw(SVD.white, view)
        segment_SW.draw(SVD.white, view)
        rect_SE.draw(SVD.white, view)

        view.flip()
        SCC.wait(wait_time)

    for i in range(half_n_steps):
        view.draw_viewport_rect(SVD.grey)
        
        kyklos.posn += spacestep * e1
        kyklos.draw(SVD.black, view)

        cone_C, cone_S, cone_P = generate_cones(circle_NW, segment_SW, rect_SE, kyklos, anticipation)
        draw_cones(cone_C, cone_S, cone_P)

        circle_NW.draw(SVD.white, view)
        segment_SW.draw(SVD.white, view)
        rect_SE.draw(SVD.white, view)

        view.flip()
        SCC.wait(wait_time)

    for i in range(n_steps):
        view.draw_viewport_rect(SVD.grey)
        
        kyklos.posn += spacestep * e2
        kyklos.draw(SVD.black, view)

        cone_C, cone_S, cone_P = generate_cones(circle_NW, segment_SW, rect_SE, kyklos, anticipation)
        draw_cones(cone_C, cone_S, cone_P)

        circle_NW.draw(SVD.white, view)
        segment_SW.draw(SVD.white, view)
        rect_SE.draw(SVD.white, view)

        view.flip()
        SCC.wait(wait_time)

