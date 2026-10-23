# import sys
import numpy as np
# import scipy as sp
# import scipy.stats as ss
# import pygame as pg

import model.geometry.shape as SMGS
import model.defs as SMD
import model.configuration as SMC
import view.defs as SVD
import view.view as SVV
import controller.controller as SCC
import model.numerical.defs as SMND

width  = 180 # 637.5
height = 180 # 637.5

scale_factor   = 4
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

centre_E = 0.5 * (corner_SE + corner_NE)
centre_N = 0.5 * (corner_NE + corner_NW)
centre_W = 0.5 * (corner_NW + corner_SW)
centre_S = 0.5 * (corner_SW + corner_SE)

centre_NE = 0.5 * (centre_true + corner_NE)
centre_NW = 0.5 * (centre_true + corner_NW)
centre_SW = 0.5 * (centre_true + corner_SW)
centre_SE = 0.5 * (centre_true + corner_SE)



l = 60
w = 24
rect_VO = SMGS.Rectangle(centre_true, e1, l, w, 3)

u1 = l * rect_VO.e_major
u2 = w * rect_VO.e_minor


# rect_SE_pts = [centre_SE + l_vec + w_vec, centre_SE - l_vec + w_vec, centre_SE - l_vec - w_vec, centre_SE + l_vec - w_vec]
# rect_SE = SMGS.Polygon(rect_SE_pts, 6)

anticipation = 1
wait_time    = 600
spacestep    = 0.25
half_n_steps = 900
n_steps      = 2 * half_n_steps


def find_first_spline_intersection(spline1, spline2):
    dx1    = spline1[3] - spline1[0]
    L1, e1 = SMND.calc_polar_decomp(dx1)

    dx2    = spline2[3] - spline2[0]
    L2, e2 = SMND.calc_polar_decomp(dx2)

    det  = SMND.calc_det2x2(e1, e2)
    if det == 0:
        return None

    rcp_det = 1. / det

    det1 = SMND.calc_det2x2(spline2[0] - spline1[0], e2)
    det2 = SMND.calc_det2x2(e1, spline2[0] - spline1[0])

    lambda1 = det1 * rcp_det
    lambda2 = -det2 * rcp_det

    if lambda1 < 0. - SMND.EPS_m_sqr or lambda1 > L1 + SMND.EPS_m_sqr:
        return None

    if lambda2 < 0. - SMND.EPS_m_sqr or lambda2 > L2 + SMND.EPS_m_sqr:
        return None

    #
    #
    #
    # print("Spline {} and spline {} should have an intersection".format(spline1, spline2))

    e1_perp = SMND.rotate_90ACW(e1)
    e2_perp = SMND.rotate_90ACW(e2)

    dev1_sqr = max((np.dot(spline1[1] - spline1[0], e1_perp))**2, (np.dot(spline1[2] - spline1[0], e1_perp))**2)
    dev2_sqr = max((np.dot(spline2[1] - spline2[0], e2_perp))**2, (np.dot(spline2[2] - spline2[0], e2_perp))**2)

    if max(dev1_sqr, dev2_sqr) < 0.25:
        x1 = spline1[0] + lambda1 * e1
        x2 = spline2[0] + lambda2 * e2
        # print("Spline {} and spline {} have an intersection at 0.5 * ({} + {})".format(spline1, spline2, x1, x2))
        s1 = lambda1 / L1
        s2 = lambda2 / L2

        return 0.5 * (x1 + x2), s1, s2, e1, e2
    elif dev1_sqr <= dev2_sqr:
        #
        # Split more curvaceous spline
        #
        s2A, s2B = SMND.subdivide_cubic_spline(spline2, 0.5)

        #
        # Look for solution in the low half of the parameter space.
        #
        res = find_first_spline_intersection(spline1, s2A)
        if not res is None:
            x, s1, s2, e1, e2 = res
            return x, s1, 0.5 * s2, e1, e2

        #
        # Look for solution in the high half of the parameter space.
        #
        res = find_first_spline_intersection(spline1, s2B)
        if not res is None:
            x, s1, s2, e1, e2 = res
            return x, s1, 0.5 * (1 + s2), e1, e2
        #
        # If neither half contained an intersection, there isn't one.
        #
        return None

    else:
        #
        # Split more curvaceous spline
        #
        s1A, s1B = SMND.subdivide_cubic_spline(spline1, 0.5)

        #
        # Look for solution in the low half of the parameter space.
        #
        res = find_first_spline_intersection(s1A, spline2)
        if not res is None:
            x, s1, s2, e1, e2 = res
            return x, 0.5 * s1, s2, e1, e2

        #
        # Look for solution in the high half of the parameter space.
        #
        res = find_first_spline_intersection(s1B, spline2)
        if not res is None:
            x, s1, s2, e1, e2 = res
            return x, 0.5 * (1 + s1), s2, e1, e2


def draw_extended_spline(spline, clr, radius, view):
    dx0 = spline[1] - spline[0]
    lA, eA = SMND.calc_polar_decomp(dx0)
    eA_perp = SMND.rotate_90ACW(eA)

    dx2 = spline[3] - spline[2]
    lB, eB = SMND.calc_polar_decomp(dx2)
    eB_perp = SMND.rotate_90ACW(eB)

    view.draw_cubic_spline(clr, spline, 0.5, radius)
    view.draw_line(clr, spline[0] - 3 * eA_perp, spline[0] + 3 * eA_perp, radius)
    view.draw_line(clr, spline[3], spline[3] - 3 * eB_perp - 3 * eB, radius)
    view.draw_line(clr, spline[3], spline[3] + 3 * eB_perp - 3 * eB, radius)

    return


if __name__ == "__main__":
    centres = [centre_N, centre_NW, centre_W, centre_SW, centre_S, centre_SE, centre_E, centre_NE]
    for idx_i in range(8):
        src1 = centres[idx_i]
        for idx_j in range(1, 8):
            dst1 = centres[(idx_i + idx_j) % 8]
            if (dst1 == src1).all():
                assert(False)
                continue

            cp11 = src1 + (2/3) * (centre_true - src1)
            cp12 = dst1 + (2/3) * (centre_true - dst1)
            spline1 = [src1, cp11, cp12, dst1]

            for idx_k in range(8):
                src2 = centres[idx_k]

                # view.draw_viewport_rect(SVD.grey)
                # view.draw_cubic_spline(SVD.blue, spline1, 0.5, 1)
                # draw_extended_spline(spline1, SVD.blue, 1, view)
                # view.flip()
                # SCC.wait(wait_time)

                for idx_l in range(1, 8):
                    dst2 = centres[(idx_k + idx_l) % 8]
                    if (dst2 == src2).all():
                        assert(False)
                        continue

                    cp21 = src2 + (2/3) * (centre_true - src2)
                    cp22 = dst2 + (2/3) * (centre_true - dst2)
                    spline2 = [src2, cp21, cp22, dst2]

                    dx1    = dst1 - src1
                    L1, e1 = SMND.calc_polar_decomp(dx1)
                    dx2    = dst2 - src2
                    L2, e2 = SMND.calc_polar_decomp(dx2)

                    e1_perp = SMND.rotate_90ACW(e1)
                    e2_perp = SMND.rotate_90ACW(e2)

                    det  = SMND.calc_det2x2(e1, -e2)
                    if det != 0:
                        view.draw_viewport_rect(SVD.grey)

                        dsrc = src2 - src1
                        det1 = SMND.calc_det2x2(dsrc, e2)
                        det2 = SMND.calc_det2x2(e1, dsrc)
                        rcp_det = 1. / det

                        lambda1 =  det1 * rcp_det
                        lambda2 = -det2 * rcp_det

                        epsilon = 0.125

                        draw_extended_spline(spline1, SVD.blue,  2, view)
                        draw_extended_spline(spline2, SVD.yellow, 1, view)

                        x1 = src1 + lambda1 * e1
                        x2 = src2 + lambda2 * e2

                        circle1 = SMGS.Circle(x1, 2)
                        circle2 = SMGS.Circle(x2, 1)
                        circle1.draw(SVD.red, view)
                        circle2.draw(SVD.black, view)

                        res = find_first_spline_intersection(spline1, spline2)
                        if not res is None:
                            x_soln, sA, sB, eA, eB = res
                            circle_soln = SMGS.Circle(x_soln, 2)
                            circle_soln.draw(SVD.red, view)

                            spline1A, spline1B = SMND.subdivide_cubic_spline(spline1, sA)
                            view.draw_cubic_spline(SVD.magenta, spline1A, 0.5, 2)
                            view.draw_cubic_spline(SVD.cyan, spline1B, 0.5, 2)

                            spline2A, spline2B = SMND.subdivide_cubic_spline(spline2, sB)
                            view.draw_cubic_spline(SVD.red, spline2A, 0.5, 1)
                            view.draw_cubic_spline(SVD.green, spline2B, 0.5, 1)

                            l = 2.2
                            w = 0.9
                            x_stop1 = x_soln - (l + w + SMC.d_safe_IA_maximum) * eA
                            dx_stop1 = (w + SMC.d_safe_IA_maximum) * eB
                            stop1 = SMGS.Segment(x_stop1 - dx_stop1, x_stop1 + dx_stop1, 1)
                            stop1.draw(SVD.yellow, view)

                            x_stop2 = x_soln - (l + w + SMC.d_safe_IA_maximum) * eB
                            dx_stop2 = (w + SMC.d_safe_IA_maximum) * eA
                            stop2 = SMGS.Segment(x_stop2 - dx_stop2, x_stop2 + dx_stop2, 1)
                            stop2.draw(SVD.blue, view)
                    else:
                        # continue
                        view.draw_viewport_rect(SVD.black)
                        draw_extended_spline(spline1, SVD.blue, 1, view)
                        draw_extended_spline(spline2, SVD.yellow, 0.5, view)

                    view.flip()
                    SCC.wait(wait_time)
