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

width  = 637.5
height = 637.5

scale_factor   = 1
screen_width   = scale_factor * width
screen_height  = scale_factor * height
screen_size    = (screen_width, screen_height)

layer_flags = SVD.LayerFlagset.GEOMETRY | SVD.LayerFlagset.TOPOLOGY | SVD.LayerFlagset.MACROSCOPIC | SVD.LayerFlagset.MICROSCOPIC
view = SVV.View(width, height, screen_size, scale_factor, layer_flags)


def generate_splines(agent_posn, e_dirn, tcone):
    d_third_L = tcone.l_left / 3
    d_third_R = tcone.l_right / 3

    pt0 = agent_posn

    pt1_L = agent_posn + d_third_L * e_dirn
    pt1_R = agent_posn + d_third_R * e_dirn

    pt3_L = agent_posn + tcone.l_left * tcone.e_left
    pt3_R = agent_posn + tcone.l_right * tcone.e_right 

    pt2_L = pt3_L - d_third_L * tcone.e_left
    pt2_R = pt3_R - d_third_R * tcone.e_right

    spline_L = [pt0, pt1_L, pt2_L, pt3_L]
    spline_R = [pt0, pt1_R, pt2_R, pt3_R]
    
    return [spline_L, spline_R]


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
    base_agent_posn = centre_SW

    agent_posn = base_agent_posn
    dirn = np.array([1, 0])
    for i in range(8):
        phi = i * np.pi / 4
        C = np.cos(phi)
        S = np.sin(phi)
        shadow_rect = SMGS.Rectangle(agent_posn, dirn, l, w, 3)
        agent_posn = centre_true + SMND.rotate_csACW(base_agent_posn - centre_true, C, S)
        for j in range(144):
            wait_time    = 50
            
            view.draw_viewport_rect(SVD.grey)

            theta = j * np.pi / 36
            dirn = np.array([np.cos(theta), np.sin(theta)])
            # dirn_perp = np.array([-dirn[1], dirn[0]])

            rect_agent = SMGS.Rectangle(agent_posn, dirn, l, w, 3)
            l_vec = l * dirn
            w_vec = w * SMND.rotate_90ACW(dirn)
            tri_pts = [agent_posn + w_vec, agent_posn - w_vec, agent_posn + l_vec]
            tri_agent = SMGS.Polygon(tri_pts, 1)

            rect_VO.draw(SVD.blue, view)
            if shadow_rect != None:
                #
                # Since our agent may appear to have teleported, I added
                # some markup to make the agent's transition easier
                # to follow.
                #
                shadow_rect.draw(SVD.dark_grey, view)
                bamf_segment = SMGS.Segment(shadow_rect.posn, agent_posn, 1)
                bamf_segment.draw(SVD.dark_grey, view)
                shadow_rect  = None
                wait_time    = 200

            rect_agent.draw(SVD.yellow, view)
            tri_agent.draw(SVD.blue, view)

            mds = SMGS.construct_minkowski_difference_rect_rect(rect_agent, rect_VO)
            min_relposn, s, nv, tv = mds.get_nearest_point_info(np.zeros(2))
            mds.translate(rect_agent.posn)

            # spd_max    = 25 / 3 # 30km/h
            spd_max    = 25 / 4.5 # 20km/h
            spd_pref   = 25 / 6 # 15km/h
            T_deadline   = s / spd_pref * np.dot(-nv, rect_agent.e_major) # 4
            # T_deadline   = cone_length[K] / spd_pref


            md_clr = SVD.white
            cosine = np.dot(rect_agent.e_major, rect_VO.e_major)
            sine   = np.dot(rect_agent.e_major, rect_VO.e_minor)
            if cosine**2 < SMND.EPS_m_sqr:
                md_clr = SVD.red
                wait_time    = 200
            elif sine**2 < SMND.EPS_m_sqr:
                md_clr = SVD.green
                wait_time    = 200
            mds.draw(md_clr, view)

            kyklos = SMGS.Circle(rect_agent.posn, SMC.d_safe_IA_maximum)
            tcone = SMGS.construct_tcone(kyklos, mds, anticipation, True)
            tcone.draw_parts(np.zeros(2), SVD.yellow, SVD.blue, view)

            cone_leftdirn_dual = SMND.rotate_90ACW(tcone.e_left)
            cone_rightdirn_dual = SMND.rotate_90ACW(tcone.e_right)
            cone_length = [tcone.l_left, tcone.l_right]
            if np.dot(dirn, cone_leftdirn_dual) < 0 and np.dot(dirn, cone_rightdirn_dual) > 0:
                tri_agent.draw(SVD.red, view)
                splines = generate_splines(agent_posn, dirn, tcone)
                spline_colours = [SVD.blue, SVD.green]

                # view.draw_thin_line(SVD.blue, pt1, pt2)
                for K in range(len(splines)):
                    spline = splines[K]

                    omega_max = np.pi / 6
                    E, L, T = SMND.analyse_cubic_spline(spline, omega_max, spd_max, T_deadline)
                    print("If {} <= {}, then the spline should be black?!".format(T, T_deadline))

                    clr = None
                    if T <= T_deadline:
                        clr = SVD.black
                    else:
                        clr = SVD.red

                    # for k in range(len(spline) - 1):
                        # view.draw_thin_line(SVD.black, spline[k], spline[k + 1])
                    # view.draw_cubic_spline(SVD.black, spline, SMND.EPS_m_sqr, 2)
                    view.draw_cubic_spline(clr, spline, 0.5, 2)

                
                wait_time    = 400
            
            view.flip()
            SCC.wait(wait_time)
