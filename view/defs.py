import numpy as np
import pygame as pg
import pygame.freetype as pgft

import model.numerical.defs as smnd

def colour(r, g, b, a = 255):
    return pg.Color(r, g, b, a)

# def delay(n_ms):
    # pg.time.delay(n_ms)
    # return

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

#
# @todo:
#   ALL          = -1
#   NONE         =  0
#
#   # Geometry
#   CENTROID      = 1
#   WAYPOINT      = 2
#   ZONE          = 4
#
#   # Topology
#   NODE       =     8
#   CONNECTION =  0x10
#
#   # Microscopic::SFM
#   ATTRACTION =  0x20
#   REPULSION  =  0x40
#
#   # Microscopic::GHRVO
#   CONES      =  0x80
#   POLYGON    = 0x100
#
#   # Mesoscopic
#   TRAJECTORY = 0x200
#   DETOUR     = 0x400
#
#   GEOMETRY_MASK    =     7
#   TOPOLOGY_MASK    =  0x18
#   SFM_MASK         =  0x60
#   GHRVO_MASK       = 0x180
#   MICROSCOPIC_MASK = 0x1e0
#   MESOSCOPIC_MASK  = 0x600
#
from enum import Enum, IntEnum
class LayerFlagset(IntEnum):
    ALL         =    -1
    NONE        =     0
    GEOMETRY    =     1 # for zone in model: zone.draw(); for node in model: node.draw()
    TOPOLOGY    =     2 # model.draw_topology()
    MACROSCOPIC =     4 # model.grid.draw_background()
    MICROSCOPIC =     8 # model.grid.draw_foreground(); for vhcl in model: vhcl.draw()
    ATTRACTION  =  0x10 # for agent in model: model.draw_nodal_attraction(agent, view)
    INTERACTION =  0x20 # for IAI in model.interagent_interactions: IAI.draw()
                        # for CZI in model.contrazone_interactions: CZI_draw()
                        # Interaction subtypes:
    INTRAMODAL  =  0x40 #   intramodal interagent_interactions IAI
    INTERMODAL  =  0x80 #   intermodal interagent_interactions IAI
    CONTRAZONE  = 0x100 #   contrazone_interactions CZI
    CONES       = 0x200 # conflict cones
    TRAJECTORY  = 0x400 # for agent in model: agent.draw_conflict_trajectory()


#
# Useful Colours
#

#
# Monochrome system
#
white          = pg.Color(255, 255, 255)
black          = pg.Color(0, 0, 0)

#
# RGB basis
#
dark_red       = pg.Color(128, 0, 0)
dark_green     = pg.Color(0, 128, 0)
dark_blue      = pg.Color(0, 0, 128)
red            = pg.Color(255, 0, 0)
green          = pg.Color(0, 255, 0)
blue           = pg.Color(0, 0, 255)
light_red      = pg.Color(255, 128, 128)
light_green    = pg.Color(128, 255, 128)
light_blue     = pg.Color(128, 128, 255)

#
# CMY basis, CMYK system needs K = black too
#
dark_cyan      = pg.Color(0, 128, 128)
dark_magenta   = pg.Color(128, 0, 128)
dark_yellow    = pg.Color(128, 128, 0)
cyan           = pg.Color(0, 255, 255)
magenta        = pg.Color(255, 0, 255)
yellow         = pg.Color(255, 255, 0)
light_cyan     = pg.Color(128, 255, 255)
light_magenta  = pg.Color(255, 128, 255)
light_yellow   = pg.Color(255, 255, 128)

#
# Useful greyscale options
#
light_grey     = pg.Color(192, 192, 192)
grey           = pg.Color(128, 128, 128)
dark_grey      = pg.Color(64, 64, 64)
darker_grey    = pg.Color(32, 32, 32)

#
# Useful chromatic options
#
silver         = grey.lerp(white, 0.5)
gold           = yellow.lerp(grey, 0.25)
black_and_gold = black.lerp(gold, 0.5)
orange         = pg.Color(128, 255, 0)
