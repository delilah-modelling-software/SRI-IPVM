#
# Shape class, two simple subclasses, Circle and Segment, and a composite subclass, Polygon, a closed shape comprising a set of small circles and intermediate segments
#
# @todo: Add a composite subclass, Polysegment, an open shape comprising a set of small circles and intermediate segments.
# @todo: Add an annular arc.
#

import copy as cp
import numpy as np
import view.view as svv
# import model.force as smf
import model.defs as smd
import model.numerical.defs as smnd
import model.geometry.grid as smgg

class Shape:
    def __init__(self):
        self.posn = None # np.array([-1, -1])
        return

    def contains(self, posn):
        assert(False)
        return False

    def fill(self, colour, view):
        smd.dbgprint("Shape::draw")
        return
    
    def get_nearest_point_info(self, posn):
        return None

    def translate(self, dx):
        assert(False)
        return

    def calc_bbox(self):
        #
        # Returns: [bottom_left_point, top_right_point]
        #
        return None

    def get_nearest_approach_info(self, posn, vel, t_max):
        #
        # Returns: [shape_posn, agent_posn, d, normal, tangent, t]
        #
        return None

    def construct_truncated_cone(self, posn, radius):
        return None


# e.g. Repulsive: tree/fountain/sculpture, pool, pillar
# e.g. Attractive: tree/fountain/sculpture, pool, public transport stops, seats
# e.g. Obscuring: tree/fountain/sculpture, pillar
class Circle(Shape):
    def __init__(self, posn, r):
        super(Circle, self).__init__()

        self.posn   = posn
        self.radius = r

        self.bounded = True
        self.convex  = True
        
        return

    def calc_bbox(self):
        #
        # Returns: [bottom_left_point, top_right_point]
        #
        delta = np.array([self.radius, self.radius])

        return smgg.BoundingBox(self.posn - delta, self.posn + delta)

        # return (self.posn - delta, self.posn + delta)

    def get_nearest_point_info(self, posn):
    # def get_nearest_point_info(self, posn, vel, t):
        x_omega = self.posn
        d, nv   = smnd.calc_polar_decomp(posn - x_omega)
        tv      = smnd.rotate_90ACW(nv)
        return (x_omega, d, nv, tv)

    def translate(self, dx):
        self.posn += dx
        return

    def translated(self, u):
        xlated = cp.deepcopy(self)
        xlated.posn += u
        return xlated

    def get_nearest_approach_info(self, posn, vel, t_max):
        x_omega = self.posn

        rel_agent_posn = posn - x_omega
        if np.dot(rel_agent_posn, vel) >= 0:
            d, nv   = smnd.calc_polar_decomp(rel_agent_posn)
            tv      = smnd.rotate_90ACW(nv)
            return (x_omega, posn, d, nv, tv, 0)

        #
        # Should not need to handle t <= 0 case hereafter.
        #
        t = np.dot(x_omega - posn, vel) / smnd.calc_norm_sqr(vel)
        smd.dbgassert(t > 0)  
        
        future_posn = posn + t * vel 
        d, nv   = smnd.calc_polar_decomp(future_posn - x_omega)
        tv      = smnd.rotate_90ACW(nv)
        
        return (x_omega, future_posn, d, nv, tv, t)

    def DEPRECATED_get_closest_point_info(self, posn, vel, T_max):
        rel_posn = posn - self.posn
        t        = T_max

        #
        # @todo: Consider loosening the temporal discretisation.
        #
        # spd = smf.calc_norm(vel)
        # v   = vel / spd
        # t   = -np.dot(rel_posn / spd, v)
        # if t < 0:
            # t = 0
        # elif t > T_max:
            # t = T_max
        # t = 
        
        # s_omega = posn - self.posn
        s_omega = rel_posn + t * vel

        nv = smf.calc_unit_vector(s_omega)
        tv = np.array([-nv[1], nv[0]])
        x_omega = self.posn

        return (x_omega, nv, tv, T_max)

    def on(self, posn):
        return np.dot(self.posn - posn, self.posn - posn) <= self.radius**2
   
    def contains(self, posn):
        return np.dot(self.posn - posn, self.posn - posn) <= self.radius**2
   
    def draw(self, colour, view):
        # smd.dbgprint("Circle::draw", self.posn, self.radius)
        view.fill_circle(colour, self.posn, self.radius)
        return

def calc_segment_approach(posn1, posn2, posn, vel, t_max):
        rel_posn1 = posn - posn1
        rel_posn2 = posn - posn2

        #
        # Handle the [quasi-]degenerate cases first:
        #     the discriminant will be zero iff the agent is travelling parallel
        #     to the surface of the segment.
        #
        length, unit_dp = smnd.calc_polar_decomp(posn2 - posn1)
        disc = smnd.calc_det2x2(unit_dp, -vel)
        if disc**2 < smnd.EPS_m_sqr:
            s = np.dot(rel_posn1, unit_dp)
            # s = np.dot(rel_posn, unit_dp) * unit_dp
            t = 0 
            return (s, t) # (x_omega, posn, d, nv, tv, 0)
            
        #
        # Cramer's Rule
        #
        s = smnd.calc_det2x2(rel_posn1, -vel)    / disc
        t = smnd.calc_det2x2(unit_dp, rel_posn1) / disc

        smd.dbgprint("smgs::calc_segment_approach: Post-Cramer, s = {}, t = {}".format(s, t))

        if s < 0:
            smd.dbgprint("smgs::calc_segment_approach: infra-s, rel_posn2 = {}, vel = {}".format(rel_posn1, vel))
            s = 0
            t = -np.dot(rel_posn1, unit_dp) / np.dot(vel, unit_dp)
        elif s > length:
            smd.dbgprint("smgs::calc_segment_approach: ultra-s, rel_posn2 = {}, vel = {}".format(rel_posn2, vel))
            s = length
            t = -np.dot(rel_posn2, unit_dp) / np.dot(vel, unit_dp)

        smd.dbgprint("smgs::calc_segment_approach: Post-s-clip, s = {}, t = {}".format(s, t))
        if t <= 0:
            t = 0
            s = np.dot(rel_posn1, unit_dp)
        elif t > t_max:
            t = t_max
            s = -np.dot(rel_posn1 + t * vel, unit_dp)

        smd.dbgprint("smgs::calc_segment_approach: Post-t-clip, s = {}, t = {}".format(s, t))

        return (s, t)

        # x_omega = self.posn1 + s * tv
        # posn    = posn + t * vel
        # d, nv   = smnd.calc_polar_decomp(rel_agent_posn)
        # tv      = smnd.rotate_90ACW(nv)
        # return (x_omega, posn, d, nv, tv, t)

class Halfplane(Shape):
    def __init__(self, posn, normal):
        super(Halfplane, self).__init__()

        self.posn   = posn
        self.normal = normal

    def contains(self, posn):
        return np.dot(posn - self.posn, self.normal) >= 0.

# e.g. Repulsive: wall, railing, road, train/tram tracks
# e.g. Attractive: pool, shop window, flower beds, sports TV screen, bar
# e.g. Obscuring: building, wall, fog/mist/haze
class Segment(Shape):
    def __init__(self, posn1, posn2, radius):
        super(Segment, self).__init__()

        self.bounded = True
        self.convex  = True

        self.posn1  = posn1
        self.posn2  = posn2
        self.posn = 0.5 * (posn1 + posn2)
        self.radius = radius

        delta = self.posn2 - self.posn1
        l, tv = smnd.calc_polar_decomp(delta)
        self.length = l
        self.tv = tv
        
        self.nv = np.array([-self.tv[1], self.tv[0]])

        self.posn = 0.5 * (self.posn1 + self.posn2)

        return

    def calc_bbox(self):
        #
        # Returns: [bottom_left_point, top_right_point]
        #
        blp = np.array(min(self.posn1[0], self.posn2[0]), min(self.posn1[1], self.posn2[1]))
        trp = np.array(max(self.posn1[0], self.posn2[0]), max(self.posn1[1], self.posn2[1]))
        delta = np.array([self.radius, self.radius])

        return smgg.BoundingBox(blp - delta, trp + delta)

        return (blp - delta, trp + delta)

    def get_nearest_point_info(self, posn):
        s = np.dot(posn - self.posn1, self.tv)
        # s = np.dot(posn + t * vel - self.posn1, self.tv) * self.tv
        if s < 0:
            s = 0
        elif s >= self.length:
            s = self.length
        
        x_omega = self.posn1 + s * self.tv
        d, nv   = smnd.calc_polar_decomp(posn - x_omega)
        tv      = smnd.rotate_90ACW(nv)
        return (x_omega, d, nv, tv)

    def translate(self, dx):
        self.posn1 += dx
        self.posn2 += dx
        self.posn += dx
        return

    def translated(self, u):
        xlated = cp.deepcopy(self)
        xlated.posn1 += dx
        xlated.posn2 += dx
        xlated.posn += dx
        return xlated

    def get_nearest_approach_info(self, posn, vel, t_max):
        s, t         = calc_segment_approach(self.posn1, self.posn2, posn, vel, t_max)
        x_omega      = self.posn1 + s * self.tv
        nearest_posn = posn + t * vel
        d, nv        = smnd.calc_polar_decomp(nearest_posn - x_omega)
        tv           = smnd.rotate_90ACW(nv)

        return (x_omega, nearest_posn, d, nv, tv, t)
            

        rel_agent_posn = posn - self.posn1

        #
        # Handle the [quasi-]degenerate cases first:
        #     the discriminant will be quasi-zero iff the agent is travelling parallel to the surface of the segment.
        #
        disc = smnd.calc_det2x2(self.tv, -vel)
        if disc**2 < smnd.EPS_m_sqr:
            return (x_omega, posn, d, nv, tv, 0)
            
        #
        # Cramer's Rule
        #
        s = smnd.calc_det2x2(rel_posn, -vel) /disc
        t = smnd.calc_det2x2(self.tv, rel_posn) / disc

        if s < 0:
            s = 0
            t = -np.dot(rel_agent_posn, vel) / smnd.calc_norm_sqr(vel)
        elif s > self.length:
            s = self.length
            t = -np.dot(posn - self.posn2, vel) / smnd.calc_norm_sqr(vel)

        if t <= 0:
            t = 0
            s = np.dot(rel_agent_posn, tv)
        elif t > t_max:
            t = t_max
            s = np.dot(rel_agent_posn + t_max, tv)

        x_omega = self.posn1 + s * tv
        posn    = posn + t * vel
        d, nv   = smnd.calc_polar_decomp(rel_agent_posn)
        tv      = smnd.rotate_90ACW(nv)

        return (x_omega, posn, d, nv, tv, t)
   
    def on(self, posn):
        x_omega = posn + np.dot(posn - self.posn1, self.tv) * self.tv
        return np.dot(x_omega - posn, x_omega - posn) <= self.radius**2

    def contains(self, posn):
        x_omega = posn + np.dot(posn - self.posn1, self.tv) * self.tv
        return np.dot(x_omega - posn, x_omega - posn) <= self.radius**2

    def draw(self, colour, view):
        # smd.dbgprint("Segment::draw", self.posn1, self.posn2, self.radius)
        view.draw_line(colour, self.posn1, self.posn2, self.radius)
        view.fill_circle(colour, self.posn1, self.radius)
        view.fill_circle(colour, self.posn2, self.radius)
        return

def get_nearest_point_info(pts, posn):
    # print("Shape - get_nearest_point_info - {}".format(self.pts))

    nearest_pts = []
    n = len(pts)
    for i in range(n):
        delta = posn - pts[i]
        item  = (delta, i, np.dot(delta, delta))
        nearest_pts.append(item)
    nearest_pts.sort(key = lambda item: item[2])
        
    idx      = nearest_pts[0][1]
    prev_idx = (idx + n - 1) % n
    next_idx = (idx + 1)     % n
    # print("Polygon::get_nearest_point_info - i - 1, i, i + 1 %{} {}, {}, {}".format(n, prev_idx, idx, next_idx))

    #
    # @todo: Determine which segment posn is closer to.
    #
    dp0 = pts[idx]      - pts[prev_idx]
    dp1 = pts[next_idx] - pts[idx]
    # print("Polygon::get_nearest_point_info - {} = {} - {}, {} = {} - {}".format(dp0, self.pts[idx], self.pts[prev_idx], dp1, self.pts[next_idx], self.pts[idx]))

    l0, e0 = smnd.calc_polar_decomp(dp0)
    l1, e1 = smnd.calc_polar_decomp(dp1)
    # print("Polygon::get_nearest_point_info - {}*{} = {}, {}*{} = {}".format(l0, e0, dp0, l1, e1, dp1))

    #
    # @todo: Not all of these are always necessary. Consider lazier evaluation or keeping them during the ?
    #
    adv_posn = posn
    # adv_posn = posn + t * vel
    u0 = adv_posn - pts[prev_idx]
    u1 = adv_posn - pts[idx]
    u2 = adv_posn - pts[next_idx]
    # print("Polygon::get_nearest_point_info - u_i {}, {}, {}".format(u0, u1, u2))

    s0 = np.dot(u0, e0) #  / l0
    s1 = np.dot(u1, e1) # / l1

    # print("Polygon::get_nearest_point_info - {} = {}^T{}, {} = {}^T{}, l0 = {}".format(s0, u0, e0, s1, u1, e1, l0))
    if s0 >= l0 and s1 <= 0.:
        #
        # Closer to self.pts[idx] than either segment.
        #
        x_omega = pts[idx]
        d, nv = smnd.calc_polar_decomp(u1)
        tv = smnd.rotate_90ACW(nv)
        return (x_omega, d, nv, tv)

    elif s0 <= l0:
        #
        # Closest to segment between self.pts[prev_idx], self.pts[idx]
        #
        smd.dbgassert(s0 >= 0.5 * l0)
            
        x_omega = pts[prev_idx] + s0 * e0
        d, nv   = smnd.calc_polar_decomp(adv_posn - x_omega)
        tv      = smnd.rotate_90ACW(nv)
        return (x_omega, d, nv, tv)

    elif s1 >= 0.:
        #
        # Closest to segment between self.pts[idx], self.pts[next_idx]
        #
        smd.dbgassert(s1 <= 0.5 * l1)
            
        x_omega = pts[idx] + s1 * e1
        d, nv   = smnd.calc_polar_decomp(adv_posn - x_omega)
        tv      = smnd.rotate_90ACW(nv)
        return (x_omega, d, nv, tv)
    else:
        assert(False)
        #
        # Inside the polygon?
        # @todo: Consider whether a more appropriate answer should be considered instead.
        #
        x_omega = pts[idx]
        d, nv = smnd.calc_polar_decomp(u1)
        tv = smnd.rotate_90ACW([-nv[1], nv[0]])
        return (x_omega, d, nv, tv)

    return None

#
# Note: Only convex polygons are currently supported.
#
class Polygon(Shape):
    def __init__(self, pts, radius):
        super(Polygon, self).__init__()

        self.pts    = pts
        self.radius = radius

        # self.posn   = np.mean(self.pts)
        s = np.zeros(2)
        for i in range(len(self.pts)):
            s += self.pts[i]
        self.posn = s / len(self.pts)
        
        self.bounded   = True
        self.convex    = True
        self.clockwise = None

        # return

        cur_u  = self.pts[0] - self.posn
        next_u = self.pts[1] - self.posn
        cur_ACW = (smnd.calc_det2x2(cur_u, next_u) > 0)
        for i in range(2, len(self.pts)):
            cur_u  = next_u
            next_u = self.pts[i] - self.posn
            next_ACW = (smnd.calc_det2x2(cur_u, next_u) > 0)
            if next_ACW != cur_ACW:
                self.convex = False
                break

        if self.convex:
            self.clockwise = not cur_ACW

        return

    def translate(self, u):
        for i in range(len(self.pts)):
            self.pts[i] += u
        self.posn += u
        return
    
    def translated(self, u):
        xlated = cp.deepcopy(self)
        for i in range(len(xlated.pts)):
            xlated.pts[i] += u
        xlated.posn += u
        return xlated

    def calc_bbox(self):
        #
        #
        #
        
        blp = self.pts[0].copy()
        trp = self.pts[0].copy()
        for i in range(1, len(self.pts)):
            pt = self.pts[i]
            if pt[0] <= blp[0]:
                blp[0] = pt[0]
            elif pt[0] >= trp[0]:
                trp[0] = pt[0]
                
            if pt[1] <= blp[1]:
                blp[1] = pt[1]
            elif pt[1] >= trp[1]:
                trp[1] = pt[1]
        delta = np.array([self.radius, self.radius])

        return smgg.BoundingBox(blp - delta, trp + delta)

        # return (blp - delta, trp + delta)

    #
    # Polygon is expected to be convex and anti-clockwise. If the assertion triggers, it isn't!
    #
    def validate(self):
        for i in range(len(pts) - 2):
            p0 = self.pts[i]
            p1 = self.pts[i + 1]
            p2 = self.pts[i + 2]
            u  = p1 - p0
            v  = p2 - p1
            assert(u[0] * v[1] - u[1] * v[0] > 0)
            
    def get_nearest_point_info(self, posn):
        # print("Polygon::get_nearest_point_info - {}".format(self.pts))

        return get_nearest_point_info(self.pts, posn)

        nearest_pts = []
        n = len(self.pts)
        for i in range(n):
            delta = posn - self.pts[i]
            item  = (delta, i, np.dot(delta, delta))
            nearest_pts.append(item)
        nearest_pts.sort(key = lambda item: item[2])
        
        idx      = nearest_pts[0][1]
        prev_idx = (idx + n - 1) % n
        next_idx = (idx + 1)     % n
        # print("Polygon::get_nearest_point_info - i - 1, i, i + 1 %{} {}, {}, {}".format(n, prev_idx, idx, next_idx))

        #
        # @todo: Determine which segment posn is closer to.
        #
        dp0 = self.pts[idx]      - self.pts[prev_idx]
        dp1 = self.pts[next_idx] - self.pts[idx]
        # print("Polygon::get_nearest_point_info - {} = {} - {}, {} = {} - {}".format(dp0, self.pts[idx], self.pts[prev_idx], dp1, self.pts[next_idx], self.pts[idx]))

        l0, e0 = smnd.calc_polar_decomp(dp0)
        l1, e1 = smnd.calc_polar_decomp(dp1)
        # print("Polygon::get_nearest_point_info - {}*{} = {}, {}*{} = {}".format(l0, e0, dp0, l1, e1, dp1))

        #
        # @todo: Not all of these are always necessary. Consider lazier evaluation or keeping them during the ?
        #
        adv_posn = posn
        # adv_posn = posn + t * vel
        u0 = adv_posn - self.pts[prev_idx]
        u1 = adv_posn - self.pts[idx]
        u2 = adv_posn - self.pts[next_idx]
        # print("Polygon::get_nearest_point_info - u_i {}, {}, {}".format(u0, u1, u2))

        s0 = np.dot(u0, e0) #  / l0
        s1 = np.dot(u1, e1) # / l1

        # print("Polygon::get_nearest_point_info - {} = {}^T{}, {} = {}^T{}, l0 = {}".format(s0, u0, e0, s1, u1, e1, l0))
        if s0 >= l0 and s1 <= 0.:
            #
            # Closer to self.pts[idx] than either segment.
            #
            x_omega = self.pts[idx]
            d, nv = smnd.calc_polar_decomp(u1)
            tv = smnd.rotate_90ACW(nv)
            return (x_omega, d, nv, tv)

        elif s0 <= l0:
            #
            # Closest to segment between self.pts[prev_idx], self.pts[idx]
            #
            smd.dbgassert(s0 >= 0.5 * l0)
            
            x_omega = self.pts[prev_idx] + s0 * e0
            d, nv   = smnd.calc_polar_decomp(adv_posn - x_omega)
            tv      = smnd.rotate_90ACW(nv)
            return (x_omega, d, nv, tv)

        elif s1 >= 0.:
            #
            # Closest to segment between self.pts[idx], self.pts[next_idx]
            #
            smd.dbgassert(s1 <= 0.5 * l1)
            
            x_omega = self.pts[idx] + s1 * e1
            d, nv   = smnd.calc_polar_decomp(adv_posn - x_omega)
            tv      = smnd.rotate_90ACW(nv)
            return (x_omega, d, nv, tv)
        else:
            #
            # Inside the polygon!
            # @todo: Consider whether a more appropriate answer should be considered instead.
            #
            x_omega = self.pts[idx]
            d, nv = smnd.calc_polar_decomp(u1)
            tv = smnd.rotate_90ACW([-nv[1], nv[0]])
            return (x_omega, d, nv, tv)

    def get_nearest_approach_info(self, posn, vel, t_max):
        smd.dbgassert(self.convex)
        
        nearest_pts = []
        for i in range(len(self.pts)):
            delta = posn - self.pts[i]
            item  = (delta, i, np.dot(delta, delta))
            nearest_pts.append(item)
        nearest_pts.sort(key = lambda item: item[2])
        
        n = len(self.pts)
        idx      = nearest_pts[0][1]
        prev_idx = (idx + n - 1) % n
        next_idx = (idx + 1)     % n

        smd.dbgprint("Polygon::get_nearest_approach_info of {} points, {}, {}, {} are the closest".format(n, prev_idx, idx, next_idx))

        #
        # @todo: Determine which segment posn is closer to.
        #
        dp0 = self.pts[idx]      - self.pts[prev_idx]
        dp1 = self.pts[next_idx] - self.pts[idx]

        l0, e0 = smnd.calc_polar_decomp(dp0)
        l1, e1 = smnd.calc_polar_decomp(dp1)

        #
        # @todo: Not all of these are always necessary. Consider lazier evaluation
        # or eliminating some of them once the code is working.
        #
        u0 = posn - self.pts[prev_idx]
        u1 = posn - self.pts[idx]
        u2 = posn - self.pts[next_idx]

        s0 = np.dot(u0, e0) # / l0
        s1 = np.dot(u1, e1) # / l1

        smd.dbgprint("Polygon::get_nearest_approach_info of {} + t {} points, starting point s0 {}, l0 {}; s1 {}, l1 {}".format(posn, vel, s0, l0, s1, l1))
        smd.dbgassert(s0 >= 0.5 * l0)
        smd.dbgassert(s1 <= 0.5 * l1)

        beyond0 = s0 >= l0 # Starting point behind 0th segment
        beyond1 = s1 <= 0  # Starting point before 1st segment

        #
        # If both flags are set, the agent's or particle's starting point is
        # closer to the polygon's nearest point than to either of the
        # neighbouring segments. Determine whether the agent/particle is
        # non-convergent, otherwise determine which segment the agent/particle
        # will cross. We may be omitting
        #
        FIX_2025_12_09 = True
        if beyond0 and beyond1:
            x_omega = self.pts[idx]
            if np.dot(u1, vel) >= 0:
                # Non-convergent
                t = 0
                d, nv = smnd.calc_polar_decomp(u1)
                tv = smnd.rotate_90ACW(nv)
                return (x_omega, posn, d, nv, tv, t)
            elif smnd.calc_det2x2(vel, -u1) > 0:
                #
                # Will pass to right of nearest point.
                #
                assert(smnd.calc_det2x2(dp0, dp1) >= 0)
                if FIX_2025_12_09:
                    beyond0 = False
                else:
                    beyond1 = False
            else:
                #
                # Will pass to left of nearest point (or go through it if det2x2 \approx 0).
                #
                assert(smnd.calc_det2x2(dp0, dp1) >= 0)
                if FIX_2025_12_09:
                    beyond1 = False
                else:
                    beyond0 = False
        elif beyond1:
            x_omega = self.pts[prev_idx] + np.dot(posn - self.pts[prev_idx], e0) * e0
        elif beyond0:
            x_omega = self.pts[idx] + np.dot(posn - self.pts[idx], e1) * e1
        else:
            #
            # Assuming convexity, the point is inside the polygon
            #
            assert(self.convex)
            d0 = s1
            d1 = l0 - s0
            if (d1 - d0)**2 < smnd.EPS_m_sqr:
                x_omega = self.pts[idx]
                nv = -u1
                tv = smnd.rotate_90ACW(nv)
                d = -nearest_pts[0][2]
            elif d0 < d1:
                x_omega = self.pts[prev_idx] + np.dot(posn - self.pts[prev_idx], e0) * e0
                nv = -e1
                tv = -e0
                d  = -d0
            else:
                assert(d1 < d0)
                x_omega = self.pts[idx] + np.dot(posn - self.pts[idx], e1) * e1
                nv = -e0
                tv = -e1
                d  = -d1

            t= 0.
            new_posn = posn
            return (x_omega, new_posn, d, nv, tv, t)

        if not beyond0:
            #
            # Let's do a quick back-face cull. If t = 0, x_omega has already been set.
            #
            if smnd.calc_det2x2(vel, dp0) >= 0:
                # Non-convergent
                t = 0
                d, nv = smnd.calc_polar_decomp(posn - x_omega)
                tv = smnd.rotate_90ACW(nv)
                return (x_omega, posn, d, nv, tv, t)
            else:
                s, t     = calc_segment_approach(self.pts[prev_idx], self.pts[idx], posn, vel, t_max)
                x_omega  = self.pts[prev_idx] + s * e0
                new_posn = posn + t * vel
                # d, nv = smnd.calc_polar_decomp(new_posn - x_omega)
                # tv = smnd.rotate_90ACW(nv)
                d  = smnd.calc_norm(new_posn - x_omega)
                nv = smnd.rotate_90ACW(e0)
                if np.dot(posn - x_omega, nv) < 0.:
                    nv = -nv
                tv = e0
                return (x_omega, new_posn, d, nv, tv, t)
        else:
            smd.dbgassert(not beyond1)
            #
            # Let's do a quick back-face cull. If t = 0, x_omega has already been set.
            #
            if smnd.calc_det2x2(vel, dp1) >= 0:
                # Non-convergent
                t = 0
                d, nv = smnd.calc_polar_decomp(posn - x_omega)
                tv = smnd.rotate_90ACW(nv)
                return (x_omega, posn, d, nv, tv, t)
            else:
                s, t     = calc_segment_approach(self.pts[idx], self.pts[next_idx], posn, vel, t_max)
                x_omega  = self.pts[prev_idx] + s * e1
                new_posn = posn + t * vel
                # d, nv = smnd.calc_polar_decomp(new_posn - x_omega)
                # tv = smnd.rotate_90ACW(nv)
                d  = smnd.calc_norm(new_posn - x_omega)
                nv = smnd.rotate_90ACW(e1)
                if np.dot(posn - x_omega, nv) < 0.:
                    nv = -nv
                tv = e1
                return (x_omega, new_posn, d, nv, tv, t)

        return None




    def on(self, posn):
        d = self.get_nearest_point_info(posn)[1]
        return d < self.radius

    def contains(self, posn):
        x_omega, d = self.get_nearest_point_info(posn)[0:2]
        return d < self.radius or smnd.calc_norm_sqr(self.posn - posn) <= smnd.calc_norm_sqr(x_omega - posn) 
    
    def draw(self, colour, view):
        # smd.dbgprint("Polygon::draw - pts {} with radius".format(self.pts, self.radius))
        view.stroke_polygon(colour, self.pts, self.radius)
        for posn in self.pts:
            view.fill_circle(colour, posn, self.radius)
        return

class Rectangle(Polygon):
    def __init__(self, posn, e_major, halflength, halfwidth, radius):
        self.e_major = e_major
        self.e_minor = smnd.rotate_90ACW(e_major)
        self.halflength = halflength
        self.halfwidth = halfwidth
        
        major_axis = halflength * self.e_major
        minor_axis = halfwidth  * self.e_minor
        self.posn  = posn
        
        pts = [posn + major_axis + minor_axis, posn - major_axis + minor_axis, posn - major_axis - minor_axis, posn + major_axis - minor_axis]

        super(Rectangle, self).__init__(pts, radius)
        return

    def translate(self, dx):
        super(Rectangle, self).translate(dx)
        return

#
# Note: Only convex polygons were supported. Because the Polysegment is the first
# non-convex shape to be supported, it is likely to be a source of problems.
#
class Polysegment(Shape):
    def __init__(self, pts, radius):
        super(Polysegment, self).__init__()

        self.pts    = pts
        self.radius = radius

        # self.posn   = np.mean(self.pts)
        s = np.zeros(2)
        for i in range(len(self.pts)):
            s += self.pts[i]
        self.posn = s / len(self.pts)
        
        self.convex = False
        self.bounded = True

        return

    def translate(self, u):
        for i in range(len(self.pts)):
            self.pts[i] += u
        self.posn += u
        return
    
    def translated(self, u):
        xlated = cp.deepcopy(self)
        for i in range(len(xlated.pts)):
            xlated.pts[i] += u
        xlated.posn += u
        return xlated

    def calc_bbox(self):
        #
        #
        #
        
        blp = self.pts[0].copy()
        trp = self.pts[0].copy()
        for i in range(1, len(self.pts)):
            pt = self.pts[i]
            if pt[0] <= blp[0]:
                blp[0] = pt[0]
            elif pt[0] >= trp[0]:
                trp[0] = pt[0]
                
            if pt[1] <= blp[1]:
                blp[1] = pt[1]
            elif pt[1] >= trp[1]:
                trp[1] = pt[1]
        delta = np.array([self.radius, self.radius])

        return smgg.BoundingBox(blp - delta, trp + delta)
            
    def get_nearest_point_info(self, posn):
        print("Polysegment::get_nearest_point_info - {}".format(self.pts))

        nearest_pts = []
        n = len(self.pts)
        for i in range(n):
            delta = posn - self.pts[i]
            item  = (delta, i, np.dot(delta, delta))
            nearest_pts.append(item)
        nearest_pts.sort(key = lambda item: item[2])
        
        idx      = nearest_pts[0][1]
        if idx == 0:
            next_idx = idx + 1
            dp1      = self.pts[next_idx] - self.pts[idx]
            l1, e1 = smnd.calc_polar_decomp(dp1)

            #
            # @todo: Not all of these are always necessary. Consider lazier evaluation or keeping them during the ?
            #
            adv_posn = posn
            u1 = adv_posn - self.pts[idx]
            s1 = np.dot(u1, e1)
            if s1 < 0.:
                s1 = 0.
            elif s1 > l1:
                s1 = l1

            x_omega = self.pts[prev_idx] + s1 * e1
            d, nv   = smnd.calc_polar_decomp(adv_posn - x_omega)
            tv      = smnd.rotate_90ACW(nv)
            return (x_omega, d, nv, tv)

        elif idx == len(self.pts) - 1:
            prev_idx = idx - 1
            dp0      = self.pts[idx]      - self.pts[prev_idx]
            l0, e0 = smnd.calc_polar_decomp(dp0)

            #
            # @todo: Not all of these are always necessary. Consider lazier evaluation or keeping them during the ?
            #
            adv_posn = posn
            u0 = adv_posn - self.pts[prev_idx]
            s0 = np.dot(u0, e0) #  / l0
            if s0 < 0.:
                s0 = 0.
            elif s0 > l1:
                s0 = l1

            x_omega = self.pts[prev_idx] + s0 * e0
            d, nv   = smnd.calc_polar_decomp(adv_posn - x_omega)
            tv      = smnd.rotate_90ACW(nv)
            return (x_omega, d, nv, tv)

        else:        
            prev_idx = (idx + n - 1) % n
            next_idx = (idx + 1)     % n
            print("Polysegment::get_nearest_point_info - i - 1, i, i + 1 %{} {}, {}, {}".format(n, prev_idx, idx, next_idx))

            #
            # @todo: Determine which segment posn is closer to.
            #
            dp0 = self.pts[idx]      - self.pts[prev_idx]
            dp1 = self.pts[next_idx] - self.pts[idx]
            print("Polysegment::get_nearest_point_info - {} = {} - {}, {} = {} - {}".format(dp0, self.pts[idx], self.pts[prev_idx], dp1, self.pts[next_idx], self.pts[idx]))

            l0, e0 = smnd.calc_polar_decomp(dp0)
            l1, e1 = smnd.calc_polar_decomp(dp1)
            print("Polysegment::get_nearest_point_info - {}*{} = {}, {}*{} = {}".format(l0, e0, dp0, l1, e1, dp1))

            #
            # @todo: Not all of these are always necessary. Consider lazier evaluation or keeping them during the ?
            #
            adv_posn = posn
            u0 = adv_posn - self.pts[prev_idx]
            u1 = adv_posn - self.pts[idx]
            u2 = adv_posn - self.pts[next_idx]
            print("Polysegment::get_nearest_point_info - u_i {}, {}, {}".format(u0, u1, u2))

            s0 = np.dot(u0, e0) #  / l0
            s1 = np.dot(u1, e1) # / l1

            print("Polysegment::get_nearest_point_info - {} = {}^T{}, {} = {}^T{}, l0 = {}".format(s0, u0, e0, s1, u1, e1, l0))
            if s0 >= l0 and s1 <= 0.:
                #
                # Closer to self.pts[idx] than either segment.
                #
                x_omega = self.pts[idx]
                d, nv = smnd.calc_polar_decomp(u1)
                tv = smnd.rotate_90ACW(nv)
                return (x_omega, d, nv, tv)

            elif s0 <= l0:
                #
                # Closest to segment between self.pts[prev_idx], self.pts[idx]
                #
                smd.dbgassert(s0 >= 0.5 * l0)
                
                x_omega = self.pts[prev_idx] + s0 * e0
                d, nv   = smnd.calc_polar_decomp(adv_posn - x_omega)
                tv      = smnd.rotate_90ACW(nv)
                return (x_omega, d, nv, tv)

            elif s1 >= 0.:
                #
                # Closest to segment between self.pts[idx], self.pts[next_idx]
                #
                smd.dbgassert(s1 <= 0.5 * l1)
                
                x_omega = self.pts[idx] + s1 * e1
                d, nv   = smnd.calc_polar_decomp(adv_posn - x_omega)
                tv      = smnd.rotate_90ACW(nv)
                return (x_omega, d, nv, tv)
            else:
                #
                # Inside the polygon!
                # @todo: Consider whether a more appropriate answer should be considered instead.
                #
                x_omega = self.pts[idx]
                d, nv = smnd.calc_polar_decomp(u1)
                tv = smnd.rotate_90ACW([-nv[1], nv[0]])
                return (x_omega, d, nv, tv)

    def get_nearest_approach_info(self, posn, vel, t_max):
        smd.dbgassert(self.convex)
        
        nearest_pts = []
        for i in range(len(self.pts)):
            delta = posn - self.pts[i]
            item  = (delta, i, np.dot(delta, delta))
            nearest_pts.append(item)
        nearest_pts.sort(key = lambda item: item[2])
        
        n = len(self.pts)
        idx      = nearest_pts[0][1]
        prev_idx = (idx + n - 1) % n
        next_idx = (idx + 1)     % n

        smd.dbgprint("Polygon::get_nearest_approach_info of {} points, {}, {}, {} are the closest".format(n, prev_idx, idx, next_idx))

        #
        # @todo: Determine which segment posn is closer to.
        #
        dp0 = self.pts[idx]      - self.pts[prev_idx]
        dp1 = self.pts[next_idx] - self.pts[idx]

        l0, e0 = smnd.calc_polar_decomp(dp0)
        l1, e1 = smnd.calc_polar_decomp(dp1)

        #
        # @todo: Not all of these are always necessary. Consider lazier evaluation
        # or eliminating some of them once the code is working.
        #
        u0 = posn - self.pts[prev_idx]
        u1 = posn - self.pts[idx]
        u2 = posn - self.pts[next_idx]

        s0 = np.dot(u0, e0) # / l0
        s1 = np.dot(u1, e1) # / l1

        smd.dbgprint("Polygon::get_nearest_approach_info of {} + t {} points, starting point s0 {}, l0 {}; s1 {}, l1 {}".format(posn, vel, s0, l0, s1, l1))
        smd.dbgassert(s0 >= 0.5 * l0)
        smd.dbgassert(s1 <= 0.5 * l1)
        
        beyond0 = s0 >= l0 # Starting point behind 0th segment
        beyond1 = s1 <= 0  # Starting point before 1st segment

        #
        # If both flags are set, the agent's or particle's starting point is
        # closer to the polygon's nearest point than to either of the
        # neighbouring segments. Determine whether the agent/particle is
        # non-convergent, otherwise determine which segment the agent/particle
        # will cross. We may be omitting 
        #
        if beyond0 and beyond1:
            x_omega = self.pts[idx]
            if np.dot(u1, vel) >= 0:
                # Non-convergent
                t = 0
                d, nv = smnd.calc_polar_decomp(u1)
                tv = smnd.rotate_90ACW(nv)
                return (x_omega, posn, d, nv, tv, t)
            elif smnd.calc_det2x2(vel, -u1) >= 0:
                beyond1 = False
            else:
                beyond0 = False
        elif beyond1:
            x_omega = self.pts[prev_idx] + np.dot(posn - self.pts[prev_idx], e0) * e0
        else:
            smd.dbgassert(beyond0)
            x_omega = self.pts[idx] + np.dot(posn - self.pts[idx], e1) * e1

        if not beyond0:
            #
            # Let's do a quick back-face cull. If t = 0, x_omega has already been set.
            #
            if smnd.calc_det2x2(vel, dp0) >= 0:
                # Non-convergent
                t = 0
                d, nv = smnd.calc_polar_decomp(posn - x_omega)
                tv = smnd.rotate_90ACW(nv)
                return (x_omega, posn, d, nv, tv, t)
            else:
                s, t     = calc_segment_approach(self.pts[prev_idx], self.pts[idx], posn, vel, t_max)
                x_omega  = self.pts[prev_idx] + s * e0
                new_posn = posn + t * vel 
                d, nv = smnd.calc_polar_decomp(new_posn - x_omega)
                tv = smnd.rotate_90ACW(nv)
                return (x_omega, new_posn, d, nv, tv, t)
        else:
            smd.dbgassert(not beyond1)
            #
            # Let's do a quick back-face cull. If t = 0, x_omega has already been set.
            #
            if smnd.calc_det2x2(vel, dp1) >= 0:
                # Non-convergent
                t = 0
                d, nv = smnd.calc_polar_decomp(posn - x_omega)
                tv = smnd.rotate_90ACW(nv)
                return (x_omega, posn, d, nv, tv, t)
            else:
                s, t     = calc_segment_approach(self.pts[idx], self.pts[next_idx], posn, vel, t_max)
                x_omega  = self.pts[prev_idx] + s * e1
                new_posn = posn + t * vel 
                d, nv = smnd.calc_polar_decomp(new_posn - x_omega)
                tv = smnd.rotate_90ACW(nv)
                return (x_omega, new_posn, d, nv, tv, t)

        return None







    def on(self, posn):
        d = self.get_nearest_point_info(posn)[1]
        return d < self.radius

    def contains(self, posn):
        return self.on(posn)
        # x_omega, d = self.get_nearest_point_info(posn)[0:2]
        # return d < self.radius or smnd.calc_norm_sqr(self.posn - posn) <= smnd.calc_norm_sqr(x_omega - posn) 
    
    def draw(self, colour, view):
        # smd.dbgprint("Polysegment::draw - pts {} with radius".format(self.pts, self.radius))
        view.stroke_polysegment(colour, self.pts, self.radius)
        for posn in self.pts:
            view.fill_circle(colour, posn, self.radius)
        return

#
# Note: Only convex polygons were supported. Because the Cone is only the second
# unbounded shape to be supported, it is likely to be a source of problems.
#
class Cone(Shape):
    def __init__(self, apex, left_dirn, right_dirn):
        super(Cone, self).__init__()

        self.apex       = apex.copy()
        self.left_dirn  = left_dirn
        self.right_dirn = right_dirn

        self.posn = apex

        self.convex  = True
        self.bounded = False

        return

    def translate(self, u):
        self.apex += u
        return
    
    def translated(self, u):
        xlated = cp.deepcopy(self)
        xlated.apex += u
        return xlated
            
    def get_nearest_point_info(self, posn):
        print("Cone::get_nearest_point_info - cone {} vs posn {}".format(self, posn))

        # nearest_pts = []
        # for i in range(len(self.pts)):
            # delta = posn - self.pts[i]
            # item  = (delta, i, np.dot(delta, delta))
            # nearest_pts.append(item)
        # nearest_pts.sort(key = lambda item: item[2])       

        dp   = posn - self.apex
        d, e = smnd.calc_polar_decomp(dp)

        cos0 = np.dot(self.left_dirn,  e)
        cos1 = np.dot(self.right_dirn, e)

        if cos0 > cos1:
            if cos0 > 0:
                x_omega = self.apex + d * cos0 * self.left_dirn
                tv = self.left_dirn
                nv = smnd.rotate_90ACW(tv)
                d  = smnd.calc_norm(posn - x_omega)
                return (x_omega, d, nv, tv)
        else:
            if cos1 > 0:
                x_omega = self.apex + d * cos1 * self.right_dirn
                tv = self.right_dirn
                nv = -smnd.rotate_90ACW(tv)
                d  = smnd.calc_norm(posn - x_omega)
                return (x_omega, d, nv, tv)
        x_omega = self.apex
        d, nv = smnd.calc_polar_decomp(posn - x_omega)
        tv = smnd.rotate_90ACW(nv)
        return (x_omega, d, nv, tv)

    def get_nearest_approach_info(self, posn, vel, t_max):
        return None

    def on(self, posn):
        d = self.get_nearest_point_info(posn)[1]
        return d < self.radius

    def contains(self, posn):
        return None
    
        #
        # @todo: 
        #
        return self.on(posn)
        # x_omega, d = self.get_nearest_point_info(posn)[0:2]
        # return d < self.radius or smnd.calc_norm_sqr(self.posn - posn) <= smnd.calc_norm_sqr(x_omega - posn) 
    
    def draw(self, colour, view):
        # smd.dbgprint("Polysegment::draw - pts {} with radius".format(self.pts, self.radius))
        # view.stroke_polysegment(colour, self.pts, self.radius)
        # for posn in self.pts:
            # view.fill_circle(colour, posn, self.radius)
        return

#
# Note: Only convex polygons are supported. An unbounded shape.
#
class TruncatedCone(Shape):
    def __init__(self, apex, pts, radius, l_left, l_right, e_left, e_right, convex):
        self.apex    = apex.copy()
        self.pts     = pts
        self.radius  = radius

        self.posn = self.apex

        self.e_left  = e_left
        self.e_right = e_right

        self.l_left  = l_left
        self.l_right = l_right

        self.convex = convex
        self.bounded = False

        return

    def scale(self, s):
        for i in range(len(self.pts)):
            self.pts[i] *= s
        self.radius *= s
        self.l_left *= s
        self.l_right *= s

    def scaled(self, s):
        new_tcone = cp.deepcopy(self)
        for i in range(len(self.pts)):
            new_tcone.pts[i] *= s
        new_tcone.radius *= s
        new_tcone.l_left *= s
        new_tcone.l_right *= s

    def translate(self, u):
        self.apex = self.apex + u
        return
    
    def translated(self, u):
        xlated = cp.deepcopy(self)
        xlated.apex += u
        return xlated
            
    def get_nearest_point_info(self, posn):
        print("TruncatedCone::get_nearest_point_info - {}".format(self.pts))

        dp   = posn - self.apex
        d, e = smnd.calc_polar_decomp(dp)

        cos0 = np.dot(self.e_left,  e)
        cos1 = np.dot(self.e_right, e)

        if cos0 > cos1:
            if cos0 > 0:
                x_omega = self.apex + d * cos0 * self.e_left
                tv = self.e_left
                nv = smnd.rotate_90ACW(tv)
                d  = smnd.calc_norm(posn - x_omega)
                return (x_omega, d, nv, tv)
        else:
            if cos1 > 0:
                x_omega = self.apex + d * cos1 * self.e_right
                tv = self.e_right
                nv = -smnd.rotate_90ACW(tv)
                d  = smnd.calc_norm(posn - x_omega)
                return (x_omega, d, nv, tv)

        nearest_pts = []
        for i in range(len(self.pts)):
            delta = posn - self.pts[i]
            item  = (delta, i, np.dot(delta, delta))
            nearest_pts.append(item)
        nearest_pts.sort(key = lambda item: item[2])

        

        x_omega = self.apex
        d, nv = smnd.calc_polar_decomp(posn - x_omega)
        tv = smnd.rotate_90ACW(nv)
        return (x_omega, d, nv, tv)

    def get_nearest_approach_info(self, posn, vel, t_max):
        return None

    def on(self, posn):
        d = self.get_nearest_point_info(posn)[1]
        return d < self.radius

    def contains(self, posn):
        return None
    
        #
        # @todo: 
        #
        return self.on(posn)
        # x_omega, d = self.get_nearest_point_info(posn)[0:2]
        # return d < self.radius or smnd.calc_norm_sqr(self.posn - posn) <= smnd.calc_norm_sqr(x_omega - posn) 
    
    def draw(self, colour, view):
        # smd.dbgprint("Polysegment::draw - pts {} with radius".format(self.pts, self.radius))
        
        # if len(self.pts) > 1:
            # view.stroke_polysegment(colour, self.pts, self.radius)

        left_pt = self.apex_posn + self.l_left * self.e_left
        right_pt = self.apex_posn + self.l_right * self.e_right
        view.draw_line(colour, self.apex_posn, left_pt, 0)
        view.draw_line(colour, self.apex_posn, right_pt, 0)

        cur_relposn = self.pts[0]
        cur_posn    = self.apex_posn + cur_relposn
        view.fill_circle(colour, cur_posn, self.radius)
        for i in range(1, len(self.pts)):
            next_relposn = self.pts[i]

            next_posn = self.apex_posn + next_relposn
            view.fill_circle(colour, next_posn, self.radius)
            view.draw_line(colour, cur_posn, next_posn, self.radius)
            
            cur_posn = next_posn
        return

    def draw_parts(self, shift_vector, colour1, colour2, view):
        # assert(np.dot(self.apex, self.apex) < smnd.EPS_m_sqr)
        # assert(np.dot(shift_vector, shift_vector) > smnd.EPS_m_sqr)

        # smd.dbgprint("Polysegment::draw - pts {} with radius".format(self.pts, self.radius))

        # if len(self.pts) > 1:
            # view.stroke_polysegment(colour1, self.pts, self.radius)

        if len(self.pts) == 0:
            return

        left_pt = self.apex + shift_vector + self.l_left * self.e_left
        right_pt = self.apex + shift_vector + self.l_right * self.e_right
        view.draw_line(colour1, self.apex + shift_vector, left_pt, 0)
        view.draw_line(colour1, self.apex + shift_vector, right_pt, 0)
        
        cur_relposn = self.pts[0]
        cur_posn    = self.apex + shift_vector + cur_relposn
        view.stroke_circle(colour2, cur_posn, self.radius)
        for i in range(1, len(self.pts)):
            next_relposn = self.pts[i]

            next_posn = self.apex + shift_vector + next_relposn
            view.stroke_circle(colour2, next_posn, self.radius)
            # view.draw_line(colour2, cur_posn, next_posn, self.radius)

            L, e1 = smnd.calc_polar_decomp(next_posn - cur_posn)
            w     = self.radius
            e2    = smnd.rotate_90ACW(e1)
            delta = w * e2
            # view.draw_thin_line(colour2, cur_posn + delta, next_posn + delta)
            view.draw_thin_line(colour2, cur_posn - delta, next_posn - delta)
            
            cur_posn = next_posn

        lefter_pt = self.apex + shift_vector + 2 * self.l_left * self.e_left
        righter_pt = self.apex + shift_vector + 2 * self.l_right * self.e_right
        view.draw_line(colour2, left_pt, lefter_pt, 0)
        view.draw_line(colour2, right_pt, righter_pt, 0)

        return

def construct_tcone_circle_circle(src, dst, T, convex):
    assert(T > 0)
    assert(convex)

    r_total = src.radius + dst.radius
    relposn = dst.posn - src.posn
    if smnd.calc_norm_sqr(relposn) < r_total**2:
        return None
           
    rcp_T = 1 / T

    radius    = rcp_T * r_total
    pt        = rcp_T * relposn
    pts = [pt]
    apex = src.posn.copy()

    d, e_main = smnd.calc_polar_decomp(pt)

    rsqr = radius**2
    dsqr = d**2
    if dsqr - rsqr < 0:
        return None

    l    = np.sqrt(dsqr - rsqr)

    l_left = l_right = l
    
    cos_theta = l / d
    sin_theta = radius / d

    e_perp = smnd.rotate_90ACW(e_main)
    e_left  = cos_theta * e_main + sin_theta * e_perp
    e_right = cos_theta * e_main - sin_theta * e_perp

    return TruncatedCone(apex, pts, radius, l_left, l_right, e_left, e_right, convex)

def construct_tcone_circle_segment(src, dst, T, convex):
    assert(T > 0)
    assert(convex)

    apex = src.posn.copy()
            
    rcp_t = 1 / T
    
    radius = rcp_t * (src.radius + dst.radius)

    relposn1 = rcp_t * (dst.posn1 - src.posn)
    relposn2 = rcp_t * (dst.posn2 - src.posn)
    relposn = 0.5 * (relposn1 + relposn2)

    L_seg, e_seg = smnd.calc_polar_decomp(relposn2 - relposn1)
    e_seg_perp = smnd.rotate_90ACW(e_seg)

    l_seg = 0.5 * L_seg
    w_seg = radius

    pts = []
    rpw = np.dot(relposn, e_seg_perp)
    if rpw**2 > radius**2: 
        if rpw < -radius:
            pts.append(relposn2)
            pts.append(relposn1)
        else:
            assert(rpw > radius)
            pts.append(relposn1)
            pts.append(relposn2)

        rsqr = radius**2

        d_left, e_leftmain   = smnd.calc_polar_decomp(pts[0])
        d_right, e_rightmain = smnd.calc_polar_decomp(pts[-1])

        if d_left < radius or d_right < radius:
            return None
                
        dsqr_left = d_left**2
        l_left = np.sqrt(dsqr_left - rsqr)
        cos_left = l_left / d_left
        sin_left = radius / d_left

        dsqr_right = d_right**2
        l_right = np.sqrt(dsqr_right - rsqr)
        cos_right = l_right / d_right
        sin_right = radius / d_right

        e_leftmain_perp = smnd.rotate_90ACW(e_leftmain)
        e_left = cos_left * e_leftmain + sin_left * e_leftmain_perp

        e_rightmain_perp = smnd.rotate_90ACW(e_rightmain)
        e_right = cos_right * e_rightmain - sin_right * e_rightmain_perp
    else:
        rpl = np.dot(relposn, e_seg)
        if rpl < -l_seg:
            pts.append(relposn2)
        elif rpl > l_seg:
            pts.append(relposn1)
        else:
            return None

        d, e_main = smnd.calc_polar_decomp(pts[0])
        if d < radius:
            return None

        rsqr = radius**2
        dsqr = d**2
        l = np.sqrt(dsqr - rsqr)
        l_left = l_right = l
                
        cos_theta = l / d
        sin_theta = radius / d

        e_perp = smnd.rotate_90ACW(e_main)
        e_left = cos_theta * e_main + sin_theta * e_perp
        e_right = cos_theta * e_main - sin_theta * e_perp

    return TruncatedCone(apex, pts, radius, l_left, l_right, e_left, e_right, convex)

def construct_tcone_circle_rectangle(src, dst, T, convex):
    assert(T > 0)

    apex = src.posn.copy()
            
    rcp_t  = 1 / T
    radius = (src.radius + dst.radius)

    l_rect = dst.halflength + radius
    w_rect = dst.halfwidth  + radius
            
    e_rect      = dst.e_major
    e_rect_perp = dst.e_minor


    relposn = dst.posn - src.posn
    rpl = np.dot(relposn, e_rect)
    rpw = np.dot(relposn, e_rect_perp)

    radius = rcp_t * (src.radius + dst.radius)

    pts = []
    if rpl < -l_rect:
        #
        # Front
        #
        if rpw < -w_rect:
            #
            # Left
            #
            pts.append(rcp_t * (dst.pts[3] - apex))
            pts.append(rcp_t * (dst.pts[0] - apex))
            pts.append(rcp_t * (dst.pts[1] - apex))
        elif rpw > w_rect:
            #
            # Right
            #
            pts.append(rcp_t * (dst.pts[2] - apex))
            pts.append(rcp_t * (dst.pts[3] - apex))
            pts.append(rcp_t * (dst.pts[0] - apex))
        else:
            #
            # Centre (Across)
            #
            pts.append(rcp_t * (dst.pts[3] - apex))
            pts.append(rcp_t * (dst.pts[0] - apex))
    elif rpl > l_rect:
        #
        # Rear
        #
        if rpw < -w_rect:
            #
            # Left
            #
            pts.append(rcp_t * (dst.pts[0] - apex))
            pts.append(rcp_t * (dst.pts[1] - apex))
            pts.append(rcp_t * (dst.pts[2] - apex))
        elif rpw > w_rect:
            #
            # Right
            #
            pts.append(rcp_t * (dst.pts[1] - apex))
            pts.append(rcp_t * (dst.pts[2] - apex))
            pts.append(rcp_t * (dst.pts[3] - apex))
        else:
            #
            # Centre (Across)
            #
            pts.append(rcp_t * (dst.pts[1] - apex))
            pts.append(rcp_t * (dst.pts[2] - apex))
    else:
        #
        # Centre (Along)
        #
        if rpw < -w_rect:
            #
            # Left
            #
            pts.append(rcp_t * (dst.pts[0] - apex))
            pts.append(rcp_t * (dst.pts[1] - apex))
        elif rpw > w_rect:
            #
            # Right
            #
            pts.append(rcp_t * (dst.pts[2] - apex))
            pts.append(rcp_t * (dst.pts[3] - apex))
        else:
            return None

    rsqr = radius**2

    d_left, e_leftmain   = smnd.calc_polar_decomp(pts[0])
    d_right, e_rightmain = smnd.calc_polar_decomp(pts[-1])

    if d_left < radius or d_right < radius:
        return None

    dsqr_left = d_left**2
    l_left = np.sqrt(dsqr_left - rsqr)
    cos_left = l_left / d_left
    sin_left = radius / d_left

    dsqr_right = d_right**2
    l_right = np.sqrt(dsqr_right - rsqr)
    cos_right = l_right / d_right
    sin_right = radius / d_right

    e_leftmain_perp = smnd.rotate_90ACW(e_leftmain)
    e_left = cos_left * e_leftmain + sin_left * e_leftmain_perp
    e_rightmain_perp = smnd.rotate_90ACW(e_rightmain)
    e_right = cos_right * e_rightmain - sin_right * e_rightmain_perp

    convex  = True

    return TruncatedCone(apex, pts, radius, l_left, l_right, e_left, e_right, convex)

def construct_tcone_circle_polygon(src, dst, T, convex):
    # print("Shape::construct_tcone_circle_polygon - src {}, dst {}, T {}".format(src, dst, T))
    assert(T > 0)

    apex = src.posn.copy()
            
    rcp_t  = 1 / T

    relposn = dst.posn - src.posn
    
    nearest_pts = []
    n = len(dst.pts)
    # for i in range(n):
        # delta = src.posn - dst.pts[i]
        # item  = (delta, i, np.dot(delta, delta))
        # nearest_pts.append(item)
    # nearest_pts.sort(key = lambda item: item[2])

    #
    # @todo: Consider binary search for large
    #
    min_delta_sqr = smnd.calc_norm_sqr(relposn)
    main_idx       = -1
    for i in range(n):
        delta_sqr = smnd.calc_norm_sqr(dst.pts[i] - src.posn)
        if delta_sqr < min_delta_sqr:
            main_idx      = i
            min_delta_sqr = delta_sqr

    main_delta = dst.pts[main_idx] - src.posn

    d_main, e_main  = smnd.calc_polar_decomp(main_delta)
    # e_leftmain      = e_rightmain = e_main
    e_main_perp = smnd.rotate_90ACW(e_main)

    left_idx = right_idx = main_idx
    print("Shape::construct_tcone_circle_polygon - main idx = {}".format(main_idx))
    
    lefter_idx = left_idx - 1
    if lefter_idx == -1:
        lefter_idx = n - 1
    lefter_delta = dst.pts[lefter_idx] - src.posn
    d_lefter, e_lefter = smnd.calc_polar_decomp(lefter_delta)
    
    old_scalar_cross  = 0 # 
    new_scalar_cross = np.dot(e_main_perp, e_lefter)
    while new_scalar_cross > old_scalar_cross:
        assert(new_scalar_cross > 0)
        left_idx = lefter_idx

        lefter_idx = left_idx - 1
        if lefter_idx == -1:
            lefter_idx = n - 1

        lefter_delta = dst.pts[lefter_idx] - src.posn
        d_lefter, e_lefter = smnd.calc_polar_decomp(lefter_delta)
        
        old_scalar_cross = new_scalar_cross
        new_scalar_cross = np.dot(e_main_perp, e_lefter)

    # print("Shape::construct_tcone_circle_polygon - left idx = {}".format(left_idx))

    righter_idx = right_idx + 1
    if righter_idx == n:
        righter_idx = 0
    righter_delta = dst.pts[righter_idx] - src.posn
    d_righter, e_righter = smnd.calc_polar_decomp(righter_delta)

    old_scalar_cross  = 0 # 
    new_scalar_cross = np.dot(e_main_perp, e_righter)
    while new_scalar_cross < old_scalar_cross:
        assert(new_scalar_cross < 0)
        right_idx = righter_idx

        righter_idx = right_idx + 1
        if righter_idx == n:
            righter_idx = 0

        righter_delta = dst.pts[righter_idx] - src.posn
        d_righter, e_righter = smnd.calc_polar_decomp(righter_delta)
        
        old_scalar_cross = new_scalar_cross
        new_scalar_cross = np.dot(e_main_perp, e_righter)

    # print("Shape::construct_tcone_circle_polygon - right idx = {}".format(right_idx))
    
    pts = []
    idx = left_idx
    while idx != right_idx:
        # pts.append(rcp_t * dst.pts[idx])
        pts.append(rcp_t * (dst.pts[idx] - src.posn))

        idx += 1
        if idx >= n:
            idx = 0
        
    assert(idx == right_idx)
    pts.append(rcp_t * (dst.pts[idx] - src.posn))
        
    radius = rcp_t * (src.radius + dst.radius)

    d_left, e_leftmain   = smnd.calc_polar_decomp(pts[0])
    d_right, e_rightmain = smnd.calc_polar_decomp(pts[-1])

    if d_left < radius or d_right < radius:
        return None

    rsqr   = radius**2

    dsqr_left = d_left**2
    l_left = np.sqrt(dsqr_left - rsqr)
    cos_left = l_left / d_left
    sin_left = radius / d_left

    dsqr_right = d_right**2
    l_right = np.sqrt(dsqr_right - rsqr)
    cos_right = l_right / d_right
    sin_right = radius / d_right

    e_leftmain_perp = smnd.rotate_90ACW(e_leftmain)
    e_left = cos_left * e_leftmain + sin_left * e_leftmain_perp
    e_rightmain_perp = smnd.rotate_90ACW(e_rightmain)
    e_right = cos_right * e_rightmain - sin_right * e_rightmain_perp

    convex  = True

    return TruncatedCone(apex, pts, radius, l_left, l_right, e_left, e_right, convex)

def construct_tcone(src, dst, T, convex):
    assert(isinstance(src, Circle))
    assert(isinstance(dst, Shape))
    # assert(isinstance(dst, Circle) or isinstance(dst, Segment) or isinstance(dst, Rectangle))
    assert(isinstance(dst, Circle) or isinstance(dst, Segment) or isinstance(dst, Rectangle) or isinstance(dst, Polygon))

    assert(T > 0)
    if T <= 0:
        return None

    if isinstance(dst, Circle):
        return construct_tcone_circle_circle(src, dst, T, convex)
    elif isinstance(dst, Segment):
        return construct_tcone_circle_segment(src, dst, T, convex)
    elif isinstance(dst, Rectangle):
        return construct_tcone_circle_rectangle(src, dst, T, convex)
    elif isinstance(dst, Polygon):
        return construct_tcone_circle_polygon(src, dst, T, convex)
    else:
        assert(False)
        return None

    return None

#
# Compute shape S s.t. S = {y - x | x in shape1, y in shape2}
#
def construct_minkowski_difference_rect_rect(shape1, shape2):
    assert(isinstance(shape1, Shape))
    assert(isinstance(shape2, Shape))

    assert(isinstance(shape1, Rectangle))
    assert(isinstance(shape2, Rectangle))

    polycentre = shape2.posn - shape1.posn
    c = np.dot(shape1.e_major, shape2.e_major)
    s = np.dot(shape1.e_major, shape2.e_minor)

    if c**2 < smnd.EPS_m_sqr:
        l_res = shape2.halflength + shape1.halfwidth
        w_res = shape2.halfwidth + shape1.halflength
        newshape = Rectangle(polycentre, shape2.e_major, l_res, w_res, shape1.radius + shape2.radius)
    elif s**2 < smnd.EPS_m_sqr:
        l_res = shape2.halflength + shape1.halflength
        w_res = shape2.halfwidth + shape1.halfwidth
        newshape = Rectangle(polycentre, shape2.e_major, l_res, w_res, shape1.radius + shape2.radius)
    else:
        pts = []
        if c > 0 and s >= 0:
            v1 = shape1.halflength * shape1.e_major
            v2 = shape1.halfwidth  * shape1.e_minor
        elif c <= 0 and s > 0:
            v1 = -shape1.halfwidth * shape1.e_minor
            v2 = shape1.halflength * shape1.e_major
        elif c < 0 and s <= 0:
            v1 = -shape1.halflength * shape1.e_major
            v2 = -shape1.halfwidth  * shape1.e_minor
        elif c >= 0 and s < 0:
            v1 =  shape1.halfwidth  * shape1.e_minor
            v2 = -shape1.halflength * shape1.e_major

        u1 = shape2.halflength * shape2.e_major
        u2 = shape2.halfwidth  * shape2.e_minor

        #
        # For each of shape2's four corners, generate two offset points
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

        newshape = Polygon(pts, shape1.radius + shape2.radius)

    return newshape

def construct_minkowski_difference(shape1, shape2):
    assert(isinstance(shape1, Shape))
    assert(isinstance(shape2, Shape))

    assert(isinstance(shape1, Rectangle))
    assert(isinstance(shape2, Rectangle))

    return construct_minkowski_difference_rect_rect(shape1, shape2)


#
# Segment-construction
#
def create_segment(centre, u, halfwidth, halfdepth):
    return Segment(centre + halfwidth * u, centre - halfwidth * u, halfdepth)

#
# Polygon-construction
#
def create_square(centre, u, halfwidth, corner_radius):
    v = smnd.rotate_90ACW(u)
    pts    = [centre + halfwidth * (v + u), centre + halfwidth * (v - u), centre - halfwidth * (v + u), centre - halfwidth * (v - u)]
    return Polygon(pts, corner_radius)

def create_rect(centre, u, halflength, halfwidth, corner_radius):
    v = smnd.rotate_90ACW(u)
    fwd_centre = centre + halflength * u
    bwd_centre = centre - halflength * u
    pts    = [fwd_centre + halfwidth * v, bwd_centre + halfwidth * v, bwd_centre - halfwidth * v, fwd_centre - halfwidth * v]
    return Polygon(pts, corner_radius)

def create_hexagon(centre, u, halflength, halfwidth, corner_radius):
    v = smnd.rotate_90ACW(u)
    fwd_centre = centre + (halflength - halfwidth) * u
    bwd_centre = centre - (halflength - halfwidth) * u
    pts = [fwd_centre - halfwidth * v, fwd_centre + halfwidth * u, fwd_centre + halfwidth * v, bwd_centre + halfwidth * v, bwd_centre - halfwidth * u, bwd_centre - halfwidth * v]
    # pts    = [fwd_centre + halfwidth * v, bwd_centre + halfwidth * v, bwd_centre - halfwidth * v, fwd_centre - halfwidth * v]
    return Polygon(pts, corner_radius)

def create_regular_octagon(centre, u, halfwidth, corner_radius):
    v = smnd.rotate_90ACW(u)
    pts    = [centre + halfwidth * (v + u), centre + halfwidth * (v - u), centre - halfwidth * (v + u), centre - halfwidth * (v - u)]
    return Polygon(pts, corner_radius)

def construct_halfpolygon(keyed_halfplanes):
    keyed_halfplanes.sort(key = lambda item: item[0])

    i = 0
    key, halfplane = keyed_halfplanes[i]
    posn = halfplane.posn
    normal = halfplane.normal

    j = i + 1

    while i < len(keyed_halfplanes) - 1 and j < len(keyed_halfplanes):
        alt_key, alt_halfplane = keyed_halfplanes[j]
        alt_posn = alt_halfplane.posn
        alt_normal = alt_halfplane.normal

        if (alt_key - key) ** 2 >= smnd.EPS_m_sqr:
            break

        if np.dot(alt_posn - posn, normal) > 0:
            posn = alt_posn
            normal = alt_normal
            i = j

        j = j + 1

    #
    # Expected state:
    #   i is the index of the first halfplane
    #   j is the index of the following halfplane
    #
    assert(i >= 0)
    assert(i < j)
    assert(i < len(keyed_halfplanes))
    assert(j <= len(keyed_halfplanes))

    result_KHL = [keyed_halfplanes[i]]
    result_NL = []

    entry_dirn = -smnd.rotate_90ACW(normal)
    if j == len(keyed_halfplanes):
        return (result_KHL, result_NL, entry_dirn, exit_dirn)

    while i < len(keyed_halfplanes) - 1 and j < len(keyed_halfplanes):
        key, halfplane = keyed_halfplanes[i]
        posn = halfplane.posn
        normal = halfplane.normal
        dirn   = -smnd.rotate_90ACW(normal)

        next_key, next_halfplane = keyed_halfplanes[j]
        next_posn = next_halfplane.posn
        next_normal = next_halfplane.normal
        next_dirn = -smnd.rotate_90ACW(next_normal)

        dx = next_posn - posn
        num1 = smnd.calc_det2x2(dx, next_dirn)
        num2 = smnd.calc_det2x2(dirn, dx)
        dnm = smnd.calc_det2x2(dirn, next_dirn)
        assert (dnm >= smnd.EPS_m_sqr)

        rcp_dnm = 1 / dnm
        next_Lq1 = num1 * rcp_dnm
        next_Lq2 = num2 * rcp_dnm

        for k in range(j + 1, len(keyed_halfplanes)):
            alt_key, alt_halfplane = keyed_halfplanes[k]
            alt_posn = alt_halfplane.posn
            alt_normal = alt_halfplane.normal
            alt_dirn = -smnd.rotate_90ACW(alt_normal)

            dx = alt_posn - posn
            num1 = smnd.calc_det2x2(dx, alt_dirn)
            num2 = smnd.calc_det2x2(dirn, dx)
            dnm = smnd.calc_det2x2(dirn, alt_dirn)
            assert(dnm >= smnd.EPS_m_sqr)

            rcp_dnm = 1 / dnm
            alt_Lq1 = num1 * rcp_dnm
            alt_Lq2 = num2 * rcp_dnm

            if alt_Lq1 < next_Lq1:
                j = k

                next_posn   = alt_posn
                next_normal = alt_normal
                next_dirn   = alt_dirn
        assert j > i
        assert j < len(keyed_halfplanes)

        result_KHL.append(keyed_halfplanes[j])
        result_NL.append(posn + next_Lq1 * dirn)
        i = j
        j = i + 1

    exit_dirn = next_dirn

    return (result_KHL, result_NL, entry_dirn, exit_dirn)

def DEPRECATED_join_halfpolygons(first_half, second_half):
    first_KHL,  first_NL,  first_e0,  first_e1   = first_half
    second_KHL, second_NL, second_e0, second_e1 = second_half

    dx = second_KHL[0][1].posn - first_KHL[-1][1].posn
    num1 = smnd.calc_det2x2(dx, second_e0)
    num2 = smnd.calc_det2x2(first_e1, dx)
    dnm  = smnd.calc_det2x2(first_e1, second_e0)
    assert (dnm >= smnd.EPS_m_sqr)

    rcp_dnm = 1 / dnm
    L1 = num1 * rcp_dnm
    L2 = num2 * rcp_dnm

    while len(first_KHL) > 0 and len(second_KHL) > 0:
        discard_first_end    = np.dot(first_NL[-1] - first_KHL[-1][1].posn, first_e1) >= L1
        discard_second_start = np.dot(second_KHL[0][1].posn - second_NL[0], second_e0) <= L2
        if not discard_first_end and not discard_second_start:
            break

        if discard_first_end:
            first_NL  = first_NL[:-1]
            first_KHL = first_KHL[:-1]

        if discard_second_start:
            second_NL  = second_NL[1:]
            second_KHL = second_KHL[1:]

        dx = second_KHL[0][1].posn - high_KHL[-1][1].posn
        num1 = smnd.calc_det2x2(dx, second_e0)
        num2 = smnd.calc_det2x2(first_e1, dx)
        dnm  = smnd.calc_det2x2(first_e1, second_e0)
        assert (dnm >= smnd.EPS_m_sqr)

        rcp_dnm = 1 / dnm
        L1 = num1 * rcp_dnm
        L2 = num2 * rcp_dnm

    join_node = None
    if len(first_KHL) > 0 and len(second_KHL) > 0:
        join_node = first_KHL[-1][1].posn + L1 * first_e0
        return join_node, first_KHL,  first_NL, second_KHL, second_NL
    else:
        return None

def DEPRECATED_merge_halfpolygons(upper_half, lower_half):
    lohi_joininfo = join_halfpolygons(lower_half, upper_half)
    if lohi_joininfo == None:
        return []
    hilo_joininfo = join_halfpolygons(lohi_joininfo[2], lohi_joininfo[1])
    if hilo_joininfo == None:
        return []

    return hilo_joininfo[2] + [lohi_joininfo[0]] + hilo_joininfo[1] + [hilo_joininfo[0]]

def calc_rotary_key(nrml):
    c = nrml[0]
    s = nrml[1]
    v = 1 - c
    if (s + 1)**2 < smnd.EPS_m_sqr:
        return -2
    elif s < 0:
        return -v
    else:
        return v


def DEPRECATED_construct_polygon(halfplanes):
    keyed_upper_halfplanes = []
    keyed_lower_halfplanes = []
    for halfplane in halfplanes:
        key = calc_rotary_key(halfplane.normal)
        if key < 0:
            keyed_upper_halfplanes.append((key, halfplane))
        else:
            keyed_lower_halfplanes.append((key, halfplane))

    upper_half = construct_halfpolygon(keyed_upper_halfplanes)
    assert(len(upper_half[0]) > 0)

    lower_half = construct_halfpolygon(keyed_lower_halfplanes)
    assert(len(lower_half[0]) > 0)

    pts = merge_halfpolygons(upper_half, lower_half)
    return SMGS.Polygon(pts)
