#
# Shape class and two simple subclasses, Circle and Segment
#
# @todo: Add a composite subclass, Polysegment, comprising a set of small circles and intermediate segments.
#

import numpy as np
import multimodal_network.view as mnv
import multimodal_network.agent as mna

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

    def get_closest_point_info(self, posn):
        s_omega = posn - self.posn

        nv = mna.calc_unit_vector(s_omega)
        tv = np.array([-nv[1], nv[0]])
        x_omega = self.posn

        return (x_omega, nv, tv)

    def draw(self, colour, view):
        # print("Circle::draw", self.posn, self.radius)
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

        delta = self.posn2 - self.posn1

        self.length = np.sqrt(np.dot(delta, delta))

        self.tv = delta / self.length
        self.nv = np.array([-self.tv[1], self.tv[0]])

        return

    def get_closest_point_info(self, posn):
        x_omega = posn + np.dot(posn - self.posn1, self.tv) * self.tv
        return (x_omega, self.nv, self.tv)
    
    def draw(self, colour, view):
        # print("Segment::draw", self.posn1, self.posn2, self.radius)
        view.draw_line(colour, self.posn1, self.posn2, self.radius)
        return

class Polygon(Shape):
    def __init__(self, pts, radius):
        super(Polygon, self).__init__()

        self.pts    = pts
        self.radius = radius

        return

    #
    # Polygon is expected to be convex and clockwise.
    #
    def validate(self):
        for i in range(len(pts) - 2):
            p0 = self.pts[i]
            p1 = self.pts[i + 1]
            p2 = self.pts[i + 2]
            u  = p1 - p0
            v  = p2 - p1
            assert(u[0] * v[1] - u[1] - v[0] > 0)
            
    def get_closest_point_info(self, posn):
        nearest_pts = []
        for i in range(len(self.pts)):
            delta = posn - self.pts[i]
            item  = (i, np.dot(delta, delta))
            nearest_pts.append(item)
        nearest_pts.sort(key = lambda item: item[1])

        n        = len(self.pts)
        idx      = nearest_pts[0][0]
        prev_idx = (idx + n - 1) % n
        next_idx = (idx + 1)     % n
        
        u = posn - self.pts[prev_idx]
        v = posn - self.pts[idx]
        w0 = self.pts[idx]      - self.pts[prev_idx]
        w1 = self.pts[next_idx] - self.pts[idx]

        l0 = mna.calc_norm(w0)
        l1 = mna.calc_norm(w1)
        e0 = w0 / l0
        e1 = w1 / l1

        # if u[0] * v[1] - u[1] - v[0] > 0:
            # print("Polygon::get_closest_point_info - {} is inside {}", posn, self.pts)
            # return None
        s0 = np.dot(u, e0) / l0
        s1 = np.dot(u, e1) / l1

        assert(s0 >= 0.25) # Should be >= 0.5, but mollification weakens such certainties.
        assert(s1 <= 0.75) # Should be <= 0.5, but mollification weakens such certainties.

        if s0 >= 1. and s1 <= 0.:
            nv = mna.calc_unit_vector(v)
            tv = np.array([-nv[1], nv[0]])
            x_omega = self.pts[idx]
        elif s1 <= 0.: # and s0 < 1.: ??
            x_omega = self.pts[prev_idx] + s0 * e0
            tv      = e0
            nv      = np.array([tv[1], -tv[0]])
        elif s0 >= 1.: # and s1 > 0.: ??
            x_omega = self.pts[idx] + s1 * e1
            tv      = e1
            nv      = np.array([tv[1], -tv[0]])
            
        else:
            return None
            #
            # @todo: Consider returning a small array of points so that
            # concave polygons and the inside of convex polygons can also be
            # supported.
            #
            # x1_omega = self_pts[prev_idx] + s0 * e0 
            # x2_omega = self_pts[idx] + s1 * e1
        
        return (x_omega, nv, tv)

    def draw(self, colour, view):
        # print("Polygon::draw - pts {} with radius".format(self.pts, self.radius))
        view.stroke_polygon(colour, self.pts, self.radius)
        for posn in self.pts:
            view.fill_circle(colour, posn, self.radius)
        return
