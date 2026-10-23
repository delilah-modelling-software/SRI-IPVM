#
# Shape class, two simple subclasses, Circle and Segment, and a composite subclass, Polygon, a closed shape comprising a set of small circles and intermediate segments
#
# @todo: Add a composite subclass, Polysegment, an open shape comprising a set of small circles and intermediate segments.
# @todo: Add an annular arc.
#

# import copy as cp
import numpy as np
import view.view as svv
# import model.force as smf
# import model.defs as smd
# import model.numerical.defs as smnd
import model.geometry.shape as smgs


class Union(smgs.Shape):
    def __init__(self, subshapes, outer_shape = None):
        super(Construction, self).__init__()

        self.inner_shapes = subshapes
        self.outer_shape  = outer_shape

        #
        # @todo: This is a na{\: i}ve implementation. Consider organising the
        # inner shapes into quadtree or n-d tree if this class's methods are
        # ever identified (preferably through profiling) as a performance issue.
        #

        if outer_shape == None:
            self.posn    = np.zeros(2)

            delta = np.array([1 << 15, 1 << 15])
            self.bbox = smgg.BoundingBox(delta, -delta)
            
            n_shapes     = len(self.inner_shapes)
            rcp_n_shapes = 1. / n
            self.bounded = True
            for i in range(n):
                cur_shape = self.inner_shapes[i]
                self.posn += rcp_n_shapes * cur_shape.posn

                self.bounded = self.bounded and cur_shape.bounded
                cur_bbox = cur_shape.get_bbox()
                if cur_bbox.posn1[0] < self.bbox.posn1[0]:
                    self.bbox.posn1[0] = cur_bbox.posn1[0]
                if cur_bbox.posn1[1] < self.bbox.posn1[1]:
                    self.bbox.posn1[1] = cur_bbox.posn1[1]

                if cur_bbox.posn2[0] > self.bbox.posn2[0]:
                    self.bbox.posn2[0] = cur_bbox.posn2[0]
                if cur_bbox.posn2[1] > self.bbox.posn2[1]:
                    self.bbox.posn2[1] = cur_bbox.posn2[1]
                
        else:
            self.posn    = outer_shape.posn
            self.bounded = outer_shape.bounded
            if self.bounded:
                self.bbox    = outer_shape.get_bbox()
        return
    
    def contains(self, posn):
        n_shapes     = len(self.inner_shapes)
        for i in range(n_shapes):
            cur_shape = self.inner_shapes[i]
            if cur_shape.contains(posn):
                return True
        return False

    def fill(self, colour, view):
        if self.outer_shape != None:
            self.outer_shape.fill(colour, view)
            colour += colour
        for i in range(len(self.inner_shapes)):
            cur_shape = self.inner_shapes[i]
            cur_shape.fill(colour, view)
        return
    
    def get_nearest_point_info(self, posn):
        n_shapes   = len(self.inner_shapes)
        d_sqr      = smnd.calc_norm_sqr(posn - self.posn)
        argmin_idx = -1
        for i in range(n_shapes):
            cur_shape       = self.inner_shapes[i]
            d_sqr_candidate = smnd.calc_norm_sqr
            if d_sqr_candidate < d_sqr:
                d_sqr      = d_sqr_candidate
                argmin_idx = i
                
        assert(argmin_idx >= 0)
        return get_nearest_point_info(self.inner_shapes[argmin_idx], posn)
    
    def translate(self, dx):
        for shape in self.inner_shapes:
            shape.translate(dx)
        if self.outer_shape != None:
            self.outer_shape.translate(dx)
        self.posn += dx
        return

    def calc_bbox(self):
        #
        # Returns: [bottom_left_point, top_right_point]
        #
        return self.bbox

    def get_nearest_approach_info(self, posn, vel, t_max):
        #
        # Returns: [shape_posn, agent_posn, d, normal, tangent, t]
        #
        return xxx # None
