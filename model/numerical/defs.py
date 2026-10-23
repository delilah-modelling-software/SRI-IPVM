#
# shared-spaces/model/numerical/defs.py
#
# @todo: 
#

import numpy as np
import model.configuration as smc

mollification = True


#
# Mollification
#
EPS_m_sqr = 0.0001

if mollification:
    def calc_norm_sqr(v):
        return np.dot(v, v) + EPS_m_sqr

    def calc_norm(v):
        return np.sqrt(np.dot(v, v) + EPS_m_sqr)

    def mollifier(a, a_max):
        #
        # Generalised Friedrichs' mollifier from Koester+ (2013)
        # with p = 2.
        #
        if a >= a_max:
            return 0.
        
        r = a / a_max
        r_pow = (r * r)**2
        m = np.exp(-r_pow / (1 - r_pow))
        return m
    
    def clip_magnitude(a, a_max):
        if a >= a_max:
            return a_max
        
        m = mollifier(a, a_max)
        # assert(m >= 0)
        # assert(m <= 1)
        return m * a + (1 - m) * a_max

    def clip_magnitude_softmax(a, a_softmax, a_hardmax):
        if a <= a_softmax:
            return a
        else:
            # m = mollifier(a - a_softmax, a_hardmax - a_softmax)
            return a_softmax + clip_magnitude(a - a_softmax, a_hardmax - a_softmax)
else:
    def calc_norm_sqr(v):
        return np.dot(v, v)

    def calc_norm(v):
        return np.sqrt(np.dot(v, v))

    def clip_magnitude(a, a_max):
        return min(a, a_max)

def clip_magnitude_asymptotic(a, a_softmax, a_asymptote):
    if a <= a_softmax:
        return a

    dmax = a_asymptote - a_softmax
    z = (a - a_softmax) / dmax
    return a_softmax + dmax * np.tanh(z)

def calc_unit_vector(v):
    return v / calc_norm(v)

def rotate_90ACW(v):
    return np.array((-v[1], v[0]))

def rotate_csACW(v, c, s):
    assert((c**2 + s**2 - 1) < EPS_m_sqr)
    perp_v = rotate_90ACW(v)
    
    return c * v + s * perp_v 

def calc_det2x2(u, v):
    #
    # Note: equivalent to -np.dot(u, rotate_90ACW(v))
    #
    return u[0] * v[1] - v[0] * u[1]

def calc_polar_decomp(v):
    #
    # Implementing the following in assembler/intrinsics
    # should be quicker, but that would not be a trivial task:
    #     dnm_sqr = v^2;
    #     inv_dnm = rsqrt(dnm_sqr);
    #     # Use an frsqrt instruction to get a starting estimate of inv_dnm.
    #     # Let a = dnm_sqr, x = inv_dnm
    #     # Execute up to two Newton iterations [x = 0.5x (1 + ax^2)]
    #     # Let inv_dnm = x
    #     (nrm, unit) = inv_dnm * (dnm_sqr, v)
    #
    nrm = calc_norm(v)
    unit = v / nrm
    return nrm, unit

def calc_cubic_spline_lengths(x):
    L_chord = calc_norm(x[3] - x[0])
    L_poly  = calc_norm(x[1] - x[0]) + calc_norm(x[2] - x[1]) + calc_norm(x[3] - x[2])
    return L_chord, L_poly

#
# Use the de Casteljeau algorithm to subdivide the spline into two splines.
#
def subdivide_cubic_spline(x, t):
    # x0 = x[0]
    # x1 = x[1]
    # x2 = x[2]
    # x3 = x[3]

    # print("smnd.subdivide_cubic_spline will subdivide {} at t={}".format(x, t))

    mx = list()
    for i in range(3):
        dxi = x[i + 1] - x[i]
        mxi = x[i] + t * dxi
        mx.append(mxi)
    # print("Weighted averages {} at t={}".format(mx, t))

    nx = list()
    for i in range(2):
        dmxi = mx[i + 1] - mx[i]
        nxi = mx[i] + t * dmxi
        nx.append(nxi)
    # print("Weighted averages of weighted averages {} at t={}".format(nx, t))

    dnx = nx[1] - nx[0]
    ox = nx[0] + t * dnx
    # print("Join {} at t={}".format(ox, t))
    
    s0 = [x[0], mx[0], nx[0], ox]
    s1 = [ox, nx[1], mx[2], x[3]]
    # print("First spline {}".format(s0))
    # print("Second spline {}".format(s1))

    # assert(False)

    return s0, s1


    
    y0 = x0
    z3 = x3

    y1 = x0 + t * (x1 - x0)
    m  = x1 + t * (x2 - x1)
    z2 = x2 + t * (x3 - x2)

    y2 = y1 + t * (m - y1)
    z1 = m  + t * (z2 - m)
    
    y3 = z0 = y2 + t * (z1 - y2)
    return [y0, y1, y2, y3], [z0, z1, z2, z3]

def find_first_spline_intersection(spline1, spline2, l1, l2):
    dx1    = spline1[3] - spline1[0]
    L1, e1 = calc_polar_decomp(dx1)

    dx2    = spline2[3] - spline2[0]
    L2, e2 = calc_polar_decomp(dx2)

    det  = calc_det2x2(e1, e2)
    if det == 0:
        return None

    rcp_det = 1. / det

    det1 = calc_det2x2(spline2[0] - spline1[0], e2)
    det2 = calc_det2x2(e1, spline2[0] - spline1[0])

    lambda1 = det1 * rcp_det
    lambda2 = -det2 * rcp_det

    d1_safe = l1 + smc.d_safe_IA_maximum + EPS_m_sqr
    d2_safe = l2 + smc.d_safe_IA_maximum + EPS_m_sqr

    if lambda1 < 0. - d1_safe or lambda1 > L1 + d1_safe:
        return None

    if lambda2 < 0. - d2_safe or lambda2 > L2 + d2_safe:
        return None

    #
    #
    #
    # print("Spline {} and spline {} should have an intersection".format(spline1, spline2))

    e1_perp = rotate_90ACW(e1)
    e2_perp = rotate_90ACW(e2)

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
        s2A, s2B = subdivide_cubic_spline(spline2, 0.5)

        #
        # Look for solution in the low half of the parameter space.
        #
        res = find_first_spline_intersection(spline1, s2A, l1, l2)
        if not res is None:
            x, s1, s2, e1, e2 = res
            return x, s1, 0.5 * s2, e1, e2

        #
        # Look for solution in the high half of the parameter space.
        #
        res = find_first_spline_intersection(spline1, s2B, l1, l2)
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
        s1A, s1B = subdivide_cubic_spline(spline1, 0.5)

        #
        # Look for solution in the low half of the parameter space.
        #
        res = find_first_spline_intersection(s1A, spline2, l1, l2)
        if not res is None:
            x, s1, s2, e1, e2 = res
            return x, 0.5 * s1, s2, e1, e2

        #
        # Look for solution in the high half of the parameter space.
        #
        res = find_first_spline_intersection(s1B, spline2, l1, l2)
        if not res is None:
            x, s1, s2, e1, e2 = res
            return x, 0.5 * (1 + s1), s2, e1, e2

        #
        # If neither half contained an intersection, there isn't one.
        #
        return None

def construct_curve_line_difference(x_curve_pts, y_line_pts):
    z0 = x_curve_pts[0] - y_line_pts[0]
    z3 = x_curve_pts[3] - y_line_pts[1]

    dy = (y_line_pts[1] - y_line_pts[0]) / 3
    z1 = x_curve_pts[1] - (y_line_pts[0] + dy)
    z2 = x_curve_pts[2] - (y_line_pts[1] - dy)
    
    return [z0, z1, z2, z3]

#
# Use the de Casteljeau algorithm to determine the value of the spline at a point.
#
def evaluate_cubic_spline(x, t):
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]

    m0 = x0 + t * (x1 - x0)
    m1 = x1 + t * (x2 - x1)
    m2 = x2 + t * (x3 - x2)

    n0 = m0 + t * (m1 - m0)
    n1 = m1 + t * (m2 - m1)

    return n0 + t * (n1 - n0)
    
#
# Gravesen algorithm for arclength, truncated at one subdivision.
#
def calc_spline_arclength(x):
    assert(len(x) == 4)

    L0_chord, L0_poly = calc_cubic_spline_lengths(x)
    L0 =  0.5 * (L0_chord + L0_poly)
    maxerr0 = 0.5 * (L0_poly - L0_chord)

    y, z                = subdivide_cubic_spline(x, 0.5)
    L1a_chord, L1a_poly = calc_cubic_spline_lengths(y)
    L1b_chord, L1b_poly = calc_cubic_spline_lengths(z)
    L1_chord            = L1a_chord + L1b_chord
    L1_poly             = L1a_poly  + L1b_poly

    L1 = 0.5 * (L1_chord + L1_poly)

    dL = (L1 - L0) / 15

    L = L1 - dL

    return L


def analyse_cubic_spline(x, omega_max, spd_max, T_curve):
    L0_chord, L0_poly = calc_cubic_spline_lengths(x)

    L0 =  0.5 * (L0_chord + L0_poly)
    E0 = 6   * (L0_poly - L0_chord) / L0_chord**2
    maxerr0 = 0.5 * (L0_poly - L0_chord)
    assert(maxerr0 >= 0)

    print("Spline has control points {}, {}, {}, and {}".format(x[0], x[1], x[2], x[3]))
    print("Spline has approx. arc-length 0.5 * ({} + {}) ".format(L0_chord, L0_poly))
    print("Spline has approx. arc-length {} and curvature energy {}".format(L0, E0))

    y, z                = subdivide_cubic_spline(x, 0.5)
    L1a_chord, L1a_poly = calc_cubic_spline_lengths(y)
    L1b_chord, L1b_poly = calc_cubic_spline_lengths(z)
    L1_chord            = L1a_chord + L1b_chord
    L1_poly             = L1a_poly  + L1b_poly

    print("Spline has approx. arc-length 0.5 * [({} + {}) + ({} + {})] ".format(L1a_chord, L1a_poly, L1b_chord, L1b_poly))
    print("Hence, spline has approx. arc-length 0.5 * ({} + {}) ".format(L1_chord, L1_poly))

    L1 = 0.5 * (L1_chord + L1_poly)

    E1a = 6 * (L1a_poly - L1a_chord) / L1a_chord**2
    E1b = 6 * (L1b_poly - L1b_chord) / L1b_chord**2
    E1 = E1a + E1b

    print("Spline has improved approx. arc-length 0.5 * ({} + {}) ".format(L1_chord, L1_poly))
    print("Spline has improved approx. arc-length {} and curvature energy {}".format(L1, E1))

    #
    # Now we use a little Richardson Extrapolation, as per Gravesen (1997),
    # to get the best possible estimates of the arc-length and energy of the
    # spline.
    #
    dL = (L1 - L0) / 15
    dE = (E1 - E0) /  3

    L = L1 - dL
    E = E1 - dE

    print("Spline has refined approx. arc-length {} and curvature energy {}".format(L, E))
    # assert(False)

    #
    # @todo: Now score it!
    #
    T_moving = L / spd_max
    # T_curving = spd_max * np.sqrt(E) / omega_max**2
    # T_score = max(T_moving, T_curving)
    T_score = T_moving
    
    # if E > omega_max**2 * T_score / spd_max:
        # T_curving = spd_max * np.sqrt(E) / omega_max**2
        # assert(T_curving > T_score)
        # T_score = max(T_moving, T_curving)

    print("SMND.analyse_cubic_spline {} has arc-length {}, energy {}, T_moving {}, T_curving {}, T_score {}".format(x, L, E, T_moving, T_curve, T_score))

    return E, L, T_score

#
# Use the de Casteljeau algorithm to determine the value of the spline at a point.
#
def evaluate_quadratic_spline(x, t):
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]

    n0 = x0 + t * (x1 - x0)
    n1 = x1 + t * (x2 - x1)

    return n0 + t * (n1 - n0)

def evaluate_line(x, t):
    x0 = x[0]
    x1 = x[1]

    return x0 + t * (x1 - x0)

