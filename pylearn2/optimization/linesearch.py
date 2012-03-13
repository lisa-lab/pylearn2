"""
Note: this code is a Theano translation of the linesearch implemented in
scipy.optimize.linesearch

See :
    https://github.com/scipy/scipy/blob/master/scipy/optimize/linesearch.py
"""

import theano
import theano.tensor as TT
from theano.ifelse import ifelse
from theano.sandbox.scan import scan
import numpy
import time

one = TT.constant(numpy.asarray(1, dtype=theano.config.floatX))
zero = TT.constant(numpy.asarray(0, dtype=theano.config.floatX))
nan = TT.constant(numpy.asarray(numpy.nan, dtype=theano.config.floatX))

true = TT.constant(numpy.asarray(1, dtype='int8'))
false = TT.constant(numpy.asarray(0, dtype='int8'))


def lazy_or(name='none', *args):
    def apply_me(args):
        if len(args) == 1:
            return args[0]
        else:
            rval = ifelse(args[0], true, apply_me(args[1:]),
                          name=name + str(len(args)))
            return rval
    return apply_me(args)


def lazy_and(name='node', *args):
    def apply_me(args):
        if len(args) == 1:
            return args[0]
        else:
            rval = ifelse(TT.eq(args[0], zero), false, apply_me(args[1:]),
                         name=name + str(len(args)))
            return rval
    return apply_me(args)


def my_not(arg):
    return TT.eq(arg, zero)

def constant(value):
    return TT.constant(numpy.asarray(value, dtype=theano.config.floatX))

def scalar_armijo_search(phi, phi0, derphi0, c1=constant(1e-4),
                         n_iters=10, profile=0):
    alpha0 = one
    phi_a0 = phi(alpha0)
    alpha1 = -(derphi0) * alpha0 ** 2 / 2.0 /\
            (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    csol1 = phi_a0 <= phi0 + c1 * derphi0
    csol2 = phi_a1 <= phi0 + c1 * alpha1 * derphi0

    def armijo(alpha0, alpha1, phi_a0, phi_a1):
        factor = alpha0 ** 2 * alpha1 ** 2 * (alpha1 - alpha0)
        a = alpha0 ** 2 * (phi_a1 - phi0 - derphi0 * alpha1) - \
            alpha1 ** 2 * (phi_a0 - phi0 - derphi0 * alpha0)
        a = a / factor
        b = -alpha0 ** 3 * (phi_a1 - phi0 - derphi0 * alpha1) + \
            alpha1 ** 3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor

        alpha2 = (-b + TT.sqrt(abs(b ** 2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2)

        end_condition = phi_a2 <= phi0 + c1 * alpha2 * derphi0
        end_condition = TT.bitwise_or(
            TT.isnan(alpha2), end_condition)
        end_condition = TT.bitwise_or(
            TT.isinf(alpha2), end_condition)
        alpha2 = TT.switch(
            TT.bitwise_or(alpha1 - alpha2 > alpha1 / constant(2.),
                  one - alpha2 / alpha1 < 0.96),
            alpha1 / constant(2.),
            alpha2)
        return [alpha1, alpha2, phi_a1, phi_a2], \
                theano.scan_module.until(end_condition)

    states = []
    states += [TT.unbroadcast(TT.shape_padleft(alpha0), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(alpha1), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(phi_a0), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(phi_a1), 0)]
    # print 'armijo'
    rvals, _ = scan(
                armijo,
                states=states,
                n_steps=n_iters,
                name='armijo',
                mode=theano.Mode(linker='cvm'),
                profile=profile)

    sol_scan = rvals[1][0]
    a_opt = ifelse(csol1, one,
                ifelse(csol2, alpha1,
                    sol_scan))
    score = ifelse(csol1, phi_a0,
                   ifelse(csol2, phi_a1,
                          rvals[2][0]))
    return a_opt, score


def scalar_search_wolfe2(phi,
                         derphi,
                         phi0=None,
                         old_phi0=None,
                         derphi0=None,
                         n_iters=20,
                         c1=1e-4,
                         c2=0.9,
                        profile=False):
    """Find alpha that satisfies strong Wolfe conditions.

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
        phi : callable f(x)
            Objective scalar function.

        derphi : callable f'(x)
            Objective function derivative (can be None)
        phi0 : float, optional
            Value of phi at s=0
        old_phi0 : float, optional
            Value of phi at previous point
        derphi0 : float, optional
            Value of derphi at s=0
        c1 : float
            Parameter for Armijo condition rule.
        c2 : float
            Parameter for curvature condition rule.
        profile : flag (boolean)
            True if you want printouts of profiling information

    Returns
    -------
        alpha_star : float
            Best alpha
        phi_star
            phi at alpha_star
        phi0
            phi at 0
        derphi_star
            derphi at alpha_star

    Notes
    -----
        Uses the line search algorithm to enforce strong Wolfe
        conditions.  See Wright and Nocedal, 'Numerical Optimization',
        1999, pg. 59-60.

        For the zoom phase it uses an algorithm by [...].

    """

    if phi0 is None:
        phi0 = phi(zero)
    else:
        phi0 = phi0

    if derphi0 is None and derphi is not None:
        derphi0 = derphi(zero)
    else:
        derphi0 = derphi0

    alpha0 = zero
    alpha0.name = 'alpha0'
    if old_phi0 is not None:
        alpha1 = TT.minimum(one,
                            numpy.asarray(1.01, dtype=theano.config.floatX) *
                            numpy.asarray(2, dtype=theano.config.floatX) * \
                            (phi0 - old_phi0) / derphi0)
    else:
        old_phi0 = nan
        alpha1 = one

    alpha1 = TT.switch(alpha1 < zero, one, alpha1)
    alpha1.name = 'alpha1'

    # This shouldn't happen. Perhaps the increment has slipped below
    # machine precision?  For now, set the return variables skip the
    # useless while loop, and raise warnflag=2 due to possible imprecision.
    phi0 = TT.switch(TT.eq(alpha1, zero), old_phi0, phi0)
    # I need a lazyif for alpha1 == 0 !!!
    phi_a1 = ifelse(TT.eq(alpha1, zero), phi0,
                    phi(alpha1), name='phi_a1')
    phi_a1.name = 'phi_a1'

    phi_a0 = phi0
    phi_a0.name = 'phi_a0'
    derphi_a0 = derphi0
    derphi_a0.name = 'derphi_a0'
    # Make sure variables are tensors otherwise strange things happen
    c1 = TT.as_tensor_variable(c1)
    c2 = TT.as_tensor_variable(c2)
    maxiter = n_iters

    def while_search(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, i_t,
                    alpha_star, phi_star, derphi_star):
        derphi_a1 = derphi(alpha1)
        cond1 = TT.bitwise_or(phi_a1 > phi0 + c1 * alpha1 * derphi0,
                              TT.bitwise_and(phi_a1 >= phi_a0, i_t > zero))
        cond2 = abs(derphi_a1) <= -c2 * derphi0
        cond3 = derphi_a1 >= zero
        alpha_star_c1, phi_star_c1, derphi_star_c1 = \
                _zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0,
                      phi, derphi, phi0, derphi0, c1, c2,
                     profile=profile)
        alpha_star_c3, phi_star_c3, derphi_star_c3 = \
                _zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi,
                      derphi, phi0, derphi0, c1, c2,
                     profile=profile)
        nw_alpha1 = alpha1 * numpy.asarray(2, dtype=theano.config.floatX)
        nw_phi = phi(nw_alpha1)
        alpha_star, phi_star, derphi_star = \
                ifelse(cond1,
                          (alpha_star_c1, phi_star_c1, derphi_star_c1),
                ifelse(cond2,
                          (alpha1, phi_a1, derphi_a1),
                ifelse(cond3,
                          (alpha_star_c3, phi_star_c3, derphi_star_c3),
                           (nw_alpha1, nw_phi, nan),
                      name='alphastar_c3'),
                      name='alphastar_c2'),
                      name='alphastar_c1')

        return ([alpha1,
                 nw_alpha1,
                 phi_a1,
                 ifelse(lazy_or('allconds',
                                cond1,
                                cond2,
                                cond3),
                        phi_a1,
                        nw_phi,
                        name='nwphi1'),
                 ifelse(cond1, derphi_a0, derphi_a1, name='derphi'),
                 i_t + one,
                 alpha_star,
                 phi_star,
                 derphi_star],
                theano.scan_module.scan_utils.until(
                    lazy_or('until_cond_',
                            TT.eq(nw_alpha1, zero),
                            cond1,
                            cond2,
                            cond3)))
    states = []
    states += [TT.unbroadcast(TT.shape_padleft(alpha0), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(alpha1), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(phi_a0), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(phi_a1), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(derphi_a0), 0)]
    # i_t
    states += [TT.unbroadcast(TT.shape_padleft(zero), 0)]
    # alpha_star
    states += [TT.unbroadcast(TT.shape_padleft(zero), 0)]
    # phi_star
    states += [TT.unbroadcast(TT.shape_padleft(zero), 0)]
    # derphi_star
    states += [TT.unbroadcast(TT.shape_padleft(zero), 0)]
    # print 'while_search'
    outs, updates = scan(while_search,
                         states=states,
                         n_steps=maxiter,
                         name='while_search',
                         mode=theano.Mode(linker='cvm_nogc'),
                         profile=profile)
    # print 'done_while_search'
    out3 = outs[-3][0]
    out2 = outs[-2][0]
    out1 = outs[-1][0]
    alpha_star, phi_star, derphi_star = \
            ifelse(TT.eq(alpha1, zero),
                        (nan, phi0, nan),
                        (out3, out2, out1), name='main_alphastar')
    return alpha_star, phi_star,  phi0, derphi_star


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

    If no minimizer can be found return None

    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
    a.name = 'a'
    fa.name = 'fa'
    fpa.name = 'fpa'
    fb.name = 'fb'
    fc.name = 'fc'
    C = fpa
    D = fa
    db = b - a
    dc = c - a

    denom = (db * dc) ** 2 * (db - dc)
    d1_00 = dc ** 2
    d1_01 = -db ** 2
    d1_10 = -dc ** 3
    d1_11 = db ** 3
    t1_0 = fb - fa - C * db
    t1_1 = fc - fa - C * dc
    A = d1_00 * t1_0 + d1_01 * t1_1
    B = d1_10 * t1_0 + d1_11 * t1_1
    A /= denom
    B /= denom
    radical = B * B - 3 * A * C
    radical.name = 'radical'
    db.name = 'db'
    dc.name = 'dc'
    b.name = 'b'
    c.name = 'c'
    A.name = 'A'
    #cond = TT.bitwise_or(radical < zero,
    #       TT.bitwise_or(TT.eq(db,zero),
    #       TT.bitwise_or(TT.eq(dc,zero),
    #       TT.bitwise_or(TT.eq(b, c),
    #                    TT.eq(A, zero)))))

    cond = lazy_or('cubicmin',
                   radical < zero,
                   TT.eq(db, zero),
                   TT.eq(dc, zero),
                   TT.eq(b, c),
                   TT.eq(A, zero))
    # Note: `lazy if` would make more sense, but it is not
    #       implemented in C right now
    xmin = TT.switch(cond, constant(numpy.nan),
                         a + (-B + TT.sqrt(radical)) / (3 * A))
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa,

    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    D = fa
    C = fpa
    db = b - a * one

    B = (fb - D - C * db) / (db * db)
    # Note : `lazy if` would make more sense, but it is not
    #        implemented in C right now
    # lazy_or('quadmin',TT.eq(db , zero), (B <= zero)),
    # xmin = TT.switch(TT.bitwise_or(TT.eq(db,zero), B <= zero),
    xmin = TT.switch(lazy_or(TT.eq(db, zero), B <= zero),
                     nan,
                     a - C /\
                     (numpy.asarray(2, dtype=theano.config.floatX) * B))
    return xmin


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2,
          n_iters=10,
          profile=False):
    """
    TODO: re-write me

    Part of the optimization algorithm in `scalar_search_wolfe2`.
    a_lo : scalar (step size)
    a_hi : scalar (step size)
    phi_lo : scalar (value of f at a_lo)
    phi_hi : scalar ( value of f at a_hi)
    derphi_lo : scalar ( value of derivative at a_lo)
    phi : callable -> generates computational graph
    derphi: callable -> generates computational graph
    phi0 : scalar ( value of f at 0)
    derphi0 : scalar (value of the derivative at 0)
    c1 : scalar  (wolfe parameter)
    c2 : scalar  (wolfe parameter)
    profile: if you want printouts of profiling information
    """
    # Function reprensenting the computations of one step of the while loop
    def while_zoom(phi_rec, a_rec, a_lo, a_hi, phi_hi,
                   phi_lo, derphi_lo, a_star, val_star, valprime):
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here.  Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection
        dalpha = a_hi - a_lo
        a = TT.switch(dalpha < zero, a_hi, a_lo)
        b = TT.switch(dalpha < zero, a_lo, a_hi)

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval) then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is stil too close to the
        # end points (or out of the interval) then use bisection

        # cubic interpolation
        cchk = delta1 * dalpha
        a_j_cubic = _cubicmin(a_lo, phi_lo, derphi_lo,
                              a_hi, phi_hi, a_rec, phi_rec)
        # quadric interpolation
        qchk = delta2 * dalpha
        a_j_quad = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
        cond_q = lazy_or('condq',
                         TT.isnan(a_j_quad),
                         a_j_quad > b - qchk,
                         a_j_quad < a + qchk)
        a_j_quad = TT.switch(cond_q, a_lo +
                             numpy.asarray(0.5, dtype=theano.config.floatX) * \
                             dalpha, a_j_quad)

        # pick between the two ..
        cond_c = lazy_or('condc',
                         TT.isnan(a_j_cubic),
                         TT.bitwise_or(a_j_cubic > b - cchk,
                                       a_j_cubic < a + cchk))
        # this lazy if actually decides if we need to run the quadric
        # interpolation
        a_j = TT.switch(cond_c, a_j_quad, a_j_cubic)
        #a_j = ifelse(cond_c, a_j_quad,  a_j_cubic)

        # Check new value of a_j
        phi_aj = phi(a_j)
        derphi_aj = derphi(a_j)

        stop = lazy_and('stop',
                        TT.bitwise_and(phi_aj <= phi0 + c1 * a_j * derphi0,
                                       phi_aj < phi_lo),
                        abs(derphi_aj) <= -c2 * derphi0)

        cond1 = TT.bitwise_or(phi_aj > phi0 + c1 * a_j * derphi0,
                              phi_aj >= phi_lo)
        cond2 = derphi_aj * (a_hi - a_lo) >= zero

        # Switches just make more sense here because they have a C
        # implementation and they get composed
        phi_rec = ifelse(cond1,
                         phi_hi,
                         TT.switch(cond2, phi_hi, phi_lo),
                         name='phi_rec')
        a_rec = ifelse(cond1,
                       a_hi,
                       TT.switch(cond2, a_hi, a_lo),
                         name='a_rec')
        a_hi = ifelse(cond1, a_j,
                      TT.switch(cond2, a_lo, a_hi),
                      name='a_hi')
        phi_hi = ifelse(cond1, phi_aj,
                        TT.switch(cond2, phi_lo, phi_hi),
                        name='phi_hi')

        a_lo = TT.switch(cond1, a_lo, a_j)
        phi_lo = TT.switch(cond1, phi_lo, phi_aj)
        derphi_lo = ifelse(cond1, derphi_lo, derphi_aj, name='derphi_lo')

        a_star = a_j
        val_star = phi_aj
        valprime = ifelse(cond1, nan,
                          TT.switch(cond2, derphi_aj, nan), name='valprime')

        return ([phi_rec,
                 a_rec,
                 a_lo,
                 a_hi,
                 phi_hi,
                 phi_lo,
                 derphi_lo,
                 a_star,
                 val_star,
                 valprime],
                theano.scan_module.scan_utils.until(stop))

    maxiter = n_iters
    # cubic interpolant check
    delta1 = TT.constant(numpy.asarray(0.2,
                                       dtype=theano.config.floatX))
    # quadratic interpolant check
    delta2 = TT.constant(numpy.asarray(0.1,
                                       dtype=theano.config.floatX))
    phi_rec = phi0
    a_rec = zero

    # Initial iteration

    dalpha = a_hi - a_lo
    a = TT.switch(dalpha < zero, a_hi, a_lo)
    b = TT.switch(dalpha < zero, a_lo, a_hi)
    #a = ifelse(dalpha < 0, a_hi, a_lo)
    #b = ifelse(dalpha < 0, a_lo, a_hi)

    # minimizer of cubic interpolant
    # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
    #
    # if the result is too close to the end points (or out of the
    # interval) then use quadratic interpolation with phi_lo,
    # derphi_lo and phi_hi if the result is stil too close to the
    # end points (or out of the interval) then use bisection

    # quadric interpolation
    qchk = delta2 * dalpha
    a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
    cond_q = lazy_or('mcond_q',
                     TT.isnan(a_j),
                     TT.bitwise_or(a_j > b - qchk,
                                   a_j < a + qchk))

    a_j = TT.switch(cond_q, a_lo +
                    numpy.asarray(0.5, dtype=theano.config.floatX) * \
                    dalpha, a_j)

    # Check new value of a_j
    phi_aj = phi(a_j)
    derphi_aj = derphi(a_j)

    cond1 = TT.bitwise_or(phi_aj > phi0 + c1 * a_j * derphi0,
                          phi_aj >= phi_lo)
    cond2 = derphi_aj * (a_hi - a_lo) >= zero

    # Switches just make more sense here because they have a C
    # implementation and they get composed
    phi_rec = ifelse(cond1,
                     phi_hi,
                     TT.switch(cond2, phi_hi, phi_lo),
                     name='mphirec')
    a_rec = ifelse(cond1,
                   a_hi,
                   TT.switch(cond2, a_hi, a_lo),
                   name='marec')
    a_hi = ifelse(cond1,
                  a_j,
                  TT.switch(cond2, a_lo, a_hi),
                  name='mahi')
    phi_hi = ifelse(cond1,
                    phi_aj,
                    TT.switch(cond2, phi_lo, phi_hi),
                    name='mphihi')

    onlyif = lazy_and('only_if',
                      TT.bitwise_and(phi_aj <= phi0 + c1 * a_j * derphi0,
                                     phi_aj < phi_lo),
                      abs(derphi_aj) <= -c2 * derphi0)

    a_lo = TT.switch(cond1, a_lo, a_j)
    phi_lo = TT.switch(cond1, phi_lo, phi_aj)
    derphi_lo = ifelse(cond1, derphi_lo, derphi_aj, name='derphi_lo_main')
    phi_rec.name = 'phi_rec'
    a_rec.name = 'a_rec'
    a_lo.name = 'a_lo'
    a_hi.name = 'a_hi'
    phi_hi.name = 'phi_hi'
    phi_lo.name = 'phi_lo'
    derphi_lo.name = 'derphi_lo'
    vderphi_aj = ifelse(cond1, nan, TT.switch(cond2, derphi_aj, nan),
                        name='vderphi_aj')
    states = []
    states += [TT.unbroadcast(TT.shape_padleft(phi_rec), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(a_rec), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(a_lo), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(a_hi), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(phi_hi), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(phi_lo), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(derphi_lo), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(zero), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(zero), 0)]
    states += [TT.unbroadcast(TT.shape_padleft(zero), 0)]
    # print'while_zoom'
    outs, updates = scan(while_zoom,
                         states=states,
                         n_steps=maxiter,
                         name='while_zoom',
                         mode=theano.Mode(linker='cvm_nogc'),
                         profile=profile)
    # print 'done_while'
    a_star = ifelse(onlyif, a_j, outs[7][0], name='astar')
    val_star = ifelse(onlyif, phi_aj, outs[8][0], name='valstar')
    valprime = ifelse(onlyif, vderphi_aj, outs[9][0], name='valprime')

    ## WARNING !! I ignore updates given by scan which I should not do !!!
    return a_star, val_star, valprime
