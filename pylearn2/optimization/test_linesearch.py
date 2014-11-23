from __future__ import print_function

import time
import warnings

import theano
import theano.tensor as TT
import numpy
from theano.compat.six.moves import xrange
from .linesearch import scalar_armijo_search
from .linesearch import scalar_search_wolfe2


def line_search_armijo(ftemp, derphi0, old_fval, args=(), c1=1e-4, alpha0=1,
                      minAlpha=1e-20):
    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        return ftemp(alpha1, *args)

    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval  # compute f(xk) -- done in past loop

    alpha, phi1 = scalar_search_armijo(phi,
                                       phi0,
                                       derphi0,
                                       c1=c1,
                                       alpha0=alpha0,
                                       minAlpha=minAlpha)
    return alpha, fc[0], phi1


def scalar_search_armijo(phi,
                         phi0,
                         derphi0,
                         c1=1e-4,
                         alpha0=1,
                         amin=0,
                         minAlpha=1e-20):
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return alpha0, phi_a0

    # Otherwise compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0 ** 2 / 2.0 / \
            (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)
    if (phi_a1 <= phi0 + c1 * alpha1 * derphi0):
        return alpha1, phi_a1

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satifies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0 ** 2 * alpha1 ** 2 * (alpha1 - alpha0)
        a = alpha0 ** 2 * (phi_a1 - phi0 - derphi0 * alpha1) - \
            alpha1 ** 2 * (phi_a0 - phi0 - derphi0 * alpha0)
        a = a / factor
        b = -alpha0 ** 3 * (phi_a1 - phi0 - derphi0 * alpha1) + \
            alpha1 ** 3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor

        alpha2 = (-b + numpy.sqrt(abs(b ** 2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2)

        if (phi_a2 <= phi0 + c1 * alpha2 * derphi0):
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
            alpha2 = alpha1 / 2.0
        if alpha2 < minAlpha:
            return alpha2, phi_a2

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1


def test():
    ## TEST ME
    def random_tensor(name, *args):
        return theano.shared(
            (numpy.random.uniform(size=args) * 1e-1).astype(
                theano.config.floatX), name=name)
    W = random_tensor('W', 784, 500)
    v = random_tensor('v', 40, 784)
    y = random_tensor('y', 40, 500)

    gv = TT.grad(((TT.tanh(TT.dot(v, W)) - y) ** 2).sum(), v)

    def phi(x):
        return ((TT.tanh(TT.dot(v - gv * x, W)) - y) ** 2).sum()

    def derphi(x):
        return TT.grad(((TT.tanh(TT.dot(v - gv * x, W)) - y) ** 2).sum(), x)

    x = TT.scalar('x')
    func = theano.function([x],
                           phi(x),
                           name='func',
                           profile=0,
                           mode=theano.Mode(linker='cvm'),
                           allow_input_downcast=True,
                           on_unused_input='ignore')
    grad = theano.function([x],
                           TT.sum(-gv * v),
                           allow_input_downcast=True,
                           name='grad',
                           profile=0,
                           mode=theano.Mode(linker='cvm'),
                           on_unused_input='ignore')
    phi0 = theano.shared(numpy.asarray(func(0),
                                       dtype=theano.config.floatX),
                         name='phi0')
    derphi0 = theano.shared(numpy.asarray(grad(0),
                                          dtype=theano.config.floatX),
                                  name='derphi0')

    outs = scalar_armijo_search(phi, phi0, derphi0, profile=0)

    f = theano.function([],
                        outs,
                        profile=0,
                        name='test_scalar_search',
                        mode=theano.Mode(linker='cvm'))

    rvals = scalar_search_wolfe2(phi,
                                 derphi,
                                 phi0,
                                 derphi0,
                                 profile=0)

    f2 = theano.function([],
                         rvals,
                         profile=0,
                         name='test_wolfe',
                         mode=theano.Mode(linker='cvm'))

    t_py = 0
    f0 = func(0)
    g0 = grad(0)
    for k in xrange(10):
        t0 = time.time()
        rval = scalar_search_armijo(func, f0, g0)
        t_py += time.time() - t0

    t_th = 0
    thrval = []
    for k in xrange(10):
        t0 = time.time()
        thrval = f()
        t_th += time.time() - t0

    t_th2 = 0
    for k in xrange(10):
        t0 = time.time()
        thrval2 = f2()
        t_th2 += time.time() - t0

    print('THEANO (armijo) output :: ', thrval)
    print('THEANO (wolfe)  output :: ', thrval2)
    print('NUMPY  (armijo) output :: ', rval)
    print()
    print('Timings')
    print()
    print('theano (armijo)---------> time %e' % t_th)
    print('theano (wolfe) ---------> time %e' % t_th2)
    print('numpy  (armijo)---------> time %e' % t_py)


if __name__ == '__main__':
    test()
