import theano
import theano.tensor as TT
import numpy
import minres

from collections import OrderedDict

def test_1():
    n = 100
    on = numpy.ones((n, 1), dtype=theano.config.floatX)
    A = numpy.zeros((n, n), dtype=theano.config.floatX)
    for k in xrange(n):
        A[k, k] = 4.
        if k > 0:
            A[k - 1, k] = -2.
            A[k, k - 1] = -2.
    b = A.sum(axis=1)
    rtol = numpy.asarray(1e-10, dtype=theano.config.floatX)
    maxit = 50
    M = numpy.ones((n,), dtype=theano.config.floatX) * 4.
    tA = theano.shared(A.astype(theano.config.floatX))
    tb = theano.shared(b.astype(theano.config.floatX))
    tM = theano.shared(M.astype(theano.config.floatX))
    compute_Av = lambda x: ([TT.dot(tA, x)], OrderedDict())
    xs, flag, iters, relres, relAres, Anorm, Acond, xnorm, Axnorm, updates = \
            minres.minres(compute_Av,
                   [tb],
                   rtol=rtol,
                   maxit=maxit,
                   Ms=[tM],
                   profile=0)

    func = theano.function([],
                           xs + [flag, iters, relres, relAres, Anorm, Acond,
                                 xnorm, Axnorm],
                          name='func',
                          profile=0,
                          updates = updates,
                          mode=theano.Mode(linker='cvm'))
    rvals = func()
    print 'flag', rvals[1]
    print minres.messages[int(rvals[1])]
    print 'iters', rvals[2]
    print 'relres', rvals[3]
    print 'relAres', rvals[4]
    print 'Anorm', rvals[5]
    print 'Acond', rvals[6]
    print 'xnorm', rvals[7]
    print 'Axnorm', rvals[8]
    print 'error', numpy.sqrt(numpy.sum((numpy.dot(rvals[0], A) - b) ** 2))
    print


def test_2():
    h = 1
    a = -10
    b = -a
    n = 2 * b // h + 1
    A = numpy.zeros((n, n), dtype=theano.config.floatX)
    A = numpy.zeros((n, n), dtype=theano.config.floatX)
    v = a
    for k in xrange(n):
        A[k, k] = v
        v += h
    b = numpy.ones((n,), dtype=theano.config.floatX)
    rtol = numpy.asarray(1e-6, theano.config.floatX)
    maxxnorm = 1e8
    maxit = 50
    tA = theano.shared(A.astype(theano.config.floatX))
    tb = theano.shared(b.astype(theano.config.floatX))
    compute_Av = lambda x: ([TT.dot(tA, x)], OrderedDict())
    xs, flag, iters, relres, relAres, Anorm, Acond, xnorm, Axnorm, updates = \
            minres.minres(compute_Av,
                   [tb],
                   rtol=rtol,
                   maxit=maxit,
                   maxxnorm=maxxnorm,
                   profile=0)

    func = theano.function([],
                           xs + [flag, iters, relres, relAres, Anorm, Acond,
                                 xnorm, Axnorm],
                          name='func',
                          profile=0,
                          updates = updates,
                          mode=theano.Mode(linker='cvm'))
    rvals = func()
    print 'flag', rvals[1]
    print minres.messages[int(rvals[1])]
    print 'iters', rvals[2]
    print 'relres', rvals[3]
    print 'relAres', rvals[4]
    print 'Anorm', rvals[5]
    print 'Acond', rvals[6]
    print 'xnorm', rvals[7]
    print 'Axnorm', rvals[8]
    print rvals[0]

if __name__ == '__main__':
    test_1()
    test_2()
