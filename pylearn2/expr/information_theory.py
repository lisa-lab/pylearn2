"""
.. todo::

    WRITEME
"""
import theano.tensor as T
from theano.gof.op import get_debug_values
from theano.gof.op import debug_assert
import numpy as np
from theano.tensor.xlogx import xlogx
from pylearn2.utils import contains_nan, isfinite


def entropy_binary_vector(P):
    """
    .. todo::

        WRITEME properly

    If P[i,j] represents the probability of some binary random variable X[i,j]
    being 1, then rval[i] gives the entropy of the random vector X[i,:]
    """

    for Pv in get_debug_values(P):
        assert Pv.min() >= 0.0
        assert Pv.max() <= 1.0

    oneMinusP = 1. - P

    PlogP = xlogx(P)
    omPlogOmP = xlogx(oneMinusP)

    term1 = - T.sum(PlogP, axis=1)
    assert len(term1.type.broadcastable) == 1

    term2 = - T.sum(omPlogOmP, axis=1)
    assert len(term2.type.broadcastable) == 1

    rval = term1 + term2

    debug_vals = get_debug_values(PlogP, omPlogOmP, term1, term2, rval)
    for plp, olo, t1, t2, rv in debug_vals:
        debug_assert(isfinite(plp))
        debug_assert(isfinite(olo))

        debug_assert(not contains_nan(t1))
        debug_assert(not contains_nan(t2))
        debug_assert(not contains_nan(rv))

    return rval
