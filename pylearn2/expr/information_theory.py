import theano.tensor as T
from theano.gof.op import get_debug_values
from theano.gof.op import debug_assert
import numpy as np
from theano.tensor.xlogx import xlogx

def entropy_binary_vector(P):
    """
        if P[i,j] represents the probability
            of some binary random variable X[i,j] being 1
        then rval[i] gives the entropy of the random vector
        X[i,:]
    """

    for Pv in get_debug_values(P):
        assert Pv.min() >= 0.0
        assert Pv.max() <= 1.0

    oneMinusP = 1.-P

    PlogP = xlogx(P)
    omPlogOmP = xlogx(oneMinusP)

    term1 = - T.sum( PlogP , axis=1)
    assert len(term1.type.broadcastable) == 1

    term2 = - T.sum( omPlogOmP , axis =1 )
    assert len(term2.type.broadcastable) == 1

    rval = term1 + term2

    for plp, olo, t1, t2, rv in get_debug_values(PlogP, omPlogOmP, term1, term2, rval):
        debug_assert(not np.any(np.isnan(plp)))
        debug_assert(not np.any(np.isinf(olo)))
        debug_assert(not np.any(np.isnan(plp)))
        debug_assert(not np.any(np.isinf(olo)))

        debug_assert(not np.any(np.isnan(t1)))
        debug_assert(not np.any(np.isnan(t2)))
        debug_assert(not np.any(np.isnan(rv)))

    return rval
