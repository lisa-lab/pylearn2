""" Expressions for encoding features """

import theano.tensor as T
from theano.printing import Print


def triangle_code(X, centroids):
    """
    .. todo::

        WRITEME properly

    Compute the triangle activation function used in Adam Coates' AISTATS 2011
    paper

    Parameters
    ----------
    X : WRITEME
        design matrix
    centroids : WRITEME
        k-means dictionary, one centroid in each row

    Returns
    -------
    WRITEME
        A design matrix of triangle code activations
    """

    X_sqr = T.sqr(X).sum(axis=1).dimshuffle(0,'x')
    c_sqr = T.sqr(centroids).sum(axis=1).dimshuffle('x',0)
    c_sqr.name = 'c_sqr'
    Xc = T.dot(X, centroids.T)
    Xc.name = 'Xc'

    sq_dists =  c_sqr + X_sqr - 2. * Xc

    #TODO: why do I have to do this and Adam doesn't?
    #is it just because he uses float64 and I usually use
    #float32? or are our libraries numerically unstable somehow,
    #or does matlab handle sqrt differently?
    sq_dists_safe = T.clip(sq_dists,0.,1e30)

    Z = T.sqrt( sq_dists_safe)
    Z.name = 'Z'

    mu = Z.mean(axis=1)
    mu.name = 'mu'

    mu = mu.dimshuffle(0,'x')
    mu.name = 'mu_broadcasted'

    rval = T.clip( mu - Z, 0., 1e30)
    rval.name = 'triangle_code'

    return rval


