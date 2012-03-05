""" Expressions for encoding features """

import theano.tensor as T

def triangle_code(X, centroids):
    """ Compute the triangle activation function used
        in Adam Coates' AISTATS 2011 paper

        X: a design matrix
        centroids: a k-means dictionary, one centroid in each row

        Returns a design matrix of triangle code activations
    """

    X_sqr = T.sqr(X).sum(axis=1).dimshuffle(0,'x')
    c_sqr = T.sqr(centroids).sum(axis=1).dimshuffle('x',0)
    c_sqr.name = 'c_sqr'
    Xc = T.dot(X, centroids.T)
    Xc.name = 'Xc'

    Z = T.sqrt( c_sqr + X_sqr - 2. * Xc)
    Z.name = 'Z'

    mu = Z.mean(axis=1)
    mu.name = 'mu'

    mu = mu.dimshuffle(0,'x')
    mu.name = 'mu_broadcasted'

    rval = T.clip( mu - Z, 0., 1e30)
    rval.name = 'triangle_code'

    return rval
