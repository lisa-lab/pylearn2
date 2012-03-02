""" Expressions for encoding features """

import theano.tensor as T

def triangle_code(X, centroids):
    """ Compute the triangle activation function used
        in Adam Coates' AISTATS 2011 paper

        X: a design matrix
        centroids: a k-means dictionary, one centroid in each row

        Returns a design matrix of triangle code activations
    """

    X_sqr = T.sqr(X).sum(axis=1)
    c_sqr = T.sqr(centroids).sum(axis=1)
    Xc = T.dot(X, centroids)

    Z = T.sqrt( c_sqr + X_sqr - 2. * Xc)

    mu = Z.mean(axis=1)

    mu = mu.dimshuffle(0,'x')

    rval = T.clip( mu - Z, 0., 1e30)

    return rval
