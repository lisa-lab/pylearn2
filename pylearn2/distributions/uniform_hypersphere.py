"""
.. todo::

    WRITEME
"""
uthors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
import numpy as N
import theano.tensor as T
from theano import config
from scipy.special import gammaln
from pylearn2.utils.rng import make_theano_rng


class UniformHypersphere(object):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, dim, radius):
        self.dim = dim
        self.radius = radius
        self.s_rng = make_theano_rng(None, 42, which_method='normal')
        log_C = ((float(self.dim) / 2.) * N.log(N.pi) -
                 gammaln(1. + float(self.dim) / 2.))
        self.logZ = N.log(self.dim) + log_C + (self.dim - 1) * N.log(radius)
        assert not N.isnan(self.logZ)
        assert not N.isinf(self.logZ)

    def free_energy(self, X):
        """
        .. todo::

            WRITEME properly

        Parameters
        ----------
        X : WRITEME
            Must contain only examples that lie on the hypersphere
        """
        #design matrix format

        return T.zeros_like(X[:, 0])

    def log_prob(self, X):
        """
        .. todo::

            WRITEME
        """
        return - self.free_energy(X) - self.logZ

    def random_design_matrix(self, m):
        """
        .. todo::

            WRITEME
        """
        Z = self.s_rng.normal(size=(m, self.dim),
                              avg=0., std=1., dtype=config.floatX)
        Z.name = 'UH.rdm.Z'
        sq_norm_Z = T.sum(T.sqr(Z), axis=1)
        sq_norm_Z.name = 'UH.rdm.sq_norm_Z'
        eps = 1e-6
        mask = sq_norm_Z < eps
        mask.name = 'UH.rdm.mask'
        Z = (Z.T * (1. - mask) + mask).T
        Z.name = 'UH.rdm.Z2'
        sq_norm_Z = sq_norm_Z * (1. - mask) + self.dim * mask
        sq_norm_Z.name = 'UH.rdm.sq_norm_Z2'
        norm_Z = T.sqrt(sq_norm_Z)
        norm_Z.name = 'UH.rdm.sq_norm_Z2'
        rval = self.radius * (Z.T / norm_Z).T
        rval.name = 'UH.rdm.rval'
        return rval
