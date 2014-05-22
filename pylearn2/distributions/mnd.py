"""A Multivariate Normal Distribution."""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
import warnings
try:
    from scipy.linalg import cholesky, det, solve
except ImportError:
    warnings.warn("Could not import some scipy.linalg functions")
import theano.tensor as T
from theano import config
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_theano_rng
import numpy as np
N = np


class MND(object):
    """
    A Multivariate Normal Distribution

    .. todo::

        WRITEME properly
    
    Parameters
    -----------
    sigma : WRITEME
        A numpy ndarray of shape (n,n)
    mu : WRITEME
        A numpy ndarray of shape (n,)
    seed : WRITEME
        The seed for the theano random number generator used to sample from
        this distribution
    """
    def __init__(self, sigma, mu, seed=42):
        self.sigma = sigma
        self.mu = mu
        if not (len(mu.shape) == 1):
            raise Exception('mu has shape ' + str(mu.shape) +
                            ' (it should be a vector)')

        self.sigma_inv = solve(self.sigma, N.identity(mu.shape[0]),
                               sym_pos=True)
        self.L = cholesky(self.sigma)

        self.s_rng = make_theano_rng(seed, which_method='normal')

        #Compute logZ
        #log Z = log 1/( (2pi)^(-k/2) |sigma|^-1/2 )
        # = log 1 - log (2pi^)(-k/2) |sigma|^-1/2
        # = 0 - log (2pi)^(-k/2) - log |sigma|^-1/2
        # = (k/2) * log(2pi) + (1/2) * log |sigma|
        k = float(self.mu.shape[0])
        self.logZ = 0.5 * (k * N.log(2. * N.pi) + N.log(det(sigma)))

    def free_energy(self, X):
        """
        .. todo::

            WRITEME
        """
        #design matrix format
        return .5 * T.sum(T.dot(X - self.mu,
                                T.dot(self.sigma_inv,
                                      T.transpose(X - self.mu))))

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
        Z = self.s_rng.normal(size=(m, self.mu.shape[0]),
                              avg=0., std=1., dtype=config.floatX)
        return self.mu + T.dot(Z, self.L.T)


def fit(dataset, n_samples=None):
    """
    .. todo::

        WRITEME properly
    
    Returns an MND fit to n_samples drawn from dataset.

    Not a class method because we currently don't have a means
    of calling class methods from YAML files.
    """
    if n_samples is not None:
        X = dataset.get_batch_design(n_samples)
    else:
        X = dataset.get_design_matrix()
    return MND(sigma=N.cov(X.T), mu=X.mean(axis=0))


class AdditiveDiagonalMND:
    """
    A conditional distribution that adds gaussian noise with diagonal precision
    matrix beta to another variable that it conditions on

    Parameters
    ----------
    init_beta : WRITEME
    nvis : WRITEME
    """

    def __init__(self, init_beta, nvis):
        self.__dict__.update(locals())
        del self.self

        self.beta = sharedX(np.ones((nvis,))*init_beta)
        assert self.beta.ndim == 1

        self.s_rng = make_theano_rng(None, 17, which_method='normal')

    def random_design_matrix(self, X):
        """
        .. todo::

            WRITEME properly

        Parameters
        ----------
        X : WRITEME
            A theano variable containing a design matrix of 
            observations of the random vector to condition on.
        """
        Z = self.s_rng.normal(size=X.shape,
                              avg=X, std=1./T.sqrt(self.beta), dtype=config.floatX)
        return Z

    def is_symmetric(self):
        """
        .. todo::

            WRITEME properly

        A property of conditional distributions
        P(Y|X)
        Return true if P(y|x) = P(x|y) for all x,y
        """

        return True
