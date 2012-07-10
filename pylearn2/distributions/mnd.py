"""A Multivariate Normal Distribution."""
from scipy.linalg import cholesky, det, solve
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from pylearn2.utils import sharedX
import numpy as np
N = np


class MND(object):
    """A Multivariate Normal Distribution"""
    def __init__(self, sigma, mu, seed=42):
        """
        Parameters
        -----------
        sigma: a numpy ndarray of shape (n,n)
        mu: a numpy ndarray of shape (n,)
        seed: the seed for the theano random number generator used to sample from this distribution"""
        self.sigma = sigma
        self.mu = mu
        if not (len(mu.shape) == 1):
            raise Exception('mu has shape ' + str(mu.shape) +
                            ' (it should be a vector)')

        self.sigma_inv = solve(self.sigma, N.identity(mu.shape[0]),
                               sym_pos=True)
        self.L = cholesky(self.sigma)
        self.s_rng = RandomStreams(seed)

        #Compute logZ
        #log Z = log 1/( (2pi)^(-k/2) |sigma|^-1/2 )
        # = log 1 - log (2pi^)(-k/2) |sigma|^-1/2
        # = 0 - log (2pi)^(-k/2) - log |sigma|^-1/2
        # = (k/2) * log(2pi) + (1/2) * log |sigma|
        k = float(self.mu.shape[0])
        self.logZ = 0.5 * (k * N.log(2. * N.pi) + N.log(det(sigma)))

    def free_energy(self, X):
        #design matrix format
        return .5 * T.sum(T.dot(X - self.mu,
                                T.dot(self.sigma_inv,
                                      T.transpose(X - self.mu))))

    def log_prob(self, X):
        return - self.free_energy(X) - self.logZ

    def random_design_matrix(self, m):
        Z = self.s_rng.normal(size=(m, self.mu.shape[0]),
                              avg=0., std=1., dtype=config.floatX)
        return self.mu + T.dot(Z, self.L.T)


def fit(dataset, n_samples=None):
    """
    Returns an MND fit to n_samples drawn from dataset.

    Not a class method because we currently don't have a means
    of calling class methods from YAML files.
    """
    if n_samples is not None:
        X = dataset.get_batch_design(n_samples)
    else:
        X = dataset.get_design_matrix()
    return MND(sigma=N.cov(X.T), mu=X.mean(axis=0))
