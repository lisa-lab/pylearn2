from scipy.linalg import cholesky, det, solve
import numpy as N
import theano.tensor as T
from theano import config, shared
import theano.sandbox.rng_mrg
RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams
floatX = config.floatX


class MND(object):
    def __init__(self, sigma, mu, seed=42):
        self.sigma = sigma
        self.mu = mu
        if not (len(mu.shape) == 1):
            raise Exception('mu has shape ' + str(mu.shape) +
                            ' (it should be a vector)')

        self.sigma_inv = solve(self.sigma, N.identity(mu.shape[0]),
                               sym_pos=True)
        self.L = cholesky(self.sigma)

        self.s_rng = RandomStreams(42)

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
                              avg=0., std=1., dtype=floatX)
        return self.mu + T.dot(Z, self.L.T)


def fit(dataset, n_samples=None):
    if n_samples is not None:
        X = dataset.get_batch_design(n_samples)
    else:
        X = dataset.get_design_matrix()

    return MND(sigma=N.cov(X.T), mu=X.mean(axis=0))
