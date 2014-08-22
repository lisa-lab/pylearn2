"""
Subclasses of `BaseVAE` implementing latent-related methods.

The following methods need to be implemented by classes in this module:

* _initialize_parameters
* _sample_from_p_z
* _sample_from_q_z_given_x
* _sample_from_epsilon
* _kl_divergence_term
* _encode_phi
* _log_q_z_given_x
* _log_p_z

The following methods are optionally implemented by classes in this module:

* _per_component_kl_divergence_term
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"
__email__ = "pylearn-dev@googlegroups"

import numpy
import theano
import theano.tensor as T
from pylearn2.models.vae import BaseVAE
from pylearn2.utils.rng import make_theano_rng
from pylearn2.utils import sharedX, wraps

theano_rng = make_theano_rng(default_seed=1234125)
pi = sharedX(numpy.pi)


class DiagonalGaussianPrior(BaseVAE):
    """
    Subclass implementing latent-related methods for a gaussian prior and a
    gaussian posterior with diagonal covariance matrices
    """
    @wraps(BaseVAE._initialize_parameters)
    def _initialize_parameters(self):
        self.prior_mu = sharedX(numpy.zeros(self.nhid), name="prior_mu")
        self.log_prior_sigma = sharedX(numpy.zeros(self.nhid),
                                       name="prior_log_sigma")
        self._decoding_parameters.extend([self.prior_mu, self.log_prior_sigma])
        self._params.extend([self.prior_mu, self.log_prior_sigma])

    @wraps(BaseVAE._sample_from_p_z)
    def _sample_from_p_z(self, num_samples):
        return theano_rng.normal(size=(num_samples, self.nhid),
                                 avg=self.prior_mu,
                                 std=T.exp(self.log_prior_sigma),
                                 dtype=theano.config.floatX)

    @wraps(BaseVAE._sample_from_q_z_given_x)
    def _sample_from_q_z_given_x(self, epsilon, phi):
        (mu_e, log_sigma_e) = phi
        if epsilon.ndim == 3:
            return (
                mu_e.dimshuffle('x', 0, 1) +
                T.exp(log_sigma_e.dimshuffle('x', 0, 1)) * epsilon
            )
        else:
            return mu_e + T.exp(log_sigma_e) * epsilon

    @wraps(BaseVAE._sample_from_epsilon)
    def _sample_from_epsilon(self, shape):
        return theano_rng.normal(size=shape, dtype=theano.config.floatX)

    @wraps(BaseVAE._kl_divergence_term)
    def _kl_divergence_term(self, X, phi):
        return self._per_component_kl_divergence_term(X, phi).sum(axis=1)

    @wraps(BaseVAE._per_component_kl_divergence_term)
    def _per_component_kl_divergence_term(self, X, phi):
        (mu_e, log_sigma_e) = phi
        log_prior_sigma = self.log_prior_sigma
        prior_mu = self.prior_mu
        return (
            log_prior_sigma - log_sigma_e +
            0.5 * (T.exp(2 * log_sigma_e) + (mu_e - prior_mu) ** 2) /
            T.exp(2 * log_prior_sigma) - 0.5
        )

    @wraps(BaseVAE._encode_phi)
    def _encode_phi(self, X):
        h_e = self._encoding_fprop(X)
        mu_e = T.dot(h_e, self.W_mu_e) + self.b_mu_e
        log_sigma_e = 0.5 * (T.dot(h_e, self.W_sigma_e) + self.b_sigma_e)
        return (mu_e, log_sigma_e)

    @wraps(BaseVAE._log_q_z_given_x)
    def _log_q_z_given_x(self, z, phi):
        (mu_e, log_sigma_e) = phi
        return -0.5 * (
            T.log(2 * pi) + 2 * log_sigma_e.dimshuffle(('x', 0, 1)) +
            (z - mu_e.dimshuffle(('x', 0, 1))) ** 2 /
            T.exp(2 * log_sigma_e.dimshuffle(('x', 0, 1)))
        ).sum(axis=2)

    @wraps(BaseVAE._log_p_z)
    def _log_p_z(self, z):
        return -0.5 * (
            T.log(2 * pi * T.exp(2 * self.log_prior_sigma)) +
            ((z - self.prior_mu) / self.prior_sigma) ** 2
        ).sum(axis=2)
