"""
Subclasses of `BaseVAE` implementing visible-related methods.

The following methods need to be implemented by classes in this module:

* _initialize_parameters
* _sample_from_p_x_given_z
* _expectation_term
* _decode_theta
* _means_from_theta
* _log_p_x_given_z
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
from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.models.vae import BaseVAE
from pylearn2.utils.rng import make_theano_rng
from pylearn2.utils import wraps, sharedX

theano_rng = make_theano_rng(default_seed=1234125)
pi = sharedX(numpy.pi)


class BinaryVisible(BaseVAE):
    """
    Subclass implementing visible-related methods for binary inputs
    """
    @wraps(BaseVAE._initialize_parameters)
    def _initialize_parameters(self):
        # Encoder parameters
        W_mu_e_value = numpy.random.normal(loc=0, scale=self.isigma,
                                           size=(self.nenc, self.nhid))
        self.W_mu_e = sharedX(W_mu_e_value, name='W_mu_e')
        b_mu_e_value = numpy.random.normal(loc=0, scale=self.isigma,
                                           size=self.nhid)
        self.b_mu_e = sharedX(b_mu_e_value, name='b_mu_e')

        W_sigma_e_value = numpy.random.normal(loc=0, scale=self.isigma,
                                              size=(self.nenc, self.nhid))
        self.W_sigma_e = sharedX(W_sigma_e_value, name='W_sigma_e')
        b_sigma_e_value = numpy.random.normal(loc=0, scale=self.isigma,
                                              size=self.nhid)
        self.b_sigma_e = sharedX(b_sigma_e_value, name='b_sigma_e')

        # Decoder parameters
        W_mu_d_value = numpy.random.normal(loc=0, scale=self.isigma,
                                           size=(self.ndec, self.nvis))
        self.W_mu_d = sharedX(W_mu_d_value, name='W_mu_d')
        b_mu_d_value = numpy.random.normal(loc=0, scale=self.isigma,
                                           size=self.nvis)
        self.b_mu_d = sharedX(b_mu_d_value, name='b_mu_d')

        self._params.extend([self.W_mu_e, self.b_mu_e, self.W_sigma_e,
                                 self.b_sigma_e, self.W_mu_d, self.b_mu_d])
        self._encoding_parameters.extend([self.W_mu_e, self.b_mu_e,
                                          self.W_sigma_e, self.b_sigma_e])
        self._decoding_parameters.extend([self.W_mu_d, self.b_mu_d])

    @wraps(BaseVAE._sample_from_p_x_given_z)
    def _sample_from_p_x_given_z(self, num_samples, theta):
        (p_x_given_z,) = theta
        return theano_rng.uniform(
            size=(num_samples, self.nvis),
            dtype=theano.config.floatX
        ) < p_x_given_z

    @wraps(BaseVAE._expectation_term)
    def _expectation_term(self, X, theta):
        # We express the expectation term in terms of the pre-sigmoid
        # activations, which lets us apply the log sigmoid(x) -> -softplus(-x)
        # optimization manually; see `_decode_theta` for more details.
        (S,) = theta
        return -(X * T.nnet.softplus(-S) + (1 - X) * T.nnet.softplus(S))

    @wraps(BaseVAE._decode_theta)
    def _decode_theta(self, z):
        h_d = self._decoding_fprop(z)
        # We express theta in terms of the pre-sigmoid activation for numerical
        # stability reasons: this lets us easily replace log sigmoid(x) with
        # -softplus(-x). Allowing more than one sample per data point makes it
        # hard for Theano to automatically apply the optimization (because of
        # the reshapes involved in encoding / decoding), hence this workaround.
        S = (T.dot(h_d, self.W_mu_d) + self.b_mu_d)
        return (S,)

    @wraps(BaseVAE._means_from_theta)
    def _means_from_theta(self, theta):
        # Theta is composed of pre-sigmoid activations; see `_decode_theta` for
        # more details.
        return T.nnet.sigmoid(theta[0])

    @wraps(BaseVAE._log_p_x_given_z)
    def _log_p_x_given_z(self, X, theta):
        # We express the probability in terms of the pre-sigmoid activations,
        # which lets us apply the log sigmoid(x) -> -softplus(-x)
        # optimization manually; see `_decode_theta` for more details.
        (S,) = theta
        return -(
            X * T.nnet.softplus(-S) + (1 - X) * T.nnet.softplus(S)
        ).sum(axis=2)


class ContinuousVisible(BaseVAE):
    """
    Subclass implementing visible-related methods for real-valued inputs
    """
    @wraps(BaseVAE._initialize_parameters)
    def _initialize_parameters(self):
        # Encoder parameters
        W_mu_e_value = numpy.random.normal(loc=0, scale=self.isigma,
                                           size=(self.nenc, self.nhid))
        self.W_mu_e = sharedX(W_mu_e_value, name='W_mu_e')
        b_mu_e_value = numpy.random.normal(loc=0, scale=self.isigma,
                                           size=self.nhid)
        self.b_mu_e = sharedX(b_mu_e_value, name='b_mu_e')

        W_sigma_e_value = numpy.random.normal(loc=0, scale=self.isigma,
                                              size=(self.nenc, self.nhid))
        self.W_sigma_e = sharedX(W_sigma_e_value, name='W_sigma_e')
        b_sigma_e_value = numpy.random.normal(loc=0, scale=self.isigma,
                                              size=self.nhid)
        self.b_sigma_e = sharedX(b_sigma_e_value, name='b_sigma_e')

        # Decoder parameters
        W_mu_d_value = numpy.random.normal(loc=0, scale=self.isigma,
                                           size=(self.ndec, self.nvis))
        self.W_mu_d = sharedX(W_mu_d_value, name='W_mu_d')
        if self.data_mean is not None:
            b_mu_d_value = inverse_sigmoid_numpy(
                numpy.clip(self.data_mean.get_value(), 1e-7, 1-1e-7)
            )
        else:
            b_mu_d_value = numpy.random.normal(loc=0, scale=self.isigma,
                                               size=self.nvis)
        self.b_mu_d = sharedX(b_mu_d_value, name='b_mu_d')

        W_sigma_d_value = numpy.random.normal(loc=0, scale=self.isigma,
                                              size=(self.ndec, self.nvis))
        self.W_sigma_d = sharedX(W_sigma_d_value, name='W_sigma_d')
        b_sigma_d_value = numpy.random.normal(loc=0, scale=self.isigma,
                                              size=self.nvis)
        self.b_sigma_d = sharedX(b_sigma_d_value, name='b_sigma_d')
        self.log_sigma_d = sharedX(1.0, name='log_sigma_d')

        self._params.extend([self.W_mu_e, self.b_mu_e, self.W_sigma_e,
                                 self.b_sigma_e, self.W_mu_d, self.b_mu_d,
                                 self.W_sigma_d, self.b_sigma_d,
                                 self.log_sigma_d])
        self._encoding_parameters.extend([self.W_mu_e, self.b_mu_e,
                                          self.W_sigma_e, self.b_sigma_e])
        self._decoding_parameters.extend([self.W_mu_d, self.b_mu_d,
                                          self.W_sigma_d, self.b_sigma_d,
                                          self.log_sigma_d])

    @wraps(BaseVAE._sample_from_p_x_given_z)
    def _sample_from_p_x_given_z(self, num_samples, theta):
        (mu_d, log_sigma_d) = theta
        return theano_rng.normal(avg=mu_d,
                                 std=T.exp(log_sigma_d),
                                 dtype=theano.config.floatX)

    @wraps(BaseVAE._expectation_term)
    def _expectation_term(self, X, theta):
        (mu_d, log_sigma_d) = theta
        return -0.5 * (T.log(2 * pi) + 2 * log_sigma_d +
                       (X - mu_d) ** 2 / T.exp(2 * log_sigma_d))

    @wraps(BaseVAE._decode_theta)
    def _decode_theta(self, z):
        h_d = self._decoding_fprop(z)
        mu_d = T.nnet.sigmoid(T.dot(h_d, self.W_mu_d) + self.b_mu_d)
        log_sigma_d = T.ones_like(mu_d) * self.log_sigma_d
        return (mu_d, log_sigma_d)

    @wraps(BaseVAE._means_from_theta)
    def _means_from_theta(self, theta):
        return theta[0]

    @wraps(BaseVAE._log_p_x_given_z)
    def _log_p_x_given_z(self, X, theta):
        (mu_d, log_sigma_d) = theta
        return -0.5 * (T.log(2 * pi) + 2 * log_sigma_d +
                       (X - mu_d) ** 2 / T.exp(2 * log_sigma_d)).sum(axis=2)
