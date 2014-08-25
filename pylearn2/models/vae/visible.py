"""
Classes implementing visible space-related methods for the VAE framework,
namely

* initialize_parameters
* sample_from_p_x_given_z
* expectation_term
* decode_theta
* means_from_theta
* log_p_x_given_z
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
from pylearn2.space import VectorSpace
from pylearn2.utils.rng import make_theano_rng
from pylearn2.utils import wraps, sharedX

theano_rng = make_theano_rng(default_seed=1234125)
pi = sharedX(numpy.pi)


class Visible(object):
    """
    Abstract class implementing visible space-related methods for the VAE
    framework.

    Parameteters
    ------------
    decoding_model : pylearn2.models.mlp.MLP
        An MLP representing the generative network, whose output will be used
        to compute :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})`
    """
    def __init__(self, decoding_model):
        self.decoding_model = decoding_model
        self.ndec = decoding_model.get_output_space().get_total_dimension()

    def initialize_parameters(self, decoder_input_space, nvis):
        """
        Initialize model parameters.
        """
        self.nvis = nvis
        self.decoder_input_space = decoder_input_space
        self.decoder_output_space = VectorSpace(dim=self.ndec)

    def get_params(self):
        """
        Return the visible space-related parameters
        """
        return self._params

    def fprop(self, z):
        """
        Wraps around the decoding model's `fprop` method to feed it data and
        return its output in the right spaces.

        Parameters
        ----------
        z : tensor_like
            Input to the decoding network
        """
        z = self.decoder_input_space.format_as(
            batch=z,
            space=self.decoding_model.get_input_space()
        )
        h_d = self.decoding_model.fprop(z)
        return self.decoding_model.get_output_space().format_as(
            batch=h_d,
            space=self.decoder_output_space
        )

    def decode_theta(self, z):
        """
        Maps latent variable `z` to a tuple of parameters of the
        :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})` distribution

        Parameters
        ----------
        z : tensor_like
            Latent sample

        Returns
        -------
        theta : tuple of tensor_like
            Tuple of parameters for the conditional distribution
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "decode_theta.")

    def means_from_theta(self, theta):
        """
        Given a tuple of parameters of the
        :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})` distribution,
        returns the expected value of `x`.

        Parameters
        ----------
        theta : tuple of tensor_like
            Tuple of parameters for the conditional distribution

        Returns
        -------
        means : tensor_like
            Expected value of `x`
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "means_from_theta.")

    def sample_from_p_x_given_z(self, num_samples, theta):
        """
        Given a tuple of parameters, samples from the
        :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})` conditional
        distribution

        Parameters
        ----------
        num_samples : int
            Number of samples
        theta : tuple of tensor_like
            Tuple of parameters for the conditional distribution

        Returns
        -------
        x : tensor_like
            Samples
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "sample_from_p_x_given_z.")

    def expectation_term(self, X, theta):
        """
        Computes an approximation of :math:`\\mathrm{E}_{q_\\phi(\\mathbf{z}
        \\mid \\mathbf{x})} [\\log p_\\theta(\\mathbf{x} \\mid \\mathbf{z})]`

        Parameters
        ----------
        X : tensor_like
            Input
        theta : tuple of tensor_like
            Tuple of parameters for the conditional distribution

        Returns
        -------
        expectation_term : tensor_like
            Expectation term
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "expectation_term.")

    def log_p_x_given_z(self, X, theta):
        """
        Computes the log-conditional probabilities of `X`

        Parameters
        ----------
        X : tensor_like
            Input
        theta : tuple of tensor_like
            Tuple of parameters for the contitional distribution

        Returns
        -------
        log_p_x_z : tensor_like
            Log-prior probabilities
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "log_p_x_given_z.")


class BinaryVisible(Visible):
    """
    Subclass implementing visible-related methods for binary inputs

    Parameters
    ----------
    decoding_model : pylearn2.models.mlp.MLP
        An MLP representing the generative network, whose output will be used
        to compute :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})`
    isigma : float, optional
        Standard deviation on the zero-mean distribution from which parameters
        initialized by the model itself will be drawn. Defaults to 0.01.
    """
    def __init__(self, decoding_model, isigma=0.01):
        super(BinaryVisible, self).__init__(decoding_model)
        self.isigma = isigma

    @wraps(Visible.initialize_parameters)
    def initialize_parameters(self, decoder_input_space, nvis):
        super(BinaryVisible, self).initialize_parameters(decoder_input_space,
                                                         nvis)

        W_mu_d_value = numpy.random.normal(loc=0, scale=self.isigma,
                                           size=(self.ndec, self.nvis))
        self.W_mu_d = sharedX(W_mu_d_value, name='W_mu_d')
        b_mu_d_value = numpy.random.normal(loc=0, scale=self.isigma,
                                           size=self.nvis)
        self.b_mu_d = sharedX(b_mu_d_value, name='b_mu_d')

        self._params = [self.W_mu_d, self.b_mu_d]

    @wraps(Visible.sample_from_p_x_given_z)
    def sample_from_p_x_given_z(self, num_samples, theta):
        (p_x_given_z,) = theta
        return theano_rng.uniform(
            size=(num_samples, self.nvis),
            dtype=theano.config.floatX
        ) < p_x_given_z

    @wraps(Visible.expectation_term)
    def expectation_term(self, X, theta):
        # We express the expectation term in terms of the pre-sigmoid
        # activations, which lets us apply the log sigmoid(x) -> -softplus(-x)
        # optimization manually; see `_decode_theta` for more details.
        (S,) = theta
        return -(X * T.nnet.softplus(-S) + (1 - X) * T.nnet.softplus(S))

    @wraps(Visible.decode_theta)
    def decode_theta(self, z):
        h_d = self.fprop(z)
        # We express theta in terms of the pre-sigmoid activation for numerical
        # stability reasons: this lets us easily replace log sigmoid(x) with
        # -softplus(-x). Allowing more than one sample per data point makes it
        # hard for Theano to automatically apply the optimization (because of
        # the reshapes involved in encoding / decoding), hence this workaround.
        S = (T.dot(h_d, self.W_mu_d) + self.b_mu_d)
        return (S,)

    @wraps(Visible.means_from_theta)
    def means_from_theta(self, theta):
        # Theta is composed of pre-sigmoid activations; see `_decode_theta` for
        # more details.
        return T.nnet.sigmoid(theta[0])

    @wraps(Visible.log_p_x_given_z)
    def log_p_x_given_z(self, X, theta):
        # We express the probability in terms of the pre-sigmoid activations,
        # which lets us apply the log sigmoid(x) -> -softplus(-x)
        # optimization manually; see `_decode_theta` for more details.
        (S,) = theta
        return -(
            X * T.nnet.softplus(-S) + (1 - X) * T.nnet.softplus(S)
        ).sum(axis=2)


class ContinuousVisible(Visible):
    """
    Subclass implementing visible-related methods for real-valued inputs

    Parameters
    ----------
    decoding_model : pylearn2.models.mlp.MLP
        An MLP representing the generative network, whose output will be used
        to compute :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})`
    isigma : float, optional
        Standard deviation on the zero-mean distribution from which parameters
        initialized by the model itself will be drawn. Defaults to 0.01.
    """
    def __init__(self, decoding_model, isigma=0.01):
        super(ContinuousVisible, self).__init__(decoding_model)
        self.isigma = isigma

    @wraps(Visible.initialize_parameters)
    def initialize_parameters(self, decoder_input_space, nvis):
        super(BinaryVisible, self).initialize_parameters(decoder_input_space,
                                                         nvis)
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

    @wraps(Visible.sample_from_p_x_given_z)
    def sample_from_p_x_given_z(self, num_samples, theta):
        (mu_d, log_sigma_d) = theta
        return theano_rng.normal(avg=mu_d,
                                 std=T.exp(log_sigma_d),
                                 dtype=theano.config.floatX)

    @wraps(Visible.expectation_term)
    def expectation_term(self, X, theta):
        (mu_d, log_sigma_d) = theta
        return -0.5 * (T.log(2 * pi) + 2 * log_sigma_d +
                       (X - mu_d) ** 2 / T.exp(2 * log_sigma_d))

    @wraps(Visible.decode_theta)
    def decode_theta(self, z):
        h_d = self.fprop(z)
        mu_d = T.nnet.sigmoid(T.dot(h_d, self.W_mu_d) + self.b_mu_d)
        log_sigma_d = T.ones_like(mu_d) * self.log_sigma_d
        return (mu_d, log_sigma_d)

    @wraps(Visible.means_from_theta)
    def means_from_theta(self, theta):
        return theta[0]

    @wraps(Visible.log_p_x_given_z)
    def log_p_x_given_z(self, X, theta):
        (mu_d, log_sigma_d) = theta
        return -0.5 * (T.log(2 * pi) + 2 * log_sigma_d +
                       (X - mu_d) ** 2 / T.exp(2 * log_sigma_d)).sum(axis=2)
