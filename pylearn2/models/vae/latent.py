"""
Classes implementing latent space-related methods for the VAE framework, namely

* initialize_parameters
* sample_from_p_z
* sample_from_q_z_given_x
* sample_from_epsilon
* kl_divergence_term
* encode_phi
* log_q_z_given_x
* log_p_z

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
from theano.compat.python2x import OrderedDict
from pylearn2.utils.rng import make_theano_rng
from pylearn2.utils import sharedX, wraps
from pylearn2.space import VectorSpace, NullSpace

theano_rng = make_theano_rng(default_seed=1234125)
pi = sharedX(numpy.pi)


class Latent(object):
    """
    Abstract class implementing latent space-related methods for the VAE
    framework.

    Parameteters
    ------------
    encoding_model : pylearn2.models.mlp.MLP
        An MLP representing the recognition network, whose output will be used
        to compute :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`
    """
    def __init__(self, encoding_model):
        self.encoding_model = encoding_model
        self.nenc = encoding_model.get_output_space().get_total_dimension()

    def get_monitoring_data_specs(self):
        """
        Get the data_specs describing the data for get_monitoring_channels.

        By default, nothing is requested for computing monitoring channels.
        """
        return (NullSpace(), '')

    def get_monitoring_channels(self, data):
        """
        Get monitoring channels for this latent component.

        By default, no monitoring channel is computed.
        """
        space, source = self.get_monitoring_data_specs()
        space.validate(data)
        return OrderedDict()

    def modify_updates(self, updates):
        """
        Modifies the parameters before a learning update is applied.

        By default, does nothing.
        """
        pass

    def initialize_parameters(self, encoder_input_space, nhid):
        """
        Initialize model parameters.
        """
        self.nhid = nhid
        self.encoder_input_space = encoder_input_space
        self.encoder_output_space = VectorSpace(dim=self.nenc)

    def get_params(self):
        """
        Return the latent space-related parameters
        """
        return self._params

    def fprop(self, X):
        """
        Wraps around the encoding model's `fprop` method to feed it data and
        return its output in the right spaces.

        Parameters
        ----------
        X : tensor_like
            Input to the encoding network
        """
        X = self.encoder_input_space.format_as(
            batch=X,
            space=self.encoding_model.get_input_space()
        )
        h_e = self.encoding_model.fprop(X)
        return self.encoding_model.get_output_space().format_as(
            batch=h_e,
            space=self.encoder_output_space
        )

    def encode_phi(self, X):
        """
        Maps input `X` to a tuple of parameters of the
        :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})` posterior distribution

        Parameters
        ----------
        X : tensor_like
            Input

        Returns
        -------
        phi : tuple of tensor_like
            Tuple of parameters for the posterior distribution
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "encode_phi.")

    def sample_from_p_z(self, num_samples, **kwargs):
        """
        Samples from the prior distribution :math:`p_\\theta(\\mathbf{z})`

        Parameters
        ----------
        num_samples : int
            Number of samples

        Returns
        -------
        z : tensor_like
            Sample from the prior distribution
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "sample_from_p_z.")

    def sample_from_q_z_given_x(self, epsilon, phi):
        """
        Given a tuple of parameters and an epsilon noise sample, generates
        samples from the :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`
        posterior distribution using the reparametrization trick

        Parameters
        ----------
        epsilon : tensor_like
            Noise sample
        phi : tuple of tensor_like
            Tuple of parameters for the posterior distribution

        Returns
        -------
        z : tensor_like
            Posterior sample
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "sample_from_q_z_given_x.")

    def sample_from_epsilon(self, shape):
        """
        Samples from a canonical noise distribution from which posterior
        samples will be drawn using the reparametrization trick (see
        `_sample_from_q_z_given_x`)

        Parameters
        ----------
        shape : tuple of int
            Shape of the requested samples

        Returns
        -------
        epsilon : tensor_like
            Noise samples
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "sample_from_epsilon.")

    def kl_divergence_term(self, X, per_component=False, **kwargs):
        """
        Computes the KL-divergence term of the VAE criterion.

        Parameters
        ----------
        X : tensor_like
            Input
        per_component : bool, optional
            If the prior/posterior combination leads to a KL that's a sum of
            individual KLs across latent dimensions, the user can request the
            vector of indivitual KLs instead of the sum by setting this to
            `True`. Defaults to `False`.
        """
        phi = self.encode_phi(X)
        if per_component:
            return self._per_component_kl_divergence_term(X=X, phi=phi,
                                                          **kwargs)
        else:
            return self._kl_divergence_term(X=X, phi=phi, **kwargs)

    def _per_component_kl_divergence_term(self, X, phi, **kwargs):
        """
        If the prior/posterior combination allows it, analytically computes the
        per-latent-dimension KL divergences between the prior distribution
        :math:`p_\\theta(\\mathbf{z})` and :math:`q_\\phi(\\mathbf{z} \\mid
        \\mathbf{x})`

        Parameters
        ----------
        X : tensor_like
            Input
        phi : tuple of tensor_like
            Tuple of parameters for the posterior distribution

        Returns
        -------
        kl_divergence_term : tensor_like
            Per-component KL-divergence terms
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_per_component_kl_divergence_term.")

    def _kl_divergence_term(self, X, phi, **kwargs):
        """
        Analytically computes the KL divergence between the prior distribution
        :math:`p_\\theta(\\mathbf{z})` and :math:`q_\\phi(\\mathbf{z} \\mid
        \\mathbf{x})`

        Parameters
        ----------
        X : tensor_like
            Input
        phi : tuple of tensor_like
            Tuple of parameters for the posterior distribution

        Returns
        -------
        kl_divergence_term : tensor_like
            KL-divergence term
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_kl_divergence_term.")

    def log_q_z_given_x(self, z, phi):
        """
        Computes the log-posterior probabilities of `z`

        Parameters
        ----------
        z : tensor_like
            Posterior samples
        phi : tuple of tensor_like
            Tuple of parameters for the posterior distribution

        Returns
        -------
        log_q_z_x : tensor_like
            Log-posterior probabilities
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "log_q_z_given_x.")

    def log_p_z(self, z):
        """
        Computes the log-prior probabilities of `z`

        Parameters
        ----------
        z : tensor_like
            Posterior samples

        Returns
        -------
        log_p_z : tensor_like
            Log-prior probabilities
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "log_p_z.")


class DiagonalGaussianPrior(Latent):
    """
    Implements a gaussian prior and a gaussian posterior with diagonal
    covariance matrices

    Parameters
    ----------
    encoding_model : pylearn2.models.mlp.MLP
        An MLP representing the recognition network, whose output will be used
        to compute :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`
    isigma : float, optional
        Standard deviation on the zero-mean distribution from which parameters
        initialized by the model itself will be drawn. Defaults to 0.01.
    """
    def __init__(self, encoding_model, isigma=0.01):
        super(DiagonalGaussianPrior, self).__init__(encoding_model)
        self.isigma = isigma

    @wraps(Latent.initialize_parameters)
    def initialize_parameters(self, encoder_input_space, nhid):
        super(DiagonalGaussianPrior, self).initialize_parameters(
            encoder_input_space,
            nhid
        )

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

        self.prior_mu = sharedX(numpy.zeros(self.nhid), name="prior_mu")
        self.log_prior_sigma = sharedX(numpy.zeros(self.nhid),
                                       name="prior_log_sigma")
        self._params = [self.W_mu_e, self.b_mu_e, self.W_sigma_e,
                        self.b_sigma_e, self.prior_mu, self.log_prior_sigma]
        self._params += self.encoding_model.get_params()

    @wraps(Latent.sample_from_p_z)
    def sample_from_p_z(self, num_samples):
        return theano_rng.normal(size=(num_samples, self.nhid),
                                 avg=self.prior_mu,
                                 std=T.exp(self.log_prior_sigma),
                                 dtype=theano.config.floatX)

    @wraps(Latent.sample_from_q_z_given_x)
    def sample_from_q_z_given_x(self, epsilon, phi):
        (mu_e, log_sigma_e) = phi
        if epsilon.ndim == 3:
            return (
                mu_e.dimshuffle('x', 0, 1) +
                T.exp(log_sigma_e.dimshuffle('x', 0, 1)) * epsilon
            )
        else:
            return mu_e + T.exp(log_sigma_e) * epsilon

    @wraps(Latent.sample_from_epsilon)
    def sample_from_epsilon(self, shape):
        return theano_rng.normal(size=shape, dtype=theano.config.floatX)

    @wraps(Latent.kl_divergence_term)
    def _kl_divergence_term(self, X, phi):
        return self._per_component_kl_divergence_term(X, phi).sum(axis=1)

    @wraps(Latent._per_component_kl_divergence_term)
    def _per_component_kl_divergence_term(self, X, phi):
        (mu_e, log_sigma_e) = phi
        log_prior_sigma = self.log_prior_sigma
        prior_mu = self.prior_mu
        return (
            log_prior_sigma - log_sigma_e +
            0.5 * (T.exp(2 * log_sigma_e) + (mu_e - prior_mu) ** 2) /
            T.exp(2 * log_prior_sigma) - 0.5
        )

    @wraps(Latent.encode_phi)
    def encode_phi(self, X):
        h_e = self.fprop(X)
        mu_e = T.dot(h_e, self.W_mu_e) + self.b_mu_e
        log_sigma_e = 0.5 * (T.dot(h_e, self.W_sigma_e) + self.b_sigma_e)
        return (mu_e, log_sigma_e)

    @wraps(Latent.log_q_z_given_x)
    def log_q_z_given_x(self, z, phi):
        (mu_e, log_sigma_e) = phi
        return -0.5 * (
            T.log(2 * pi) + 2 * log_sigma_e.dimshuffle(('x', 0, 1)) +
            (z - mu_e.dimshuffle(('x', 0, 1))) ** 2 /
            T.exp(2 * log_sigma_e.dimshuffle(('x', 0, 1)))
        ).sum(axis=2)

    @wraps(Latent.log_p_z)
    def log_p_z(self, z):
        return -0.5 * (
            T.log(2 * pi * T.exp(2 * self.log_prior_sigma)) +
            ((z - self.prior_mu) / T.exp(self.log_prior_sigma)) ** 2
        ).sum(axis=2)
