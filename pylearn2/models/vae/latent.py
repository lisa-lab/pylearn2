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
* _get_default_output_layer
* _get_output_space
* _validate_encoding_model

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
import warnings
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from pylearn2.models.mlp import Linear, CompositeLayer
from pylearn2.utils.rng import make_theano_rng
from pylearn2.utils import sharedX, wraps
from pylearn2.space import VectorSpace, NullSpace, CompositeSpace

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
    output_layer_required : bool, optional
        If `True`, the encoding model's output is the last hidden
        representation from which parameters of :math:`q_\\phi(\\mathbf{z}
        \\mid \\mathbf{x})` will be computed, and `Latent` will add its own
        default output layer to the `encoding_model` MLP. If `False`, the MLP's
        last layer **is** the output layer. Defaults to `True`.
    """
    def __init__(self, encoding_model, output_layer_required=True):
        self.encoding_model = encoding_model
        self.output_layer_required = output_layer_required

    def _get_default_output_layer(self):
        """
        Returns a default `Layer` mapping the encoding model's last hidden
        representation to parameters of :math:`q_\\phi(\\mathbf{z} \\mid
        \\mathbf{x})`
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_get_default_output_layer")

    def _get_output_space(self):
        """
        Returns the expected output space of the decoding model, i.e. a
        description of how the parameters output by the encoding model should
        look like.
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_get_output_space")

    def _validate_encoding_model(self):
        """
        Makes sure the encoding model's output layer is compatible with the
        parameters expected by :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`
        """
        expected_output_space = self._get_output_space()
        model_output_space = self.encoding_model.get_output_space()
        if not model_output_space == expected_output_space:
            raise ValueError("the specified encoding model's output space is "
                             "incompatible with " + str(self.__class__) + ": "
                             "expected " + str(expected_output_space) + " but "
                             "encoding model's output space is " +
                             str(model_output_space))

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

    def get_vae(self):
        """
        Returns the VAE that this `Latent` instance belongs to, or None
        if it has not been assigned to a VAE yet.
        """
        if hasattr(self, 'vae'):
            return self.vae
        else:
            return None

    def set_vae(self, vae):
        """
        Assigns this `Latent` instance to a VAE.

        Parameters
        ----------
        vae : pylearn2.models.vae.VAE
            VAE to assign to
        """
        if self.get_vae() is not None:
            raise RuntimeError("this `Latent` instance already belongs to "
                               "another VAE")
        self.vae = vae

    def initialize_parameters(self, encoder_input_space, nhid):
        """
        Initialize model parameters.
        """
        self.nhid = nhid
        self.encoder_input_space = encoder_input_space
        if self.output_layer_required:
            self.encoding_model.add_layer(self._get_default_output_layer())
        self._validate_encoding_model()
        self._params = self.encoding_model.get_params()
        self._initialize_prior_parameters()

    def _initialize_prior_parameters(self):
        """
        Initialize parameters for the prior distribution
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_initialize_prior_parameters")

    def get_params(self):
        """
        Return the latent space-related parameters
        """
        return self._params

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
        X = self.encoder_input_space.format_as(
            batch=X,
            space=self.encoding_model.get_input_space()
        )
        rval = self.encoding_model.fprop(X)
        if not type(rval) == tuple:
            rval = (rval, )
        return rval

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

    def kl_divergence_term(self, phi, approximate=False, epsilon=None,
                           **kwargs):
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
        if approximate:
            return self._approximate_kl_divergence_term(phi, epsilon)
        else:
            try:
                return self._kl_divergence_term(phi, **kwargs)
            except NotImplementedError:
                warnings.warn("analytic KL is not supported by this prior/"
                              "posterior combination, approximating the KL "
                              "term instead")
                return self._approximate_kl_divergence_term(phi, epsilon)

    def _kl_divergence_term(self, phi, **kwargs):
        """
        Analytically computes the KL divergence between the prior distribution
        :math:`p_\\theta(\\mathbf{z})` and :math:`q_\\phi(\\mathbf{z} \\mid
        \\mathbf{x})`

        Parameters
        ----------
        phi : tuple of tensor_like
            Tuple of parameters for the posterior distribution

        Returns
        -------
        kl_divergence_term : tensor_like
            KL-divergence term
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_kl_divergence_term.")

    def _approximate_kl_divergence_term(self, phi, epsilon):
        """
        Returns a Monte Carlo approximation of the KL divergence term.

        Parameters
        ----------
        phi : tuple of tensor_like
            Tuple of parameters for the posterior distribution
        epsilon : tensor_like
            Noise term from which z is computed
        """
        if epsilon is None:
            raise ValueError("stochastic KL is requested but no epsilon is "
                             "given")
        z = self.sample_from_q_z_given_x(epsilon=epsilon, phi=phi)
        log_q_z_x = self.log_q_z_given_x(z=z, phi=phi)
        log_p_z = self.log_p_z(z)
        return (log_q_z_x - log_p_z).mean(axis=0)

    def per_component_kl_divergence_term(self, phi, **kwargs):
        """
        If the prior/posterior combination allows it, analytically computes the
        per-latent-dimension KL divergences between the prior distribution
        :math:`p_\\theta(\\mathbf{z})` and :math:`q_\\phi(\\mathbf{z} \\mid
        \\mathbf{x})`

        Parameters
        ----------
        phi : tuple of tensor_like
            Tuple of parameters for the posterior distribution

        Returns
        -------
        kl_divergence_term : tensor_like
            Per-component KL-divergence terms
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "per_component_kl_divergence_term.")

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
    output_layer_required : bool, optional
        If `True`, the encoding model's output is the last hidden
        representation from which parameters of :math:`q_\\phi(\\mathbf{z}
        \\mid \\mathbf{x})` will be computed, and `Latent` will add its own
        default output layer to the `encoding_model` MLP. If `False`, the MLP's
        last layer **is** the output layer. Defaults to `True`.
    isigma : float, optional
        Standard deviation on the zero-mean distribution from which parameters
        initialized by the model itself will be drawn. Defaults to 0.01.
    """
    def __init__(self, encoding_model, output_layer_required=True,
                 isigma=0.01):
        super(DiagonalGaussianPrior, self).__init__(encoding_model,
                                                    output_layer_required)
        self.isigma = isigma

    @wraps(Latent._get_default_output_layer)
    def _get_default_output_layer(self):
        return CompositeLayer(
            layer_name='phi',
            layers=[Linear(dim=self.nhid, layer_name='mu', irange=0.01),
                    Linear(dim=self.nhid, layer_name='log_sigma',
                           irange=0.01)]
        )

    @wraps(Latent._get_output_space)
    def _get_output_space(self):
        return CompositeSpace([VectorSpace(dim=self.nhid),
                               VectorSpace(dim=self.nhid)])

    @wraps(Latent._initialize_prior_parameters)
    def _initialize_prior_parameters(self):
        self.prior_mu = sharedX(numpy.zeros(self.nhid), name="prior_mu")
        self.log_prior_sigma = sharedX(numpy.zeros(self.nhid),
                                       name="prior_log_sigma")
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

    @wraps(Latent._kl_divergence_term)
    def _kl_divergence_term(self, phi):
        return self.per_component_kl_divergence_term(phi).sum(axis=1)

    @wraps(Latent.per_component_kl_divergence_term)
    def per_component_kl_divergence_term(self, phi):
        (mu_e, log_sigma_e) = phi
        log_prior_sigma = self.log_prior_sigma
        prior_mu = self.prior_mu
        return (
            log_prior_sigma - log_sigma_e +
            0.5 * (T.exp(2 * log_sigma_e) + (mu_e - prior_mu) ** 2) /
            T.exp(2 * log_prior_sigma) - 0.5
        )

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
