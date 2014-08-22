"""
Variational autoencoder (VAE) implementation.

The `BaseVAE` base class defines the interface which all instances should
conform to. Two type of subclasses implement abstract methods defined in
`BaseVAE`:

1. Subclasses in the `visible` module handle functionalities related to visible
   space, like its conditional distribution :math:`p_\\theta(\\mathbf{x} \\mid
   \\mathbf{z})`.
2. Subclasses in the `latent` module handle functionalities related to latent
   space, such as its prior distribution :math:`p_\\theta(\\mathbf{z})` and its
   posterior distribution :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`.

A VAE is instantiated through the `VAE` method, which takes the names of a
`visible` subclass and a `latent` subclass and dynamically returns an instance
with the appropriate superclasses.
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"
__email__ = "pylearn-dev@googlegroups"

import sys
import inspect
import numpy
import theano.tensor as T
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import wraps, sharedX
from pylearn2.utils.rng import make_theano_rng
from sheldon.code.pylearn2.expr.basic import log_sum_exp

theano_rng = make_theano_rng(default_seed=2341)
pi = sharedX(numpy.pi)


class BaseVAE(Model):
    """
    Basic abstract class for AEVB models.

    Parameters
    ----------
    nvis : int
        Number of dimensions in the input data
    encoding_model : pylearn2.models.mlp.MLP
        An MLP representing the recognition network, whose output will be used
        to compute :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`
    decoding_model : pylearn2.models.mlp.MLP
        An MLP representing the generative network, whose output will be used
        to compute :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})`
    nhid : int
        Number of dimensions in latent space, i.e. the space in which :math:`z`
        lives
    num_samples : int, optional
        Number of posterior samples per datapoint used to compute the
        log-likelihood lower bound. Defaults to 1.
    isigma : float, optional
        Standard deviation on the zero-mean distribution from which parameters
        initialized by the model itself will be drawn. Defaults to 0.01.
    data_mean : numpy.ndarray, optional
        Data mean. Defaults to None.
    data_std : numpy.ndarray, optional
        Data standard deviation. Defaults to None.
    """
    def __init__(self, nvis, encoding_model, decoding_model, nhid,
                 num_samples=1, isigma=0.01, data_mean=None, data_std=None):
        super(BaseVAE, self).__init__()

        self.__dict__.update(locals())
        del self.self

        self.nenc = encoding_model.get_output_space().get_total_dimension()
        self.ndec = decoding_model.get_output_space().get_total_dimension()

        if self.data_mean is not None:
            self.data_mean = sharedX(self.data_mean)
        if self.data_std is not None:
            self.data_std = sharedX(self.data_std)

        # Space initialization
        self.input_space = VectorSpace(dim=self.nvis)
        self.input_source = 'features'
        self.encoder_input_space = self.input_space
        self.encoder_output_space = VectorSpace(dim=self.nenc)
        self.decoder_input_space = VectorSpace(dim=self.nhid)
        self.decoder_output_space = VectorSpace(dim=self.ndec)

        self._params = []
        self._encoding_parameters = []
        self._decoding_parameters = []
        self._initialize_parameters()

    def get_decoding_params(self):
        """
        Returns the model's decoder-related parameters
        """
        return self._decoding_parameters + self.decoding_model.get_params()

    def get_encoding_params(self):
        """
        Returns the model's encoder-related parameters
        """
        return self._encoding_parameters + self.encoding_model.get_params()

    def sample(self, num_samples, return_sample_means=True, **kwargs):
        """
        Sample from the model's learned distribution

        Parameters
        ----------
        num_samples : int
            Number of samples
        return_sample_means : bool, optional
            Whether to return the conditional expectations
            :math:`\\mathbb{E}[p_\\theta(\\mathbf{x} \\mid \\mathbf{h})]` in
            addition to the actual samples. Defaults to `False`.

        Returns
        -------
        rval : tensor_like or tuple of tensor_like
            Samples, and optionally conditional expectations
        """
        # Sample from p(z)
        z = self._sample_from_p_z(num_samples=num_samples **kwargs)
        # Decode theta
        theta = self._decode_theta(z)
        # Sample from p(x | z)
        X = self._sample_from_p_x_given_z(num_samples=num_samples, theta=theta)

        if return_sample_means:
            return (X, self._means_from_theta(theta))
        else:
            return X

    def reconstruct(self, X, return_sample_means=True):
        """
        Given an input, generates its reconstruction by propagating it through
        the encoder network **without adding noise** and projecting it back
        through the decoder network.

        Parameters
        ----------
        X : tensor_like
            Input to reconstruct
        return_sample_means : bool, optional
            Whether to return the conditional expectations
            :math:`\\mathbb{E}[p_\\theta(\\mathbf{x} \\mid \\mathbf{h})]` in
            addition to the actual samples. Defaults to `False`.

        Returns
        -------
        rval : tensor_like or tuple of tensor_like
            Samples, and optionally conditional expectations
        """
        # Substract mean (if provided)
        if self.data_mean is None:
            mean = sharedX(numpy.zeros(self.nvis))
        else:
            mean = self.data_mean
        X = X - mean
        # Sample noise
        # TODO: For now this covers our use cases, but we need something more
        # robust for the future.
        epsilon = T.zeros((X.shape[0], self.nhid))
        # Encode q(z | x) parameters
        phi = self._encode_phi(X)
        # Compute z
        z = self._sample_from_q_z_given_x(epsilon=epsilon, phi=phi)
        # Compute expectation term
        theta = self._decode_theta(z)
        reconstructed_X = self._sample_from_p_x_given_z(
            num_samples=X.shape[0],
            theta=theta
        )
        if return_sample_means:
            return (reconstructed_X, self._means_from_theta(theta))
        else:
            return reconstructed_X

    def log_likelihood_lower_bound(self, X):
        """
        Computes the VAE lower-bound on the marginal log-likelihood of X.

        Parameters
        ----------
        X : tensor_like
            Input

        Returns
        -------
        lower_bound : tensor_like
            Lower-bound on the marginal log-likelihood
        """
        # Substract mean (if provided). We do not overwrite X, as X is the
        # reconstruction target.
        if self.data_mean is None:
            mean = sharedX(numpy.zeros(self.nvis))
        else:
            mean = self.data_mean
        Y = X - mean
        # Sample noise
        epsilon_shape = (self.num_samples, Y.shape[0], self.nhid)
        epsilon = self._sample_from_epsilon(shape=epsilon_shape)
        # Encode q(z | x) parameters
        phi = self._encode_phi(Y)
        # Compute z
        z = self._sample_from_q_z_given_x(epsilon=epsilon, phi=phi)
        # Compute KL divergence term
        kl_divergence_term = self._kl_divergence_term(X=X, phi=phi)
        # Compute expectation term
        # (z is flattened out in order to be MLP-compatible, and the parameters
        #  output by the decoder network are reshaped to the right shape)
        z = z.reshape((epsilon.shape[0] * epsilon.shape[1], epsilon.shape[2]))
        theta = self._decode_theta(z)
        theta = tuple(
            theta_i.reshape((epsilon.shape[0], epsilon.shape[1],
                             theta_i.shape[1]))
            for theta_i in theta
        )
        expectation_term = self._expectation_term(
            X=X.dimshuffle('x', 0, 1),
            theta=theta
        ).mean(axis=0).sum(axis=1)

        return -kl_divergence_term + expectation_term

    def log_likelihood_approximation(self, X, num_samples):
        """
        Computes the importance sampling approximation to the marginal
        log-likelihood of X, using the reparametrization trick.

        Parameters
        ----------
        X : tensor_like
            Input

        Returns
        -------
        approximation : tensor_like
            Approximation on the marginal log-likelihood
        """
        # Substract mean (if provided). We do not overwrite X, as X is used to
        # compute the conditional log-likelihood log p(x | z).
        if self.data_mean is None:
            mean = sharedX(numpy.zeros(self.nvis))
        else:
            mean = self.data_mean
        Y = X - mean
        # Sample noise
        epsilon_shape = (num_samples, Y.shape[0], self.nhid)
        epsilon = self._sample_from_epsilon(shape=epsilon_shape)
        # Encode q(z | x) parameters
        phi = self._encode_phi(Y)
        # Compute z
        z = self._sample_from_q_z_given_x(epsilon=epsilon, phi=phi)
        # Decode p(x | z) parameters
        # (z is flattened out in order to be MLP-compatible, and the parameters
        #  output by the decoder network are reshaped to the right shape)
        flat_z = z.reshape((epsilon.shape[0] * epsilon.shape[1],
                            epsilon.shape[2]))
        theta = self._decode_theta(flat_z)
        theta = tuple(
            theta_i.reshape((epsilon.shape[0], epsilon.shape[1],
                             theta_i.shape[1]))
            for theta_i in theta
        )
        # Compute log-probabilities
        log_q_z_x = self._log_q_z_given_x(z=z, phi=phi)
        log_p_z = self._log_p_z(z)
        log_p_x_z = self._log_p_x_given_z(
            X=X.dimshuffle(('x', 0, 1)),
            theta=theta
        )

        return log_sum_exp(
            log_p_z + log_p_x_z - log_q_z_x,
            axis=0
        ) - T.log(num_samples)

    @wraps(Model.get_weights)
    def get_weights(self):
        # TODO: This choice is arbitrary. It's something that's useful to
        # visualize, but is it the most intuitive choice?
        return self.decoding_model.get_weights()

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
        phi = self._encode_phi(X)
        if per_component:
            return self._per_component_kl_divergence_term(X=X, phi=phi,
                                                          **kwargs)
        else:
            return self._kl_divergence_term(X=X, phi=phi, **kwargs)

    def _initialize_parameters(self):
        """
        Initialize model parameters.
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_initialize_parameters.")

    def _encoding_fprop(self, X):
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

    def _decoding_fprop(self, z):
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

    def _encode_phi(self, X):
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
                                  "_encode_phi.")

    def _decode_theta(self, z):
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
                                  "_decode_theta.")

    def _means_from_theta(self, theta):
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
                                  "_means_from_theta.")

    def _sample_from_p_z(self, num_samples, **kwargs):
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
                                  "_sample_from_p_z.")

    def _sample_from_p_x_given_z(self, num_samples, theta):
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
                                  "_sample_from_p_x_given_z.")

    def _sample_from_q_z_given_x(self, epsilon, phi):
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
                                  "_sample_from_q_z_given_x.")

    def _sample_from_epsilon(self, shape):
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
                                  "_sample_from_epsilon.")

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

    def _expectation_term(self, X, theta):
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
                                  "_expectation_term.")

    def _log_q_z_given_x(self, z, phi):
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
                                  "_log_q_z_given_x.")

    def _log_p_z(self, z):
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
                                  "_log_p_z.")

    def _log_p_x_given_z(self, X, theta):
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
                                  "_log_p_x_given_z.")


def VAE(visible, latent, args, kwargs):
    """
    Dynamically instantiates a `BaseVAE` subclass with `visible` and `latent`
    as its superclasses.

    Parameters
    ----------
    visible : str
        Name of the superclass implementing visible-related methods
    latent : str
        Name of the superclass implementing latent-related methods
    """
    # Parse subclass names
    import pylearn2.models.vae.visible as visible_module
    visible_classes = dict(inspect.getmembers(
        sys.modules[visible_module.__name__],
        inspect.isclass
    ))
    try:
        visible_class = visible_classes[visible]
    except KeyError:
        raise ValueError(visible + " is not a 'visible' subclass")
    import pylearn2.models.vae.latent as latent_module
    latent_classes = dict(inspect.getmembers(
        sys.modules[latent_module.__name__],
        inspect.isclass
    ))
    try:
        latent_class = latent_classes[latent]
    except KeyError:
        raise ValueError(latent + " is not a 'latent' subclass")

    # Instantiate anonymous subclass
    class CustomVAE(visible_class, latent_class):
        def _initialize_parameters(self):
            for base in self.__class__.__bases__:
                base._initialize_parameters(self)
    return CustomVAE(*args, **kwargs)
