"""
Variational autoencoder (VAE) implementation.

`VAE` expects to receive two objects to do its job properly:

1. An instance of `Visible` (`pylearn2.models.vae.visible` module), which
   handles methods related to visible space, like its conditional distribution
   :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})`.
2. An instance of `Latent` (`pylearn2.models.vae.latent` module), which handles
   methods related to latent space, such as its prior distribution
   :math:`p_\\theta(\\mathbf{z})` and its posterior distribution
   :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`.
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"
__email__ = "pylearn-dev@googlegroups"

import numpy
import theano.tensor as T
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import wraps, sharedX
from pylearn2.utils.rng import make_theano_rng

theano_rng = make_theano_rng(default_seed=2341)
pi = sharedX(numpy.pi)


def log_sum_exp(A, axis=None):
    A_max = T.max(A, axis=axis, keepdims=True)
    B = T.log(T.sum(T.exp(A - A_max), axis=axis, keepdims=True)) + A_max
    # TODO: find a cleaner way to get rid of the summed axis
    return B.sum(axis=axis)


class VAE(Model):
    """
    Implementation of the variational autoencoder (VAE).

    Parameters
    ----------
    nvis : int
        Number of dimensions in the input data
    visible : pylearn2.models.vae.visible.Visible
        Handles the visible space-related methods necessary for `VAE` to work
    latent : pylearn2.models.vae.latent.Latent
        Handles the latent space-related methods necessary for `VAE` to work
    nhid : int
        Number of dimensions in latent space, i.e. the space in which :math:`z`
        lives
    data_mean : numpy.ndarray, optional
        Data mean. Defaults to None.
    data_std : numpy.ndarray, optional
        Data standard deviation. Defaults to None.
    """
    def __init__(self, nvis, visible, latent, nhid, data_mean=None,
                 data_std=None):
        super(VAE, self).__init__()

        self.__dict__.update(locals())
        del self.self

        if self.data_mean is not None:
            self.data_mean = sharedX(self.data_mean)
        if self.data_std is not None:
            self.data_std = sharedX(self.data_std)

        # Space initialization
        self.input_space = VectorSpace(dim=self.nvis)
        self.input_source = 'features'
        self.latent_space = VectorSpace(dim=self.nhid)

        # Parameter initialization
        self.visible.initialize_parameters(
            decoder_input_space=self.latent_space,
            nvis=self.nvis
        )
        self.latent.initialize_parameters(
            encoder_input_space=self.input_space,
            nhid=self.nhid
        )
        self._encoding_params = self.latent.get_params()
        self._decoding_params = self.visible.get_params()
        self._params = self._encoding_params + self._decoding_params

    def get_decoding_params(self):
        """
        Returns the model's decoder-related parameters
        """
        return self._decoding_params

    def get_encoding_params(self):
        """
        Returns the model's encoder-related parameters
        """
        return self._encoding_params

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
        z = self.latent.sample_from_p_z(num_samples=num_samples, **kwargs)
        # Decode theta
        theta = self.visible.decode_theta(z)
        # Sample from p(x | z)
        X = self.visible.sample_from_p_x_given_z(num_samples=num_samples,
                                                 theta=theta)

        if return_sample_means:
            return (X, self.visible.means_from_theta(theta))
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
        epsilon = 0 * self.latent.sample_from_epsilon((X.shape[0], self.nhid))
        # Encode q(z | x) parameters
        phi = self.latent.encode_phi(X)
        # Compute z
        z = self.latent.sample_from_q_z_given_x(epsilon=epsilon, phi=phi)
        # Compute expectation term
        theta = self.visible.decode_theta(z)
        reconstructed_X = self.visible.sample_from_p_x_given_z(
            num_samples=X.shape[0],
            theta=theta
        )
        if return_sample_means:
            return (reconstructed_X, self.visible.means_from_theta(theta))
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
        epsilon_shape = (self.latent.num_samples, Y.shape[0], self.nhid)
        epsilon = self.latent.sample_from_epsilon(shape=epsilon_shape)
        # Encode q(z | x) parameters
        phi = self.latent.encode_phi(Y)
        # Compute z
        z = self.latent.sample_from_q_z_given_x(epsilon=epsilon, phi=phi)
        # Compute KL divergence term
        kl_divergence_term = self.latent.kl_divergence_term(
            X=X,
            per_component=False
        )
        # Compute expectation term
        # (z is flattened out in order to be MLP-compatible, and the parameters
        #  output by the decoder network are reshaped to the right shape)
        z = z.reshape((epsilon.shape[0] * epsilon.shape[1], epsilon.shape[2]))
        theta = self.visible.decode_theta(z)
        theta = tuple(
            theta_i.reshape((epsilon.shape[0], epsilon.shape[1],
                             theta_i.shape[1]))
            for theta_i in theta
        )
        expectation_term = self.visible.expectation_term(
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
        epsilon = self.latent.sample_from_epsilon(shape=epsilon_shape)
        # Encode q(z | x) parameters
        phi = self.latent.encode_phi(Y)
        # Compute z
        z = self.latent.sample_from_q_z_given_x(epsilon=epsilon, phi=phi)
        # Decode p(x | z) parameters
        # (z is flattened out in order to be MLP-compatible, and the parameters
        #  output by the decoder network are reshaped to the right shape)
        flat_z = z.reshape((epsilon.shape[0] * epsilon.shape[1],
                            epsilon.shape[2]))
        theta = self.visible.decode_theta(flat_z)
        theta = tuple(
            theta_i.reshape((epsilon.shape[0], epsilon.shape[1],
                             theta_i.shape[1]))
            for theta_i in theta
        )
        # Compute log-probabilities
        log_q_z_x = self.latent.log_q_z_given_x(z=z, phi=phi)
        log_p_z = self.latent.log_p_z(z)
        log_p_x_z = self.visible.log_p_x_given_z(
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
        return self.visible.get_weights()
