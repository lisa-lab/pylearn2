"""
Variational autoencoder (VAE) implementation, as described in

    Kingma, D. and Welling, M. Auto-Encoding Variational Bayes

`VAE` expects to receive two objects to do its job properly:

1. An instance of `Visible` (`pylearn2.models.vae.visible` module), which
   handles methods related to visible space, like its conditional distribution
   :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})`.
2. An instance of `Latent` (`pylearn2.models.vae.latent` module), which handles
   methods related to latent space, such as its prior distribution
   :math:`p_\\theta(\\mathbf{z})` and its posterior distribution
   :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`.

For an example on how to use the VAE framework, see
`pylearn2/scripts/tutorials/variational_autoencoder/vae.yaml`.
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"
__email__ = "pylearn-dev@googlegroups"

import numpy
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import wraps, sharedX, safe_update
from pylearn2.utils.rng import make_np_rng

default_seed = 2014 + 9 + 20
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
    seed : int or list of int
        Seed for the VAE's numpy RNG used by its subcomponents
    """
    def __init__(self, nvis, visible, latent, nhid, batch_size=None,
                 seed=None):
        super(VAE, self).__init__()

        self.__dict__.update(locals())
        del self.self

        self.rng = make_np_rng(self.seed, default_seed,
                               ['uniform', 'randint', 'randn'])

        self.visible.set_vae(self)
        self.latent.set_vae(self)

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
        names = []
        for param in self._params:
            if param.name not in names:
                names.append(param.name)
            else:
                raise Exception("no two parameters must share the same name: "
                                + param.name)

    @wraps(Model.get_monitoring_data_specs)
    def get_monitoring_data_specs(self):
        return self.input_space, self.input_source

    @wraps(Model.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        """
        Notes
        -----
        Monitors quantities related to the approximate posterior parameters phi
        and the distributions parameters theta.
        """
        space, source = self.get_monitoring_data_specs()
        space.validate(data)

        rval = OrderedDict()

        X = data
        epsilon_shape = (1, X.shape[0], self.nhid)
        epsilon = self.sample_from_epsilon(shape=epsilon_shape)
        phi = self.encode_phi(X)
        z = self.sample_from_q_z_given_x(epsilon=epsilon, phi=phi)
        z = z.reshape((epsilon.shape[0] * epsilon.shape[1], epsilon.shape[2]))
        theta = self.decode_theta(z)

        latent_channels = self.latent.monitoring_channels_from_phi(phi)
        safe_update(rval, latent_channels)

        visible_channels = self.visible.monitoring_channels_from_theta(theta)
        safe_update(rval, visible_channels)

        return rval

    @wraps(Model.get_lr_scalers)
    def get_lr_scalers(self):
        rval = self.visible.get_lr_scalers()
        safe_update(rval, self.latent.get_lr_scalers())
        return rval

    @wraps(Model._modify_updates)
    def _modify_updates(self, updates):
        self.visible.modify_updates(updates)
        self.latent.modify_updates(updates)

    @wraps(Model.get_weights)
    def get_weights(self):
        # TODO: This choice is arbitrary. It's something that's useful to
        # visualize, but is it the most intuitive choice?
        return self.visible.get_weights()

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
        z = self.sample_from_p_z(num_samples=num_samples, **kwargs)
        # Decode theta
        theta = self.decode_theta(z)
        # Sample from p(x | z)
        X = self.sample_from_p_x_given_z(num_samples=num_samples, theta=theta)

        if return_sample_means:
            return (X, self.means_from_theta(theta))
        else:
            return X

    def reconstruct(self, X, noisy_encoding=False, return_sample_means=True):
        """
        Given an input, generates its reconstruction by propagating it through
        the encoder network and projecting it back through the decoder network.

        Parameters
        ----------
        X : tensor_like
            Input to reconstruct
        noisy_encoding : bool, optional
            If `True`, sample z from the posterior distribution. If `False`,
            take the expected value. Defaults to `False`.
        return_sample_means : bool, optional
            Whether to return the conditional expectations
            :math:`\\mathbb{E}[p_\\theta(\\mathbf{x} \\mid \\mathbf{h})]` in
            addition to the actual samples. Defaults to `False`.

        Returns
        -------
        rval : tensor_like or tuple of tensor_like
            Samples, and optionally conditional expectations
        """
        # Sample noise
        # TODO: For now this covers our use cases, but we need something more
        # robust for the future.
        epsilon = self.sample_from_epsilon((X.shape[0], self.nhid))
        if not noisy_encoding:
            epsilon *= 0
        # Encode q(z | x) parameters
        phi = self.encode_phi(X)
        # Compute z
        z = self.sample_from_q_z_given_x(epsilon=epsilon, phi=phi)
        # Compute expectation term
        theta = self.decode_theta(z)
        reconstructed_X = self.sample_from_p_x_given_z(
            num_samples=X.shape[0],
            theta=theta
        )
        if return_sample_means:
            return (reconstructed_X, self.means_from_theta(theta))
        else:
            return reconstructed_X

    def log_likelihood_lower_bound(self, X, num_samples):
        """
        Computes the VAE lower-bound on the marginal log-likelihood of X.

        Parameters
        ----------
        X : tensor_like
            Input
        num_samples : int
            Number of posterior samples per data point, e.g. number of times z
            is sampled for each x.

        Returns
        -------
        lower_bound : tensor_like
            Lower-bound on the marginal log-likelihood
        """
        # Sample noise
        epsilon_shape = (num_samples, X.shape[0], self.nhid)
        epsilon = self.sample_from_epsilon(shape=epsilon_shape)
        # Encode q(z | x) parameters
        phi = self.encode_phi(X)
        # Compute z
        z = self.sample_from_q_z_given_x(epsilon=epsilon, phi=phi)
        # Compute KL divergence term
        kl_divergence_term = self.kl_divergence_term(phi=phi,
                                                     approximate=False,
                                                     epsilon=epsilon)
        # Compute expectation term
        # (z is flattened out in order to be MLP-compatible, and the parameters
        #  output by the decoder network are reshaped to the right shape)
        z = z.reshape((epsilon.shape[0] * epsilon.shape[1], epsilon.shape[2]))
        theta = self.decode_theta(z)
        theta = tuple(
            theta_i.reshape((epsilon.shape[0], epsilon.shape[1],
                             theta_i.shape[1]))
            for theta_i in theta
        )
        expectation_term = self.expectation_term(
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
        num_samples : int
            Number of posterior samples per data point, e.g. number of times z
            is sampled for each x.

        Returns
        -------
        approximation : tensor_like
            Approximation on the marginal log-likelihood
        """
        # Sample noise
        epsilon_shape = (num_samples, X.shape[0], self.nhid)
        epsilon = self.sample_from_epsilon(shape=epsilon_shape)
        # Encode q(z | x) parameters
        phi = self.encode_phi(X)
        # Compute z
        z = self.sample_from_q_z_given_x(epsilon=epsilon, phi=phi)
        # Decode p(x | z) parameters
        # (z is flattened out in order to be MLP-compatible, and the parameters
        #  output by the decoder network are reshaped to the right shape)
        flat_z = z.reshape((epsilon.shape[0] * epsilon.shape[1],
                            epsilon.shape[2]))
        theta = self.decode_theta(flat_z)
        theta = tuple(
            theta_i.reshape((epsilon.shape[0], epsilon.shape[1],
                             theta_i.shape[1]))
            for theta_i in theta
        )
        # Compute log-probabilities
        log_q_z_x = self.log_q_z_given_x(z=z, phi=phi)
        log_p_z = self.log_p_z(z)
        log_p_x_z = self.log_p_x_given_z(
            X=X.dimshuffle(('x', 0, 1)),
            theta=theta
        )

        return log_sum_exp(
            log_p_z + log_p_x_z - log_q_z_x,
            axis=0
        ) - T.log(num_samples)

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
        return self.latent.encode_phi(X)

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
        return self.visible.decode_theta(z)

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
        return self.visible.means_from_theta(theta)

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
        return self.visible.expectation_term(X, theta)

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
        return self.latent.kl_divergence_term(phi, approximate, epsilon,
                                              **kwargs)

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
        return self.latent.per_component_kl_divergence_term(phi, **kwargs)

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
        return self.visible.sample_from_p_x_given_z(num_samples, theta)

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
        return self.latent.sample_from_p_z(num_samples, **kwargs)

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
        return self.latent.sample_from_q_z_given_x(epsilon, phi)

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
        return self.latent.sample_from_epsilon(shape)

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
        return self.visible.log_p_x_given_z(X, theta)

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
        return self.latent.kl_divergence_term(phi, approximate, epsilon,
                                              **kwargs)

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
        return self.latent.per_component_kl_divergence_term(phi, **kwargs)

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
        return self.latent.log_q_z_given_x(z, phi)

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
        return self.latent.log_p_z(z)
