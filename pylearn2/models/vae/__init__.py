"""
Variational autoencoder (VAE) implementation, as described in

    Kingma, D. and Welling, M. Auto-Encoding Variational Bayes

`VAE` expects to receive three objects to do its job properly:

1. An instance of `Prior` (`pylearn2.models.vae.prior` module), which
   handles methods related to the prior distribution
   :math:`p_\\theta(\\mathbf{z})`.
2. An instance of `Conditional` (`pylearn2.models.vae.conditional`
   module), which handles methods related to the conditional distribution
   :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})`.
1. An instance of `Conditional` (`pylearn2.models.vae.conditional`
   module), which handles methods related to the posterior distribution
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

import warnings
import numpy
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from pylearn2.expr.basic import log_sum_exp
from pylearn2.models.model import Model
from pylearn2.models.vae.kl import find_integrator_for
from pylearn2.space import VectorSpace
from pylearn2.utils import wraps, sharedX, safe_update
from pylearn2.utils.rng import make_np_rng

default_seed = 2014 + 9 + 20
pi = sharedX(numpy.pi)


class VAE(Model):
    """
    Implementation of the variational autoencoder (VAE).

    Parameters
    ----------
    nvis : int
        Number of dimensions in the input data
    prior : pylearn2.models.vae.prior.Prior
        Represents the prior distribution :math:`p_\\theta(\\mathbf{z})`
    conditional : pylearn2.models.vae.conditional.Conditional
        Represents the conditional distribution
        :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})`
    posterior : pylearn2.models.vae.conditional.Conditional
        Represents the posterior distribution
        :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`
    nhid : int
        Number of dimensions in latent space, i.e. the space in which :math:`z`
        lives
    learn_prior : bool, optional
        Whether to learn the prior distribution p(z). Defaults to `True`.
    kl_integrator : pylearn2.models.vae.kl.KLIntegrator, optional
        Object providing methods for computing KL-related quantities. Defaults
        to `None`, in which case the approximate KL is computed instead.
    batch_size : int, optional
        Sometimes required for some MLPs representing encoding/decoding models.
        Defaults to `None`.
    seed : int or list of int
        Seed for the VAE's numpy RNG used by its subcomponents
    """
    def __init__(self, nvis, prior, conditional, posterior, nhid,
                 learn_prior=True, kl_integrator=None, batch_size=None,
                 seed=None):
        super(VAE, self).__init__()

        self.__dict__.update(locals())
        del self.self

        self.rng = make_np_rng(self.seed, default_seed,
                               ['uniform', 'randint', 'randn'])

        self.prior.set_vae(self)
        self.conditional.set_vae(self)
        self.posterior.set_vae(self)

        self.learn_prior = learn_prior

        # Space initialization
        self.input_space = VectorSpace(dim=self.nvis)
        self.input_source = 'features'
        self.latent_space = VectorSpace(dim=self.nhid)

        # Parameter initialization
        self.prior.initialize_parameters(nhid=self.nhid)
        self.conditional.initialize_parameters(
            input_space=self.latent_space,
            ndim=self.nvis
        )
        self.posterior.initialize_parameters(
            input_space=self.input_space,
            ndim=self.nhid
        )
        self._params = (self.get_posterior_params() +
                        self.get_conditional_params())
        if self.learn_prior:
            self._params += self.get_prior_params()

        names = []
        for param in self._params:
            if param.name not in names:
                names.append(param.name)
            else:
                raise Exception("no two parameters must share the same name: "
                                + param.name)

        # Look for the right KLIntegrator if it's not specified
        if self.kl_integrator is None:
            self.kl_integrator = find_integrator_for(self.prior,
                                                     self.posterior)

    @wraps(Model.get_monitoring_data_specs)
    def get_monitoring_data_specs(self):
        return self.input_space, self.input_source

    @wraps(Model.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        """
        Notes
        -----
        Monitors quantities related to the approximate posterior parameters phi
        and the conditional and prior parameters theta.
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

        posterior_channels = \
            self.posterior.monitoring_channels_from_conditional_params(phi)
        safe_update(rval, posterior_channels)

        conditional_channels = \
            self.conditional.monitoring_channels_from_conditional_params(theta)
        safe_update(rval, conditional_channels)

        prior_channels = self.prior.monitoring_channels_from_prior_params()
        safe_update(rval, prior_channels)

        return rval

    @wraps(Model.get_lr_scalers)
    def get_lr_scalers(self):
        rval = self.prior.get_lr_scalers()
        safe_update(rval, self.conditional.get_lr_scalers())
        safe_update(rval, self.posterior.get_lr_scalers())
        return rval

    @wraps(Model._modify_updates)
    def _modify_updates(self, updates):
        self.prior.modify_updates(updates)
        self.conditional.modify_updates(updates)
        self.posterior.modify_updates(updates)

    @wraps(Model.get_weights)
    def get_weights(self):
        # TODO: This choice is arbitrary. It's something that's useful to
        # visualize, but is it the most intuitive choice?
        return self.conditional.get_weights()

    def get_prior_params(self):
        """
        Returns the model's prior distribution parameters
        """
        return self.prior.get_params()

    def get_conditional_params(self):
        """
        Returns the model's conditional distribution parameters
        """
        return self.conditional.get_params()

    def get_posterior_params(self):
        """
        Returns the model's posterior distribution parameters
        """
        return self.posterior.get_params()

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

    def log_likelihood_lower_bound(self, X, num_samples, approximate_kl=False,
                                   return_individual_terms=False):
        """
        Computes the VAE lower-bound on the marginal log-likelihood of X.

        Parameters
        ----------
        X : tensor_like
            Input
        num_samples : int
            Number of posterior samples per data point, e.g. number of times z
            is sampled for each x.
        approximate_kl : bool, optional
            Whether to compute a stochastic approximation of the KL divergence
            term. Defaults to `False`.
        return_individual_terms : bool, optional
            If `True`, return `(kl_divergence_term, expectation_term)` instead.
            Defaults to `False`.

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
        # Get prior parameters
        prior_theta = self.get_prior_theta()
        # Compute z
        z = self.sample_from_q_z_given_x(epsilon=epsilon, phi=phi)
        # Compute KL divergence term
        kl_divergence_term = self.kl_divergence_term(
            phi=phi,
            theta=prior_theta,
            approximate=approximate_kl,
            epsilon=epsilon
        )
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
        ).mean(axis=0)

        if return_individual_terms:
            return (kl_divergence_term, expectation_term)
        else:
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
        return self.posterior.encode_conditional_params(X)

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
        return self.conditional.encode_conditional_params(z)

    def get_prior_theta(self):
        """
        Returns parameters of the prior distribution
        :math:`p_\\theta(\\mathbf{z})`
        """
        return self.prior.get_params()

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
        return self.conditional.conditional_expectation(theta)

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
        return self.log_p_x_given_z(X, theta)

    def kl_divergence_term(self, phi, theta, approximate=False, epsilon=None):
        """
        Computes the KL-divergence term of the VAE criterion.

        Parameters
        ----------
        phi : tuple of tensor_like
            Parameters of the distribution
            :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`
        theta : tuple of tensor_like
            Parameters of the distribution :math:`p_\\theta(\\mathbf{z})`
        approximate_kl : bool, optional
            Whether to compute a stochastic approximation of the KL divergence
            term. Defaults to `False`.
        epsilon : tensor_like, optional
            Noise samples used to compute the approximate KL term. Defaults to
            `None`.
        """
        if self.kl_integrator is None:
            warnings.warn("computing the analytical KL divergence term is not "
                          "supported for this prior/posterior combination, "
                          "computing a stochastic approximate KL instead")
            return self._approximate_kl_divergence_term(phi, epsilon)
        if approximate:
            return self._approximate_kl_divergence_term(phi, epsilon)
        else:
            return self.kl_integrator.kl_divergence(
                phi=phi,
                theta=theta,
                prior=self.prior,
                posterior=self.posterior
            )

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

    def per_component_kl_divergence_term(self, phi, theta):
        """
        If the prior/posterior combination allows it, analytically computes the
        per-latent-dimension KL divergences between the prior distribution
        :math:`p_\\theta(\\mathbf{z})` and :math:`q_\\phi(\\mathbf{z} \\mid
        \\mathbf{x})`

        Parameters
        ----------
        phi : tuple of tensor_like
            Parameters of the distribution
            :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`
        theta : tuple of tensor_like
            Parameters of the distribution :math:`p_\\theta(\\mathbf{z})`
        """
        if self.kl_integrator is None:
            raise NotImplementedError("impossible to compute the analytical "
                                      "KL divergence")
        else:
            return self.kl_integrator.per_component_kl_divergence(
                phi=phi,
                theta=theta,
                prior=self.prior,
                posterior=self.posterior
            )

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
        return self.conditional.sample_from_conditional(
            conditional_params=theta,
            num_samples=num_samples
        )

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
        return self.prior.sample_from_p_z(num_samples, **kwargs)

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
        return self.posterior.sample_from_conditional(
            conditional_params=phi,
            epsilon=epsilon
        )

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
        return self.posterior.sample_from_epsilon(shape)

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
        return self.prior.log_p_z(z)

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
        return self.conditional.log_conditional(X, theta)

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
        return self.posterior.log_conditional(z, phi)
