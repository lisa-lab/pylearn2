"""
Classes implementing logic related to the analytical computation of the KL
divergence between :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})` and
:math:`p_\\theta(\\mathbf{z})` in the VAE framework
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"
__email__ = "pylearn-dev@googlegroups"

import sys
import inspect
import theano.tensor as T
from pylearn2.utils import wraps
from pylearn2.models.vae import prior, conditional


class ImpossiblePrior(prior.Prior):
    """
    A Prior that's incompatible with everything

    Parameters
    ----------
    See `Prior`
    """


class ImpossiblePosterior(conditional.Conditional):
    """
    A Conditional that's incompatible with everything

    Parameters
    ----------
    See `Conditional`
    """


class KLIntegrator(object):
    """
    Class responsible for computing the analytical KL divergence term in the
    VAE criterion
    """
    prior_class = ImpossiblePrior
    posterior_class = ImpossiblePosterior

    def kl_divergence(self, phi, theta, prior, posterior):
        """
        Computes the KL-divergence term of the VAE criterion.

        Parameters
        ----------
        phi : tuple of tensor_like
            Parameters of the distribution
            :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`
        theta : tuple of tensor_like
            Parameters of the distribution :math:`p_\\theta(\\mathbf{z})`
        """
        raise NotImplementedError(str(self.__class__) + " does not " +
                                  "implement kl_divergence")

    def per_component_kl_divergence(self, phi, theta, prior, posterior):
        """
        If the prior/posterior combination allows it, computes the
        per-component KL divergence term

        Parameters
        ----------
        phi : tuple of tensor_like
            Parameters of the distribution
            :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`
        theta : tuple of tensor_like
            Parameters of the distribution :math:`p_\\theta(\\mathbf{z})`
        """
        raise NotImplementedError(str(self.__class__) + " does not " +
                                  "implement per_component_kl_divergence")

    def _validate_prior_posterior(self, prior, posterior):
        """
        Checks that the prior/posterior combination is what the integrator
        expects and raises an exception otherwise

        Parameters
        ----------
        prior : pylearn2.models.vae.prior.Prior
            Prior distribution on z
        posterior : pylearn2.models.vae.posterior.Posterior
            Posterior distribution on z given x
        """
        if self.prior_class is None or self.posterior_class is None:
            raise NotImplementedError(str(self.__class__) + " has not set " +
                                      "the required 'prior_class' and " +
                                      "'posterior_class' class attributes")
        if not isinstance(prior, self.prior_class):
            raise ValueError("prior class " + str(prior.__class__) + " is " +
                             "incompatible with expected prior class " +
                             str(self.prior_class.__class__))
        if not isinstance(posterior, self.posterior_class):
            raise ValueError("posterior class " + str(posterior.__class__) +
                             " is incompatible with expected posterior " +
                             "class " + str(self.prior_class.__class__))


class DiagonalGaussianPriorPosteriorKL(KLIntegrator):
    """
    Computes the analytical KL between a diagonal gaussian prior and a diagonal
    gaussian posterior
    """
    prior_class = prior.DiagonalGaussianPrior
    posterior_class = conditional.DiagonalGaussian

    @wraps(KLIntegrator.kl_divergence)
    def kl_divergence(self, phi, theta, prior, posterior):
        return self.per_component_kl_divergence(
            phi=phi,
            theta=theta,
            prior=prior,
            posterior=posterior
        ).sum(axis=1)

    @wraps(KLIntegrator.per_component_kl_divergence)
    def per_component_kl_divergence(self, phi, theta, prior, posterior):
        self._validate_prior_posterior(prior, posterior)
        (posterior_mu, posterior_log_sigma) = phi
        (prior_mu, prior_log_sigma) = theta
        return (
            prior_log_sigma - posterior_log_sigma +
            0.5 * (T.exp(2 * posterior_log_sigma) +
                   (posterior_mu - prior_mu) ** 2) /
            T.exp(2 * prior_log_sigma) - 0.5
        )


def find_integrator_for(prior, posterior):
    """
    Returns a KLIntegrator instance compatible with 'prior' and 'posterior', or
    None if nothing is compatible.

    Parameters
    ----------
    prior : pylearn2.models.vae.prior.Prior
        Object representing the prior p(z)
    posterior : pylearn2.models.vae.conditional.Conditional
        Object representing the approximate posterior q(z | x)
    """
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and issubclass(obj, KLIntegrator):
            corresponds = (isinstance(prior, obj.prior_class) and
                           isinstance(posterior, obj.posterior_class))
            if corresponds:
                return obj()
    return None
