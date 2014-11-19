"""
Classes implementing logic related to the conditional distributions
in the VAE framework
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
from pylearn2.compat import OrderedDict
from pylearn2.models import Model
from pylearn2.models.mlp import Linear, CompositeLayer
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils.rng import make_theano_rng
from pylearn2.utils import wraps, sharedX

pi = sharedX(numpy.pi)


class Conditional(Model):
    """
    Abstract class implementing methods related to a conditional distribution
    :math:`f_\\omega(\\mathbf{a} \\mid \\mathbf{b})`. Used in the VAE framework
    for the conditional :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})` and
    the posterior :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`.

    Parameters
    ----------
    mlp : pylearn2.models.mlp.MLP
        An MLP mapping the variable conditioned on (e.g. x for the posterior
        distribution or z for the conditional distribution in the VAE
        framework) to the distribution parameters. Note that the MLP must be
        **nested**, meaning that its input space must not have already been
        defined, as `Conditional` will do it automatically.
    name : str
        A string identifier for this conditional distribution (e.g. "posterior"
        or "conditional")
    output_layer_required : bool, optional
        If `True`, the MLP's output is the last hidden representation from
        which parameters of the conditional distribution will be computed, and
        `Conditional` will add its own default output layer to the MLP. If
        `False`, the MLP's last layer **is** the output layer. Defaults to
        `True`.
    """
    def __init__(self, mlp, name, output_layer_required=True):
        super(Conditional, self).__init__()
        if not mlp._nested:
            raise ValueError(str(self.__class__) + " expects an MLP whose " +
                             "input space has not been defined yet. You " +
                             "should not specify 'nvis' or 'input_space' " +
                             "when instantiating the MLP.")
        self.mlp = mlp
        self.name = name
        self.output_layer_required = output_layer_required

    def get_weights(self):
        """
        Returns its MLP's weights
        """
        return self.mlp.get_weights()

    def get_lr_scalers(self):
        """
        Returns the encoding model's learning rate scalers
        """
        return self.mlp.get_lr_scalers()

    def _get_default_output_layer(self):
        """
        Returns a default `Layer` mapping the MLP's last hidden representation
        to parameters of the conditional distribution
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_get_default_output_layer")

    def _get_required_mlp_output_space(self):
        """
        Returns the expected output space of the MLP, i.e. a description of how
        the parameters output by the MLP should look like.
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_get_required_mlp_output_space")

    def _validate_mlp(self):
        """
        Makes sure the MLP's output layer is compatible with the parameters
        expected by the conditional distribution
        """
        expected_output_space = self._get_required_mlp_output_space()
        mlp_output_space = self.mlp.get_output_space()
        if not mlp_output_space == expected_output_space:
            raise ValueError("the specified MLP's output space is " +
                             "incompatible with " + str(self.__class__) + ": "
                             "expected " + str(expected_output_space) + " but "
                             "encoding model's output space is " +
                             str(mlp_output_space))

    def monitoring_channels_from_conditional_params(self, conditional_params):
        """
        Get monitoring channels from the parameters of the conditional
        distribution.

        By default, no monitoring channel is computed.

        Parameters
        ----------
        conditional_params : tuple of tensor_like
            Parameters of the conditional distribution
        """
        return OrderedDict()

    def _modify_updates(self, updates):
        """
        Modifies the parameters before a learning update is applied.

        By default, only calls the MLP's `modify_updates` method.
        """
        self.mlp.modify_updates(updates)

    def get_vae(self):
        """
        Returns the VAE that this `Conditional` instance belongs to, or None
        if it has not been assigned to a VAE yet.
        """
        if hasattr(self, 'vae'):
            return self.vae
        else:
            return None

    def set_vae(self, vae):
        """
        Assigns this `Conditional` instance to a VAE.

        Parameters
        ----------
        vae : pylearn2.models.vae.VAE
            VAE to assign to
        """
        if self.get_vae() is not None:
            raise RuntimeError("this " + str(self.__class__) + " instance " +
                               "already belongs to another VAE")
        self.vae = vae
        self.rng = self.vae.rng
        self.theano_rng = make_theano_rng(int(self.rng.randint(2 ** 30)),
                                          which_method=["normal", "uniform"])
        self.batch_size = vae.batch_size

    def initialize_parameters(self, input_space, ndim):
        """
        Initialize model parameters.

        Parameters
        ----------
        input_space : pylearn2.space.Space
            The input space for the MLP
        ndim : int
            Number of units of a in f(a | b)
        """
        self.ndim = ndim
        self.input_space = input_space

        if self.output_layer_required:
            self.mlp.add_layers([self._get_default_output_layer()])
        self.mlp.set_mlp(self)
        self.mlp.set_input_space(self.input_space)

        self._validate_mlp()

        self._params = self.mlp.get_params()
        for param in self._params:
            param.name = self.name + "_" + param.name

    def encode_conditional_params(self, X):
        """
        Maps input `X` to a tuple of parameters of the conditional distribution

        Parameters
        ----------
        X : tensor_like
            Input

        Returns
        -------
        conditional_params : tuple of tensor_like
            Tuple of parameters for the conditional distribution
        """
        conditional_params = self.mlp.fprop(X)
        if not type(conditional_params) == tuple:
            conditional_params = (conditional_params, )
        return conditional_params

    def conditional_expectation(self, conditional_params):
        """
        Given parameters of the conditional distribution, returns the
        expected value of a in p(a | b).

        Parameters
        ----------
        conditional_params : tuple of tensor_like
            Tuple of parameters for the conditional distribution
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "conditional_expectation.")

    def sample_from_conditional(self, conditional_params, epsilon=None,
                                num_samples=None):
        """
        Given a tuple of conditional parameters and an epsilon noise sample,
        generates samples from the conditional distribution.

        Parameters
        ----------
        conditional_params : tuple of tensor_like
            Tuple of parameters for the conditional distribution
        epsilon : tensor_like, optional
            Noise sample used to sample with the reparametrization trick. If
            `None`, sampling will be done without the reparametrization trick.
            Defaults to `None`.
        num_samples : int, optional
            Number of requested samples, in case the reparametrization trick is
            not used

        Returns
        -------
        rval : tensor_like
            Samples
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "sample_from_conditional.")

    def sample_from_epsilon(self, shape):
        """
        Samples from a canonical noise distribution from which conditional
        samples will be drawn using the reparametrization trick.

        Parameters
        ----------
        shape : tuple of int
            Shape of the requested samples

        Returns
        -------
        epsilon : tensor_like
            Noise samples

        Notes
        -----
        If using the reparametrization trick is not possible for this
        particular conditional distribution, will raise an exception.

        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "sample_from_epsilon, which probably "
                                  "means it is not able to sample using the "
                                  "reparametrization trick.")

    def log_conditional(self, samples, conditional_params):
        """
        Given the conditional parameters, computes the log-conditional
        probabilities of samples of this distribution.

        Parameters
        ----------
        samples : tensor_like
            Conditional samples
        conditional_params : tuple of tensor_like
            Tuple of parameters for the conditional distribution

        Returns
        -------
        log_conditonal : tensor_like
            Log-conditional probabilities
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "log_conditional.")


class BernoulliVector(Conditional):
    """
    Implements a vectorial bernoulli conditional distribution, i.e.

    .. math::
        f_\\omega(\\mathbf{a} \\mid \\mathbf{b})
        = \\prod_i \\mu_i(\\mathbf{b})^{a_i}
                   (1 - \\mu_i(\\mathbf{b}))^{(1 - a_i)}

    Parameters
    ----------
    See `Conditional`
    """
    @wraps(Conditional._get_default_output_layer)
    def _get_default_output_layer(self):
        return Linear(dim=self.ndim, layer_name='mu', irange=0.01)

    @wraps(Conditional._get_required_mlp_output_space)
    def _get_required_mlp_output_space(self):
        return VectorSpace(dim=self.ndim)

    @wraps(Conditional.sample_from_conditional)
    def sample_from_conditional(self, conditional_params, epsilon=None,
                                num_samples=None):
        if epsilon is not None:
            raise ValueError(str(self.__class__) + " is not able to sample " +
                             "using the reparametrization trick.")
        if num_samples is None:
            raise ValueError("number of requested samples needs to be given.")
        # We express mu in terms of the pre-sigmoid activations. See
        # `log_conditional` for more details.
        conditional_probs = T.nnet.sigmoid(conditional_params[0])
        return self.theano_rng.uniform(
            size=(num_samples, self.ndim),
            dtype=theano.config.floatX
        ) < conditional_probs

    @wraps(Conditional.conditional_expectation)
    def conditional_expectation(self, conditional_params):
        # `conditional_params` is composed of pre-sigmoid activations; see
        # `log_conditional` for more details.
        return T.nnet.sigmoid(conditional_params[0])

    @wraps(Conditional.log_conditional)
    def log_conditional(self, samples, conditional_params):
        # We express the probability in terms of the pre-sigmoid activations,
        # which lets us apply the log sigmoid(x) -> -softplus(-x)
        # optimization manually
        (S,) = conditional_params
        # If there are multiple samples per data point, make sure mu and
        # log_sigma are broadcasted correctly.
        if samples.ndim == 3:
            if S.ndim == 2:
                S = S.dimshuffle('x', 0, 1)
        return -(
            samples * T.nnet.softplus(-S) + (1 - samples) * T.nnet.softplus(S)
        ).sum(axis=2)


class DiagonalGaussian(Conditional):
    """
    Implements a normal conditional distribution with diagonal covariance
    matrix, i.e.

    .. math::
        f_\\omega(\\mathbf{a} \\mid \\mathbf{b})
        = \\prod_i \\exp(-(a_i - \\mu_i(\\mathbf{b}))^2 /
                         (2\\sigma_i(\\mathbf{b})^2 ) /
                   (\\sqrt{2 \\pi} \\sigma_i(\\mathbf{b}))

    Parameters
    ----------
    See `Conditional`
    """
    @wraps(Conditional._get_default_output_layer)
    def _get_default_output_layer(self):
        return CompositeLayer(
            layer_name='conditional',
            layers=[Linear(dim=self.ndim, layer_name='mu', irange=0.01),
                    Linear(dim=self.ndim, layer_name='log_sigma', irange=0.01)]
        )

    @wraps(Conditional._get_required_mlp_output_space)
    def _get_required_mlp_output_space(self):
        return CompositeSpace([VectorSpace(dim=self.ndim),
                               VectorSpace(dim=self.ndim)])

    @wraps(Conditional.monitoring_channels_from_conditional_params)
    def monitoring_channels_from_conditional_params(self, conditional_params):
        rval = OrderedDict()

        mu, log_sigma = conditional_params
        rval[self.name + '_sigma_min'] = T.exp(log_sigma).min()
        rval[self.name + '_sigma_max'] = T.exp(log_sigma).max()
        rval[self.name + '_sigma_mean'] = T.exp(log_sigma).mean()
        rval[self.name + '_sigma_std'] = T.exp(log_sigma).std()

        return rval

    @wraps(Conditional.sample_from_conditional)
    def sample_from_conditional(self, conditional_params, epsilon=None,
                                num_samples=None):
        (mu, log_sigma) = conditional_params
        if epsilon is None:
            if num_samples is None:
                raise ValueError("number of requested samples needs to be "
                                 "given.")
            return self.theano_rng.normal(size=mu.shape,
                                          avg=mu,
                                          std=T.exp(log_sigma),
                                          dtype=theano.config.floatX)
        else:
            # If there are multiple samples per data point, make sure mu and
            # log_sigma are broadcasted correctly.
            if epsilon.ndim == 3:
                if mu.ndim == 2:
                    mu = mu.dimshuffle('x', 0, 1)
                if log_sigma.ndim == 2:
                    log_sigma = log_sigma.dimshuffle('x', 0, 1)
            return mu + T.exp(log_sigma) * epsilon

    @wraps(Conditional.sample_from_epsilon)
    def sample_from_epsilon(self, shape):
        return self.theano_rng.normal(size=shape, dtype=theano.config.floatX)

    @wraps(Conditional.conditional_expectation)
    def conditional_expectation(self, conditional_params):
        return conditional_params[0]

    @wraps(Conditional.log_conditional)
    def log_conditional(self, samples, conditional_params):
        (mu, log_sigma) = conditional_params
        # If there are multiple samples per data point, make sure mu and
        # log_sigma are broadcasted correctly.
        if samples.ndim == 3:
            if log_sigma.ndim == 2:
                log_sigma = log_sigma.dimshuffle('x', 0, 1)
            if mu.ndim == 2:
                mu = mu.dimshuffle('x', 0, 1)
        return -0.5 * (
            T.log(2 * pi) + 2 * log_sigma + (samples - mu) ** 2 /
            T.exp(2 * log_sigma)
        ).sum(axis=2)
