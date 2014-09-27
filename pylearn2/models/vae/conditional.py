"""
Classes implementing logic related to the conditional distribution
:math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})` in the VAE framework
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
from pylearn2.models import Model
from pylearn2.models.mlp import Linear, Sigmoid, CompositeLayer
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils.rng import make_theano_rng
from pylearn2.utils import wraps, sharedX

pi = sharedX(numpy.pi)


class Conditional(Model):
    """
    Abstract class implementing methods related to the conditional distribution
    :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})` for the VAE framework.

    Parameteters
    ------------
    decoding_model : pylearn2.models.mlp.MLP
        An MLP representing the generative network, whose output will be used
        to compute :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})`. Note that
        the MLP must be **nested**, meaning that its input space must not have
        already been defined, as `Conditional` will do it automatically.
    output_layer_required : bool, optional
        If `True`, the decoding model's output is the last hidden
        representation from which parameters of :math:`p_\\theta(\\mathbf{x}
        \\mid \\mathbf{z})` will be computed, and `Conditional` will add its
        own default output layer to the `decoding_model` MLP. If `False`, the
        MLP's last layer **is** the output layer. Defaults to `True`.
    """
    def __init__(self, decoding_model, output_layer_required=True):
        super(Conditional, self).__init__()
        if not decoding_model._nested:
            raise ValueError("Conditional expects an MLP whose input space "
                             "has not been defined yet. You should not "
                             "specify 'nvis' or 'input_space' when "
                             "instantiating the MLP.")
        self.decoding_model = decoding_model
        self.output_layer_required = output_layer_required

    def get_lr_scalers(self):
        """
        Returns the decoding model's learning rate scalers
        """
        return self.decoding_model.get_lr_scalers()

    def _get_default_output_layer(self):
        """
        Returns a default `Layer` mapping the decoding model's last hidden
        representation to parameters of :math:`p_\\theta(\\mathbf{x} \\mid
        \\mathbf{z})`
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_get_default_output_layer")

    def _get_required_decoder_output_space(self):
        """
        Returns the expected output space of the decoding model, i.e. a
        description of how the parameters output by the decoding model should
        look like.
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_get_required_decoder_output_space")

    def _validate_decoding_model(self):
        """
        Makes sure the decoding model's output layer is compatible with the
        parameters expected by :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})`
        """
        expected_output_space = self._get_required_decoder_output_space()
        decoder_output_space = self.decoding_model.get_output_space()
        if not decoder_output_space == expected_output_space:
            raise ValueError("the specified decoding model's output space is "
                             "incompatible with " + str(self.__class__) + ": "
                             "expected " + str(expected_output_space) + " but "
                             "decoding model's output space is " +
                             str(model_output_space))

    @wraps(Model.get_weights)
    def get_weights(self):
        return self.decoding_model.get_weights()

    def monitoring_channels_from_theta(self, theta):
        """
        Get monitoring channels for this visible component.

        By default, no monitoring channel is computed.
        """
        return OrderedDict()

    def _modify_updates(self, updates):
        """
        Modifies the parameters before a learning update is applied.

        By default, only calls the decoding model's `modify_updates` method.
        """
        self.decoding_model.modify_updates(updates)

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
            raise RuntimeError("this `Conditional` instance already belongs "
                               "to another VAE")
        self.vae = vae
        self.rng = self.vae.rng
        self.theano_rng = make_theano_rng(int(self.rng.randint(2 ** 30)),
                                          which_method=["normal", "uniform"])
        self.batch_size = vae.batch_size

    def initialize_parameters(self, decoder_input_space, nvis):
        """
        Initialize model parameters.
        """
        self.nvis = nvis
        self.decoder_input_space = decoder_input_space

        if self.output_layer_required:
            self.decoding_model.add_layers([self._get_default_output_layer()])
        self.decoding_model.set_mlp(self)
        self.decoding_model.set_input_space(self.decoder_input_space)

        self._validate_decoding_model()

        self._conditional_params = self.decoding_model.get_params()
        for param in self._conditional_params:
            param.name = 'conditional_' + param.name

    def get_conditional_params(self):
        """
        Return the visible space-related parameters
        """
        return self._conditional_params

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
        rval = self.decoding_model.fprop(z)
        if not type(rval) == tuple:
            rval = (rval, )
        return rval

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


class BernoulliVector(Conditional):
    """
    Implements a vectorial bernoulli conditional distribution, i.e.
    
    .. math::
        `p_\\theta(\\mathbf{x} \\mid \\mathbf{z})
        = \\prod_i \\mu_i(\\mathbf{z})^{x_i}
                   (1 - \\mu_i(\\mathbf{z}))^{(1 - x_i)}
    """
    @wraps(Conditional._get_default_output_layer)
    def _get_default_output_layer(self):
        return Linear(dim=self.nvis, layer_name='mu', irange=0.01)

    @wraps(Conditional._get_required_decoder_output_space)
    def _get_required_decoder_output_space(self):
        return VectorSpace(dim=self.nvis)

    @wraps(Conditional.sample_from_p_x_given_z)
    def sample_from_p_x_given_z(self, num_samples, theta):
        # We express mu in terms of the pre-sigmoid activations. See
        # `log_p_x_given_z` for more details.
        p_x_given_z = T.nnet.sigmoid(theta[0])
        return self.theano_rng.uniform(
            size=(num_samples, self.nvis),
            dtype=theano.config.floatX
        ) < p_x_given_z

    @wraps(Conditional.expectation_term)
    def expectation_term(self, X, theta):
        # We express the expectation term in terms of the pre-sigmoid
        # activations, which lets us apply the log sigmoid(x) -> -softplus(-x)
        # optimization manually.
        (S,) = theta
        return -(X * T.nnet.softplus(-S) + (1 - X) * T.nnet.softplus(S))

    @wraps(Conditional.means_from_theta)
    def means_from_theta(self, theta):
        # Theta is composed of pre-sigmoid activations; see `_log_p_x_given_z`
        # for more details.
        return T.nnet.sigmoid(theta[0])

    @wraps(Conditional.log_p_x_given_z)
    def log_p_x_given_z(self, X, theta):
        # We express the probability in terms of the pre-sigmoid activations,
        # which lets us apply the log sigmoid(x) -> -softplus(-x)
        # optimization manually; see `_decode_theta` for more details.
        (S,) = theta
        return -(
            X * T.nnet.softplus(-S) + (1 - X) * T.nnet.softplus(S)
        ).sum(axis=2)


class DiagonalGaussianConditional(Conditional):
    """
    Implements a normal conditional distribution with diagonal covariance
    matrix, i.e.
    
    .. math::
        `p_\\theta(\\mathbf{x} \\mid \\mathbf{z})
        = \\prod_i \\exp(-(x_i - \\mu_i(\\mathbf{z}))^2 /
                         (2\\sigma_i(\\mathbf{z})^2 ) /
                   (\\sqrt{2 \\pi} \\sigma_i(\\mathbf{z}))
    """
    @wraps(Conditional._get_default_output_layer)
    def _get_default_output_layer(self):
        return CompositeLayer(
            layer_name='theta',
            layers=[Sigmoid(dim=self.nvis, layer_name='mu', irange=0.01),
                    Linear(dim=self.nvis, layer_name='log_sigma', irange=0.01)]
        )

    @wraps(Conditional._get_required_decoder_output_space)
    def _get_required_decoder_output_space(self):
        return CompositeSpace([VectorSpace(dim=self.nvis),
                               VectorSpace(dim=self.nvis)])

    @wraps(Conditional.monitoring_channels_from_theta)
    def monitoring_channels_from_theta(self, theta):
        rval = OrderedDict()

        mu, log_sigma = theta
        rval['sigma_theta_min'] = T.exp(log_sigma).min()
        rval['sigma_theta_max'] = T.exp(log_sigma).max()
        rval['sigma_theta_mean'] = T.exp(log_sigma).mean()
        rval['sigma_theta_std'] = T.exp(log_sigma).std()

        return rval

    @wraps(Conditional.sample_from_p_x_given_z)
    def sample_from_p_x_given_z(self, num_samples, theta):
        (mu_d, log_sigma_d) = theta
        return self.theano_rng.normal(size=mu_d.shape,
                                      avg=mu_d,
                                      std=T.exp(log_sigma_d),
                                      dtype=theano.config.floatX)

    @wraps(Conditional.expectation_term)
    def expectation_term(self, X, theta):
        (mu_d, log_sigma_d) = theta
        return -0.5 * (T.log(2 * pi) + 2 * log_sigma_d +
                       (X - mu_d) ** 2 / T.exp(2 * log_sigma_d))

    @wraps(Conditional.means_from_theta)
    def means_from_theta(self, theta):
        return theta[0]

    @wraps(Conditional.log_p_x_given_z)
    def log_p_x_given_z(self, X, theta):
        (mu_d, log_sigma_d) = theta
        return -0.5 * (T.log(2 * pi) + 2 * log_sigma_d +
                       (X - mu_d) ** 2 / T.exp(2 * log_sigma_d)).sum(axis=2)
