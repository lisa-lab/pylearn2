"""
Classes implementing visible space-related methods for the VAE framework,
namely

* initialize_parameters
* sample_from_p_x_given_z
* expectation_term
* decode_theta
* means_from_theta
* log_p_x_given_z
* _get_default_output_layer
* _get_output_space
* _validate_decoding_model
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
from pylearn2.models.mlp import Linear, Sigmoid, CompositeLayer
from pylearn2.space import VectorSpace, NullSpace, CompositeSpace
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
    output_layer_required : bool, optional
        If `True`, the decoding model's output is the last hidden
        representation from which parameters of :math:`p_\\theta(\\mathbf{x}
        \\mid \\mathbf{z})` will be computed, and `Visible` will add its own
        default output layer to the `decoding_model` MLP. If `False`, the MLP's
        last layer **is** the output layer. Defaults to `True`.
    """
    def __init__(self, decoding_model, output_layer_required=True):
        self.decoding_model = decoding_model
        self.output_layer_required = output_layer_required

    def _get_default_output_layer(self):
        """
        Returns a default `Layer` mapping the decoding model's last hidden
        representation to parameters of :math:`p_\\theta(\\mathbf{x} \\mid
        \\mathbf{z})`
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_get_default_output_layer")

    def _get_output_space(self):
        """
        Returns the expected output space of the decoding model, i.e. a
        description of how the parameters output by the decoding model should
        look like.
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_get_output_space")

    def _validate_decoding_model(self):
        """
        Makes sure the decoding model's output layer is compatible with the
        parameters expected by :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})`
        """
        expected_output_space = self._get_output_space()
        model_output_space = self.decoding_model.get_output_space()
        if not model_output_space == expected_output_space:
            raise ValueError("the specified decoding model's output space is "
                             "incompatible with " + str(self.__class__) + ": "
                             "expected " + str(expected_output_space) + " but "
                             "decoding model's output space is " +
                             str(model_output_space))

    def get_weights(self):
        return self.decoding_model.get_weights()

    def get_monitoring_data_specs(self):
        """
        Get the data_specs describing the data for get_monitoring_channels.

        By default, nothing is requested for computing monitoring channels.
        """
        return (NullSpace(), '')

    def get_monitoring_channels(self, data):
        """
        Get monitoring channels for this visible component.

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
        Returns the VAE that this `Visible` instance belongs to, or None
        if it has not been assigned to a VAE yet.
        """
        if hasattr(self, 'vae'):
            return self.vae
        else:
            return None

    def set_vae(self, vae):
        """
        Assigns this `Visible` instance to a VAE.

        Parameters
        ----------
        vae : pylearn2.models.vae.VAE
            VAE to assign to
        """
        if self.get_vae() is not None:
            raise RuntimeError("this `Visible` instance already belongs to "
                               "another VAE")
        self.vae = vae

    def initialize_parameters(self, decoder_input_space, nvis):
        """
        Initialize model parameters.
        """
        self.nvis = nvis
        self.decoder_input_space = decoder_input_space
        if self.output_layer_required:
            self.decoding_model.add_layer(self._get_default_output_layer())
        self._validate_decoding_model()
        self._params = self.decoding_model.get_params()

    def get_params(self):
        """
        Return the visible space-related parameters
        """
        return self._params

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
        z = self.decoder_input_space.format_as(
            batch=z,
            space=self.decoding_model.get_input_space()
        )
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


class BinaryVisible(Visible):
    """
    Subclass implementing visible-related methods for binary inputs

    Parameters
    ----------
    decoding_model : pylearn2.models.mlp.MLP
        An MLP representing the generative network, whose output will be used
        to compute :math:`p_\\theta(\\mathbf{x} \\mid \\mathbf{z})`
    output_layer_required : bool, optional
        If `True`, the decoding model's output is the last hidden
        representation from which parameters of :math:`p_\\theta(\\mathbf{x}
        \\mid \\mathbf{z})` will be computed, and `Visible` will add its own
        default output layer to the `decoding_model` MLP. If `False`, the MLP's
        last layer **is** the output layer. Defaults to `True`.
    isigma : float, optional
        Standard deviation on the zero-mean distribution from which parameters
        initialized by the model itself will be drawn. Defaults to 0.01.
    """
    def __init__(self, decoding_model, output_layer_required=True,
                 isigma=0.01):
        super(BinaryVisible, self).__init__(decoding_model,
                                            output_layer_required)
        self.isigma = isigma

    @wraps(Visible._get_default_output_layer)
    def _get_default_output_layer(self):
        return Linear(dim=self.nvis, layer_name='mu', irange=0.01)

    @wraps(Visible._get_output_space)
    def _get_output_space(self):
        return VectorSpace(dim=self.nvis)

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
    output_layer_required : bool, optional
        If `True`, the decoding model's output is the last hidden
        representation from which parameters of :math:`p_\\theta(\\mathbf{x}
        \\mid \\mathbf{z})` will be computed, and `Visible` will add its own
        default output layer to the `decoding_model` MLP. If `False`, the MLP's
        last layer **is** the output layer. Defaults to `True`.
    isigma : float, optional
        Standard deviation on the zero-mean distribution from which parameters
        initialized by the model itself will be drawn. Defaults to 0.01.
    """
    def __init__(self, decoding_model, output_layer_required=True,
                 isigma=0.01):
        super(ContinuousVisible, self).__init__(decoding_model,
                                                output_layer_required)
        self.isigma = isigma

    @wraps(Visible._get_default_output_layer)
    def _get_default_output_layer(self):
        return CompositeLayer(
            layer_name='phi',
            layers=[Sigmoid(dim=self.nvis, layer_name='mu', irange=0.01),
                    Linear(dim=self.nvis, layer_name='log_sigma', irange=0.01)]
        )

    @wraps(Visible._get_output_space)
    def _get_output_space(self):
        return CompositeSpace([VectorSpace(dim=self.nvis),
                               VectorSpace(dim=self.nvis)])

    @wraps(Visible.sample_from_p_x_given_z)
    def sample_from_p_x_given_z(self, num_samples, theta):
        (mu_d, log_sigma_d) = theta
        return theano_rng.normal(size=mu_d.shape,
                                 avg=mu_d,
                                 std=T.exp(log_sigma_d),
                                 dtype=theano.config.floatX)

    @wraps(Visible.expectation_term)
    def expectation_term(self, X, theta):
        (mu_d, log_sigma_d) = theta
        return -0.5 * (T.log(2 * pi) + 2 * log_sigma_d +
                       (X - mu_d) ** 2 / T.exp(2 * log_sigma_d))

    @wraps(Visible.means_from_theta)
    def means_from_theta(self, theta):
        return theta[0]

    @wraps(Visible.log_p_x_given_z)
    def log_p_x_given_z(self, X, theta):
        (mu_d, log_sigma_d) = theta
        return -0.5 * (T.log(2 * pi) + 2 * log_sigma_d +
                       (X - mu_d) ** 2 / T.exp(2 * log_sigma_d)).sum(axis=2)
