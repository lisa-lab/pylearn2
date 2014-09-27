"""
Classes implementing logic related to the posterior distribution
:math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})` in the VAE framework
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
from pylearn2.models import Model
from pylearn2.models.mlp import Linear, CompositeLayer
from pylearn2.utils.rng import make_theano_rng
from pylearn2.utils import sharedX, wraps
from pylearn2.space import VectorSpace, CompositeSpace

pi = sharedX(numpy.pi)


class Posterior(Model):
    """
    Abstract class implementing methods related to the posterior distribution
    :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})` for the VAE framework

    Parameteters
    ------------
    encoding_model : pylearn2.models.mlp.MLP
        An MLP representing the recognition network, whose output will be used
        to compute :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`. Note that
        the MLP must be **nested**, meaning that its input space must not have
        already been defined, as `Posterior` will do it automatically.
    output_layer_required : bool, optional
        If `True`, the encoding model's output is the last hidden
        representation from which parameters of :math:`q_\\phi(\\mathbf{z}
        \\mid \\mathbf{x})` will be computed, and `Posterior` will add its own
        default output layer to the `encoding_model` MLP. If `False`, the MLP's
        last layer **is** the output layer. Defaults to `True`.
    """
    def __init__(self, encoding_model, output_layer_required=True):
        super(Posterior, self).__init__()
        if not encoding_model._nested:
            raise ValueError("`Posterior` expects an MLP whose input space "
                             "has not been defined yet. You should not "
                             "specify 'nvis' or 'input_space' when "
                             "instantiating the MLP.")
        self.encoding_model = encoding_model
        self.output_layer_required = output_layer_required

    def get_lr_scalers(self):
        """
        Returns the encoding model's learning rate scalers
        """
        return self.encoding_model.get_lr_scalers()

    def _get_default_output_layer(self):
        """
        Returns a default `Layer` mapping the encoding model's last hidden
        representation to parameters of :math:`q_\\phi(\\mathbf{z} \\mid
        \\mathbf{x})`
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_get_default_output_layer")

    def _get_required_encoder_output_space(self):
        """
        Returns the expected output space of the encoding model, i.e. a
        description of how the parameters output by the encoding model should
        look like.
        """
        raise NotImplementedError(str(self.__class__) + " does not implement "
                                  "_get_required_encoder_output_space")

    def _validate_encoding_model(self):
        """
        Makes sure the encoding model's output layer is compatible with the
        parameters expected by :math:`q_\\phi(\\mathbf{z} \\mid \\mathbf{x})`
        """
        expected_output_space = self._get_required_encoder_output_space()
        encoder_output_space = self.encoding_model.get_output_space()
        if not encoder_output_space == expected_output_space:
            raise ValueError("the specified encoding model's output space is "
                             "incompatible with " + str(self.__class__) + ": "
                             "expected " + str(expected_output_space) + " but "
                             "encoding model's output space is " +
                             str(encoder_output_space))

    def monitoring_channels_from_phi(self, phi):
        """
        Get monitoring channels for this latent component.

        By default, no monitoring channel is computed.
        """
        return OrderedDict()

    def _modify_updates(self, updates):
        """
        Modifies the parameters before a learning update is applied.

        By default, only calls the encoding model's `modify_updates` method.
        """
        self.encoding_model.modify_updates(updates)

    def get_vae(self):
        """
        Returns the VAE that this `Posterior` instance belongs to, or None
        if it has not been assigned to a VAE yet.
        """
        if hasattr(self, 'vae'):
            return self.vae
        else:
            return None

    def set_vae(self, vae):
        """
        Assigns this `Posterior` instance to a VAE.

        Parameters
        ----------
        vae : pylearn2.models.vae.VAE
            VAE to assign to
        """
        if self.get_vae() is not None:
            raise RuntimeError("this `Posterior` instance already belongs to "
                               "another VAE")
        self.vae = vae
        self.rng = self.vae.rng
        self.theano_rng = make_theano_rng(int(self.rng.randint(2 ** 30)),
                                          which_method=["normal", "uniform"])
        self.batch_size = vae.batch_size

    def initialize_parameters(self, encoder_input_space, nhid):
        """
        Initialize model parameters.

        Parameters
        ----------
        encoder_input_space : pylearn2.space.Space
            The input space for the encoding model
        nhid : int
            Number of latent units for z
        """
        self.nhid = nhid
        self.encoder_input_space = encoder_input_space

        if self.output_layer_required:
            self.encoding_model.add_layers([self._get_default_output_layer()])
        self.encoding_model.set_mlp(self)
        self.encoding_model.set_input_space(self.encoder_input_space)

        self._validate_encoding_model()

        self._params = self.encoding_model.get_params()
        for param in self._params:
            param.name = 'posterior_' + param.name

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
        rval = self.encoding_model.fprop(X)
        if not type(rval) == tuple:
            rval = (rval, )
        return rval

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


class DiagonalGaussianPosterior(Posterior):
    """
    Implements a gaussian posterior diagonal covariance matrix, i.e.

    .. math::
        q_\\phi(\\mathbf{z} \\mid \\mathbf{x})
        = \\prod_i \\exp(-(z_i - \\mu_i(\\mathbf{x}))^2 /
                         (2\\sigma_i(\\mathbf{x})^2 ) /
                   (\\sqrt{2 \\pi} \\sigma_i(\\mathbf{x}))
    """
    @wraps(Posterior.monitoring_channels_from_phi)
    def monitoring_channels_from_phi(self, phi):
        mu, log_sigma = phi
        return OrderedDict([
            ('mu_phi_min', mu.min()),
            ('mu_phi_max', mu.max()),
            ('mu_phi_mean', mu.mean()),
            ('mu_phi_std', mu.std()),
            ('sigma_phi_min', T.exp(log_sigma).min()),
            ('sigma_phi_max', T.exp(log_sigma).max()),
            ('sigma_phi_mean', T.exp(log_sigma).mean()),
            ('sigma_phi_std', T.exp(log_sigma).std())
        ])

    @wraps(Posterior._get_default_output_layer)
    def _get_default_output_layer(self):
        return CompositeLayer(
            layer_name='phi',
            layers=[Linear(dim=self.nhid, layer_name='mu', irange=0.01),
                    Linear(dim=self.nhid, layer_name='log_sigma',
                           irange=0.01)]
        )

    @wraps(Posterior._get_required_encoder_output_space)
    def _get_required_encoder_output_space(self):
        return CompositeSpace([VectorSpace(dim=self.nhid),
                               VectorSpace(dim=self.nhid)])

    @wraps(Posterior.sample_from_q_z_given_x)
    def sample_from_q_z_given_x(self, epsilon, phi):
        (mu_e, log_sigma_e) = phi
        if epsilon.ndim == 3:
            return (
                mu_e.dimshuffle('x', 0, 1) +
                T.exp(log_sigma_e.dimshuffle('x', 0, 1)) * epsilon
            )
        else:
            return mu_e + T.exp(log_sigma_e) * epsilon

    @wraps(Posterior.sample_from_epsilon)
    def sample_from_epsilon(self, shape):
        return self.theano_rng.normal(size=shape, dtype=theano.config.floatX)

    @wraps(Posterior.log_q_z_given_x)
    def log_q_z_given_x(self, z, phi):
        (mu_e, log_sigma_e) = phi
        return -0.5 * (
            T.log(2 * pi) + 2 * log_sigma_e.dimshuffle(('x', 0, 1)) +
            (z - mu_e.dimshuffle(('x', 0, 1))) ** 2 /
            T.exp(2 * log_sigma_e.dimshuffle(('x', 0, 1)))
        ).sum(axis=2)
