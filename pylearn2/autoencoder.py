"""Autoencoders, denoising autoencoders, and stacked DAEs."""
# Standard library imports
import functools
from itertools import izip
import operator

# Third-party imports
import numpy
import theano
from theano import tensor

# Local imports
from pylearn2.base import Block, StackedBlocks
from pylearn2.models import Model
from pylearn2.utils import sharedX
from pylearn2.utils.theano_graph import is_pure_elemwise
from pylearn2.space import VectorSpace

theano.config.warn.sum_div_dimshuffle_bug = False

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams


class Autoencoder(Block, Model):
    """
    Base class implementing ordinary autoencoders.

    More exotic variants (denoising, contracting autoencoders) can inherit
    much of the necessary functionality and override what they need.
    """
    def __init__(self, nvis, nhid, act_enc, act_dec,
                 tied_weights=False, irange=1e-3, rng=9001):
        """
        Allocate an autoencoder object.

        Parameters
        ----------
        nvis : int
            Number of visible units (input dimensions) in this model.
            A value of 0 indicates that this block will be left partially
            initialized until later (e.g., when the dataset is loaded and
            its dimensionality is known)
        nhid : int
            Number of hidden units in this model.
        act_enc : callable or string
            Activation function (elementwise nonlinearity) to use for the
            encoder. Strings (e.g. 'tanh' or 'sigmoid') will be looked up as
            functions in `theano.tensor.nnet` and `theano.tensor`. Use `None`
            for linear units.
        act_dec : callable or string
            Activation function (elementwise nonlinearity) to use for the
            decoder. Strings (e.g. 'tanh' or 'sigmoid') will be looked up as
            functions in `theano.tensor.nnet` and `theano.tensor`. Use `None`
            for linear units.
        tied_weights : bool, optional
            If `False` (default), a separate set of weights will be allocated
            (and learned) for the encoder and the decoder function. If `True`,
            the decoder weight matrix will be constrained to be equal to the
            transpose of the encoder weight matrix.
        irange : float, optional
            Width of the initial range around 0 from which to sample initial
            values for the weights.
        rng : RandomState object or seed
            NumPy random number generator object (or seed to create one) used
            to initialize the model parameters.
        """
        super(Autoencoder, self).__init__()
        assert nvis >= 0, "Number of visible units must be non-negative"
        assert nhid > 0, "Number of hidden units must be positive"

        self.input_space = VectorSpace(nvis)
        self.output_space = VectorSpace(nhid)

        # Save a few parameters needed for resizing
        self.nhid = nhid
        self.irange = irange
        self.tied_weights = tied_weights
        if not hasattr(rng, 'randn'):
            self.rng = numpy.random.RandomState(rng)
        else:
            self.rng = rng
        self._initialize_hidbias()
        if nvis > 0:
            self._initialize_visbias(nvis)
            self._initialize_weights(nvis)
        else:
            self.visbias = None
            self.weights = None

        seed = int(self.rng.randint(2 ** 30))
        self.s_rng = RandomStreams(seed)
        if tied_weights:
            self.w_prime = self.weights.T
        else:
            self._initialize_w_prime(nvis)

        def _resolve_callable(conf, conf_attr):
            if conf[conf_attr] is None or conf[conf_attr] == "linear":
                return None
            # If it's a callable, use it directly.
            if hasattr(conf[conf_attr], '__call__'):
                return conf[conf_attr]
            elif (conf[conf_attr] in globals()
                  and hasattr(globals()[conf[conf_attr]], '__call__')):
                return globals()[conf[conf_attr]]
            elif hasattr(tensor.nnet, conf[conf_attr]):
                return getattr(tensor.nnet, conf[conf_attr])
            elif hasattr(tensor, conf[conf_attr]):
                return getattr(tensor, conf[conf_attr])
            else:
                raise ValueError("Couldn't interpret %s value: '%s'" %
                                    (conf_attr, conf[conf_attr]))

        self.act_enc = _resolve_callable(locals(), 'act_enc')
        self.act_dec = _resolve_callable(locals(), 'act_dec')
        self._params = [
            self.visbias,
            self.hidbias,
            self.weights,
        ]
        if not self.tied_weights:
            self._params.append(self.w_prime)

    def _initialize_weights(self, nvis, rng=None, irange=None):
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        # TODO: use weight scaling factor if provided, Xavier's default else
        self.weights = sharedX(
            (.5 - rng.rand(nvis, self.nhid)) * irange,
            name='W',
            borrow=True
        )

    def _initialize_hidbias(self):
        self.hidbias = sharedX(
            numpy.zeros(self.nhid),
            name='hb',
            borrow=True
        )

    def _initialize_visbias(self, nvis):
        self.visbias = sharedX(
            numpy.zeros(nvis),
            name='vb',
            borrow=True
        )

    def _initialize_w_prime(self, nvis, rng=None, irange=None):
        assert not self.tied_weights, (
            "Can't initialize w_prime in tied weights model; "
            "this method shouldn't have been called"
        )
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        self.w_prime = sharedX(
            (.5 - rng.rand(self.nhid, nvis)) * irange,
            name='Wprime',
            borrow=True
        )

    def set_visible_size(self, nvis, rng=None):
        """
        Create and initialize the necessary parameters to accept
        `nvis` sized inputs.

        Parameters
        ----------
        nvis : int
            Number of visible units for the model.
        rng : RandomState object or seed, optional
            NumPy random number generator object (or seed to create one) used
            to initialize the model parameters. If not provided, the stored
            rng object (from the time of construction) will be used.
        """
        if self.weights is not None:
            raise ValueError('parameters of this model already initialized; '
                             'create a new object instead')
        if rng is not None:
            self.rng = rng
        else:
            rng = self.rng
        self._initialize_visbias(nvis)
        self._initialize_weights(nvis, rng)
        if not self.tied_weights:
            self._initialize_w_prime(nvis, rng)
        self._set_params()

    def _hidden_activation(self, x):
        """
        Single minibatch activation function.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing the input minibatch.

        Returns
        -------
        y : tensor_like
            (Symbolic) hidden unit activations given the input.
        """
        if self.act_enc is None:
            act_enc = lambda x: x
        else:
            act_enc = self.act_enc
        return act_enc(self._hidden_input(x))

    def _hidden_input(self, x):
        """
        Given a single minibatch, computes the input to the
        activation nonlinearity without applying it.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing the input minibatch.

        Returns
        -------
        y : tensor_like
            (Symbolic) input flowing into the hidden layer nonlinearity.
        """
        return self.hidbias + tensor.dot(x, self.weights)

    def encode(self, inputs):
        """
        Map inputs through the encoder function.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second indexing
            data dimensions.

        Returns
        -------
        encoded : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            minibatch(es) after encoding.
        """
        if isinstance(inputs, tensor.Variable):
            return self._hidden_activation(inputs)
        else:
            return [self.encode(v) for v in inputs]

    def decode(self, hiddens):
        """
        Map inputs through the encoder function.

        Parameters
        ----------
        hiddens : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second indexing
            data dimensions.

        Returns
        -------
        decoded : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            minibatch(es) after decoding.
        """
        if self.act_dec is None:
            act_dec = lambda x: x
        else:
            act_dec = self.act_dec
        if isinstance(hiddens, tensor.Variable):
            return act_dec(self.visbias + tensor.dot(hiddens, self.w_prime))
        else:
            return [self.decode(v) for v in hiddens]

    def reconstruct(self, inputs):
        """
        Reconstruct (decode) the inputs after mapping through the encoder.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded and reconstructed. Assumed to be
            2-tensors, with the first dimension indexing training examples and
            the second indexing data dimensions.

        Returns
        -------
        reconstructed : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            reconstructed minibatch(es) after encoding/decoding.
        """
        return self.decode(self.encode(inputs))

    def __call__(self, inputs):
        """
        Forward propagate (symbolic) input through this module, obtaining
        a representation to pass on to layers above.

        This just aliases the `encode()` function for syntactic
        sugar/convenience.
        """
        return self.encode(inputs)

    def get_weights(self, borrow=False):

        return self.weights.get_value(borrow = borrow)

    def get_weights_format(self):

        return ['v', 'h']

class DenoisingAutoencoder(Autoencoder):
    """
    A denoising autoencoder learns a representation of the input by
    reconstructing a noisy version of it.
    """
    def __init__(self, corruptor, nvis, nhid, act_enc, act_dec,
                 tied_weights=False, irange=1e-3, rng=9001):
        """
        Allocate a denoising autoencoder object.

        Parameters
        ----------
        corruptor : object
            Instance of a corruptor object to use for corrupting the
            input.

        Notes
        -----
        The remaining parameters are identical to those of the constructor
        for the Autoencoder class; see the `Autoencoder.__init__` docstring
        for details.
        """
        super(DenoisingAutoencoder, self).__init__(
            nvis,
            nhid,
            act_enc,
            act_dec,
            tied_weights,
            irange,
            rng
        )
        self.corruptor = corruptor

    def reconstruct(self, inputs):
        """
        Reconstruct the inputs after corrupting and mapping through the
        encoder and decoder.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be corrupted and reconstructed. Assumed to be
            2-tensors, with the first dimension indexing training examples and
            the second indexing data dimensions.

        Returns
        -------
        reconstructed : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            reconstructed minibatch(es) after corruption and encoding/decoding.
        """
        corrupted = self.corruptor(inputs)
        return super(DenoisingAutoencoder, self).reconstruct(corrupted)


class ContractiveAutoencoder(Autoencoder):
    """
    A contracting autoencoder works like a regular autoencoder, and adds an
    extra term to its cost function.
    """
    @functools.wraps(Autoencoder.__init__)
    def __init__(self, *args, **kwargs):
        super(ContractiveAutoencoder, self).__init__(*args, **kwargs)
        dummyinput = tensor.matrix()
        if not is_pure_elemwise(self.act_enc(dummyinput), [dummyinput]):
            raise ValueError("Invalid encoder activation function: "
                             "not an elementwise function of its input")

    def _activation_grad(self, inputs):
        """
        Calculate (symbolically) the contracting autoencoder penalty term.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) on which the penalty is calculated. Assumed to be
            2-tensors, with the first dimension indexing training examples and
            the second indexing data dimensions.

        Returns
        -------
        act_grad : tensor_like
            2-dimensional tensor representing, dh/da for every pre/postsynaptic pair,
            which we can easily do by taking the gradient of the sum of the
            hidden units activations w.r.t the presynaptic activity,
            since the gradient of hiddens.sum() with respect to hiddens is a matrix of ones!

        Notes
        -----
        Theano's differentiation capabilities do not currently allow
        (efficient) automatic evaluation of the Jacobian, mainly because
        of the immature state of the `scan` operator. Here we use a
        "semi-automatic" hack that works for hidden layers of the for
        :math:`s(Wx + b)`, where `s` is the activation function, :math:`W`
        is `self.weights`, and :math:`b` is `self.hidbias`, by only taking
        the derivative of :math:`s` with respect :math:`a = Wx + b` and
        manually constructing the Jacobian from there.

        Because of this implementation depends *critically* on the
        _hidden_inputs() method implementing only an affine transformation
        by the weights (i.e. :math:`Wx + b`), and the activation function
        `self.act_enc` applying an independent, elementwise operation.
        """

        # Compute the input flowing into the hidden units, i.e. the
        # value before applying the nonlinearity/activation function
        acts = self._hidden_input(inputs)
        # Apply the activating nonlinearity.
        hiddens = self.act_enc(acts)
        act_grad = tensor.grad(hiddens.sum(), acts)
        return act_grad

    def jacobian_h_x(self, inputs):
        """
        Calculate (symbolically) the contracting autoencoder penalty term.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) on which the penalty is calculated. Assumed to be
            2-tensors, with the first dimension indexing training examples and
            the second indexing data dimensions.

        Returns
        -------
        jacobian : tensor_like
            3-dimensional tensor representing, for each mini-batch
            example, the Jacobian matrix of the encoder
            transformation.
            You can then apply the penalty you want on it,
            or use the contraction_penalty method to have a default one.
        """
        # As long as act_enc is an elementwise operator, the Jacobian
        # of a act_enc(Wx + b) hidden layer has a Jacobian of the
        # following form.
        act_grad = self._activation_grad(inputs)
        jacobian = self.weights * act_grad.dimshuffle(0, 'x', 1)
        return jacobian

    def contraction_penalty(self, inputs):
        """
        Calculate (symbolically) the contracting autoencoder penalty term.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) on which the penalty is calculated. Assumed to be
            2-tensors, with the first dimension indexing training examples and
            the second indexing data dimensions.

        Returns
        -------
        jacobian : tensor_like
            1-dimensional tensor representing, for each mini-batch
            example, the penalty of the encoder transformation.

            Add this to the output of a Cost object, such as
            SquaredError, to penalize it.
        """
        act_grad = self._activation_grad(inputs)
        frob_norm = tensor.dot(tensor.sqr(act_grad), tensor.sqr(self.weights.sum(axis=0)))
        contract_penalty = frob_norm.sum() / inputs.shape[0]
        return contract_penalty

class HigherOrderContractiveAutoencoder(ContractiveAutoencoder):
    """Higher order contractive autoencoder.
    Adds higher orders regularization
    """
    def __init__(self, corruptor, num_corruptions, nvis, nhid, act_enc,
                    act_dec, tied_weights=False, irange=1e-3, rng=9001):
        """
        Allocate a higher order contractive autoencoder object.

        Parameters
        ----------
        corruptor : object
        Instance of a corruptor object to use for corrupting the
        input.

        num_corruptions : integer
        number of corrupted inputs to use

        Notes
        -----
        The remaining parameters are identical to those of the constructor
        for the Autoencoder class; see the `ContractiveAutoEncoder.__init__` docstring
        for details.
        """
        super(HigherOrderContractiveAutoencoder, self).__init__(
            nvis,
            nhid,
            act_enc,
            act_dec,
            tied_weights,
            irange,
            rng
        )
        self.corruptor = corruptor
        self.num_corruptions = num_corruptions


    def higher_order_penalty(self, inputs):
        """
        Stochastic approximation of Hessian Frobenius norm
        """

        corrupted_inputs = [self.corruptor(inputs) for times in\
                            range(self.num_corruptions)]

        hessian = tensor.concatenate([self.jacobian_h_x(inputs) - \
                                self.jacobian_h_x(corrupted) for\
                                corrupted in corrupted_inputs])

        return (hessian ** 2).mean()


class UntiedAutoencoder(Autoencoder):
    def __init__(self, base):
        if not base.tied_weights:
            raise ValueError("%s is not a tied-weights autoencoder" %
                             str(base))
        self.weights = tensor.shared(base.weights.get_value(borrow=False),
                                     name='weights')
        self.visbias = tensor.shared(base.visbias.get_value(borrow=False),
                                     name='vb')
        self.hidbias = tensor.shared(base.visbias.get_value(borrow=False),
                                     name='hb')
        self.w_prime = tensor.shared(base.weights.get_value(borrow=False).T,
                                     name='w_prime')
        self._params = [self.visbias, self.hidbias, self.weights, self.w_prime]


class DeepComposedAutoencoder(Autoencoder):
    """
    A deep autoencoder composed of several single-layer
    autoencoders.
    """
    def __init__(self, autoencoders):
        """
        Construct a deep autoencoder from several single layer
        autoencoders.

        Parameters
        ----------
        autoencoders : list
            A list of autoencoder objects.
        """
        # TODO: Check that the dimensions line up.
        self.autoencoders = list(autoencoders)

    @functools.wraps(Autoencoder.encode)
    def encode(self, inputs):
        current = inputs
        for encoder in self.autoencoders:
            current = encoder.encode(current)
        return current

    @functools.wraps(Autoencoder.decode)
    def decode(self, hiddens):
        current = hiddens
        for decoder in self.autoencoders[::-1]:
            current = decoder.decode(current)
        return current

    @functools.wraps(Model.get_params)
    def get_params(self):
        return reduce(operator.add,
                      [ae.get_params() for ae in self.autoencoders])


def build_stacked_ae(nvis, nhids, act_enc, act_dec,
                     tied_weights=False, irange=1e-3, rng=None,
                     corruptor=None, contracting=False):
    """Allocate a stack of autoencoders."""
    if not hasattr(rng, 'randn'):
        rng = numpy.random.RandomState(rng)
    layers = []
    final = {}
    # "Broadcast" arguments if they are singular, or accept sequences if
    # they are the same length as nhids
    for c in ['corruptor', 'contracting', 'act_enc', 'act_dec',
              'tied_weights', 'irange']:
        if type(locals()[c]) is not str and hasattr(locals()[c], '__len__'):
            assert len(nhids) == len(locals()[c])
            final[c] = locals()[c]
        else:
            final[c] = [locals()[c]] * len(nhids)
    # The number of visible units in each layer is the initial input
    # size and the first k-1 hidden unit sizes.
    nviss = [nvis] + nhids[:-1]
    seq = izip(nhids, nviss,
        final['act_enc'],
        final['act_dec'],
        final['corruptor'],
        final['contracting'],
        final['tied_weights'],
        final['irange'],
    )
    # Create each layer.
    for (nhid, nvis, act_enc, act_dec, corr, cae, tied, ir) in seq:
        args = (nvis, nhid, act_enc, act_dec, tied, ir, rng)
        if cae and corr is not None:
            raise ValueError("Can't specify denoising and contracting "
                             "objectives simultaneously")
        elif cae:
            autoenc = ContractiveAutoencoder(*args)
        elif corr is not None:
            autoenc = DenoisingAutoencoder(corr, *args)
        else:
            autoenc = Autoencoder(*args)
        layers.append(autoenc)

    # Create the stack
    return StackedBlocks(layers)
