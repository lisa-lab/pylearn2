"""
Autoencoders, denoising autoencoders, and stacked DAEs.
"""
# Standard library imports
import functools
import operator

# Third-party imports
import numpy
import theano
from theano import tensor
from theano.compat.six.moves import zip as izip, reduce

# Local imports
from pylearn2.blocks import Block, StackedBlocks
from pylearn2.models import Model
from pylearn2.utils import sharedX
from pylearn2.utils.theano_graph import is_pure_elemwise
from pylearn2.utils.rng import make_np_rng, make_theano_rng
from pylearn2.space import VectorSpace

theano.config.warn.sum_div_dimshuffle_bug = False


class AbstractAutoencoder(Model, Block):
    """
    Abstract class for autoencoders.
    """
    def __init__(self):
        super(AbstractAutoencoder, self).__init__()

    def encode(self, inputs):
        """
        Map inputs through the encoder function.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing data dimensions.

        Returns
        -------
        encoded : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            minibatch(es) after encoding.
        """
        raise NotImplementedError(
            str(type(self)) + " does not implement encode.")

    def decode(self, hiddens):
        """
        Map inputs through the encoder function.

        Parameters
        ----------
        hiddens : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing data dimensions.

        Returns
        -------
        decoded : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            minibatch(es) after decoding.
        """
        raise NotImplementedError(
            str(type(self)) + " does not implement decode.")

    def reconstruct(self, inputs):
        """
        Reconstruct (decode) the inputs after mapping through the encoder.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded and reconstructed. Assumed to be
            2-tensors, with the first dimension indexing training examples
            and the second indexing data dimensions.

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


class Autoencoder(AbstractAutoencoder):
    """
    Base class implementing ordinary autoencoders.

    More exotic variants (denoising, contracting autoencoders) can inherit
    much of the necessary functionality and override what they need.

    Parameters
    ----------
    nvis : int
        Number of visible units (input dimensions) in this model.
        A value of 0 indicates that this block will be left partially
        initialized until later (e.g., when the dataset is loaded and
        its dimensionality is known).  Note: There is currently a bug
        when nvis is set to 0. For now, you should not set nvis to 0.
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
        (and learned) for the encoder and the decoder function. If
        `True`, the decoder weight matrix will be constrained to be equal
        to the transpose of the encoder weight matrix.
    irange : float, optional
        If specified, initialized each weight randomly in U(-irange, irange).
        Must be specified if istdev is not. Defaults to 1e-3.
    istdev : float, optional
        If specified, initialize each weight randomly from N(0,istdev). Must
        be specified if irange is not. Default to None.
    rng : RandomState object or seed, optional
        NumPy random number generator object (or seed to create one) used
        to initialize the model parameters.
    """

    def __init__(self, nvis, nhid, act_enc, act_dec,
                 tied_weights=False, irange=1e-3, istdev=None, rng=9001):
        """
        WRITEME
        """
        super(Autoencoder, self).__init__()
        assert nvis > 0, "Number of visible units must be non-negative"
        assert nhid > 0, "Number of hidden units must be positive"

        self.input_space = VectorSpace(nvis)
        self.output_space = VectorSpace(nhid)

        # Save a few parameters needed for resizing
        self.nvis = nvis
        self.nhid = nhid
        self.irange = irange
        self.istdev = istdev
        self.tied_weights = tied_weights
        self.rng = make_np_rng(rng, which_method="randn")
        self._initialize_hidbias()
        if nvis > 0:
            self._initialize_visbias(nvis)
            self._initialize_weights(nvis)
        else:
            self.visbias = None
            self.weights = None

        seed = int(self.rng.randint(2 ** 30))

        # why a theano rng? should we remove it?
        self.s_rng = make_theano_rng(seed, which_method="uniform")

        if tied_weights and self.weights is not None:
            self.w_prime = self.weights.T
        else:
            self._initialize_w_prime(nvis)

        def _resolve_callable(conf, conf_attr):
            """
            .. todo::

                WRITEME
            """
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

    def _initialize_weights(self, nvis, rng=None, irange=None, istdev=None):
        """
        .. todo::

            WRITEME
        """
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        if istdev is None:
            istdev = self.istdev

        # TODO: use weight scaling factor if provided, Xavier's default else
        if irange is not None:
            assert istdev is None
            W = rng.uniform(
                -irange,
                irange,
                (nvis, self.nhid)
            )
        else:
            assert istdev is not None
            W = rng.randn(nvis, self.nhid) * istdev

        self.weights = sharedX(W, name='W', borrow=True)

    def _initialize_hidbias(self):
        """
        .. todo::

            WRITEME
        """
        self.hidbias = sharedX(
            numpy.zeros(self.nhid),
            name='hb',
            borrow=True
        )

    def _initialize_visbias(self, nvis):
        """
        .. todo::

            WRITEME
        """
        self.visbias = sharedX(
            numpy.zeros(nvis),
            name='vb',
            borrow=True
        )

    def _initialize_w_prime(self, nvis, rng=None, irange=None, istdev=None):
        """
        .. todo::

            WRITEME
        """
        assert not self.tied_weights, (
            "Can't initialize w_prime in tied weights model; "
            "this method shouldn't have been called"
        )
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        if istdev is None:
            istdev = self.istdev

        if irange is not None:
            assert istdev is None
            W = (.5 - rng.rand(self.nhid, nvis)) * irange
        else:
            assert istdev is not None
            W = rng.randn(self.nhid, nvis) * istdev

        self.w_prime = sharedX(W, name='Wprime', borrow=True)

    def set_visible_size(self, nvis, rng=None):
        """
        Create and initialize the necessary parameters to accept
        `nvis` sized inputs.

        Parameters
        ----------
        nvis : int
            Number of visible units for the model.
        rng : RandomState object or seed, optional
            NumPy random number generator object (or seed to create one) used \
            to initialize the model parameters. If not provided, the stored \
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

    def upward_pass(self, inputs):
        """
        Wrapper to Autoencoder encode function. Called when autoencoder
        is accessed by mlp.PretrainedLayer

        Parameters
        ----------
        inputs : WRITEME

        Returns
        -------
        WRITEME
        """
        return self.encode(inputs)

    def encode(self, inputs):
        """
        Map inputs through the encoder function.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second
            indexing data dimensions.

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
            first dimension indexing training examples and the second
            indexing data dimensions.

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

    def get_weights(self, borrow=False):
        """
        .. todo::

            WRITEME
        """
        return self.weights.get_value(borrow=borrow)

    def get_weights_format(self):
        """
        .. todo::

            WRITEME
        """
        return ['v', 'h']


class DenoisingAutoencoder(Autoencoder):
    """
    A denoising autoencoder learns a representation of the input by
    reconstructing a noisy version of it.

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
    def __init__(self, corruptor, nvis, nhid, act_enc, act_dec,
                 tied_weights=False, irange=1e-3, istdev=None, rng=9001):
        super(DenoisingAutoencoder, self).__init__(
            nvis=nvis,
            nhid=nhid,
            act_enc=act_enc,
            act_dec=act_dec,
            tied_weights=tied_weights,
            irange=irange,
            istdev=istdev,
            rng=rng
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
            2-tensors, with the first dimension indexing training examples
            and the second indexing data dimensions.

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
            Theano symbolic (or list thereof) representing the input \
            minibatch(es) on which the penalty is calculated. Assumed to be \
            2-tensors, with the first dimension indexing training examples \
            and the second indexing data dimensions.

        Returns
        -------
        act_grad : tensor_like
            2-dimensional tensor representing, dh/da for every \
            pre/postsynaptic pair, which we can easily do by taking the \
            gradient of the sum of the hidden units activations w.r.t the \
            presynaptic activity, since the gradient of hiddens.sum() with \
            respect to hiddens is a matrix of ones!

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
            2-tensors, with the first dimension indexing training examples
            and the second indexing data dimensions.

        Returns
        -------
        jacobian : tensor_like
            3-dimensional tensor representing, for each mini-batch example,
            the Jacobian matrix of the encoder transformation. You can then
            apply the penalty you want on it, or use the contraction_penalty
            method to have a default one.
        """
        # As long as act_enc is an elementwise operator, the Jacobian
        # of a act_enc(Wx + b) hidden layer has a Jacobian of the
        # following form.
        act_grad = self._activation_grad(inputs)
        jacobian = self.weights * act_grad.dimshuffle(0, 'x', 1)
        return jacobian

    def contraction_penalty(self, data):
        """
        Calculate (symbolically) the contracting autoencoder penalty term.

        Parameters
        ----------
        data : tuple containing one tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) on which the penalty is calculated. Assumed to be
            2-tensors, with the first dimension indexing training examples
            and the second indexing data dimensions.

        Returns
        -------
        jacobian : tensor_like
            1-dimensional tensor representing, for each mini-batch
            example, the penalty of the encoder transformation. Add this to
            the output of a Cost object, such as SquaredError, to penalize it.
        """
        X = data
        act_grad = self._activation_grad(X)
        frob_norm = tensor.dot(tensor.sqr(act_grad),
                               tensor.sqr(self.weights).sum(axis=0))
        contract_penalty = frob_norm.sum() / X.shape[0]
        return tensor.cast(contract_penalty, X .dtype)

    def contraction_penalty_data_specs(self):
        """
        .. todo::

            WRITEME
        """
        return (self.get_input_space(), self.get_input_source())


class HigherOrderContractiveAutoencoder(ContractiveAutoencoder):
    """
    Higher order contractive autoencoder. Adds higher orders regularization

    Parameters
    ----------
    corruptor : object
        Instance of a corruptor object to use for corrupting the input.
    num_corruptions : integer
        number of corrupted inputs to use

    Notes
    -----
    The remaining parameters are identical to those of the constructor
    for the Autoencoder class; see the `ContractiveAutoEncoder.__init__`
    docstring for details.
    """
    def __init__(self, corruptor, num_corruptions, nvis, nhid, act_enc,
                 act_dec, tied_weights=False, irange=1e-3, istdev=None,
                 rng=9001):
        super(HigherOrderContractiveAutoencoder, self).__init__(
            nvis=nvis,
            nhid=nhid,
            act_enc=act_enc,
            act_dec=act_dec,
            tied_weights=tied_weights,
            irange=irange,
            istdev=istdev,
            rng=rng
        )
        self.corruptor = corruptor
        self.num_corruptions = num_corruptions

    def higher_order_penalty(self, data):
        """
        Stochastic approximation of Hessian Frobenius norm

        Parameters
        ----------
        data : WRITEME

        Returns
        -------
        WRITEME
        """
        X = data

        corrupted_inputs = [self.corruptor(X) for times in
                            range(self.num_corruptions)]

        hessian = tensor.concatenate(
            [self.jacobian_h_x(X) - self.jacobian_h_x(corrupted)
             for corrupted in corrupted_inputs])

        return (hessian ** 2).mean()

    def higher_order_penalty_data_specs(self):
        """
        .. todo::

            WRITEME
        """
        return (self.get_input_space(), self.get_input_source())


class UntiedAutoencoder(Autoencoder):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    base : WRITEME
    """

    def __init__(self, base):
        if not (isinstance(base, Autoencoder) and base.tied_weights):
            raise ValueError("%s is not a tied-weights autoencoder" %
                             str(base))

        super(UntiedAutoencoder, self).__init__(
            nvis=base.nvis, nhid=base.nhid, act_enc=base.act_enc,
            act_dec=base.act_dec, tied_weights=True, irange=base.irange,
            istdev=base.istdev, rng=base.rng)

        self.weights = theano.shared(base.weights.get_value(borrow=False),
                                     name='weights')
        self.visbias = theano.shared(base.visbias.get_value(borrow=False),
                                     name='vb')
        self.hidbias = theano.shared(base.visbias.get_value(borrow=False),
                                     name='hb')
        self.w_prime = theano.shared(base.weights.get_value(borrow=False).T,
                                     name='w_prime')
        self._params = [self.visbias, self.hidbias, self.weights, self.w_prime]


class DeepComposedAutoencoder(AbstractAutoencoder):
    """
    A deep autoencoder composed of several single-layer
    autoencoders.

    Parameters
    ----------
    autoencoders : list
        A list of autoencoder objects.
    """
    def __init__(self, autoencoders):
        super(DeepComposedAutoencoder, self).__init__()
        self.fn = None
        self.cpu_only = False

        assert all(pre.get_output_space().dim == post.get_input_space().dim
                   for pre, post in izip(autoencoders[:-1], autoencoders[1:]))

        self.autoencoders = list(autoencoders)
        self.input_space = autoencoders[0].get_input_space()
        self.output_space = autoencoders[-1].get_output_space()

    @functools.wraps(Autoencoder.encode)
    def encode(self, inputs):
        """
        .. todo::

            WRITEME
        """
        current = inputs
        for encoder in self.autoencoders:
            current = encoder.encode(current)
        return current

    @functools.wraps(Autoencoder.decode)
    def decode(self, hiddens):
        """
        .. todo::

            WRITEME
        """
        current = hiddens
        for decoder in self.autoencoders[::-1]:
            current = decoder.decode(current)
        return current

    @functools.wraps(Model.get_params)
    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        return reduce(operator.add,
                      [ae.get_params() for ae in self.autoencoders])

    def _modify_updates(self, updates):
        """
        .. todo::

            WRITEME
        """
        for autoencoder in self.autoencoders:
            autoencoder.modify_updates(updates)


class StackedDenoisingAutoencoder(DeepComposedAutoencoder):
    """
    A stacked denoising autoencoder learns a representation of the input by
    reconstructing a noisy version of it.

    Parameters
    ----------
    autoencoders : list
        A list of autoencoder objects.
    corruptor : object
        Instance of a corruptor object to use for corrupting the
        input.
    """
    def __init__(self, autoencoders, corruptor):
        super(StackedDenoisingAutoencoder, self).__init__(autoencoders)
        self.corruptor = corruptor

    @functools.wraps(AbstractAutoencoder.reconstruct)
    def reconstruct(self, inputs):
        corrupted = self.corruptor(inputs)
        return super(StackedDenoisingAutoencoder, self).reconstruct(corrupted)


def build_stacked_ae(nvis, nhids, act_enc, act_dec,
                     tied_weights=False, irange=1e-3, rng=None,
                     corruptor=None, contracting=False):
    """
    .. todo::

        WRITEME properly

    Allocate a stack of autoencoders.
    """
    rng = make_np_rng(rng, which_method='randn')
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
               final['irange'],)
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
