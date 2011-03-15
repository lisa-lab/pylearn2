"""Autoencoders, denoising autoencoders, and stacked DAEs."""
# Standard library imports
from itertools import izip,imap

# Third-party imports
import numpy
import theano
from theano import tensor

# Local imports
from .base import Block, StackedBlocks
from .utils import sharedX

theano.config.warn.sum_div_dimshuffle_bug = False
floatX = theano.config.floatX

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams

class Autoencoder(Block):
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
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        self.visbias = sharedX(
            numpy.zeros(nvis),
            name='vb',
            borrow=True
        )
        self.hidbias = sharedX(
            numpy.zeros(nhid),
            name='hb',
            borrow=True
        )
        # TODO: use weight scaling factor if provided, Xavier's default else
        self.weights = sharedX(
            .5 - rng.rand(nvis, nhid) * irange,
            name='W',
            borrow=True
        )
        seed = int(rng.randint(2**30))
        self.s_rng = RandomStreams(seed)
        if tied_weights:
            self.w_prime = self.weights.T
        else:
            self.w_prime = sharedX(
                .5 - rng.rand(nhid, nvis) * irange,
                name='Wprime',
                borrow=True
            )

        def _resolve_callable(conf, conf_attr):
            try:
                if conf[conf_attr] is None or conf[conf_attr] == "linear":
                    # The identity function, for linear layers.
                    return None
                # If it's a callable, use it directly.
                if hasattr(conf[conf_attr], '__call__'):
                    return conf[conf_attr]
                elif hasattr(tensor.nnet, conf[conf_attr]):
                    return getattr(tensor.nnet, conf[conf_attr])
                elif hasattr(tensor, conf[conf_attr]):
                    return getattr(tensor, conf[conf_attr])
                else:
                    raise ValueError("Couldn't interpret %s value: '%s'" %
                                    (conf_attr, conf[conf_attr]))
            except Exception as e:
                print conf_attr,':',e.args
                raise

        self.act_enc = _resolve_callable(locals(), 'act_enc')
        self.act_dec = _resolve_callable(locals(), 'act_dec')
        self._params = [
            self.visbias,
            self.hidbias,
            self.weights,
        ]
        if not tied_weights:
            self._params.append(self.w_prime)

    def _hidden_activation(self, x):
        """Single input pattern/minibatch activation function."""
        if self.act_enc is None:
            act_enc = lambda x: x
        else:
            act_enc = self.act_enc
        self.hiddens=act_enc(self.hidbias + tensor.dot(x, self.weights))
        return self.hiddens

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
            reconstructed minibatch(es) after encoding/decoding.
        """
        if isinstance(inputs, tensor.Variable):
            return self._hidden_activation(inputs)
        else:
            return [self._hidden_activation(v) for v in inputs]

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
        hiddens = self.encode(inputs)
        if self.act_dec is None:
            act_dec = lambda x: x
        else:
            act_dec = self.act_dec
        if isinstance(inputs, tensor.Variable):
            return act_dec(self.visbias + tensor.dot(hiddens, self.w_prime))
        else:
            return [
                act_dec(self.visbias + tensor.dot(h, self.w_prime))
                for h in hiddens
            ]

    def __call__(self, inputs):
        """
        Forward propagate (symbolic) input through this module, obtaining
        a representation to pass on to layers above.

        This just aliases the `encode()` function for syntactic
        sugar/convenience.
        """
        return self.encode(inputs)

class DenoisingAutoencoder(Autoencoder):
    """
    A denoising autoencoder learns a representation of the input by
    reconstructing a noisy version of it.
    """
    def __init__(self, corruptor, *args, **kwargs):
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
        super(DenoisingAutoencoder, self).__init__(*args, **kwargs)
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

class ContractingAutoencoder(Autoencoder):
    """
    A contracting autoencoder works like a regular autoencoder, and adds an
    extra term to its cost function.
    """
    def __init__(self, *args, **kwargs):
        super(ContractingAutoencoder, self).__init__(*args, **kwargs)

    def contraction_penalty(self):
        def cp(p):
            Jacobien = self.weights * (p*(1-p)).dimshuffle(0,'x',1)
            L=tensor.mean(Jacobien**2)
            return L
        return cp(self.hiddens)

def build_stacked_DA(corruptors, nvis, nhids, act_enc, act_dec,
                     tied_weights=False, irange=1e-3, rng=None):
    """Allocate a StackedBlocks containing denoising autoencoders."""
    if not hasattr(rng, 'randn'):
        rng = numpy.random.RandomState(rng)
    layers = []
    _local = {}
    # Make sure that if we have a sequence of encoder/decoder activations
    # or corruptors, that we have exactly as many as len(n_hids)
    if hasattr(corruptors, '__len__'):
        assert len(nhids) == len(corruptors)
    else:
        corruptors = [corruptors] * len(nhids)
    for c in ['act_enc', 'act_dec']:
        if type(locals()[c]) is not str and hasattr(locals()[c], '__len__'):
            assert len(nhids) == len(locals()[c])
            _local[c] = locals()[c]
        else:
            _local[c] = [locals()[c]] * len(nhids)
    # The number of visible units in each layer is the initial input
    # size and the first k-1 hidden unit sizes.
    nviss = [nvis] + nhids[:-1]
    seq = izip(
        corruptors,
        xrange(len(nhids)),
        nhids,
        nviss,
        _local['act_enc'],
        _local['act_dec'],
    )
    # Create each layer.
    for k, nhid, nvis, act_enc, act_dec, corr in seq:
        da = DenoisingAutoencoder(corr, nvis, nhid, act_enc, act_dec,
                                  tied_weights, irange, rng)
        layers.append(da)

    # Create the stack
    return StackedBlocks(layers)


def build_denoising_stack(  corruptors,
                            nvis,
                            nhids,
                            act_enc,
                            act_dec,
                            tied_weights=False,
                            irange=1e-3,
                            rng=None,
                            contracting=None):
    """Allocate a StackedBlocks containing denoising/contrasting autoencoders."""

    if not hasattr(rng, 'randn'):
        rng = numpy.random.RandomState(rng)
    layers = []
    _local = {}

    # Make sure that if we have a sequence of encoder/decoder activations
    # or corruptors, that we have exactly as many as len(n_hids)
    if hasattr(corruptors, '__len__'):
        assert len(nhids) == len(corruptors)
    else:
        corruptors = [corruptors] * len(nhids)
    for c in ['act_enc', 'act_dec']:
        if type(locals()[c]) is not str and hasattr(locals()[c], '__len__'):
            assert len(nhids) == len(locals()[c])
            _local[c] = locals()[c]
        else:
            _local[c] = [locals()[c]] * len(nhids)
    # Make sure that if the contracting arg is used, it is consistently done so
    if hasattr(contracting,'__len__'):
        assert len(nhids) == len(contracting)
    else:
        contracting=[False]*len(nhids)

    # The number of visible units in each layer is the initial input
    # size and the first k-1 hidden unit sizes.
    nviss = [nvis] + nhids[:-1]
    seq = izip(
        xrange(len(nhids)),
        nhids,
        nviss,
        _local['act_enc'],
        _local['act_dec'],
        corruptors,
        contracting,
    )
    # Create each layer.
    for k, nhid, nvis, act_enc, act_dec, corr, is_cae in seq:
        args=nvis, nhid, act_enc, act_dec, tied_weights, irange, rng
        if is_cae:
            ae = ContractingAutoencoder(*args)
        else:
            ae = DenoisingAutoencoder(corr,*args)
        layers.append(ae)

    # Create the stack
    return StackedBlocks(layers)
