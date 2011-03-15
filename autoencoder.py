"""Autoencoders, denoising autoencoders, and stacked DAEs."""
# Standard library imports
from itertools import izip

# Third-party imports
import numpy
import theano
from theano import tensor

# Local imports
from .base import Block
from .stack import StackedBlocks
from .utils import sharedX

theano.config.warn.sum_div_dimshuffle_bug = False
floatX = theano.config.floatX

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams

class DenoisingAutoencoder(Block):
    """
    A denoising autoencoder learns a representation of the input by
    reconstructing a noisy version of it.
    """
    def __init__(self, nvis, nhid, corruptor, act_enc, act_dec,
                 tied_weights=False, irange=1e-3, rng=9001):
        """Allocate a denoising autoencoder object."""
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        self.corruptor = corruptor
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
        return act_enc(self.hidbias + tensor.dot(x, self.weights))

    def hidden_repr(self, inputs):
        """Hidden unit activations for each set of inputs."""
        if isinstance(inputs, tensor.Variable):
            return self._hidden_activation(inputs)
        else:
            return [self._hidden_activation(v) for v in inputs]

    def reconstruction(self, inputs):
        """Reconstructed inputs after corruption."""
        corrupted = self.corruptor(inputs)
        hiddens = self.hidden_repr(corrupted)
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
        """
        return self.hidden_repr(inputs)

def build_stacked_DA(nvis, nhids, corruptors, act_enc, act_dec,
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
        xrange(len(nhids)),
        nhids,
        nviss,
        _local['act_enc'],
        _local['act_dec'],
        corruptors
    )
    # Create each layer.
    for k, nhid, nvis, act_enc, act_dec, corr in seq:
        da = DenoisingAutoencoder(nvis, nhid, corr, act_enc, act_dec,
                                  tied_weights, irange, rng)
        layers.append(da)

    # Create the stack
    return StackedBlocks(layers)

class StackedDA(Block):
    """
    A class representing a stacked model. Forward propagation passes
    (symbolic) input through each layer sequentially.
    """
    def __init__(self, nvis, nhids, corruptors, act_enc, act_dec,
                 tied_weights=False, irange=1e-3, rng=None):
        """Allocate a stacked denoising autoencoder object."""
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        self._layers = []
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
            xrange(len(nhids)),
            nhids,
            nviss,
            _local['act_enc'],
            _local['act_dec'],
            corruptors
        )
        # Create each layer.
        for k, nhid, nvis, act_enc, act_dec, corr in seq:
            da = DenoisingAutoencoder(nvis, nhid, corr, act_enc, act_dec,
                                      tied_weights, irange, rng)
            self._layers.append(da)

    def layers(self):
        """
        The layers of this model: the individual denoising autoencoder
        objects, which can be individually pre-trained.
        """
        return list(self._layers)

    def params(self):
        """
        The parameters that are learned in this model, i.e. the concatenation
        of all the layers' weights and biases.
        """
        return sum([l.params() for l in self._layers], [])

    def __call__(self, inputs):
        """
        Forward propagate (symbolic) input through this module, obtaining
        a representation to pass on to layers above.
        """
        transformed = inputs
        # Pass the input through each layer of the hierarchy.
        for layer in self._layers:
            transformed = layer(transformed)
        return transformed

