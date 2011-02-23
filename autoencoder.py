"""Autoencoders, denoising autoencoders, and stacked DAEs."""
# Standard library imports
from itertools import izip

# Third-party imports
import numpy
import theano
from theano import tensor

# Local imports
from .base import Block
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
    def __init__(self, conf, corruptor, rng=None):
        """Allocate a denoising autoencoder object."""
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        self.corruptor = corruptor
        self.visbias = sharedX(
            numpy.zeros(conf['n_vis']),
            name='vb',
            borrow=True
        )
        self.hidbias = sharedX(
            numpy.zeros(conf['n_hid']),
            name='hb',
            borrow=True
        )
        # TODO: use weight scaling factor if provided, Xavier's default else
        self.weights = sharedX(
            .5 * rng.rand(conf['n_vis'], conf['n_hid']) * conf['irange'],
            name='W',
            borrow=True
        )
        seed = int(rng.randint(2**30))
        self.s_rng = RandomStreams(seed)
        if conf['tied_weights']:
            self.w_prime = self.weights.T
        else:
            self.w_prime = sharedX(
                .5 * rng.rand(conf['n_hid'], conf['n_vis']) * conf['irange'],
                name='Wprime',
                borrow=True
            )

        def _resolve_callable(conf_attr):
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

        self.act_enc = _resolve_callable('act_enc')
        self.act_dec = _resolve_callable('act_dec')
        self.conf = conf
        self._params = [
            self.visbias,
            self.hidbias,
            self.weights,
        ]
        if not conf['tied_weights']:
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
            return act_dec(self.visbias + tensor.dot(inputs, self.w_prime))
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

    def function(self):
        """ Returns a compiled theano function to compute a representation """
        inputs = tensor.matrix()
        return theano.function([inputs], self(inputs), name='da_transform_fn')

class StackedDA(Block):
    """
    A class representing a stacked model. Forward propagation passes
    (symbolic) input through each layer sequentially.
    """
    def __init__(self, conf, corruptors, rng=None):
        """Allocate a stacked denoising autoencoder object."""
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        self._layers = []
        _local = {}
        # Make sure that if we have a sequence of encoder/decoder activations
        # or corruptors, that we have exactly as many as len(conf['n_hid'])
        if hasattr(corruptors, '__len__'):
            assert len(conf['n_hid']) == len(corruptors)
        else:
            corruptors = [corruptors] * len(conf['n_hid'])
        for c in ['act_enc', 'act_dec']:
            if type(conf[c]) is not str and hasattr(conf[c], '__len__'):
                assert len(conf['n_hid']) == len(conf[c])
                _local[c] = conf[c]
            else:
                _local[c] = [conf[c]] * len(conf['n_hid'])
        n_hids = conf['n_hid']
        # The number of visible units in each layer is the initial input
        # size and the first k-1 hidden unit sizes.
        n_viss = [conf['n_vis']] + conf['n_hid'][:-1]
        seq = izip(
            xrange(len(n_hids)),
            n_hids,
            n_viss,
            _local['act_enc'],
            _local['act_dec'],
            corruptors
        )
        # Create each layer.
        for k, n_hid, n_vis, act_enc, act_dec, corr in seq:
            # Create a local configuration dictionary for this layer.
            lconf = {
                'n_hid': n_hid,
                'n_vis': n_vis,
                'act_enc': act_enc,
                'act_dec': act_dec,
                'irange': conf['irange'],
                'tied_weights': conf['tied_weights'],
            }
            da = DenoisingAutoencoder(lconf, corr, rng)
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

