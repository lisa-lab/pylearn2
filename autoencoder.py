"""Autoencoders, denoising autoencoders, and stacked DAEs."""
from itertools import izip
import numpy
import theano
from theano import tensor

#from pylearn.gd.sgd import sgd_updates
#from pylearn.algorithms.mcRBM import contrastive_cost, contrastive_grad
theano.config.warn.sum_div_dimshuffle_bug = False
floatX = theano.config.floatX
sharedX = lambda X, name : theano.shared(numpy.asarray(X, dtype=floatX), name=name)
if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams

from base import Block

class DenoisingAutoencoder(Block):
    """
    A denoising autoencoder learns a representation of the input by
    reconstructing a noisy version of it.
    """
    def __init__(self, **kwargs):
        # TODO: Do we need anything else here?
        super(DenoisingAutoencoder, self).__init__(**kwargs)

    @classmethod
    def alloc(cls, corruptor, conf, rng=None):
        self.corruptor = corruptor
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        self = cls()
        self.visbias = sharedX(
            numpy.zeros(conf['n_vis']),
            name='vb'
        )
        self.hidbias = sharedX(
            numpy.zeros(conf['n_hid']),
            name='hb'
        )
        self.weights = sharedX(
            .5 * rng.rand(conf['n_vis'], conf['n_hid']),
            name='W'
        )
        seed = int(rng.randint(2**30))
        self.s_rng = RandomStreams(seed)
        if conf['tied_weights']:
            self.w_prime = self.weights.T
        else:
            self.w_prime = sharedX(
                .5 * rng.rand(conf['n_hid'], conf['n_vis']),
                name='Wprime'
            )
        def _resolve_callable(conf_attr):
            if conf_attr is None:
                # The identity function, for linear layers.
                return lambda x: x
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
            self.weights,
            self.visbias,
            self.hidbias
        ]
        if not conf['tied_weights']:
            self._params.append(self.w_prime)

    def _hidden_activation(self, x):
        """Single input pattern/minibatch activation function."""
        return self.act_enc(self.hidbias + tensor.dot(self.weights, x))

    def hidden_repr(self, inputs):
        """Hidden unit activations for each set of inputs."""
        return [self._hidden_activation(v) for v in inputs]

    def reconstruction(self, inputs):
        """Reconstructed inputs after corruption."""
        corrupted = (self.corruptor(inp) for inp in inputs)
        hiddens = self.hidden_repr(corrupted)
        return [self.visbias + tensor.dot(self.w_prime, h) for h in hiddens]

    def __call__(self, inputs):
        """Output to pass on to layers above."""
        return self.hidden_repr(inputs)

    def mse(self, inputs):
        """
        Symbolic expression for mean-squared error between the input and the
        denoised reconstruction.
        """
        pairs = izip(inputs, self.reconstruction(inputs))
        return [((inp - rec)**2).sum(axis=-1).mean() for inp, rec in pairs]

    def cross_entropy(self, inputs):
        """
        Symbolic expression for elementwise cross-entropy between input
        and reconstruction. Use for binary-valued features (but not for,
        e.g., one-hot codes).
        """
        pairs = izip(inputs, self.reconstruction(inputs))
        ce = lambda x, z: x * tensor.log(z) + (1 - x) * tensor.log(1 - z)
        return [ce(inp, rec).sum(axis=1).mean() for inp, rec in pairs]

class StackedDA(Block):
    def __init__(self, inputs, **kwargs):
        # TODO: Do we need anything else here?
        super(StackedDA, self).__init__(inputs, **kwargs)

    def alloc(cls, conf, rng=None):
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        self = cls()
        self._layers = []
        _local = {}
        # Make sure that if we have a sequence of encoder/decoder activations
        # or corruptors, that we have exactly as many as len(conf['n_hid'])
        for c in ['act_enc', 'act_dec', 'corruptor']:
            if hasattr(conf[c], '__len__'):
                assert len(conf['n_hid']) == len(conf[c])
                _local[c] = conf[c]
            else:
                _local[c] = [conf[c]] * len(conf['n_hid'])
        n_hids = conf['n_hid']
        # The number of visible units in each layer is the initial input
        # size and the first k-1 hidden unit sizes.
        n_viss = [conf['n_vis']] + conf['n_hid'][:-1]
        first = False
        seq = izip(
            xrange(len(n_hids)),
            n_hids,
            n_viss,
            _local['act_encs'],
            _local['act_decs'],
            _local['corruptors']
        )
        # Create each layer.
        for k, n_hid, n_vis, act_enc, act_dec, corr in seq:
            # Create a local configuration dictionary for this layer.
            lconf = {
                'n_hid': n_hid,
                'n_vis': n_vis,
                'act_enc': act_enc,
                'act_dec': act_dec,
            }
            da = DenoisingAutoencoder.alloc(corr, lconf, rng)
            self._layers.append(da)

    def layers(self):
        return list(self._layers)

    def params(self):
        # TODO: Rewrite this to be more readable (don't use reduce).
        return reduce(lambda x, y: x + y, [l.params() for l in self._layers])

    def __call__(self, inputs):
        # TODO: write this.
        pass


