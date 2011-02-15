"""Autoencoders, denoising autoencoders, and stacked DAEs."""
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
    def __init__(self, inputs, **kwargs):
        # TODO: Do we need anything else here?
        super(DenoisingAutoencoder, self).__init__(inputs, **kwargs)

    @classmethod
    def alloc(cls, clean_inputs, corrupted_inputs, conf, rng=None):
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        self = cls(clean_inputs)
        self.corrupted = corrupted_inputs
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

        def _resolve_callable(conf_attr):
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
            self.w_prime,
            self.visbias,
            self.hidbias
        ]
    def _hidden_activation(self, x):
        return self.act_enc(self.hidbias + tensor.dot(self.weights, x))

    def hidden_with_corrupted_input(self):
        """Hidden unit activations when the input is corrupted."""
        return [self._hidden_activation(v) for v in self.corrupted]

    def hidden_with_clean_input(self):
        """Hidden unit activations when the input is corrupted."""
        return [self._hidden_activation(v) for v in self.inputs]

    def reconstruction(self):
        """Reconstructed inputs after corruption."""
        return [self._hidden_activation(v) for v in self.inputs]

    @property
    def outputs(self):
        """Output to pass on to layers above."""
        return [self.hidden_with_clean_input(v) for v in self.inputs]
