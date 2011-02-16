"""Base class for the components in other modules."""
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

class Block(object):
    """
    Basic building block for deep architectures.
    """
    def __init__(self, **kwargs):
        self._params = []
        self.__dict__.update(kwargs)

    def alloc(cls, conf, rng=None):
        raise NotImplementedError('alloc')

    def load(self, load_dir, load_filename):
        raise NotImplementedError('load')

    def params(self):
        """
        Returns a list of *shared* learnable parameters that
        are, in your judgment, typically learned in this
        model.
        """
        return self._params

    def outputs(self):
        """Output to pass on to layers above."""
        raise NotImplementedError('outputs')

    def __call__(self, inputs):
        raise NotImplementedError('__call__')

class Trainer(object):
    """
    Basic abstract class for training
    """
    def __init__(self, inputs, **kwargs):
        self.inputs = inputs
        self.__dict__.update(kwargs)

    def updates(self):
        """Do one step of training."""
        raise NotImplementedError()

    def save(self, save_dir, save_filename):
        raise NotImplementedError('save')
