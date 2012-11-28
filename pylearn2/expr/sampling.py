__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from theano.sandbox.rng_mrg import MRG_RandomStreams

from pylearn2.base import Block

class SampleBernoulli(Block):
    def __init__(self, theano_rng = None, seed=None):
        super(SampleBernoulli, self).__init__()
        assert theano_rng is None or seed is None
        if theano_rng is None:
            if seed is None:
                seed = 2012 + 11 + 22
            theano_rng = MRG_RandomStreams(seed)
        self.__dict__.update(locals())
        del self.self

    def __call__(self, inputs):
        return self.theano_rng.binomial(p=inputs, size=inputs.shape, dtype=inputs.dtype)
