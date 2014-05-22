"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

from theano.sandbox.rng_mrg import MRG_RandomStreams

from pylearn2.blocks import Block
from pylearn2.utils.rng import make_theano_rng


class SampleBernoulli(Block):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    theano_rng : WRITEME
    seed : WRITEME
    input_space : WRITEME
    """
    def __init__(self, theano_rng = None, seed=None, input_space=None):
        super(SampleBernoulli, self).__init__()
        assert theano_rng is None or seed is None
        theano_rng = make_theano_rng(theano_rng if theano_rng is not None else seed,
                                     2012+11+22, which_method='binomial')
        self.__dict__.update(locals())
        del self.self

    def __call__(self, inputs):
        """
        .. todo::

            WRITEME
        """
        if self.input_space:
            self.input_space.validate(inputs)
        return self.theano_rng.binomial(p=inputs, size=inputs.shape, dtype=inputs.dtype)

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME
        """
        self.input_space = space

    def get_input_space(self):
        """
        .. todo::

            WRITEME
        """
        if self.input_space is not None:
            return self.input_space
        raise ValueError("No input space was specified for this Block (%s). "
                "You can call set_input_space to correct that." % str(self))

    def get_output_space(self):
        """
        .. todo::

            WRITEME
        """
        return self.get_input_space()
