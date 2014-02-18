from pylearn2.utils.rng import make_np_rng, make_theano_rng
import numpy

import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def test_np_rng():
    """
        Tests that the four possible ways of creating
        a numpy RNG give the same results with the same seed
    """

    rngs = [make_np_rng(rng_or_seed=42, which_method='uniform'),
            make_np_rng(rng_or_seed=numpy.random.RandomState(42),
                        which_method='uniform'),
            make_np_rng(default_seed=42),
            make_np_rng()]

    random_numbers = rngs[0].uniform(size=(100,))
    equals = numpy.ones((100,))
    for rng in rngs[1:]:
        equal = random_numbers == rng.uniform(size=(100,))
        equals *= equal

    assert equals.all()


def test_theano_rng():
    """
        Tests that the four possible ways of creating
        a theano RNG give the same results with the same seed
    """

    rngs = [make_theano_rng(rng_or_seed=42, which_method='uniform'),
            make_theano_rng(rng_or_seed=RandomStreams(42),
                            which_method='uniform'),
            make_theano_rng(default_seed=42),
            make_theano_rng()]

    functions = [theano.function([], rng.uniform(size=(100,)))
                 for rng in rngs]

    random_numbers = functions[0]()
    equals = numpy.ones((100,))
    for function in functions[1:]:
        equal = random_numbers == function()
        equals *= equal

    assert equals.all()
