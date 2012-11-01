__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import numpy as np

from pylearn2.expr.basic import is_binary
from pylearn2.models.dbm import BinaryVisLayer

def check_binary_samples(value, expected_shape, expected_mean, tol):
    """
    Tests that a matrix of binary samples (observations in rows, variables
        in columns)
    1) Has the right shape
    2) Is binary
    3) Converges to the right mean
    """
    assert value.shape == expected_shape
    assert is_binary(value)
    mean = value.mean(axis=0)
    if np.abs(value-expected_mean).max() > tol:
        print value
        print expected_mean
        print np.abs(value-expected_mean).max()
        raise ValueError("Samples don't seem to have the right mean.")

def test_binary_vis_layer_make_state():

    # Verifies that BinaryVisLayer.make_state creates
    # a shared variable whose value

    n = 5
    num_samples = 100
    tol = .01

    layer = BinaryVisLayer(nvis = n)

    rng = np.random.RandomState([2012,11,1])

    init_state = layer.make_state(num_examples=num_samples,
            numpy_rng=rng)

    value = init_state.get_value()


    mean = np.ones((n,))

    check_binary_samples(value, (num_samples, n), mean, tol)


