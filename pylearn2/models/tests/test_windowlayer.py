"""
Test for WindowLayer
"""
__authors__ = "Axel Davy"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Axel Davy"]
__license__ = "3-clause BSD"
__maintainer__ = "Axel Davy"

import numpy as np
import theano
from pylearn2.models.mlp import MLP, WindowLayer
from pylearn2.space import Conv2DSpace


def build_mlp_fn(x0, y0, x1, y1, s0, s1, c, axes):
    """
    Creates an theano function to test the WindowLayer

    Parameters
    ----------
    x0: x coordinate of the left of the window
    y0: y coordinate of the top of the window
    x1: x coordinate of the right of the window
    y1: y coordinate of the bottom of the window
    s0: x shape of the images of the input space
    s1: y shape of the images of the input space
    c: number of channels of the input space
    axes: description of the axes of the input space

    Returns
    -------
    f: a theano function applicating the window layer
    of window (x0, y0, x1, y1).
    """
    mlp = MLP(layers=[WindowLayer('h0', window=(x0, y0, x1, y1))],
              input_space=Conv2DSpace(shape=(s0, s1),
                                      num_channels=c, axes=axes))
    X = mlp.get_input_space().make_batch_theano()
    f = theano.function([X], mlp.fprop(X))
    return f


def test_windowlayer():
    """
    Tests that WindowLayer reacts correctly to the error in window
    settings, and that the window gives the correct results.

    Parameters
    ----------
    No Parameter
    """
    np.testing.assert_raises(ValueError, build_mlp_fn,
                             0, 0, 20, 20, 20, 20, 3, ('c', 0, 1, 'b'))
    np.testing.assert_raises(ValueError, build_mlp_fn,
                             -1, -1, 19, 19, 20, 20, 3, ('c', 0, 1, 'b'))
    fprop = build_mlp_fn(5, 5, 10, 15, 20, 20, 2, ('b', 'c', 0, 1))
    n = np.random.rand(3, 2, 20, 20).astype(theano.config.floatX)
    r = fprop(n)
    assert r.shape == (3, 2, 6, 11)
    assert r[0, 0, 0, 0] == n[0, 0, 5, 5]
    assert r[2, 1, 5, 10] == n[2, 1, 10, 15]
    assert r[1, 1, 3, 3] == n[1, 1, 8, 8]

if __name__ == "__main__":
    test_windowlayer()
