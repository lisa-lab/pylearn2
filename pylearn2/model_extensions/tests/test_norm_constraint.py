"""
Tests of functionality in norm_constraint.py
"""

import numpy as np

from theano.compat import OrderedDict
from theano import function

from pylearn2.models.model import Model
from pylearn2.model_extensions.norm_constraint import ConstrainFilterMaxNorm
from pylearn2.model_extensions.norm_constraint import MaxL2FilterNorm
from pylearn2.utils import sharedX


class ModelWithW(Model):

    """
    A dummy model that has some weights.

    Parameters
    ----------
    W : 2-D theano shared
        The model's weights.
    """

    def __init__(self, W):
        self.W = W
        super(ModelWithW, self).__init__()


def test_max_l2_filter_norm():
    """
    Test that MaxL2FilterNorm matches a manual implementation.
    """

    limit = 1.
    ext = MaxL2FilterNorm(limit)

    W = np.zeros((2, 4))
    # Column 0 tests the case where a column has zero norm
    # Column 1 tests the case where a column is smaller than the limit
    W[0, 1] = .5
    # Column 2 tests the case where a column is on the limit
    W[0, 2] = 1.
    # Column 3 tests the case where a column is too big
    W[0, 3] = 2.

    W = sharedX(W / 2.)
    model = ModelWithW(W)
    model.extensions.append(ext)

    updates = OrderedDict()
    updates[W] = W * 2.
    model.modify_updates(updates)
    f = function([], updates=updates)
    f()
    W = W.get_value()

    assert W.shape == (2, 4)
    assert np.abs(W[1, :]).max() == 0
    assert W[0, 0] == 0.
    assert W[0, 1] == 0.5
    assert W[0, 2] == 1.
    assert W[0, 3] == 1., W[0, 3]


def test_constrain_filter_max_norm():
    """
    Test that ConstrainFilterNorm matches a manual implementation.
    """

    limit = 1.
    ext = ConstrainFilterMaxNorm(limit)

    W = np.zeros((2, 4))
    # Column 0 tests the case where an element has zero norm
    # Column 1 tests the case where an element is smaller than the limit
    W[0, 1] = .5
    # Column 2 tests the case where an element is on the limit
    W[0, 2] = 1.
    # Column 3 tests the case where an element is too big
    W[0, 3] = 2.

    W = sharedX(W / 2.)
    model = ModelWithW(W)
    model.extensions.append(ext)

    updates = OrderedDict()
    updates[W] = W * 2.
    model.modify_updates(updates)
    f = function([], updates=updates)
    f()
    W = W.get_value()

    assert W.shape == (2, 4)
    assert np.abs(W[1, :]).max() == 0
    assert W[0, 0] == 0.
    assert W[0, 1] == 0.5
    assert W[0, 2] == 1.
    assert W[0, 3] == 1., W[0, 3]
