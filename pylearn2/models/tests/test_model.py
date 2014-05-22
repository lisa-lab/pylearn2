"""
Tests for functionality from model.py
"""

import numpy as np

from pylearn2.models import Model
from pylearn2.utils import sharedX


def test_get_set_vector():
    """
    Tests that get_vector and set_vector use the same
    format.
    """

    rng = np.random.RandomState([2014, 5, 8])

    class DummyModel(Model):
        """
        A Model that exercises this test by having a few different
        parameters with different shapes and dimensionalities.

        Don't instantiate more than one of these because the parameters
        are class-level attributes.
        """

        _params = [sharedX(rng.randn(5)), sharedX(rng.randn(5, 3)),
                   sharedX(rng.randn(4, 4, 4))]

    model = DummyModel()

    vector = model.get_param_vector()
    model.set_param_vector(0. * vector)
    assert np.allclose(0. * vector, model.get_param_vector())
    model.set_param_vector(vector)
    assert np.allclose(model.get_param_vector(), vector)
