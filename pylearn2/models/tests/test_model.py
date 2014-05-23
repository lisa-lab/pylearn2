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


def test_tag():
    """Test that the tag attribute works correctly."""
    class DummyModel(Model):
        """The simplest instance of Model possible."""
    x = DummyModel()
    x.tag['foo']['bar'] = 5

    assert len(x.tag.keys()) == 1
    assert len(x.tag['foo'].keys()) == 1
    assert x.tag['foo']['bar'] == 5

    assert 'bar' not in x.tag
    x.tag['bar']['baz'] = 3
    assert 'bar' in x.tag
    assert 'baz' in x.tag['bar']
    assert len(x.tag.keys()) == 2
