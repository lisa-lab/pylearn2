"""Tests for compilation utilities."""

import os
from nose.tools import eq_, assert_raises

import pylearn2
from pylearn2.utils.image import load


def test_image_load():
    """
    Test utils.image.load
    """
    assert_raises(AssertionError, load, 1)

    path = os.path.join(pylearn2.__path__[0], 'utils',
                        'tests', 'example_image', 'mnist0.jpg')
    img = load(path)
    eq_(img.shape, (28, 28, 1))