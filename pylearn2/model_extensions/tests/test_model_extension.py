"""
Tests for model_extensions.
"""

from theano.compat.six.moves import xrange
from pylearn2.models import Model
from pylearn2.model_extensions.model_extension import ModelExtension


def test_model_extension():
    """
    Test that the base class Model passes tests when given a list of
    ModelExtension instances.
    """
    class DummyModelExtension(ModelExtension):
        """Simplest instance of ModelExtension"""
    class DummyModel(Model):
        """Simplest instance of Model"""

    extensions = (DummyModelExtension())
    try:
        """
        This should cause an assertion error for passing a tuple instead of
        a list
        """
        model = DummyModel(extensions=extensions)
    except AssertionError:
        pass

    extensions = [DummyModelExtension(), None]
    try:
        """
        This should cause an assertion error for passing a list where each
        element is not an instance of ModelExtension
        """
        model = DummyModel(extensions=extensions)
    except AssertionError:
        pass

    extensions = []
    for i in xrange(3):
        extensions.append(DummyModelExtension())
    """
    This should not cause an assertion
    """
    model = DummyModel(extensions=extensions)
