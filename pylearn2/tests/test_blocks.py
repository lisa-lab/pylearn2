"""
Unit tests for blocks
"""

from pylearn2.models.autoencoder import Autoencoder
from pylearn2.blocks import Block, StackedBlocks


def test_stackedblocks_with_params():
    """
    Test StackedBlocks when all layers have trainable params
    """

    aes = [Autoencoder(100, 50, 'tanh', 'tanh'),
           Autoencoder(50, 10, 'tanh', 'tanh')]
    sb = StackedBlocks(aes)
    _params = set([p for l in sb._layers for p in l._params])

    assert sb._params == _params


def test_stackedblocks_without_params():
    """
    Test StackedBlocks when not all layers have trainable params
    """

    sb = StackedBlocks([Block(), Block()])

    assert sb._params is None
