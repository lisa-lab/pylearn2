"""
.. note::

    pylearn2.base is deprecated. It is now called pylearn2.blocks,
    since it was not actually the base of the library. pylearn2.base
    may be removed from the library on or after 2014-09-06.
"""
import warnings

warnings.warn("pylearn2.base is deprecated. It is now called pylearn2.blocks,"
        "since it was not actually the base of the library. pylearn2.base may"
        " be removed from the library on or after 2014-09-06.", stacklevel=2)

from pylearn2.blocks import Block
from pylearn2.blocks import StackedBlocks
