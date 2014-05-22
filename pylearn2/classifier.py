"""
.. note::

    pylearn2.classifier has been deprecated and will be removed from
    the library on or after Aug 24, 2014.
"""
import warnings

warnings.warn("pylearn2.classifier has been deprecated and will be removed "
        "from the library on or after Aug 24, 2014.")

from deprecated.classifier import Block
from deprecated.classifier import CumulativeProbabilitiesLayer
from deprecated.classifier import LogisticRegressionLayer
from deprecated.classifier import Model
from deprecated.classifier import VectorSpace
from deprecated.classifier import numpy
from deprecated.classifier import sharedX
from deprecated.classifier import tensor
from deprecated.classifier import theano
