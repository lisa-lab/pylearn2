import unittest
import numpy as np
from pylearn2.datasets.avicenna import Avicenna
from pylearn2.testing.skip import skip_if_no_data


def test_avicenna():
    skip_if_no_data()
    data = Avicenna(which_set='train', standardize=False)
    data = Avicenna(which_set='valid', standardize=False)
    data = Avicenna(which_set='test', standardize=False)


def test_avicenna_standardized():
    skip_if_no_data()
    data = Avicenna(which_set='train', standardize=True)
    data = Avicenna(which_set='valid', standardize=True)
    data = Avicenna(which_set='test', standardize=True)
