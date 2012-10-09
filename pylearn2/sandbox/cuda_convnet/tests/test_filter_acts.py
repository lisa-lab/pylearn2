__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import unittest
from pylearn2.testing.skip import skip_if_no_gpu
import numpy as np
from pylearn2.utils import sharedX
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs

class TestFilterActs(unittest.TestCase):

    def setUp(self):
        skip_if_no_gpu()

    def test_match_valid_conv(self):

        # Tests that running FilterActs with no padding is the same as running
        # theano's conv2D in valid mode

        rng = np.random.RandomState([2012,10,9])

        batch_size = 5
        rows = 10
        cols = 9
        channels = 3
        filter_rows = 4
        filter_cols = filter_rows
        num_filters = 6

        images = sharedX(rng.uniform(-1., 1., (channels, rows, cols,
            batch_size)), name='images')
        filters = sharedX(rng.uniform(-1., 1., (channels, filter_rows,
            filter_cols, num_filters)), name='filters')

        output = FilterActs()(images, filters)
