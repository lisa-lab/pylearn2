__authors__ = "Ian Goodfellow, David Warde-Farley"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow, David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from pylearn2.testing.skip import skip_if_no_gpu
skip_if_no_gpu()
import numpy as np
from theano import shared
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.img_acts import ImageActs
from theano.sandbox.cuda import gpu_from_host
from theano import function


def test_reject_rect():
    for cls in (FilterActs, ImageActs):
        # Tests that running FilterActs with a non-square
        # kernel is an error
        rng = np.random.RandomState([2012, 10, 9])
        batch_size = 5
        rows = 10
        cols = 9
        channels = 3
        filter_rows = 4
        filter_cols = filter_rows + 1
        num_filters = 6

        images = shared(rng.uniform(-1., 1., (channels, rows, cols,
            batch_size)).astype('float32'), name='images')
        filters = shared(rng.uniform(-1., 1., (channels, filter_rows,
            filter_cols, num_filters)).astype('float32'), name='filters')

        gpu_images = gpu_from_host(images)
        gpu_filters = gpu_from_host(filters)

        output = cls()(gpu_images, gpu_filters)
        f = function([], output)
        try:
            output = f()
        except ValueError:
            continue
        assert False


def test_reject_bad_filt_number():
    for cls in (FilterActs, ImageActs):
        # Tests that running FilterActs with a # of filters per
        # group that is not 16 is an error
        rng = np.random.RandomState([2012, 10, 9])
        batch_size = 5
        rows = 10
        cols = 9
        channels = 3
        filter_rows = 4
        filter_cols = filter_rows
        num_filters = 6

        images = shared(rng.uniform(-1., 1., (channels, rows, cols,
            batch_size)).astype('float32'), name='images')
        filters = shared(rng.uniform(-1., 1., (channels, filter_rows,
            filter_cols, num_filters)).astype('float32'), name='filters')

        gpu_images = gpu_from_host(images)
        gpu_filters = gpu_from_host(filters)

        output = cls()(gpu_images, gpu_filters)
        f = function([], output)
        try:
            output = f()
        except ValueError:
            continue
        assert False
