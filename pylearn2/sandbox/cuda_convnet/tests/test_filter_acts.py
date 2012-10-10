__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import unittest
from pylearn2.testing.skip import skip_if_no_gpu
skip_if_no_gpu()
import numpy as np
from theano import shared
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda import gpu_from_host
from theano.tensor.nnet.conv import conv2d
from theano import function

def test_match_valid_conv():

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

    images = shared(rng.uniform(-1., 1., (channels, rows, cols,
        batch_size)).astype('float32'), name='images')
    filters = shared(rng.uniform(-1., 1., (channels, filter_rows,
        filter_cols, num_filters)).astype('float32'), name='filters')

    gpu_images = gpu_from_host(images)
    gpu_filters = gpu_from_host(filters)

    output = FilterActs()(gpu_images, gpu_filters)

    images_bc01 = images.dimshuffle(3,0,1,2)
    filters_bc01 = images.dimshuffle(3,0,1,2)

    output_conv2d = conv2d(images_bc01, filters_bc01,
            border_mode='valid')

    f = function([], [output, output_conv2d])

    output, output_conv2d = f()

    assert np.allclose(output, output_conv2d)

def test_reject_rect():

    # Tests that running FilterActs with a non-square
    # kernel is an error

    rng = np.random.RandomState([2012,10,9])

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

    output = FilterActs()(gpu_images, gpu_filters)

    images_bc01 = images.dimshuffle(3,0,1,2)
    filters_bc01 = images.dimshuffle(3,0,1,2)

    output_conv2d = conv2d(images_bc01, filters_bc01,
            border_mode='valid')

    f = function([], [output, output_conv2d])

    try:
        output, output_conv2d = f()
    except ValueError:
        return

    assert False

if __name__ == '__main__':
    test_match_valid_conv()
