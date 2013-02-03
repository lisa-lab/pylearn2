__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from pylearn2.testing.skip import skip_if_no_gpu
skip_if_no_gpu()

import numpy as np
from theano import shared
from theano.tensor import grad, constant
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda import gpu_from_host
from theano.sandbox.cuda import host_from_gpu
from theano.tensor.nnet.conv import conv2d
from theano import function
from theano import tensor as T
import warnings


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
    num_filters = 16

    images = shared(rng.uniform(-1., 1., (channels, rows, cols,
        batch_size)).astype('float32'), name='images')
    filters = shared(rng.uniform(-1., 1., (channels, filter_rows,
        filter_cols, num_filters)).astype('float32'), name='filters')

    gpu_images = gpu_from_host(images)
    gpu_filters = gpu_from_host(filters)

    output = FilterActs()(gpu_images, gpu_filters)
    output = host_from_gpu(output)

    images_bc01 = images.dimshuffle(3,0,1,2)
    filters_bc01 = filters.dimshuffle(3,0,1,2)
    filters_bc01 = filters_bc01[:,:,::-1,::-1]

    output_conv2d = conv2d(images_bc01, filters_bc01,
            border_mode='valid')

    output_conv2d = output_conv2d.dimshuffle(1,2,3,0)

    f = function([], [output, output_conv2d])

    output, output_conv2d = f()

    warnings.warn("""test_match_valid_conv success criterion is not very strict. Can we verify that this is OK?
                     One possibility is that theano is numerically unstable and Alex's code is better.
                     Probably theano CPU 64 bit is OK but it's worth checking the others.""")
    if np.abs(output - output_conv2d).max() > 2.4e-6:
        assert type(output) == type(output_conv2d)
        assert output.dtype == output_conv2d.dtype
        if output.shape != output_conv2d.shape:
            print 'cuda-convnet shape: ',output.shape
            print 'theano shape: ',output_conv2d.shape
            assert False
        err = np.abs(output - output_conv2d)
        print 'absolute error range: ', (err.min(), err.max())
        print 'mean absolute error: ', err.mean()
        print 'cuda-convnet value range: ', (output.min(), output.max())
        print 'theano value range: ', (output_conv2d.min(), output_conv2d.max())
        assert False

def test_match_valid_conv_padded():

    # Tests that running FilterActs with no padding is the same as running
    # theano's conv2D in valid mode

    rng = np.random.RandomState([2012,10,9])

    batch_size = 5
    rows = 10
    cols = 9
    channels = 3
    filter_rows = 4
    filter_cols = filter_rows
    num_filters = 16

    images = shared(rng.uniform(-1., 1., (channels, rows, cols,
        batch_size)).astype('float32'), name='images')
    filters = shared(rng.uniform(-1., 1., (channels, filter_rows,
        filter_cols, num_filters)).astype('float32'), name='filters')

    gpu_images = gpu_from_host(images)
    gpu_filters = gpu_from_host(filters)

    PAD = 3

    output = FilterActs(PAD)(gpu_images, gpu_filters)
    output = host_from_gpu(output)

    images_bc01 = T.alloc(0., batch_size, channels, rows + PAD * 2, cols + PAD * 2)

    images_bc01 = T.set_subtensor(images_bc01[:,:,PAD:-PAD,PAD:-PAD], images.dimshuffle(3,0,1,2))


    filters_bc01 = filters.dimshuffle(3,0,1,2)
    filters_bc01 = filters_bc01[:,:,::-1,::-1]

    output_conv2d = conv2d(images_bc01, filters_bc01,
            border_mode='valid')

    output_conv2d = output_conv2d.dimshuffle(1,2,3,0)

    f = function([], [output, output_conv2d])

    output, output_conv2d = f()

    warnings.warn("""test_match_valid_conv success criterion is not very strict. Can we verify that this is OK?
                     One possibility is that theano is numerically unstable and Alex's code is better.
                     Probably theano CPU 64 bit is OK but it's worth checking the others.""")

    assert output.shape == output_conv2d.shape

    if np.abs(output - output_conv2d).max() > 2.4e-6:
        assert type(output) == type(output_conv2d)
        assert output.dtype == output_conv2d.dtype
        if output.shape != output_conv2d.shape:
            print 'cuda-convnet shape: ',output.shape
            print 'theano shape: ',output_conv2d.shape
            assert False
        err = np.abs(output - output_conv2d)
        print 'absolute error range: ', (err.min(), err.max())
        print 'mean absolute error: ', err.mean()
        print 'cuda-convnet value range: ', (output.min(), output.max())
        print 'theano value range: ', (output_conv2d.min(), output_conv2d.max())
        assert False

def test_grad():
    rng = np.random.RandomState([2012, 10, 9])
    batch_size = 5
    rows = 10
    cols = 9
    channels = 3
    filter_rows = 4
    filter_cols = filter_rows
    num_filters = 16

    images = shared(rng.uniform(-1., 1., (channels, rows, cols,
        batch_size)).astype('float32'), name='images')
    filters = shared(rng.uniform(-1., 1., (channels, filter_rows,
        filter_cols, num_filters)).astype('float32'), name='filters')

    gpu_images = gpu_from_host(images)
    gpu_filters = gpu_from_host(filters)

    output = FilterActs()(gpu_images, gpu_filters)
    output = host_from_gpu(output)

    # Proper random projection, like verify_grad does.
    cost_weights = rng.normal(size=(num_filters, rows - filter_rows + 1,
                                    cols - filter_cols + 1, batch_size))
    cost = (constant(cost_weights) * output).sum()


    images_bc01 = images.dimshuffle(3,0,1,2)
    filters_bc01 = filters.dimshuffle(3,0,1,2)
    filters_bc01 = filters_bc01[:,:,::-1,::-1]

    output_conv2d = conv2d(images_bc01, filters_bc01,
            border_mode='valid')

    output_conv2d = output_conv2d.dimshuffle(1,2,3,0)
    # XXX: use verify_grad
    images_grad, filters_grad = grad(cost.sum(), [images, filters])
    reference_cost = (constant(cost_weights) * output_conv2d).sum()
    images_conv2d_grad, filters_conv2d_grad = grad(reference_cost,
                                                  [images, filters])
    f = function([], [images_grad, filters_grad,
                      images_conv2d_grad,
                      filters_conv2d_grad])

    images_grad, filters_grad, images_conv2d_grad, filters_conv2d_grad = f()

    warnings.warn("""test_match_valid_conv success criterion is not very strict. Can we verify that this is OK?
                     One possibility is that theano is numerically unstable and Alex's code is better.
                     Probably theano CPU 64 bit is OK but it's worth checking the others.""")
    # XXX: Refactor
    if np.abs(images_grad - images_conv2d_grad).max() > 7.7e-6:
        print "=== IMAGES GRADIENT ==="
        assert type(images_grad) == type(images_conv2d_grad)
        assert images_grad.dtype == images_conv2d_grad.dtype
        if images_grad.shape != images_conv2d_grad.shape:
            print 'cuda-convnet shape: ',images_grad.shape
            print 'theano shape: ',images_conv2d_grad.shape
            assert False
        err = np.abs(images_grad - images_conv2d_grad)
        print 'absolute error range: ', (err.min(), err.max())
        print 'mean absolute error: ', err.mean()
        print 'cuda-convnet value range: ', (images_grad.min(),
                                             images_grad.max())
        print 'theano value range: ', (images_conv2d_grad.min(),
                                       images_conv2d_grad.max())
        assert False
    if np.abs(filters_grad - filters_conv2d_grad).max() > 7.7e-6:
        print "=== FILTERS GRADIENT ==="
        assert type(filters_grad) == type(filters_conv2d_grad)
        assert filters_grad.dtype == filters_conv2d_grad.dtype
        if filters_grad.shape != filters_conv2d_grad.shape:
            print 'cuda-convnet shape: ',filters_grad.shape
            print 'theano shape: ',filters_conv2d_grad.shape
            assert False
        err = np.abs(filters_grad - filters_conv2d_grad)
        print 'absolute error range: ', (err.min(), err.max())
        print 'mean absolute error: ', err.mean()
        print 'cuda-convnet value range: ', (filters_grad.min(),
                                             filters_grad.max())
        print 'theano value range: ', (filters_conv2d_grad.min(),
                                       filters_conv2d_grad.max())
        assert False

if __name__ == '__main__':
    test_match_valid_conv_padded()

