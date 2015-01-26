__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

from pylearn2.testing.skip import skip_if_no_gpu
skip_if_no_gpu()
import logging
import numpy as np
from theano import shared
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda import gpu_from_host
from theano.sandbox.cuda import host_from_gpu
from theano.tensor.nnet.conv import conv2d
from theano import function


def main():
    logger = logging.getLogger(__name__)

    # Tests that running FilterActs with no padding is the same as running
    # theano's conv2D in valid mode

    rng = np.random.RandomState([2012, 10, 9])

    batch_size = 128
    rows = 32
    cols = 32
    channels = 3
    filter_rows = 7
    filter_cols = filter_rows
    num_filters = 16

    images = shared(rng.uniform(-1., 1., (channels, rows, cols,
                    batch_size)).astype('float32'), name='images')
    filters = shared(rng.uniform(-1., 1., (channels, filter_rows,
                     filter_cols, num_filters)).astype('float32'),
                     name='filters')

    gpu_images = gpu_from_host(images)
    gpu_filters = gpu_from_host(filters)

    output = FilterActs()(gpu_images, gpu_filters)
    output = host_from_gpu(output)

    images_bc01 = images.dimshuffle(3, 0, 1, 2)
    filters_bc01 = filters.dimshuffle(3, 0, 1, 2)
    filters_bc01 = filters_bc01[:, :, ::-1, ::-1]

    output_conv2d = conv2d(images_bc01, filters_bc01,
                           border_mode='valid')

    output_conv2d = output_conv2d.dimshuffle(1, 2, 3, 0)

    f = function([], [output, output_conv2d])

    def err():
        output, output_conv2d = f()
        diff = output - output_conv2d

        return np.abs(diff).max()

    prev_err = err()
    accepted_steps = 0

    while True:
        logger.debug('Current error: {0}'.format(prev_err))
        change_filters = rng.randint(2)

        if change_filters:
            target = filters
        else:
            target = images

        old_val = target.get_value()

        selector = rng.randint(2)
        if selector == 0:
            new_val = old_val + rng.uniform(-.1, .1, old_val.shape)
        else:
            idx1 = rng.randint(old_val.shape[0])
            idx2 = rng.randint(old_val.shape[1])
            idx3 = rng.randint(old_val.shape[2])
            idx4 = rng.randint(old_val.shape[3])
            new_val = old_val.copy()
            new_val[idx1, idx2, idx3, idx4] += rng.uniform(-1., 1.)
        new_val = new_val.astype(old_val.dtype)

        target.set_value(new_val)

        new_err = err()

        if new_err <= prev_err:
            logger.debug(
                'Failed to move beyond step {0}'.format(accepted_steps))
            target.set_value(old_val)
        else:
            prev_err = new_err
            accepted_steps += 1

if __name__ == "__main__":
    main()
