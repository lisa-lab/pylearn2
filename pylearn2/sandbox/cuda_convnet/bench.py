__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from pylearn2.testing.skip import skip_if_no_gpu
skip_if_no_gpu()
import numpy as np
from theano import shared
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.tensor.nnet.conv import conv2d
from theano import function
import time


# Tests that running FilterActs with no padding is the same as running
# theano's conv2D in valid mode

rng = np.random.RandomState([2012,10,9])

batch_size = 128
rows = 32
cols = 32
channels = 3
filter_rows = 7
filter_cols = filter_rows
num_filters = 16

base_image_value = rng.uniform(-1., 1., (channels, rows, cols,
    batch_size)).astype('float32')
base_filters_value = rng.uniform(-1., 1., (channels, filter_rows,
    filter_cols, num_filters)).astype('float32')
images = shared(base_image_value)
filters = shared(base_filters_value, name='filters')

# bench.py should always be run in gpu mode so we should not need a gpu_from_host here
output = FilterActs()(images, filters)

output_shared = shared( output.eval() )

cuda_convnet = function([], updates = { output_shared : output } )
cuda_convnet.name = 'cuda_convnet'

images_bc01 = base_image_value.transpose(3,0,1,2)
filters_bc01 = base_filters_value.transpose(3,0,1,2)
filters_bc01 = filters_bc01[:,:,::-1,::-1]

images_bc01 = shared(images_bc01)
filters_bc01 = shared(filters_bc01)

output_conv2d = conv2d(images_bc01, filters_bc01,
        border_mode='valid')

output_conv2d_shared = shared(output_conv2d.eval())

baseline = function([], updates = { output_conv2d_shared : output_conv2d } )
baseline.name = 'baseline'

def bench(f):
    print 'Benchmarking ',f
    print 'Warming up...'
    for i in xrange(3):
        f()
    trials = 10
    t1 = time.time()
    for i in xrange(trials):
        f()
    t2 = time.time()
    print 'Mean time: ',(t2-t1)/float(trials)

bench(baseline)
bench(cuda_convnet)
