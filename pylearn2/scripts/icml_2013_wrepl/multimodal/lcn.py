
import numpy as np
import os
import sys

from pylearn2.utils import image
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess

def usage():
    print """
Run
python lcn.py public_test
to preprocess the ICML 2013 multimodal learning contest's public test images.
or
python lcn.py private_test
to preprocess the ICML 2013 multimodal learning contest's private test images
(which will be released 72 hours before the contest ends)
"""

if len(sys.argv) != 2:
    usage()
    print '(You used the wrong number of arguments)'
    quit(-1)

_, arg = sys.argv

if arg == 'public_test':
    base = preprocess('${PYLEARN2_DATA_PATH}/icml_2013_multimodal/public_test_images')
    outdir = base[:-6] + 'lcn'
    expected_num_images = 500
elif arg == 'private_test':
    base = preprocess('${PYLEARN2_DATA_PATH}/icml_2013_multimodal/private_test_images')
    outdir = base[:-6] + 'lcn'
    expected_num_images = 500
else:
    usage()
    print 'Unrecognized argument value:',arg
    print 'Recognized values are: public_test, private_test'

serial.mkdir(outdir)

paths = os.listdir(base)
if len(paths) != expected_num_images:
    raise AssertionError("Something is wrong with your " + base \
            + "directory. It should contain " + str(expected_num_images) + \
            " image files, but contains " + str(len(paths)))

kernel_shape = 7

from theano import tensor as T
from pylearn2.utils import sharedX
from pylearn2.datasets.preprocessing import gaussian_filter
from theano.tensor.nnet import conv2d

X = T.TensorType(dtype='float32', broadcastable=(True, False, False, True))()
from theano import config
if config.compute_test_value == 'raise':
    X.tag.test_value = np.zeros((1,32,32,1),dtype=X.dtype)
orig_X = X
filter_shape = (1, 1, kernel_shape, kernel_shape)
filters = sharedX(gaussian_filter(kernel_shape).reshape(filter_shape))

X = X.dimshuffle(0, 3, 1, 2)

convout = conv2d(X, filters=filters, border_mode='full')

# For each pixel, remove mean of 9x9 neighborhood
mid = int(np.floor(kernel_shape/ 2.))
centered_X = X - convout[:,:,mid:-mid,mid:-mid]

# Scale down norm of 9x9 patch if norm is bigger than 1
sum_sqr_XX = conv2d(T.sqr(X), filters=filters, border_mode='full')

denom = T.sqrt(sum_sqr_XX[:,:,mid:-mid,mid:-mid])
per_img_mean = denom.mean(axis = [2,3])
divisor = T.largest(per_img_mean.dimshuffle(0,1, 'x', 'x'), denom)

new_X = centered_X / T.maximum(1., divisor)

new_X = new_X.dimshuffle(0, 2, 3, 1)

from theano import function
f = function([orig_X], new_X)

j = 0
for path in paths:
    if j % 100 == 0:
        print j
    try:
        raw_path = path
        path = base + '/' + path
        img = image.load(path)

        #image.show(img)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, 0:3]
        img = img.reshape(*([1]+list(img.shape))).astype('float32')
        channels = [f(img[:,:,:,i:i+1]) for i in xrange(img.shape[3])]
        if len(channels) != 3:
            assert len(channels) == 1
            channels = [channels[0] ] * 3
        img = np.concatenate(channels, axis=3)
        img = img[0,:,:,:]

        assert not np.any(np.isnan(img))
        assert not np.any(np.isinf(img))

        path = outdir + '/' + raw_path
        path = path[0:-3]
        assert path.endswith('.')
        path = path + 'npy'
        np.save(path, img)
    except Exception, e:
        raise
        print e
    j += 1

