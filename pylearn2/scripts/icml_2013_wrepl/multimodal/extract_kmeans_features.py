import numpy as np
import os
import sys

from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess

def usage():
    print """
Run
python extract_kmeans_features.py public_test
to extract features for the ICML 2013 multimodal learning contest's public test images.
or
python extract_kmeans_features.py private_test
to extract features for the ICML 2013 multimodal learning contest's private test images
(which will be released 72 hours before the contest ends)
"""

if len(sys.argv) != 2:
    usage()
    print '(You used the wrong number of arguments)'
    quit(-1)

_, arg = sys.argv

if arg == 'public_test':
    base = preprocess('${PYLEARN2_DATA_PATH}/icml_2013_multimodal/public_test_lcn')
    expected_num_images = 500
elif arg == 'private_test':
    base = preprocess('${PYLEARN2_DATA_PATH}/icml_2013_multimodal/private_test_lcn')
    expected_num_images = 500
else:
    usage()
    print 'Unrecognized argument value:',arg
    print 'Recognized values are: public_test, private_test'

outdir = base[:-3] + 'layer_1_features'
serial.mkdir(outdir)

paths = os.listdir(base)
if len(paths) != expected_num_images:
    raise AssertionError("Something is wrong with your " + base \
            + "directory. It should contain " + str(expected_num_images) + \
            " lcn-preprocessed image files, but contains " + str(len(paths)))

means = np.load('lcn_means.npy')

norms = 1e-7 + np.sqrt(np.square(means).sum(axis=3).sum(axis=2).sum(axis=1))
means = np.transpose(np.transpose(means, (1, 2, 3, 0)) / norms, (3, 0, 1, 2))

from pylearn2.utils import sharedX
kernels = sharedX(np.transpose(means, (0, 3, 1, 2)))

import theano.tensor as T

X = T.TensorType(broadcastable=(False, False, False), dtype=kernels.dtype)()

bc01 = X.dimshuffle('x', 2, 0, 1)

Z = T.nnet.conv2d(input=bc01, filters=kernels, subsample = (means.shape[1] / 2, means.shape[2] /2),
        filter_shape = kernels.get_value().shape)

F = T.signal.downsample.max_pool_2d(Z, (2, 2))

F = T.clip(F - .5, 0., 10.)

from theano import function

f = function([X], F)

for i, path in enumerate(paths):
    print i
    try:
        X = np.load(base + '/' + path)

        assert X.ndim == 3

        F = f(X)

        np.save(outdir + '/' + path, F)
    except Exception, e:
        raise

