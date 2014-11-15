from __future__ import print_function

import numpy as np
import sys

from theano.compat.six.moves import xrange
from theano import function
from theano import tensor as T

from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess

def usage():
    """
Run
python make_submission.py <model> <test set>
where <test set> is public_test or private_test
(private_test will be released 72 hours before the end of the contest)
"""

if len(sys.argv) != 3:
    usage()
    print("(You used the wrong number of arguments)")
    quit(-1)

_, model_path, test_set = sys.argv

model = serial.load(model_path)

# Load BOVW features
features_dir = preprocess('${PYLEARN2_DATA_PATH}/icml_2013_multimodal/'+test_set+'_layer_2_features')
vectors = []
for i in xrange(500):
    vectors.append(serial.load(features_dir + '/' + str(i) + '.npy'))
features = np.concatenate(vectors, axis=0)
del vectors


# Load BOW targets
f = open('wordlist.txt')
wordlist = f.readlines()
f.close()
options_dir = preprocess('${PYLEARN2_DATA_PATH}/icml_2013_multimodal/'+test_set+'_options')
def load_options(option):
    rval = np.zeros((500, 4000), dtype='float32')
    for i in xrange(500):
        f = open(options_dir + '/' + str(i) + '.option_' + str(option) + '.desc')
        l = f.readlines()
        f.close()
        for w in l:
            if w in wordlist:
                rval[i, wordlist.index(w)] = 1
    return rval
option_0, option_1 = [load_options(0), load_options(1)]

X = T.matrix()
Y0 = T.matrix()
Y1 = T.matrix()

Y_hat = model.fprop(X)

cost_0 = model.layers[-1].kl(Y=Y0, Y_hat=Y_hat)
cost_1 = model.layers[-1].kl(Y=Y1, Y_hat=Y_hat)

f = function([X, Y0, Y1], cost_1 < cost_0)

prediction = f(features, option_0, option_1)

f = open('submission.csv', 'w')

for i in xrange(500):
    f.write(str(prediction[i])+'\n')
f.close()
