from __future__ import print_function

from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys

_, model_path = sys.argv

model = serial.load(model_path)

src = model.dataset_yaml_src
batch_size = 100
model.set_batch_size(batch_size)

assert src.find('train') != -1
test = yaml_parse.load(src)
test = test.get_test_set()
assert test.X.shape[0] == 10000

test.X = test.X.astype('float32')

import theano.tensor as T

Xb = model.get_input_space().make_batch_theano()
Xb.name = 'Xb'
yb = model.get_output_space().make_batch_theano()
yb.name = 'yb'

ymf = model.fprop(Xb)
ymf.name = 'ymf'

from theano import function

yl = T.argmax(yb,axis=1)

mf1acc = 1.-T.neq(yl , T.argmax(ymf,axis=1)).mean()

batch_acc = function([Xb,yb],[mf1acc])

# The averaging math assumes batches are all same size
assert test.X.shape[0] % batch_size == 0


def accs():
    mf1_accs = []
    assert isinstance(test.X.shape[0], py_integer_types)
    assert isinstance(batch_size, py_integer_types)
    iterator = test.iterator(mode = 'even_sequential',
                            batch_size = batch_size,
                            data_specs = model.cost_from_X_data_specs())
    for item in iterator:
        x_arg, y_arg = item
        mf1_accs.append(batch_acc(x_arg, y_arg)[0])
    return sum(mf1_accs) / float(len(mf1_accs))


result = accs()

print(1. - result)

