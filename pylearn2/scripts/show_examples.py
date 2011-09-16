#!/bin/env python
import numpy as N
import sys
from pylearn2.gui import patch_viewer
from pylearn2.config import yaml_parse

assert len(sys.argv) > 1
path = sys.argv[1]

rescale = 'global'
if len(sys.argv) > 2:
    arg2 = sys.argv[2]
    assert arg2.startswith('--rescale=')
    split = arg2.split('--rescale=')
    assert len(split) == 2
    rescale = split[1]

if rescale == 'none':
    global_rescale = False
    patch_rescale = False
elif rescale == 'global':
    global_rescale = True
    patch_rescale = False
elif rescale == 'individual':
    global_rescale = False
    patch_rescale = True
else:
    assert False

assert len(sys.argv) <4

if path.endswith('.pkl'):
    from pylearn2.utils import serial
    obj = serial.load(path)
elif path.endswith('.yaml'):
    obj =yaml_parse.load_path(path)
else:
    obj = yaml_parse.load(path)

rows = 20
cols = 20

if hasattr(obj,'get_batch_topo'):
    #obj is a Dataset
    dataset = obj
    examples = dataset.get_batch_topo(rows*cols)
else:
    #obj is a Model
    model = obj
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    theano_rng = RandomStreams(42)
    design_examples_var = model.random_design_matrix(batch_size = rows * cols, theano_rng = theano_rng)
    from theano import function
    print 'compiling sampling function'
    f = function([],design_examples_var)
    print 'sampling'
    design_examples = f()
    print 'loading dataset'
    dataset = yaml_parse.load(model.dataset_yaml_src)
    examples = dataset.get_topological_view(design_examples)

norms = N.asarray( [
        N.sqrt(N.sum(N.square(examples[i,:])))
                    for i in xrange(examples.shape[0])
                    ])
print 'norms of exmaples: '
print '\tmin: ',norms.min()
print '\tmean: ',norms.mean()
print '\tmax: ',norms.max()

print 'range of elements of examples',(examples.min(),examples.max())
print 'dtype: ', examples.dtype
if global_rescale:
    examples /= N.abs(examples).max()

if len(examples.shape) != 4:
    print 'sorry, view_examples.py only supports image examples for now.'
    print 'this dataset has '+str(len(examples)-2)+' topological dimensions'
    quit(-1)
#

if examples.shape[3] == 1:
    is_color = False
elif examples.shape[3] == 3:
    is_color = True
else:
    print 'got unknown image format with '+str(examples.shape[3])+' channels'
    print 'supported formats are 1 channel greyscale or three channel RGB'
    quit(-1)
#

print examples.shape[1:3]

pv = patch_viewer.PatchViewer( (rows, cols), examples.shape[1:3], is_color = is_color)

for i in xrange(rows*cols):
    pv.add_patch(examples[i,:,:,:], activation = 0.0, rescale = patch_rescale)
#

pv.show()
