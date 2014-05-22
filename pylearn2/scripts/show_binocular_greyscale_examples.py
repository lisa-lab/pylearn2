#!/usr/bin/env python
"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
import numpy as N
from pylearn2.gui import patch_viewer
from pylearn2.config import yaml_parse
from optparse import OptionParser


def main(options, positional_args):
    """
    .. todo::

        WRITEME
    """
    assert len(positional_args) == 1

    path ,= positional_args

    out = options.out
    rescale = options.rescale

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

    if path.endswith('.pkl'):
        from pylearn2.utils import serial
        obj = serial.load(path)
    elif path.endswith('.yaml'):
        print 'Building dataset from yaml...'
        obj =yaml_parse.load_path(path)
        print '...done'
    else:
        obj = yaml_parse.load(path)

    rows = options.rows
    cols = options.cols

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
    print 'norms of examples: '
    print '\tmin: ',norms.min()
    print '\tmean: ',norms.mean()
    print '\tmax: ',norms.max()

    print 'range of elements of examples',(examples.min(),examples.max())
    print 'dtype: ', examples.dtype

    examples = dataset.adjust_for_viewer(examples)

    if global_rescale:
        examples /= N.abs(examples).max()

    if len(examples.shape) != 4:
        print 'sorry, view_examples.py only supports image examples for now.'
        print 'this dataset has '+str(len(examples.shape)-2)+' topological dimensions'
        quit(-1)

    is_color = False
    assert examples.shape[3] == 2

    print examples.shape[1:3]

    pv = patch_viewer.PatchViewer( (rows, cols * 2), examples.shape[1:3], is_color = is_color)

    for i in xrange(rows*cols):
        # Load patches in backwards order for easier cross-eyed viewing
        # (Ian can't do the magic eye thing where you focus your eyes past the screen, must
        # focus eyes in front of screen)
        pv.add_patch(examples[i,:,:,1], activation = 0.0, rescale = patch_rescale)
        pv.add_patch(examples[i,:,:,0], activation = 0.0, rescale = patch_rescale)

    if out is None:
        pv.show()
    else:
        pv.save(out)

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option('--rows', dest='rows', default=20, action='store', type='int')
    parser.add_option('--cols', dest='cols', default=20, action='store', type='int')
    parser.add_option('--rescale', dest='rescale', default='global', action='store', type='string',
            help="how to rescale the patches for display: rescale|global|individual")
    parser.add_option('--out', dest='out', default=None, action='store',type='string', help='if not specified, displays an image. otherwise saves an image to the specified path')

    (options, positional_args) = parser.parse_args()
    main(options, positional_args)
