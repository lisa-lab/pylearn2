#!/usr/bin/env python
"""
.. todo::

    WRITEME
"""
from __future__ import print_function

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import argparse
import numpy as np
from theano.compat.six.moves import xrange
from pylearn2.gui import patch_viewer
from pylearn2.config import yaml_parse


def show_examples(path, rows, cols, rescale='global', out=None):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    path : string
        The pickle or YAML file to show examples of
    rows : int
        WRITEME
    cols : int
        WRITEME
    rescale : {'rescale', 'global', 'individual'}
        Default is 'rescale', WRITEME
    out : string, optional
        WRITEME
    """

    if rescale == 'none':
        global_rescale = False
        patch_rescale = False
    elif rescale == 'global':
        global_rescale = True
        patch_rescale = False
    elif rescale == 'individual':
        global_rescale = False
        patch_rescale = True

    if path.endswith('.pkl'):
        from pylearn2.utils import serial
        obj = serial.load(path)
    elif path.endswith('.yaml'):
        print('Building dataset from yaml...')
        obj = yaml_parse.load_path(path)
        print('...done')
    else:
        obj = yaml_parse.load(path)

    if hasattr(obj, 'get_batch_topo'):
        # obj is a Dataset
        dataset = obj

        examples = dataset.get_batch_topo(rows*cols)
    else:
        # obj is a Model
        model = obj
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        theano_rng = RandomStreams(42)
        design_examples_var = model.random_design_matrix(
            batch_size=rows * cols, theano_rng=theano_rng
        )
        from theano import function
        print('compiling sampling function')
        f = function([], design_examples_var)
        print('sampling')
        design_examples = f()
        print('loading dataset')
        dataset = yaml_parse.load(model.dataset_yaml_src)
        examples = dataset.get_topological_view(design_examples)

    norms = np.asarray([np.sqrt(np.sum(np.square(examples[i, :])))
                        for i in xrange(examples.shape[0])])
    print('norms of examples: ')
    print('\tmin: ', norms.min())
    print('\tmean: ', norms.mean())
    print('\tmax: ', norms.max())

    print('range of elements of examples', (examples.min(), examples.max()))
    print('dtype: ', examples.dtype)

    examples = dataset.adjust_for_viewer(examples)

    if global_rescale:
        examples /= np.abs(examples).max()

    if len(examples.shape) != 4:
        print('sorry, view_examples.py only supports image examples for now.')
        print('this dataset has ' + str(len(examples.shape) - 2), end='')
        print('topological dimensions')
        quit(-1)

    if examples.shape[3] == 1:
        is_color = False
    elif examples.shape[3] == 3:
        is_color = True
    else:
        print('got unknown image format with', str(examples.shape[3]), end='')
        print('channels')
        print('supported formats are 1 channel greyscale or three channel RGB')
        quit(-1)

    print(examples.shape[1:3])

    pv = patch_viewer.PatchViewer((rows, cols), examples.shape[1:3],
                                  is_color=is_color)

    for i in xrange(rows*cols):
        pv.add_patch(examples[i, :, :, :], activation=0.0,
                     rescale=patch_rescale)

    if out is None:
        pv.show()
    else:
        pv.save(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--rows', default=20, type=int)
    parser.add_argument('--cols', default=20, type=int)
    parser.add_argument('--rescale', default='global',
                        choices=['none', 'global', 'individual'],
                        help="how to rescale the patches for display")
    parser.add_argument('--out',
                        help='if not specified, displays an image. '
                             'otherwise saves an image to the specified path')
    parser.add_argument("path")

    args = parser.parse_args()
    show_examples(args.path, args.rows, args.cols, args.rescale, args.out)
