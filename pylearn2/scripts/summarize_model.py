#!/usr/bin/env python
"""
This script summarizes a model by showing some statistics about
the parameters and checking whether the model completed
training succesfully
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

from pylearn2.compat import first_key
from pylearn2.utils import serial


def summarize(path):
    """
    Summarize the model

    Parameters
    ----------
    path : str
        The path to the pickled model to summarize
    """
    model = serial.load(path)
    for param in model.get_params():
        name = param.name
        if name is None:
            name = '<anon>'
        v = param.get_value()
        print(name + ': ' + str((v.min(), v.mean(), v.max())), end='')
        print(str(v.shape))
        if np.sign(v.min()) != np.sign(v.max()):
            v = np.abs(v)
            print('abs(' + name + '): ' + str((v.min(), v.mean(), v.max())))
        if v.ndim == 2:
            row_norms = np.sqrt(np.square(v).sum(axis=1))
            print(name + " row norms:", end='')
            print((row_norms.min(), row_norms.mean(), row_norms.max()))
            col_norms = np.sqrt(np.square(v).sum(axis=0))
            print(name + " col norms:", end='')
            print((col_norms.min(), col_norms.mean(), col_norms.max()))

    if hasattr(model, 'monitor'):
        print('trained on', model.monitor.get_examples_seen(), 'examples')
        print('which corresponds to ', end='')
        print(model.monitor.get_batches_seen(), 'batches')
        key = first_key(model.monitor.channels)
        hour = float(model.monitor.channels[key].time_record[-1]) / 3600.
        print('Trained for {0} hours'.format(hour))
        try:
            print(model.monitor.get_epochs_seen(), 'epochs')
        except Exception:
            pass
        if hasattr(model.monitor, 'training_succeeded'):
            if model.monitor.training_succeeded:
                print('Training succeeded')
            else:
                print('Training was not yet completed ' +
                      'at the time of this save.')
        else:
            print('This pickle file is damaged, or was made before the ' +
                  'Monitor tracked whether training completed.')


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Print some parameter statistics of a pickled model "
                    "and check if it completed training succesfully."
    )
    parser.add_argument('path',
                        help='The pickled model to summarize')
    return parser


if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    summarize(args.path)
