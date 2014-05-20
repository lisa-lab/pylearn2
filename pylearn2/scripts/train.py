#!/usr/bin/env python
"""
Script implementing the logic for training pylearn2 models.

This is a "driver" that we recommend using for all but the most unusual
training experiments.

Basic usage:

.. code-block:: none

    train.py yaml_file.yaml

The YAML file should contain a pylearn2 YAML description of a
`pylearn2.train.Train` object (or optionally, a list of Train objects to
run sequentially).

See `doc/yaml_tutorial` for a description of how to write the YAML syntax.

The following environment variables will be locally defined and available
for use within the YAML file:

- `PYLEARN2_TRAIN_BASE_NAME`: the name of the file within the directory
  (`foo/bar.yaml` -> `bar.yaml`)
- `PYLEARN2_TRAIN_DIR`: the directory containing the YAML file
  (`foo/bar.yaml` -> `foo`)
- `PYLEARN2_TRAIN_FILE_FULL_STEM`: the filepath with the file extension
  stripped off.
  `foo/bar.yaml` -> `foo/bar`)
- `PYLEARN2_TRAIN_FILE_STEM`: the stem of `PYLEARN2_TRAIN_BASE_NAME`
  (`foo/bar.yaml` -> `bar`)
- `PYLEARN2_TRAIN_PHASE` : set to `phase0`, `phase1`, etc. during iteration
  through a list of Train objects. Not defined for a single train object.

These environment variables are especially useful for setting the save
path. For example, to make sure that `foo/bar.yaml` saves to `foo/bar.pkl`,
use

.. code-block:: none

    save_path: "${PYLEARN2_TRAIN_FILE_FULL_STEM}.pkl"

This way, if you copy `foo/bar.yaml` to `foo/bar2.yaml`, the output of
`foo/bar2.yaml` won't overwrite `foo/bar.pkl`, but will automatically save
to foo/bar2.pkl.

For example configuration files that are consumable by this script, see

- `pylearn2/scripts/tutorials/grbm_smd`
- `pylearn2/scripts/tutorials/dbm_demo`
- `pylearn2/scripts/papers/maxout`

Use `train.py -h` to see an auto-generated description of advanced options.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
# Standard library imports
import argparse
import gc
import logging
import os

# Third-party imports
import numpy as np

# Local imports
from pylearn2.utils import serial
from pylearn2.utils.logger import (
    CustomStreamHandler, CustomFormatter, restore_defaults
)


class FeatureDump(object):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    encoder : WRITEME
    dataset : WRITEME
    path : WRITEME
    batch_size : WRITEME
    topo : WRITEME
    """
    def __init__(self, encoder, dataset, path, batch_size=None, topo=False):
        self.encoder = encoder
        self.dataset = dataset
        self.path = path
        self.batch_size = batch_size
        self.topo = topo

    def main_loop(self, **kwargs):
        """
        .. todo::

            WRITEME
        """
        if self.batch_size is None:
            if self.topo:
                data = self.dataset.get_topological_view()
            else:
                data = self.dataset.get_design_matrix()
            output = self.encoder.perform(data)
        else:
            myiterator = self.dataset.iterator(mode='sequential',
                                               batch_size=self.batch_size,
                                               topo=self.topo)
            chunks = []
            for data in myiterator:
                chunks.append(self.encoder.perform(data))
            output = np.concatenate(chunks)
        np.save(self.path, output)


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Launch an experiment from a YAML configuration file.",
        epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--level-name', '-L',
                        action='store_true',
                        help='Display the log level (e.g. DEBUG, INFO) '
                             'for each logged message')
    parser.add_argument('--timestamp', '-T',
                        action='store_true',
                        help='Display human-readable timestamps for '
                             'each logged message')
    parser.add_argument('--time-budget', '-t', type=int,
                        help='Time budget in seconds. Stop training at '
                             'the end of an epoch if more than this '
                             'number of seconds has elapsed.')
    parser.add_argument('--verbose-logging', '-V',
                        action='store_true',
                        help='Display timestamp, log level and source '
                             'logger for every logged message '
                             '(implies -T).')
    parser.add_argument('--debug', '-D',
                        action='store_true',
                        help='Display any DEBUG-level log messages, '
                             'suppressed by default.')
    parser.add_argument('config', action='store',
                        choices=None,
                        help='A YAML configuration file specifying the '
                             'training procedure')
    return parser


if __name__ == "__main__":
    """See module-level docstring for a description of the script."""
    parser = make_argument_parser()
    args = parser.parse_args()
    train_obj = serial.load_train_file(args.config)
    try:
        iter(train_obj)
        iterable = True
    except TypeError as e:
        iterable = False

    # Undo our custom logging setup.
    restore_defaults()
    # Set up the root logger with a custom handler that logs stdout for INFO
    # and DEBUG and stderr for WARNING, ERROR, CRITICAL.
    root_logger = logging.getLogger()
    if args.verbose_logging:
        formatter = logging.Formatter(fmt="%(asctime)s %(name)s %(levelname)s "
                                          "%(message)s")
        handler = CustomStreamHandler(formatter=formatter)
    else:
        if args.timestamp:
            prefix = '%(asctime)s '
        else:
            prefix = ''
        formatter = CustomFormatter(prefix=prefix, only_from='pylearn2')
        handler = CustomStreamHandler(formatter=formatter)
    root_logger.addHandler(handler)
    # Set the root logger level.
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)

    if iterable:
        for number, subobj in enumerate(iter(train_obj)):
            # Publish a variable indicating the training phase.
            phase_variable = 'PYLEARN2_TRAIN_PHASE'
            phase_value = 'phase%d' % (number + 1)
            os.environ[phase_variable] = phase_value

            # Execute this training phase.
            subobj.main_loop(time_budget=args.time_budget)

            # Clean up, in case there's a lot of memory used that's
            # necessary for the next phase.
            del subobj
            gc.collect()
    else:
        train_obj.main_loop(time_budget=args.time_budget)
