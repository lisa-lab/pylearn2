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

def print_monitor(args):
    from pylearn2.utils import serial
    import gc
    for model_path in args:
        if len(args) > 1:
            print(model_path)
        model = serial.load(model_path)
        monitor = model.monitor
        del model
        gc.collect()
        channels = monitor.channels
        if not hasattr(monitor, '_epochs_seen'):
            print('old file, not all fields parsed correctly')
        else:
            print('epochs seen: ', monitor._epochs_seen)
        print('time trained: ', max(channels[key].time_record[-1] for key in
              channels))
        for key in sorted(channels.keys()):
            print(key, ':', channels[key].val_record[-1])


if __name__ == '__main__':
    import sys
    print_monitor(sys.argv[1:])
