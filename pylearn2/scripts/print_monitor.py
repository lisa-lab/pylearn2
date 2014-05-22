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
import sys
from pylearn2.utils import serial
for model_path in sys.argv[1:]:
    if len(sys.argv) > 2:
        print model_path
    model = serial.load(model_path)
    monitor = model.monitor
    channels = monitor.channels
    if not hasattr(monitor, '_epochs_seen'):
        print 'old file, not all fields parsed correctly'
    else:
        print 'epochs seen: ',monitor._epochs_seen
    print 'time trained: ',max(channels[key].time_record[-1] for key in
            channels)
    for key in sorted(channels.keys()):
        print key, ':', channels[key].val_record[-1]
