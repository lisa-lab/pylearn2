#!/usr/bin/env python
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import logging
import sys
from pylearn2.utils import serial


logger = logging.getLogger(__name__)

for model_path in sys.argv[1:]:
    if len(sys.argv) > 2:
        print model_path
    model = serial.load(model_path)
    monitor = model.monitor
    channels = monitor.channels
    if not hasattr(monitor, '_epochs_seen'):
        logger.warning('old file, not all fields parsed correctly')
    else:
        logger.info('epochs seen: %s', monitor._epochs_seen)
    logger.info('time trained: %d', max(channels[key].time_record[-1]
                for key in channels))
    for key in sorted(channels.keys()):
        logger.info('%s : %d', key, channels[key].val_record[-1])
