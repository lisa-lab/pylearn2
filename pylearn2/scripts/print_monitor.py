#!/bin/env python
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import sys
_, model_path = sys.argv
from pylearn2.utils import serial
model = serial.load(model_path)
monitor = model.monitor
channels = monitor.channels
for key in sorted(channels.keys()):
    print key, ':', channels[key].val_record[-1]
