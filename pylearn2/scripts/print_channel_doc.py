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

import sys
from pylearn2.utils import serial

if __name__ == "__main__":
    for model_path in sys.argv[1:]:
        if len(sys.argv) > 2:
            print(model_path)
        model = serial.load(model_path)
        monitor = model.monitor
        channels = monitor.channels
        for key in sorted(channels.keys()):
            print(key)
            value = channels[key]
            if not hasattr(value, 'doc'):
                print("\tOld pkl file, written before doc system.")
            else:
                doc = value.doc
                if doc is None:
                    print("No doc available.")
                else:
                    print(doc)
            print()
