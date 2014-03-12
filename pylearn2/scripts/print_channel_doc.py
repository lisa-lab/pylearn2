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

if __name__ == "__main__":
    for model_path in sys.argv[1:]:
        if len(sys.argv) > 2:
            logger.info(model_path)
        model = serial.load(model_path)
        monitor = model.monitor
        channels = monitor.channels
        for key in sorted(channels.keys()):
            print key
            value = channels[key]
            if not hasattr(value, 'doc'):
                logger.warning("\tOld pkl file, written before doc system.")
            else:
                doc = value.doc
                if doc is None:
                    logger.info("No doc available.")
                else:
                    logger.info(doc)
            print
