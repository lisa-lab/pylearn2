#!/usr/bin/env python
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import logging
import numpy as np
import sys

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    path = sys.argv[1]
    from pylearn2.utils import serial
    model = serial.load(path)
    for param in model.get_params():
        name = param.name
        if name is None:
            name = '<anon>'
        v = param.get_value()
        logger.info(name + ': ' + str((v.min(), v.mean(), v.max())) +
                    ' ' + str(v.shape))
        if np.sign(v.min()) != np.sign(v.max()):
            v = np.abs(v)
            logger.info('abs(' + name + '): ' +
                        str((v.min(), v.mean(), v.max())))
        if v.ndim == 2:
            row_norms = np.sqrt(np.square(v).sum(axis=1))
            logger.info(name + " row norms: %s",
                        (row_norms.min(), row_norms.mean(), row_norms.max()))
            col_norms = np.sqrt(np.square(v).sum(axis=0))
            logger.info(name + " col norms: %s",
                        (col_norms.min(), col_norms.mean(), col_norms.max()))

    if hasattr(model,'monitor'):
        logger.info('trained on %s examples',
                    model.monitor.get_examples_seen())
        logger.info('which corresponds to %s batches',
                    model.monitor.get_batches_seen())
        try:
            logger.info('%s epochs', model.monitor.get_epochs_seen())
        except:
            pass
        if hasattr(model.monitor, 'training_succeeded'):
            if model.monitor.training_succeeded:
                logger.info('Training succeeded')
            else:
                logger.info('Training was not yet completed ' +
                            'at the time of this save.')
        else:
            logger.error('This pickle file is damaged, or was made before ' +
                         'the Monitor tracked whether training completed.')
