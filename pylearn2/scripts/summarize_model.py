#!/bin/env python
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np
import sys
path = sys.argv[1]
from pylearn2.utils import serial
model = serial.load(path)
for param in model.get_params():
    name = param.name
    if name is None:
        name = '<anon>'
    v = param.get_value()
    print name+': '+str((v.min(),v.mean(),v.max()))+' '+str(v.shape)
    if np.sign(v.min()) != np.sign(v.max()):
        v = np.abs(v)
        print 'abs('+name+'): '+str((v.min(),v.mean(),v.max()))
    if v.ndim == 2:
        row_norms = np.sqrt(np.square(v).sum(axis=1))
        print name +" row norms: ",(row_norms.min(), row_norms.mean(), row_norms.max())
        col_norms = np.sqrt(np.square(v).sum(axis=0))
        print name +" col norms: ",(col_norms.min(), col_norms.mean(), col_norms.max())

if hasattr(model,'monitor'):
    print 'trained on',model.monitor.get_examples_seen(),' examples'
    print 'which corresponds to',model.monitor.get_batches_seen(),'batches'
    try:
        print model.monitor.get_epochs_seen(),'epochs'
    except:
        pass
    if hasattr(model.monitor, 'training_succeeded'):
        if model.monitor.training_succeeded:
            print 'Training succeeded'
        else:
            print 'Training was not yet completed at the time of this save.'
    else:
        print 'This pickle file is damaged, or was made before the Monitor tracked whether training completed.'
