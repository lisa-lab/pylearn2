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
if hasattr(model,'monitor'):
    print 'trained on',model.monitor.get_examples_seen(),' examples'
    print 'which corresponds to',model.monitor.get_batches_seen(),'batches'
    print model.monitor.get_epochs_seen(),'epochs'
