#!/bin/env python
from pylearn2.utils import serial
import matplotlib.pyplot as plt
import numpy as N
import sys

model = serial.load(sys.argv[1])


for channel in model.monitor.channels.values():
    plt.plot(N.asarray(channel.example_record),N.asarray(channel.val_record))

plt.show()
