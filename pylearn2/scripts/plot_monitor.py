#!/bin/env python
from pylearn2.utils import serial
import matplotlib.pyplot as plt
import numpy as N
import sys
from theano.printing import _TagGenerator

model = serial.load(sys.argv[1])
channels = model.monitor.channels


#Make a list of short codes for each channel so user can specify them easily
tag_generator = _TagGenerator()
codebook = {}
for channel_name in sorted(channels):
    code = tag_generator.get_tag()
    codebook[code] = channel_name

#Display the codebook
for code in sorted(codebook):
    print code + '. ' + codebook[code]

print

#if there is more than one channel in the monitor ask which ones to plot
prompt = len(channels.values()) > 0

if prompt:

    response = raw_input('Enter a list of channels to plot (example: A, B)): ')

    response = response.replace(' ','')

    codes = response.split(',')
else:
    codes ,= codebook.keys()

#plot the requested channels
for code in codes:

    channel_name= codebook[code]

    channel = channels[channel_name]

    plt.plot( N.asarray(channel.example_record), \
              N.asarray(channel.val_record), \
              label = channel_name)

plt.legend()

plt.show()
