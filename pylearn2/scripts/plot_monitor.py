#!/bin/env python
"""
usage:

plot_monitor.py model_1.pkl model_2.pkl ... model_n.pkl

Loads any number of .pkl files produced by train.py. Extracts
all of their monitoring channels and prompts the user to select
a subset of them to be plotted.

"""

from pylearn2.utils import serial
import matplotlib.pyplot as plt
import numpy as np
import sys
from theano.printing import _TagGenerator
from pylearn2.utils.string_utils import number_aware_alphabetical_key

channels = {}

for i, arg in enumerate(sys.argv[1:]):
    model = serial.load(arg)
    this_model_channels = model.monitor.channels

    if len(sys.argv) > 2:
        postfix = ":model%d" % i
    else:
        postfix = ""

    for channel in this_model_channels:
        channels[channel+postfix] = this_model_channels[channel]


while True:
#Make a list of short codes for each channel so user can specify them easily
    tag_generator = _TagGenerator()
    codebook = {}
    sorted_codes = []
    for channel_name in sorted(channels, key = number_aware_alphabetical_key):
        code = tag_generator.get_tag()
        codebook[code] = channel_name
        sorted_codes.append(code)

    x_axis = 'example'

    if len(channels.values()) == 0:
        print "there are no channels to plot"
        break

    #if there is more than one channel in the monitor ask which ones to plot
    prompt = len(channels.values()) > 1

    if prompt:

        #Display the codebook
        for code in sorted_codes:
            print code + '. ' + codebook[code]

        print

        print "Put e or b in the list somewhere to plot epochs or batches, respectively."
        response = raw_input('Enter a list of channels to plot (example: A, C,F-G)) or q to quit: ')

        if response == 'q':
            break

        #Remove spaces
        response = response.replace(' ','')

        #Split into list
        codes = response.split(',')

        final_codes = set([])

        for code in codes:
            if code == 'e':
                x_axis = 'epoch'
                continue
            if code == 'b':
                x_axis = 'batch'
                continue
            if code.find('-') != -1:
                #The current list element is a range of codes

                rng = code.split('-')

                if len(rng) != 2:
                    print "Input not understood: "+code
                    quit(-1)

                found = False
                for i in xrange(len(sorted_codes)):
                    if sorted_codes[i] == rng[0]:
                        found = True
                        break

                if not found:
                    print "Invalid code: "+rng[0]
                    quit(-1)

                found = False
                for j in xrange(i,len(sorted_codes)):
                    if sorted_codes[j] == rng[1]:
                        found = True
                        break

                if not found:
                    print "Invalid code: "+rng[1]
                    quit(-1)

                final_codes = final_codes.union(set(sorted_codes[i:j+1]))
            else:
                #The current list element is just a single code
                final_codes = final_codes.union(set([code]))

    else:
        final_codes ,= set(codebook.keys())

    fig = plt.figure()
    ax = plt.subplot(1,1,1)

# Shink current axis' width by 20% so legend will still appear in the window
    box = ax.get_position()

    try:
        x0 = box.x0
        y0 = box.y0
        width = box.width
        height = box.height
    except:
        x0, width, y0, height = box


    ax.set_position([x0, y0, width * 0.8, height])

    ax.ticklabel_format( scilimits = (-3,3), axis = 'both')

    plt.xlabel('# '+x_axis+'s')


#plot the requested channels
    for code in sorted(final_codes):

        channel_name= codebook[code]

        channel = channels[channel_name]

        y = np.asarray(channel.val_record)

        if np.any(np.isnan(y)):
            print channel_name + ' contains NaNs'

        if np.any(np.isinf(y)):
            print channel_name + 'contains infinite values'

        if x_axis == 'example':
            x = np.asarray(channel.example_record)
        elif x_axis == 'batch':
            x = np.asarray(channel.batch_record)
        elif x_axis == 'epoch':
            x = np.arange(len(y))
        else:
            assert False


        plt.plot( x,
                  y,
                  label = channel_name)


    plt.legend(bbox_to_anchor=(1.05, 1),  loc=2, borderaxespad=0.)

    plt.show()

    if not prompt:
        break
