#!/bin/env python
"""
usage:

plot_monitor.py model_1.pkl model_2.pkl ... model_n.pkl

Loads any number of .pkl files produced by train.py. Extracts
all of their monitoring channels and prompts the user to select
a subset of them to be plotted.

"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

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
        codebook['<'+channel_name+'>'] = channel_name
        sorted_codes.append(code)

    x_axis = 'example'
    print 'set x_axis to example'

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

        print "Put e, b, or s in the list somewhere to plot epochs, batches, or seconds, respectively."
        response = raw_input('Enter a list of channels to plot (example: A, C,F-G, t, <test_err>) or q to quit: ')

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
            elif code == 'b':
                x_axis = 'batche'
            elif code == 's':
                x_axis = 'second'
            elif code.startswith('<'):
                assert code.endswith('>')
                final_codes.add(code)
            elif code.find('-') != -1:
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
        # end for code in codes
    else:
        final_codes ,= set(codebook.keys())

    fig = plt.figure()
    #Make 2 subplots so the legend gets a plot to itself and won't cover up the plot
    ax = plt.subplot(1,2,1)

    # Grow current axis' width by 30%
    box = ax.get_position()

    try:
        x0 = box.x0
        y0 = box.y0
        width = box.width
        height = box.height
    except:
        x0, width, y0, height = box


    ax.set_position([x0, y0, width * 1.3, height])

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
        elif x_axis == 'batche':
            x = np.asarray(channel.batch_record)
        elif x_axis == 'epoch':
            try:
                x = np.asarray(channel.epoch_record)
            except AttributeError:
                # older saved monitors won't have epoch_record
                x = np.arange(len(channel.batch_record))
        elif x_axis == 'second':
            x = np.asarray(channel.time_record)
        else:
            assert False


        plt.plot( x,
                  y,
                  label = channel_name)


    plt.legend(bbox_to_anchor=(1.05, 1),  loc=2, borderaxespad=0.)


    plt.show()

    if not prompt:
        break

