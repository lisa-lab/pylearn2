#!/usr/bin/env python
"""
usage:

plot_monitor.py model_1.pkl model_2.pkl ... model_n.pkl

Loads any number of .pkl files produced by train.py. Extracts
all of their monitoring channels and prompts the user to select
a subset of them to be plotted.

"""
from __future__ import print_function

__authors__ = "Ian Goodfellow, Harm Aarts"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import gc
import numpy as np
import sys

from theano.compat.six.moves import input, xrange
from pylearn2.utils import serial
from theano.printing import _TagGenerator
from pylearn2.utils.string_utils import number_aware_alphabetical_key
from pylearn2.utils import contains_nan, contains_inf
import argparse

channels = {}

def unique_substring(s, other, min_size=1):
    """
    .. todo::

        WRITEME
    """
    size = min(len(s), min_size)
    while size <= len(s):
        for pos in xrange(0,len(s)-size+1):
            rval = s[pos:pos+size]
            fail = False
            for o in other:
                if o.find(rval) != -1:
                    fail = True
                    break
            if not fail:
                return rval
        size += 1
    # no unique substring
    return s

def unique_substrings(l, min_size=1):
    """
    .. todo::

        WRITEME
    """
    return [unique_substring(s, [x for x in l if x is not s], min_size)
            for s in l]

def main():
    """
    .. todo::

        WRITEME
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("model_paths", nargs='+')
    parser.add_argument("--yrange", help='The y-range to be used for plotting, e.g.  0:1')
    
    options = parser.parse_args()
    model_paths = options.model_paths

    if options.out is not None:
      import matplotlib
      matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print('generating names...')
    model_names = [model_path.replace('.pkl', '!') for model_path in
            model_paths]
    model_names = unique_substrings(model_names, min_size=10)
    model_names = [model_name.replace('!','') for model_name in
            model_names]
    print('...done')

    for i, arg in enumerate(model_paths):
        try:
            model = serial.load(arg)
        except Exception:
            if arg.endswith('.yaml'):
                print(sys.stderr, arg + " is a yaml config file," + 
                      "you need to load a trained model.", file=sys.stderr)
                quit(-1)
            raise
        this_model_channels = model.monitor.channels

        if len(sys.argv) > 2:
            postfix = ":" + model_names[i]
        else:
            postfix = ""

        for channel in this_model_channels:
            channels[channel+postfix] = this_model_channels[channel]
        del model
        gc.collect()


    while True:
        # Make a list of short codes for each channel so user can specify them
        # easily
        tag_generator = _TagGenerator()
        codebook = {}
        sorted_codes = []
        for channel_name in sorted(channels,
                key = number_aware_alphabetical_key):
            code = tag_generator.get_tag()
            codebook[code] = channel_name
            codebook['<'+channel_name+'>'] = channel_name
            sorted_codes.append(code)

        x_axis = 'example'
        print('set x_axis to example')

        if len(channels.values()) == 0:
            print("there are no channels to plot")
            break

        # If there is more than one channel in the monitor ask which ones to
        # plot
        prompt = len(channels.values()) > 1

        if prompt:

            # Display the codebook
            for code in sorted_codes:
                print(code + '. ' + codebook[code])

            print()

            print("Put e, b, s or h in the list somewhere to plot " + 
                    "epochs, batches, seconds, or hours, respectively.")
            response = input('Enter a list of channels to plot ' + \
                    '(example: A, C,F-G, h, <test_err>) or q to quit' + \
                    ' or o for options: ')

            if response == 'o':
                print('1: smooth all channels')
                print('any other response: do nothing, go back to plotting')
                response = input('Enter your choice: ')
                if response == '1':
                    for channel in channels.values():
                        k = 5
                        new_val_record = []
                        for i in xrange(len(channel.val_record)):
                            new_val = 0.
                            count = 0.
                            for j in xrange(max(0, i-k), i+1):
                                new_val += channel.val_record[j]
                                count += 1.
                            new_val_record.append(new_val / count)
                        channel.val_record = new_val_record
                continue

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
                elif code == 'h':
                    x_axis = 'hour'
                elif code.startswith('<'):
                    assert code.endswith('>')
                    final_codes.add(code)
                elif code.find('-') != -1:
                    #The current list element is a range of codes

                    rng = code.split('-')

                    if len(rng) != 2:
                        print("Input not understood: "+code)
                        quit(-1)

                    found = False
                    for i in xrange(len(sorted_codes)):
                        if sorted_codes[i] == rng[0]:
                            found = True
                            break

                    if not found:
                        print("Invalid code: "+rng[0])
                        quit(-1)

                    found = False
                    for j in xrange(i,len(sorted_codes)):
                        if sorted_codes[j] == rng[1]:
                            found = True
                            break

                    if not found:
                        print("Invalid code: "+rng[1])
                        quit(-1)

                    final_codes = final_codes.union(set(sorted_codes[i:j+1]))
                else:
                    #The current list element is just a single code
                    final_codes = final_codes.union(set([code]))
            # end for code in codes
        else:
            final_codes ,= set(codebook.keys())

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        styles = list(colors)
        styles += [color+'--' for color in colors]
        styles += [color+':' for color in colors]

        fig = plt.figure()
        ax = plt.subplot(1,1,1)

        # plot the requested channels
        for idx, code in enumerate(sorted(final_codes)):

            channel_name= codebook[code]
            channel = channels[channel_name]

            y = np.asarray(channel.val_record)

            if contains_nan(y):
                print(channel_name + ' contains NaNs')

            if contains_inf(y):
                print(channel_name + 'contains infinite values')

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
            elif x_axis == 'hour':
                x = np.asarray(channel.time_record) / 3600.
            else:
                assert False


            ax.plot( x,
                      y,
                      styles[idx % len(styles)],
                      marker = '.', # add point margers to lines
                      label = channel_name)

        plt.xlabel('# '+x_axis+'s')
        ax.ticklabel_format( scilimits = (-3,3), axis = 'both')

        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc = 'upper left',
               bbox_to_anchor = (1.05, 1.02))

        # Get the axis positions and the height and width of the legend

        plt.draw()       
        ax_pos = ax.get_position()
        pad_width = ax_pos.x0 * fig.get_size_inches()[0]
        pad_height = ax_pos.y0 * fig.get_size_inches()[1]
        dpi = fig.get_dpi()
        lgd_width = ax.get_legend().get_frame().get_width() / dpi 
        lgd_height = ax.get_legend().get_frame().get_height() / dpi 

        # Adjust the bounding box to encompass both legend and axis.  Axis should be 3x3 inches.
        # I had trouble getting everything to align vertically.

        ax_width = 3
        ax_height = 3
        total_width = 2*pad_width + ax_width + lgd_width
        total_height = 2*pad_height + np.maximum(ax_height, lgd_height)

        fig.set_size_inches(total_width, total_height)
        ax.set_position([pad_width/total_width, 1-6*pad_height/total_height, ax_width/total_width, ax_height/total_height])

        if(options.yrange is not None):
            ymin, ymax = map(float, options.yrange.split(':'))
            plt.ylim(ymin, ymax)
        
        if options.out is None:
          plt.show()
        else:
          plt.savefig(options.out)

        if not prompt:
            break

if __name__ == "__main__":
    main()
