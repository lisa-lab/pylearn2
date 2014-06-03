#!/usr/bin/env python
"""
Print (average) channel values for a collection of models, such as that
serialized by TrainCV. Based on print_monitor.py

usage: print_monitor_cv.py model.pkl [-a]
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import argparse
import numpy as np

from pylearn2.utils import serial

parser = argparse.ArgumentParser()
parser.add_argument('models', nargs='+',
                    help='Model or models to analyze.')
parser.add_argument('-a', '--all', action='store_true',
                    help='Print values for all models instead of averages.')
args = parser.parse_args()


def main():
    """
    Print (average) final channel values for a collection of models.
    """
    epochs = []
    time = []
    values = {}
    for filename in args.models:
        models = serial.load(filename)
        for model in list(models):
            monitor = model.monitor
            channels = monitor.channels
            epochs.append(monitor._epochs_seen)
            time.append(max(channels[key].time_record[-1] for key in channels))
            for key in sorted(channels.keys()):
                if key not in values:
                    values[key] = []
                values[key].append(channels[key].val_record[-1])
    n_models = len(epochs)
    print 'number of models: {}'.format(n_models)
    if n_models > 1:
        if args.all:
            print '\nepochs seen:\n{}\n{} +/- {}'.format(np.asarray(epochs),
                                                         np.mean(epochs),
                                                         np.std(epochs))
            print '\ntraining time:\n{}\n{} +/- {}'.format(np.asarray(time),
                                                           np.mean(time),
                                                           np.std(time))
        else:
            print 'epochs seen: {} +/- {}'.format(np.mean(epochs),
                                                  np.std(epochs))
            print 'training time: {} +/- {}'.format(np.mean(time),
                                                    np.std(time))
        for key in sorted(values.keys()):
            if args.all:
                print '\n{}:\n{}\n{} +/- {}'.format(key,
                                                    np.asarray(values[key]),
                                                    np.mean(values[key]),
                                                    np.std(values[key]))
            else:
                print '{}: {} +/- {}'.format(key, np.mean(values[key]),
                                             np.std(values[key]))
    else:
        print 'epochs seen: {}'.format(epochs[0])
        print 'training time: {}'.format(time[0])
        for key in sorted(values.keys()):
            print '{}: {}'.format(key, values[key][0])

if __name__ == '__main__':
    main()
