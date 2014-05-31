#!/usr/bin/env python

"""
Print average channel values for a collection of models, such as that
serialized by TrainCV. Based on print_monitor.py

usage: print_monitor_average.py model.pkl
"""
__author__ = "Steven Kearnes"

import sys
from pylearn2.utils import serial
import numpy as np


def main():
    """Print average channel final values for a collection of models."""
    epochs = []
    time = []
    values = {}
    for filename in sys.argv[1:]:
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
    print 'number of models: {}'.format(len(epochs))
    if len(epochs) > 1:
        print 'epochs seen: {} +/- {}'.format(np.mean(epochs), np.std(epochs))
        print 'training time: {} +/- {}'.format(np.mean(time), np.std(time))
        for key in sorted(values.keys()):
            print '{}: {} +/- {}'.format(key, np.mean(values[key]),
                                         np.std(values[key]))
    else:
        print 'epochs seen: {}'.format(epochs[0])
        print 'training time: {}'.format(time[0])
        for key in sorted(values.keys()):
            print '{}: {}'.format(key, values[key][0])

if __name__ == '__main__':
    main()
