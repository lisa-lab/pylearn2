#!/bin/env python
"""
Script implementing the logic for training pylearn2 models.

This is intended to be a "driver" for most training experiments. A user
specifies an object hierarchy in a configuration file using a dictionary-like
syntax and this script takes care of the rest.

For example configuration files that are consumable by this script, see

    pylearn2/scripts/train_example
    pylearn2/scripts/autoencoder_example
"""
# Standard library imports
import argparse
import datetime
import gc
import os
import warnings

# Third-party imports
import numpy as np

# Local imports
import pylearn2.config.yaml_parse
from pylearn2.utils import serial
from pylearn2.utils import environ
from pylearn2.monitor import Monitor


class FeatureDump(object):
    def __init__(self, encoder, dataset, path, batch_size=None, topo=False):
        self.encoder = encoder
        self.dataset = dataset
        self.path = path
        self.batch_size = batch_size
        self.topo = topo

    def main_loop(self):
        if self.batch_size is None:
            if self.topo:
                data = self.dataset.get_topological_view()
            else:
                data = self.dataset.get_design_matrix()
            output = self.encoder.perform(data)
        else:
            myiterator = self.dataset.iterator(mode='sequential',
                                               batch_size=self.batch_size,
                                               topo=self.topo)
            chunks = []
            for data in myiterator:
                chunks.append(self.encoder.perform(data))
            output = np.concatenate(chunks)
        np.save(self.path, output)


class Train(object):
    """
    A class representing the main loop of the training script.  Trains the
    specified model using the specified algorithm on the specified dataset.
    After each call to the training algorithm, the model is saved to save_path
    and each of the registered callbacks are called.
    """
    def __init__(self, dataset, model, algorithm=None, save_path=None,
                 save_freq=0, callbacks=None):
        """
        Construct a Train instance.

        Parameters
        ----------
        dataset : object
            Object that implements the Dataset interface defined in
            `pylearn2.datasets`.
        model : object
            Object that implements the Model interface defined in
            `pylearn2.models`.
        algorithm : object, optional
            Object that implements the TrainingAlgorithm interface
            defined in `pylearn2.training_algorithms`.
        save_path : str, optional
            Path  to save the (pickled) model.
        save_freq : int, optional
            Frequency of saves, in epochs. A frequency of zero disables
            automatic saving altogether. A frequency of 1 saves every
            epoch. A frequency of 2 saves every other epoch, etc. (default=0,
            i.e. never save)
        callbacks : iterable, optional
            A collection of callbacks that are called, one at a time,
            after each epoch.
        """
        self.dataset = dataset
        self.model = model
        self.algorithm = algorithm
        if save_path is not None:
            if save_freq == 0:
                warnings.warn('save_path specified but save_freq is 0 '
                              '(never save). Is this intentional?')
            self.save_path = save_path
        else:
            if save_freq > 0:
                phase_variable = 'PYLEARN2_TRAIN_PHASE'
                if phase_variable in os.environ:
                    phase = 'phase%d' % os.environ[phase_variable]
                    tokens = [os.environ['PYLEARN2_TRAIN_FILE_NAME'],
                              phase, 'pkl']
                else:
                    tokens = os.environ['PYLEARN2_TRAIN_FILE_NAME'], 'pkl'
                self.save_path = '.'.join(tokens)
        self.save_freq = save_freq
        self.epochs = 0
        self.callbacks = callbacks if callbacks is not None else []

        if hasattr(self.dataset,'yaml_src'):
            self.model.dataset_yaml_src = self.dataset.yaml_src
        else:
            warnings.warn("dataset has no yaml src, model won't know what data it was trained on")

    def main_loop(self):
        """
        Repeatedly runs an epoch of the training algorithm, runs any
        epoch-level callbacks, and saves the model.
        """
        if self.algorithm is None:
            self.model.monitor = Monitor.get_monitor(self.model)
            self.run_callbacks_and_monitoring()
            while self.model.train(dataset=self.dataset):
                self.run_callbacks_and_monitoring()
                if self.save_freq > 0 and self.epochs % self.save_freq == 0:
                    self.save()
                self.epochs += 1
            self.run_callbacks_and_monitoring()
            if self.save_freq > 0:
                self.save()
        else:
            self.algorithm.setup(model=self.model, dataset=self.dataset)
            self.run_callbacks_and_monitoring()
            epoch_start = datetime.datetime.now()
            while self.algorithm.train(dataset=self.dataset):
                epoch_end = datetime.datetime.now()
                print 'Time this epoch:', str(epoch_end - epoch_start)
                epoch_start = datetime.datetime.now()
                self.run_callbacks_and_monitoring()
                if self.save_freq > 0 and self.epochs % self.save_freq == 0:
                    self.save()
                self.epochs += 1
            self.run_callbacks_and_monitoring()

            if self.save_freq > 0:
                self.save()

    def run_callbacks_and_monitoring(self):
        self.model.monitor()
        for callback in self.callbacks:
            try:
                callback(self.model, self.dataset, self.algorithm)
            except TypeError, e:
                print 'Failure during callback '+str(callback)
                raise


    def save(self):
        """Saves the model."""
        #TODO-- save state of dataset and training algorithm so training can be
        # resumed after a crash
        if self.save_path is not None:
            print 'saving to', self.save_path, '...'
            save_start = datetime.datetime.now()
            serial.save(self.save_path, self.model)
            save_end = datetime.datetime.now()
            delta = (save_end - save_start)
            print '...done. saving took', str(delta)


def make_argument_parser():
    parser = argparse.ArgumentParser(
        description="Launch an experiment from a YAML configuration file.",
        epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('config', action='store',
                        type=argparse.FileType('r'),
                        choices=None,
                        help='A YAML configuration file specifying the '
                             'training procedure')
    return parser


if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    train_obj = serial.load_train_file(args.config)
    try:
        iter(train_obj)
        iterable = True
    except TypeError as e:
        iterable = False
    if iterable:
        for number, subobj in enumerate(iter(train_obj)):
            # Publish a variable indicating the training phase.
            phase_variable = 'PYLEARN2_TRAIN_PHASE'
            phase_value = 'phase%d' % (number + 1)
            os.environ[phase_variable] = phase_value
            os.putenv(phase_variable, phase_value)

            # Execute this training phase.
            subobj.main_loop()

            # Clean up, in case there's a lot of memory used that's
            # necessary for the next phase.
            del subobj
            gc.collect()
    else:
        train_obj.main_loop()
