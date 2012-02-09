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
import os

# Local imports
import pylearn2.config.yaml_parse
from pylearn2.utils import serial

class MultiTrain(object):
    def __init__(self, instances):
        """
        Construct a MultiTrain instance.

        Parameters
        ----------
        instances : iterable
            A collection of Train instances that implement the
            `main_loop()` interface.
        """
        self.instances = instances

    def main_loop(self):
        # TODO: Add fine-grained checks to load previously existing
        # results if instance.save_path exists and is the result of
        # a terminated training procedure.
        for index, instance in enumerate(self.instances):
            os.environ['PYLEARN2_TRAINING_PHASE'] = str(index)
            os.putenv('PYLEARN2_TRAINING_PHASE', str(index))
            print "Entering training phase %d" % index
            instance.main_loop()


class Train(object):
    """
    A class representing the main loop of the training script.  Trains the
    specified model using the specified algorithm on the specified dataset.
    After each call to the training algorithm, the model is saved to save_path
    and each of the registered callbacks are called.
    """
    def __init__(self, dataset, model, algorithm=None, save_path=None,
                 callbacks=None):
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
            Path to save the (pickled) model.
        callbacks : iterable, optional
            A collection of callbacks that are called, one at a time,
            after each epoch.
        """
        self.dataset = dataset
        self.model = model
        self.algorithm = algorithm
        if save_path is not None:
            self.save_path = save_path
        else:
            phase_variable = 'PYLEARN2_TRAINING_PHASE'
            if phase_variable in os.environ:
                phase = 'phase%d' % os.environ[phase_variable]
                tokens = [os.environ['PYLEARN2_TRAIN_FILE_NAME'],
                          phase, '.pkl']
            else:
                tokens = os.environ['PYLEARN2_TRAIN_FILE_NAME'], '.pkl'
            self.save_path = '.'.join(tokens)
        self.callbacks = callbacks if callbacks is not None else []
        self.model.dataset_yaml_src = self.dataset.yaml_src

    def main_loop(self):
        """
        Repeatedly runs an epoch of the training algorithm, runs any
        epoch-level callbacks, and saves the model.
        """
        if self.algorithm is None:
            while self.model.train(dataset=self.dataset):
                self.save()
            self.save()
        else:
            self.algorithm.setup(model=self.model, dataset=self.dataset)
            epoch_start = datetime.datetime.now()
            while self.algorithm.train(dataset=self.dataset):
                epoch_end = datetime.datetime.now()
                print 'Time this epoch:', str(epoch_end - epoch_start)
                self.save()
                epoch_start = datetime.datetime.now()
                for callback in self.callbacks:
                    callback(self.model, self.dataset, self.algorithm)
            self.save()

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
    config_file_path = args.config.name
    suffix_to_strip = '.yaml'
    if config_file_path.endswith(suffix_to_strip):
        config_file_name = config_file_path[0:-len(suffix_to_strip)]
    else:
        config_file_name = config_file_path
    # publish the PYLEARN2_TRAIN_FILE_NAME environment variable
    varname = "PYLEARN2_TRAIN_FILE_NAME"
    # this makes it available to other sections of code in this same script
    os.environ[varname] = config_file_name
    # this make it available to any subprocesses we launch
    os.putenv(varname, config_file_name)
    train_obj = pylearn2.config.yaml_parse.load(args.config)
    train_obj.main_loop()
