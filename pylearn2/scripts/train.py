#!/bin/env python
"""
General training script
usage:
train.py <path to yaml file>

see train_example.yaml for an example


"""
# Standard library imports
import sys
import time
import os

# Local imports
import pylearn2.config.yaml_parse
from pylearn2.utils import serial

class Train(object):
    """
        A class representing the main loop of the training script.
        Trains the specified model using the specified algorithm on the specified dataset.
        After each call to the training algorithm, the model is saved to save_path and
        each of the registered callbacks are called.
    """
    def __init__(self,
                dataset,
                model,
                algorithm = None,
                save_path = None,
                callbacks = []):
        """

        parameters:
            dataset: pylearn2.datasets.dataset.Dataset
            model: pylearn2.models.model.Model
            algorithm: pylearn2.training_algorithms.training_algorithm.TrainingAlgorithm
                optionally, pass None to use the model's train method, if it has one
            save_path: string, the location to save to
            callbacks: list of pylearn2.training_callbacks.training_callback.TrainingCallback
        """
        self.dataset, self.model, self.algorithm, self.save_path  = dataset, model, algorithm, save_path
        self.model.dataset_yaml_src = self.dataset.yaml_src
        self.callbacks = callbacks
    #

    def main_loop(self):
        """
        Runs one iteration of the training algorithm (what an "iteration" means is up to
        the training algorithm to define, usually it's something like an "epoch" )

        Saves the model

        Runs the callbacks
        """

        if self.algorithm is None:
            while self.model.train(dataset = self.dataset):
                self.save()
            #
            self.save()
        else:
            self.algorithm.setup(model = self.model, dataset = self.dataset)

            t1 = time.time()
            while self.algorithm.train(dataset = self.dataset):
                t2 = time.time()
                diff_time = t2-t1
                print 'Time this epoch: '+str(diff_time)
                self.save()
                t1 = time.time()

                for callback in self.callbacks:
                    callback(self.model, self.dataset, self.algorithm)
                #
            #
            self.save()
        #
    #

    def save(self):
        """ saves the model """

        #TODO-- save state of dataset and training algorithm so training can be resumed after a crash
        if self.save_path is not None:
            print 'saving to ',self.save_path,'...'
            t1 = time.time()
            serial.save(self.save_path, self.model)
            t2 = time.time()
            print '...done. saving took ',(t2-t1),' seconds'
        #
    #

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise Exception("train.py takes exactly one argument, the path to a yaml file (see train_example.yaml for an example)")

    config_file_path = sys.argv[1]

    suffix_to_strip = '.yaml'
    if config_file_path.endswith(suffix_to_strip):
        config_file_name = config_file_path[0:-len(suffix_to_strip)]
    else:
        config_file_name = config_file_path

    #publish the PYLEARN2_TRAIN_FILE_NAME environment variable
    varname = "PYLEARN2_TRAIN_FILE_NAME"
    #this makes it available to other sections of code in this same script
    os.environ[varname] =  config_file_name
    #this make it available to any subprocesses we launch
    os.putenv(varname, config_file_name)

    train_obj = pylearn2.config.yaml_parse.load_path(config_file_path)

    train_obj.main_loop()

