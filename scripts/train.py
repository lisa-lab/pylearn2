#!/bin/env python
"""General training script"""
# Standard library imports
import sys
import time

# Local imports
import framework.config.yaml_parse
from framework.utils import serial

class Train:
    def __init__(self, dataset, model, algorithm = None, save_path = None):
        self.dataset, self.model, self.algorithm, self.save_path  = dataset, model, algorithm, save_path
        self.model.dataset_yaml_src = self.dataset.yaml_src

    def main_loop(self):
        if self.algorithm is None:
            while self.model.train(dataset = self.dataset):
                self.save()
            #
            self.save()
        else:
            t1 = time.time()
            while self.algorithm.train(model= self.model, dataset = self.dataset):
                t2 = time.time()
                diff_time = t2-t1
                print 'Time this epoch: '+str(diff_time)
                print 'quitting for profiling purposes'
                sys.exit()
                self.save()
                t1 = time.time()
            #
        #
    #

    def save(self):
        #TODO-- save state of dataset and training algorithm so training can be resumed after a crash
        if self.save_path is not None:
            print 'saving to '+self.save_path
            serial.save(self.save_path, self.model)
        #
    #

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise Exception("train.py takes exactly one argument")

    config_file_path = sys.argv[1]

    train_obj = framework.config.yaml_parse.load_path(config_file_path)

    train_obj.main_loop()

