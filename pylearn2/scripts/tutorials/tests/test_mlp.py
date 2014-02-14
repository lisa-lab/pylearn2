"""
Test for multilayer_perceptron.ipynb
The number of epochs has been limited.
"""

import os
from nose.plugins.skip import SkipTest

from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.exc import NoDataPathError
from pylearn2.config import yaml_parse
import pylearn2


def test_multilayer_perceptron():
    try:
        with open(os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials', 'mlp_tutorial_part_2.yaml'), 'r') as f:
            train = f.read()
        f.close()
        print train
        train = yaml_parse.load(train)
        train.algorithm.termination_criterion = EpochCounter(max_epochs=2)
        train.main_loop()
        with open(os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials', 'mlp_tutorial_part_3.yaml'), 'r') as f:
            train_2 = f.read()
        f.close()
        print train_2
        train_2 = yaml_parse.load(train_2)
        train_2.algorithm.termination_criterion = EpochCounter(max_epochs=2)
        train_2.main_loop()
        with open(os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials', 'mlp_tutorial_part_4.yaml'), 'r') as f:
            train_3 = f.read()
        f.close()
        print train_3
        train_3 = yaml_parse.load(train_3)
        train_3.algorithm.termination_criterion = EpochCounter(max_epochs=2)
        train_3.main_loop()
    except NoDataPathError:
        raise SkipTest("PYLEARN2_DATA_PATH environment variable not defined")

if __name__ == '__main__':
    test_multilayer_perceptron()
