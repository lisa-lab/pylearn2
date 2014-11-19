"""
Test for multilayer_perceptron.ipynb
"""

from __future__ import print_function

import os

import pylearn2
from pylearn2.termination_criteria import EpochCounter
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.config import yaml_parse


def test_nested():
    skip_if_no_data()
    with open(os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials',
              'mlp_nested.yaml'), 'r') as f:
        train_3 = f.read()
    hyper_params = {'train_stop': 50,
                    'valid_stop': 50050,
                    'dim_h0': 5,
                    'dim_h1': 20,
                    'dim_h2': 30,
                    'dim_h3': 40,
                    'sparse_init_h1': 2,
                    'max_epochs': 1}
    train_3 = train_3 % (hyper_params)
    print(train_3)
    train_3 = yaml_parse.load(train_3)
    train_3.main_loop()

if __name__ == '__main__':
    test_nested()
