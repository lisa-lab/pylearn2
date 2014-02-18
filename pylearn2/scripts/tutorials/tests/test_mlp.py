"""
Test for multilayer_perceptron.ipynb
"""

import os

import pylearn2
from pylearn2.termination_criteria import EpochCounter
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.config import yaml_parse


def test_part_2():
    skip_if_no_data()
    with open(os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials',
              'mlp_tutorial_part_2.yaml'), 'r') as f:
        train = f.read()
    f.close()
    hyper_params = {'train_stop': 50,
                    'valid_stop': 50050,
                    'dim_h0': 5,
                    'max_epochs': 1}
    train = train % (hyper_params)
    print train
    train = yaml_parse.load(train)
    train.main_loop()


def test_part_3():
    skip_if_no_data()
    with open(os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials',
              'mlp_tutorial_part_3.yaml'), 'r') as f:
        train_2 = f.read()
    f.close()
    hyper_params = {'train_stop': 50,
                    'valid_stop': 50050,
                    'dim_h0': 5,
                    'dim_h1': 10,
                    'sparse_init_h1': 2,
                    'max_epochs': 1}
    train_2 = train_2 % (hyper_params)
    print train_2
    train_2 = yaml_parse.load(train_2)
    train_2.main_loop()


def test_part_4():
    skip_if_no_data()
    with open(os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials',
              'mlp_tutorial_part_4.yaml'), 'r') as f:
        train_3 = f.read()
    f.close()
    hyper_params = {'train_stop': 50,
                    'valid_stop': 50050,
                    'dim_h0': 5,
                    'dim_h1': 10,
                    'sparse_init_h1': 2,
                    'max_epochs': 1}
    train_3 = train_3 % (hyper_params)
    print train_3
    train_3 = yaml_parse.load(train_3)
    train_3.main_loop()

if __name__ == '__main__':
    test_part_2()
    test_part_3()
    test_part_4()
