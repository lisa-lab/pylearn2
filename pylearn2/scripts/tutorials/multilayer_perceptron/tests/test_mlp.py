"""
Test for multilayer_perceptron.ipynb
"""

import os

import pylearn2
from pylearn2.termination_criteria import EpochCounter
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.config import yaml_parse


YAML_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              '..'))
SAVE_PATH = os.path.dirname(os.path.realpath(__file__))


def cleaunup(file_name):
    try:
        os.remove(os.path.join(SAVE_PATH, file_name))
    except OSError:
        pass


def test_part_2():
    skip_if_no_data()
    with open(os.path.join(YAML_FILE_PATH,
              'mlp_tutorial_part_2.yaml'), 'r') as f:
        train = f.read()
    hyper_params = {'train_stop': 50,
                    'valid_stop': 50050,
                    'dim_h0': 5,
                    'max_epochs': 1,
                    'save_path': SAVE_PATH}
    train = train % (hyper_params)
    train = yaml_parse.load(train)
    train.main_loop()
    cleaunup("mlp_best.pkl")


def test_part_3():
    skip_if_no_data()
    with open(os.path.join(YAML_FILE_PATH,
              'mlp_tutorial_part_3.yaml'), 'r') as f:
        train_2 = f.read()
    hyper_params = {'train_stop': 50,
                    'valid_stop': 50050,
                    'dim_h0': 5,
                    'dim_h1': 10,
                    'sparse_init_h1': 2,
                    'max_epochs': 1,
                    'save_path': SAVE_PATH}
    train_2 = train_2 % (hyper_params)
    train_2 = yaml_parse.load(train_2)
    train_2.main_loop()
    cleaunup("mlp_2_best.pkl")


def test_part_4():
    skip_if_no_data()
    with open(os.path.join(YAML_FILE_PATH,
              'mlp_tutorial_part_4.yaml'), 'r') as f:
        train_3 = f.read()
    hyper_params = {'train_stop': 50,
                    'valid_stop': 50050,
                    'dim_h0': 5,
                    'dim_h1': 10,
                    'sparse_init_h1': 2,
                    'max_epochs': 1,
                    'save_path': SAVE_PATH}
    train_3 = train_3 % (hyper_params)
    train_3 = yaml_parse.load(train_3)
    train_3.main_loop()
    cleaunup("mlp_3_best.pkl")
