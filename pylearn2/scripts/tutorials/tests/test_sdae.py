"""
This module tests stacked_autoencoders.ipynb
"""

import os
from nose.plugins.skip import SkipTest

from pylearn2.datasets.exc import NoDataPathError
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse

YAML_FILES_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ),
                                                                    '..'))
SAVE_PATH = os.path.dirname(os.path.realpath(__file__))

@no_debug_mode
def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)
    train.main_loop()

def train_layer1():

    yaml = open("{}/dae_l1.yaml".format(YAML_FILES_PATH), 'r').read()
    hyper_params = {'train_stop' : 50,
                    'batch_size' : 50,
                    'monitoring_batches' : 1,
                    'nhid' : 10,
                    'max_epochs' : 1,
                    'save_path' : SAVE_PATH}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)

def train_layer2():

    yaml = open("{}/dae_l2.yaml".format(YAML_FILES_PATH), 'r').read()
    hyper_params = {'train_stop' : 50,
                    'batch_size' : 50,
                    'monitoring_batches' : 1,
                    'nvis' : 10,
                    'nhid' : 10,
                    'max_epochs' : 1,
                    'save_path' : SAVE_PATH}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)

def train_mlp():

    yaml = open("{}/dae_mlp.yaml".format(YAML_FILES_PATH), 'r').read()
    hyper_params = {'train_stop' : 50,
                    'valid_stop' : 50050,
                    'batch_size' : 50,
                    'max_epochs' : 1,
                    'save_path' : SAVE_PATH}
    yaml = yaml % (hyper_params)
    train_yaml(yaml)

def test_sda():

    try:
        train_layer1()
    except NoDataPathError:
        raise SkipTest("PYLEARN2_DATA_PATH environment variable not defined")

    train_layer2()
    train_mlp()
    try:
        os.remove("{}/dae_l1.pkl".format(SAVE_PATH))
        os.remove("{}/dae_l2.pkl".format(SAVE_PATH))
    except:
        pass

if __name__ == '__main__':
    test_sda()
