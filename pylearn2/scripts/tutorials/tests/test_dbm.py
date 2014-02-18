"""
This module tests dbm_demo/rbm.yaml
"""

import os

from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse


@no_debug_mode
def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)
    train.main_loop()


def train(yaml_file_path, save_path):

    yaml = open("{0}/rbm.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'detector_layer_dim': 5,
                    'monitoring_batches': 2,
                    'train_stop': 500,
                    'max_epochs': 7,
                    'save_path': save_path}

    yaml = yaml % (hyper_params)
    train_yaml(yaml)


def test_dbm():

    skip.skip_if_no_data()

    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  '../dbm_demo'))
    save_path = os.path.dirname(os.path.realpath(__file__))

    train(yaml_file_path, save_path)

    try:
        os.remove("{}/dbm.pkl".format(save_path))
    except:
        pass

if __name__ == '__main__':
    test_dbm()
