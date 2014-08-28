"""
This module tests gae_demo/gae_random.yaml
"""

import os

from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse
from pylearn2.scripts.tutorials.gae_demo.make_random_dataset import generate


@no_debug_mode
def train_yaml(yaml_file):

    train = yaml_parse.load(yaml_file)
    train.main_loop()


def train(yaml_file_path, save_path, opc):

    yaml = open("{0}/gae_random.yaml".format(yaml_file_path), 'r').read()
    params = {'max_epochs': 3,
              'batch_size': 100,
              'recepF': 13,
              'train_data': 'train_preprocessed.pkl',
              'nvisX': 169,
              'nvisY': 169,
              'nfac': 169,
              'nmap': 50,
              'lr': 0.01}

    yaml = yaml % (params)
    train_yaml(yaml)


def test_gae(opc):
    generate(opc)
    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  '../gae_demo'))
    save_path = os.path.dirname(os.path.realpath(__file__))

    train(yaml_file_path, save_path, opc)

    try:
        os.remove("{0}/gae_169_50.pkl".format(save_path))
        os.remove("{0}/gae_169_50_best.pkl".format(save_path))
        os.remove("{0}/train_design.npy".format(save_path))
        os.remove("{0}/train_preprocessed.pkl".format(save_path))
    except OSError:
        pass

if __name__ == '__main__':
    test_gae('shifts')
    test_gae('rotations')
