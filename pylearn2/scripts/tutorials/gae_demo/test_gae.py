"""
This module tests gae_demo/gae_random.yaml
"""

import os
import logging
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse
from pylearn2.scripts.tutorials.gae_demo.make_random_dataset import generate


@no_debug_mode
def train_yaml(yaml_file):
    """
    Executes the the main_loop()

    Parameters
    ----------
    yaml_file: string
        Configuration yaml
    """
    train = yaml_parse.load(yaml_file)
    train.main_loop()


def train(yaml_file_path, save_path):
    """
    Loads the parameters used for training

    Parameters
    ----------
    yaml_file_path: string
        Path to the configuration file
    save_path: string
        Saving path
    """
    yaml = open("{0}/gae_random.yaml".format(yaml_file_path), 'r').read()
    data = os.path.join(save_path, 'train_preprocessed.pkl')
    params = {'save_path': save_path,
              'region': 13,
              'nvisX': 169,
              'nvisY': 169,
              'max_epochs': 3,
              'batch_size': 100,
              'train_data': data,
              'nfac': 196,
              'nmap': 50,
              'lr': 0.01}

    yaml = yaml % (params)
    train_yaml(yaml)


def test_gae():
    """
    The function generates a dataset and uses it to train the model.
    """
    generate('shifts')
    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  "../gae_demo"))
    save_path = os.path.dirname(os.path.realpath(__file__))

    train(yaml_file_path, save_path)

    try:
        os.remove("{0}/train_preprocessed.pkl".format(save_path))
        os.remove("{0}/gae_196_50.pkl".format(save_path))
        os.remove("{0}/gae_196_50_best.pkl".format(save_path))
    except OSError:
        logging.warning("Files not found")

if __name__ == '__main__':
    test_gae()
