"""
Test that a smaller version of convolutional_network.ipynb works.

The differences (needed for speed) are:
    * output_channels: 4 instead of 64
    * train.stop: 500 instead of 50000
    * valid.stop: 50100 instead of 60000
    * test.start: 0 instead of non-specified
    * test.stop: 100 instead of non-specified
    * termination_criterion.max_epochs: 1 instead of 500

This should make the test run in about one minute.
"""

import os
from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse


def test_convolutional_network():

    skip.skip_if_no_data()
    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  '..'))
    save_path = os.path.dirname(os.path.realpath(__file__))

    yaml = open("{0}/conv.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'train_stop': 50,
                    'valid_stop': 50050,
                    'test_stop': 50,
                    'batch_size': 50,
                    'output_channels_h2': 4,
                    'output_channels_h3': 4,
                    'max_epochs': 1,
                    'save_path': save_path}
    yaml = yaml % (hyper_params)
    train = yaml_parse.load(yaml)
    train.main_loop()

    try:
        os.remove("{}/convolutional_network_best.pkl".format(save_path))
    except OSError:
        pass

