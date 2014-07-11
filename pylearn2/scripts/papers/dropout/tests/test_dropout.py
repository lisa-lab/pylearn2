"""
Unit tests for dropout paper
"""

import os
from pylearn2.scripts.tests.yaml_testing import limited_epoch_train
from pylearn2.testing.skip import skip_if_no_data
from theano import config

yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              '..'))
save_path = os.path.dirname(os.path.realpath(__file__))


def test_mnist_valid():
    """
    Tests mnist_valid.yaml by running it for only one epoch
    """
    if config.mode == "DEBUG_MODE":
        yaml_file = 'mnist_valid_fast'
    else:
        yaml_file = 'mnist_valid'
    skip_if_no_data()
    limited_epoch_train(os.path.join(yaml_file_path, '%s.yaml' % yaml_file))
    try:
        os.remove(os.path.join(save_path, '%s.pkl' % yaml_file))
        os.remove(os.path.join(save_path, '%s_best.pkl' % yaml_file))
    except:
        pass


def test_mnist():
    """
    Tests mnist.yaml by running it for only one epoch
    """
    if config.mode == "DEBUG_MODE":
        yaml_file = 'mnist_fast'
    else:
        yaml_file = 'mnist'
    skip_if_no_data()
    limited_epoch_train(os.path.join(yaml_file_path, '%s.yaml' % yaml_file))
    try:
        os.remove(os.path.join(save_path, '%s.pkl' % yaml_file))
        os.remove(os.path.join(save_path, '%s_best.pkl' % yaml_file))
    except:
        pass
