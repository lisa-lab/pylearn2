"""
Unit tests for dropout paper
"""

import os
from pylearn2.scripts.tests.yaml_testing import limited_epoch_train
from pylearn2.testing.skip import skip_if_no_data

yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              '..'))
save_path = os.path.dirname(os.path.realpath(__file__))


def test_mnist_valid():
    """
    Tests mnist_valid.yaml by running it for only one epoch
    """

    skip_if_no_data()
    limited_epoch_train(os.path.join(yaml_file_path, 'mnist_valid.yaml'))
    try:
        os.remove(os.path.join(save_path, 'mnist_valid.pkl'))
        os.remove(os.path.join(save_path, 'mnist_valid_best.pkl'))
    except:
        pass


def test_mnist():
    """
    Tests mnist.yaml by running it for only one epoch
    """

    skip_if_no_data()
    limited_epoch_train(os.path.join(yaml_file_path, 'mnist.yaml'))
    try:
        os.remove(os.path.join(save_path, 'mnist.pkl'))
        os.remove(os.path.join(save_path, 'mnist_best.pkl'))
    except:
        pass
