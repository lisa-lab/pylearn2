"""
Unit tests for dropout paper
"""

import os
from pylearn2.scripts.tests.yaml_testing import limited_epoch_train
from pylearn2.testing.skip import skip_if_no_data
from theano import config
from theano.compile import get_default_mode

yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              '..'))
save_path = os.path.dirname(os.path.realpath(__file__))


def test_mnist_valid():
    """
    Tests mnist_valid.yaml by running it for only one epoch
    """
    skip_if_no_data()
    mode = get_default_mode()
    if hasattr(mode, 'check_py_code'):
        old_value = mode.check_py_code
        mode.check_py_code = False
    try:
        if config.mode == "DEBUG_MODE":
            yaml_file = 'mnist_valid_fast'
        else:
            yaml_file = 'mnist_valid'
        limited_epoch_train(os.path.join(yaml_file_path, '%s.yaml'
                                         % yaml_file))
        try:
            os.remove(os.path.join(save_path, '%s.pkl' % yaml_file))
            os.remove(os.path.join(save_path, '%s_best.pkl' % yaml_file))
        except Exception:
            pass
    finally:
        if hasattr(mode, 'check_py_code'):
            mode.check_py_code = old_value


def test_mnist():
    """
    Tests mnist.yaml by running it for only one epoch
    """
    skip_if_no_data()
    mode = get_default_mode()
    if hasattr(mode, 'check_py_code'):
        old_value = mode.check_py_code
        mode.check_py_code = False
    try:
        if config.mode == "DEBUG_MODE":
            yaml_file = 'mnist_fast'
        else:
            yaml_file = 'mnist'
        limited_epoch_train(os.path.join(yaml_file_path, '%s.yaml'
                                         % yaml_file))
        try:
            os.remove(os.path.join(save_path, '%s.pkl' % yaml_file))
            os.remove(os.path.join(save_path, '%s_best.pkl' % yaml_file))
        except Exception:
            pass
    finally:
        if hasattr(mode, 'check_py_code'):
            mode.check_py_code = old_value
