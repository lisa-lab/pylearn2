import os

from nose.plugins.skip import SkipTest
from theano import config

import pylearn2
from pylearn2.testing import no_debug_mode
from pylearn2.utils.serial import load_train_file


@no_debug_mode
def test_train_example():
    """ tests that the grbm_smd example script runs correctly """

    assert config.mode != "DEBUG_MODE"
    path = pylearn2.__path__[0]
    train_example_path = os.path.join(path, 'scripts', 'tutorials', 'grbm_smd')
    if not os.path.isfile(os.path.join(train_example_path,
                                       'cifar10_preprocessed_train.pkl')):
        raise SkipTest
    cwd = os.getcwd()
    try:
        os.chdir(train_example_path)
        train_yaml_path = os.path.join(train_example_path,
                                       'cifar_grbm_smd.yaml')
        train_object = load_train_file(train_yaml_path)

        # Make the termination criterion really lax so that it is quick
        train_object.algorithm.termination_criterion.prop_decrease = 0.5
        train_object.algorithm.termination_criterion.N = 1

        train_object.main_loop()
    finally:
        os.chdir(cwd)

if __name__ == '__main__':
    test_train_example()
