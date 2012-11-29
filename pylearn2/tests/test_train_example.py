import os

from nose.plugins.skip import SkipTest
import theano

import pylearn2
from pylearn2.utils.serial import load_train_file


def test_train_example():
    """ tests that the train example script runs correctly """
    if 'TRAVIS' in os.environ and os.environ['TRAVIS'] == '1':
        raise SkipTest()
    path = pylearn2.__path__[0]
    train_example_path = path + '/scripts/train_example'
    cwd = os.getcwd()

    #This test is too slow in FAST_COMPILE.
    orig_mode = theano.config.mode
    if orig_mode in ["FAST_COMPILE", "DebugMode", "DEBUG_MODE"]:
        theano.config.mode = "FAST_RUN"
    try:
        os.chdir(train_example_path)
        train_yaml_path = 'cifar_grbm_smd.yaml'
        train_object = load_train_file(train_yaml_path)

        # make the termination criterion really lax so the test won't
        # run for long
        train_object.algorithm.termination_criterion.prop_decrease = 0.5
        train_object.algorithm.termination_criterion.N = 1

        train_object.main_loop()
    finally:
        os.chdir(cwd)
        theano.config.mode = orig_mode

if __name__ == '__main__':
    test_train_example()
