from nose.plugins.skip import SkipTest

from pylearn2.utils.serial import load_train_file
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.exc import NoDataPathError

def limited_epoch_train(file_path, max_epochs = 1):
    try:
        train = load_train_file(file_path)
        train.algorithm.termination_criterion = EpochCounter(max_epochs = max_epochs)
        train.main_loop()
    except NoDataPathError:
        raise SkipTest("PYLEARN2_DATA_PATH environment variable not defined")
