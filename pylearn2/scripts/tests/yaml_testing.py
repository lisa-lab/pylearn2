from nose.plugins.skip import SkipTest

from pylearn2.config import yaml_parse
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.exc import NoDataPathError

def yaml_file_execution(file_path):
    try:
        train = yaml_parse.load_path(file_path)
        train.algorithm.termination_criterion = EpochCounter(max_epochs=2)
        train.main_loop()
    except NoDataPathError:
        raise SkipTest("PYLEARN2_DATA_PATH environment variable not defined")
