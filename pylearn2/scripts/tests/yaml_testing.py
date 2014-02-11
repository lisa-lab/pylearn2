from pylearn2.config import yaml_parse
from pylearn2.termination_criteria import EpochCounter

def test_yaml_file(file_path):
    train = yaml_parse.load_path(file_path)
    train.algorithm.termination_criterion = EpochCounter(max_epochs=3)
    train.main_loop()
