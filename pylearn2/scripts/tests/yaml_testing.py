from pylearn2.config import yaml_parse
from pylearn2.termination_criteria import EpochCounter

def yaml_file_execution(file_path):
    train = yaml_parse.load_path(file_path)
    train.algorithm.termination_criterion = EpochCounter(max_epochs=3)
    train.main_loop()
