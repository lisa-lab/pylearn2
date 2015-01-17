'''
Created on Jan 17, 2015

@author: Minh Ngoc Le
'''
import os
from pylearn2.config import yaml_parse


def test_yaml():
    """Test loading and running a complex model with ProjectionLayer."""
    test_dir = os.path.dirname(__file__)
    with open(os.path.join(test_dir, 'composite.yaml')) as f:
        train = yaml_parse.load(f.read())
        train.main_loop()
