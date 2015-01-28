"""
Tests for MLP
"""
__authors__ = "Minh Ngoc Le"
__copyright__ = "Copyright 2010-2015, Universite de Montreal"
__credits__ = ["Minh Ngoc Le"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import os
from pylearn2.config import yaml_parse


def test_projection_layer_yaml():
    """Test loading and running a complex model with ProjectionLayer."""
    test_dir = os.path.dirname(__file__)
    with open(os.path.join(test_dir, 'composite.yaml')) as f:
        train = yaml_parse.load(f.read())
        train.main_loop()
