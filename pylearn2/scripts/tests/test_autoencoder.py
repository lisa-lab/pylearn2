"""
Tests for the pylearn2 autoencoder module.
"""
import numpy as np
import os
from pylearn2.scripts.tests.yaml_testing import yaml_file_execution
import pylearn2

def test_hcae_yaml():
    yaml_file_execution(os.path.join(pylearn2.__path__[0],"scripts/autoencoder_example/hcae.yaml"))

def test_dae_yaml():
    yaml_file_execution(os.path.join(pylearn2.__path__[0],"scripts/autoencoder_example/dae.yaml"))
