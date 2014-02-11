"""
Tests for the pylearn2 autoencoder module.
"""
import numpy as np
from pylearn2.scripts.tests.yaml_testing import yaml_file_execution

def test_hcae_yaml():
    yaml_file_execution("pylearn2/scripts/autoencoder_example/hcae.yaml")

def test_dae_yaml():
    yaml_file_execution("pylearn2/scripts/autoencoder_example/dae.yaml")
