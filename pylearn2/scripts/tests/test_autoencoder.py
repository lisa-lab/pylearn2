"""
Tests for the pylearn2 autoencoder module.
"""
import numpy as np
import os
from pylearn2.scripts.tests.yaml_testing import limited_epoch_train
import pylearn2

def test_hcae_yaml():
    limited_epoch_train(os.path.join(pylearn2.__path__[0],"scripts/autoencoder_example/hcae.yaml"))

def test_dae_yaml():
    limited_epoch_train(os.path.join(pylearn2.__path__[0],"scripts/autoencoder_example/dae.yaml"))
