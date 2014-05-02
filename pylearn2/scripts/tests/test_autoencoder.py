"""
Tests for the pylearn2 autoencoder module.
"""
import os

from pylearn2.scripts.tests.yaml_testing import limited_epoch_train
import pylearn2


def test_hcae_yaml():
    """
    Train a higher order contractive autoencoder for a single epoch
    """
    limited_epoch_train(os.path.join(pylearn2.__path__[0],
                                     "scripts/autoencoder_example/hcae.yaml"))


def test_dae_yaml():
    """
    Train a denoising autoencoder for a single epoch
    """
    limited_epoch_train(os.path.join(pylearn2.__path__[0],
                                     "scripts/autoencoder_example/dae.yaml"))
