"""
Tests for the pylearn2 autoencoder module.
"""
import numpy as np

def test_hcae_yaml():
    from pylearn2.scripts.tests.yaml_testing import test_yaml_file
    X = np.random.normal(size=(1000, 300))
    np.save('garbage.npy', X)
    test_yaml_file("pylearn2/scripts/autoencoder_example/hcae.yaml")

def test_dae_yaml():
    from pylearn2.scripts.tests.yaml_testing import test_yaml_file
    X = np.random.normal(size=(1000, 300))
    np.save('garbage.npy', X)
    test_yaml_file("pylearn2/scripts/autoencoder_example/dae.yaml")
