"""
A unit test for the train.py script
"""
import os

import pylearn2
from pylearn2.scripts.train import train


def test_train_cmd():
    """
    Calls the train.py script with a help argument and
    with a short YAML file to see if it exits without
    an error
    """
    train(os.path.join(pylearn2.__path__[0],
                       "scripts/autoencoder_example/dae.yaml"))
