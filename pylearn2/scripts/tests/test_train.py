"""
A unit test for the train.py script
"""
import os
import subprocess

import pylearn2


def test_train_cmd():
    """
    Calls the train.py script with a help argument and
    with a short YAML file to see if it exits without
    an error
    """
    assert not subprocess.call(
        [os.path.join(pylearn2.__path__[0], "scripts/train.py"), "-h"]
    )
    assert not subprocess.call([
        os.path.join(pylearn2.__path__[0], "scripts/train.py"), "-L", "-T",
        "-t 60", "-V", "-D",
        os.path.join(pylearn2.__path__[0],
                     "scripts/autoencoder_example/dae.yaml")
    ])
