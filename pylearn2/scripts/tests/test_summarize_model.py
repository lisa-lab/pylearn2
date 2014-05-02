"""
A unit test for the summarize_model.py script
"""
import os
import subprocess

import pylearn2


def test_summarize_model():
    """
    Asks the summarize_model.py script to inspect a pickled model and
    check that it completes succesfully
    """
    assert not subprocess.call([
        os.path.join(pylearn2.__path__[0], "scripts/summarize_model.py"),
        os.path.join(pylearn2.__path__[0], "scripts/tests/model.pkl")
    ])
