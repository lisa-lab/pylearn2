"""
A unit test for the summarize_model.py script
"""
import os
import subprocess

import pylearn2
from pylearn2.testing.skip import skip_if_no_matplotlib


def test_summarize_model():
    """
    Asks the summarize_model.py script to inspect a pickled model and
    check that it completes succesfully
    """
    skip_if_no_matplotlib()
    assert not subprocess.call([
        os.path.join(pylearn2.__path__[0], "scripts/summarize_model.py"),
        os.path.join(pylearn2.__path__[0], "scripts/tests/model.pkl")
    ])
