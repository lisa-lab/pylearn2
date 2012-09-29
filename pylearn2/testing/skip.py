""" Helper functions for determining which tests to skip. """
from nose.plugins.skip import SkipTest
import os

scipy_works = True
try:
    import scipy
except ImportError:
    # pyflakes gets mad if you set scipy to None here
    scipy_works = False

def skip_if_no_data():
    if 'PYLEARN2_DATA_PATH' not in os.environ:
        raise SkipTest()

def skip_if_no_scipy():
    if not scipy_works:
        raise SkipTest()
