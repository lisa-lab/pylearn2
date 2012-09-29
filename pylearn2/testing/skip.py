""" Helper functions for determining which tests to skip. """
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
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
