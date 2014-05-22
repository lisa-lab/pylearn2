"""
This functionality has been moved to theano. Links here maintained to
avoid breaking old import statements
"""
from theano.tests.record import MismatchError, Record, RecordMode

import warnings
warnings.warn("pylearn2.devtools.record is deprecated and may be removed "
        "on or after Aug 20, 2014. This functionality has been moved to "
        "theano.tests.record.")
