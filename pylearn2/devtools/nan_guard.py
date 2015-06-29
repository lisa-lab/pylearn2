"""
Functionality for detecting NaNs in a Theano graph.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import logging
import theano.compile.nanguardmode

logger = logging.getLogger(__name__)


class NanGuardMode(theano.compile.nanguardmode.NanGuardMode):
    """
    A Theano compilation Mode that makes the compiled function automatically
    detect NaNs and Infs and detect an error if they occur. This mode is now in
    theano, thus it is depreciated in pylearn2.

    Parameters
    ----------
    nan_is_error : bool
        If True, raise an error anytime a NaN is encountered
    inf_is_error: bool
        If True, raise an error anytime an Inf is encountered.  Note that some
        pylearn2 modules currently use np.inf as a default value (e.g.
        mlp.max_pool) and these will cause an error if inf_is_error is True.
    big_is_error: bool
        If True, raise an error when a value greater than 1e10 is encountered.
    """
    def __init__(self, nan_is_error, inf_is_error, big_is_error=True):
        super(NanGuardMode, self).__init__(
            self, nan_is_error, inf_is_error, big_is_error=big_is_error
        )
        logger.warning("WARNING: NanGuardMode has been moved to theano. It is "
                       "depreciated in pylearn2. Importing from theano. ")
