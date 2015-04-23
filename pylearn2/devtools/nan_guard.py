"""
Functionality for detecting NaNs in a Theano graph.

Some nodes are ignored by name because of known issues:
- GPU_mrg_uniform: Theano hack (see #1465)
Please add an explanation if you add nodes to this list.
"""
__authors__ = "Ian Goodfellow, Nicu Tofan"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import logging
import re
from theano.compile import Mode
import theano
import numpy as np
from pylearn2.models.dbm import flatten
from pylearn2.utils import contains_nan, contains_inf


logger = logging.getLogger(__name__)

# Following nodes are ignored by the NanGuardMode check.
IGNORED_NODES = ['GPU_mrg_uniform']

class NanGuardMode(Mode):
    """
    A Theano compilation Mode that makes the compiled function automatically
    detect NaNs and Infs and detect an error if they occur.

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
        def do_check_on(var, nd, f, is_input):
            """
            Checks `var` for NaNs / Infs. If detected, raises an exception
            and / or prints information about `nd`, `f`, and `is_input` to
            help the user determine the cause of the invalid values.

            Parameters
            ----------
            var : numpy.ndarray
                The value to be checked.
            nd : theano.gof.Apply
                The Apply node being executed
            f : callable
                The thunk for the apply node
            is_input : bool
                If True, `var` is an input to `nd`.
                If False, it is an output.
            """
            error = False
            if nan_is_error:
                if contains_nan(var):
                    error_s = 'NaN detected'
                    error = True
            if inf_is_error:
                if contains_inf(var):
                    error_s = 'Inf detected'
                    error = True
            if big_is_error:
                if np.abs(var).max() > 1e10:
                    error_s = 'Big value detected'
                    error = True
            if error:
                 
                if is_input:
                    logger.error('%s in an input', error_s)
                else:
                    logger.error('%s in an output', error_s)
                logger.error('Inputs: ')
                for ivar, ival in zip(nd.inputs, f.inputs):
                    logger.error('var %s', str(ivar))
                    logger.error('    %s', str(theano.printing.min_informative_str(ivar)))
                    logger.error('    value: %s', str(ival))
                logger.error('Node: %s', str(nd))
                assert False, error_s 

        def nan_check(i, node, fn):
            """
            Runs `fn` while checking its inputs and outputs for NaNs / Infs

            Parameters
            ----------
            i : int
                Currently ignored (is here to match required signature for wrappers)
            node : theano.gof.Apply
                The Apply node currently being executed
            fn : callable
                The thunk to execute for this Apply node
            """

            # Some nodes are ignored; see module level documentation
            perform_checks = True
            node_match = self.__node_name_rex__.match(str(node))
            if node_match:
                if node_match.group(1) in IGNORED_NODES:
                    perform_checks = False

            if perform_checks:
                inputs = fn.inputs
                # TODO: figure out why individual inputs are themselves lists sometimes
                for x in flatten(inputs):
                    do_check_on(x, node, fn, True)

            fn()

            if perform_checks:
                outputs = fn.outputs
                for j, x in enumerate(flatten(outputs)):
                    do_check_on(x, node, fn, False)

        self.__node_name_rex__ = re.compile('^([A-Za-z0-9_]+)')
        wrap_linker = theano.gof.WrapLinkerMany([theano.gof.OpWiseCLinker()], [nan_check])
        super(NanGuardMode, self).__init__(wrap_linker, optimizer=theano.config.optimizer)
