"""
Can be run as a script or imported as a module.

Module exposes the run_pyflakes method which returns a dictionary.

As a script:

    python run_pyflakes.py <no_warnings>

    prints out all the errors in the library
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import sys
import logging

from theano.compat import six

from pylearn2.devtools.list_files import list_files
from pylearn2.utils.shell import run_shell_command


logger = logging.getLogger(__name__)


def run_pyflakes(no_warnings=False):
    """
    Return a description of all errors pyflakes finds in Pylearn2.

    Parameters
    ----------
    no_warnings : bool
        If True, omits pyflakes outputs that don't correspond to actual
        errors.

    Returns
    -------
    rval : dict
        Keys are pylearn2 .py filepaths
        Values are outputs from pyflakes
    """

    files = list_files(".py")

    rval = {}

    for filepath in files:
        output, rc = run_shell_command('pyflakes ' + filepath)
        output = output.decode(sys.getdefaultencoding())
        if u'pyflakes: not found' in output:
            # The return code alone does not make it possible to detect
            # if pyflakes is present or not. When pyflakes is not present,
            # the return code seems to always be 127, but 127 can also be
            # the result of finding 127 warnings in a file.
            # Therefore, we examine the output instead.
            raise RuntimeError("Couldn't run 'pyflakes " + filepath + "'. "
                               "Error code returned:" + str(rc) +
                               " Output was: " + output)

        output = _filter(output, no_warnings)

        if output is not None:
            rval[filepath] = output

    return rval


def _filter(output, no_warnings):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    output : str
        The output of pyflakes for a single.py file
    no_warnings: bool
        If True, removes lines corresponding to warnings rather than errors

    Returns
    -------
    rval : None or str
        `output` with blank lines and optionally lines corresponding to
        warnings removed, or, if all lines are removed, returns None.
        A return value of None indicates that the file is validly formatted.
    """
    lines = output.split('\n')

    lines = [line for line in lines
             if line != '' and line.find("undefined name 'long'") == -1]

    if no_warnings:

        lines = [line for line in lines if
                 line.find("is assigned to but never used") == -1]

        lines = [line for line in lines if
                 line.find('imported but unused') == -1]

        lines = [line for line in lines if
                 line.find('redefinition of unused ') == -1]

    if len(lines) == 0:
        return None
    return '\n'.join(lines)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        no_warnings = bool(sys.argv[1])
    else:
        no_warnings = False

    d = run_pyflakes(no_warnings=no_warnings)

    for key in d:
        logger.info('{0}:'.format(key))
        for l in d[key].split('\n'):
            logger.info('\t{0}'.format(l))
