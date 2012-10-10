""" Can be run as a script or imported as a module.

Module exposes the run_pyflakes method which returns a dictionary.

As a script:

    python run_pyflakes.py <no_warnings>

    prints out all the errors in the library
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from pylearn2.devtools.list_files import list_files
from pylearn2.utils.shell import run_shell_command

def run_pyflakes(no_warnings = False):
    """ Returns a dictionary mapping pylearn2 .py filepaths
        to outputs from pyflakes.

        Omits files for which there was no output.

        If no_warnings = True, omits pyflakes outputs that don't
        correspond to actual errors.
    """

    files = list_files(".py")

    rval = {}

    for filepath in files:
        output, rc = run_shell_command('pyflakes ' + filepath)
        if 'pyflakes: not found' in output:
            # The return code alone does not make it possible to detect
            # if pyflakes is present or not. When pyflakes is not present,
            # the return code seems to always be 127, but 127 can also be
            # the result of finding 127 warnings in a file.
            # Therefore, we examine the output instead.
            raise RuntimeError("Couldn't run 'pyflakes " + filepath + "'. "
                    "Error code returned:" + str(rc)\
                    + " Output was: " + output)

        output = _filter(output, no_warnings)

        if output is not None:
            rval[filepath] = output

    return rval

def _filter(output, no_warnings):
    lines = output.split('\n')

    lines = [ line for line in lines if line != '' ]


    if no_warnings:

        lines = [ line for line in lines if
                line.find("is assigned to but never used") == -1 ]

        lines = [ line for line in lines if
                line.find('imported but unused') == -1 ]

        lines = [ line for line in lines if
                line.find('redefinition of unused ') == -1 ]


    if len(lines) == 0:
        return None
    return '\n'.join(lines)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        no_warnings = bool(sys.argv[1])
    else:
        no_warnings = False

    d = run_pyflakes( no_warnings = no_warnings)

    for key in d:
        print key +':'
        for l in d[key].split('\n'):
            print '\t'+l
