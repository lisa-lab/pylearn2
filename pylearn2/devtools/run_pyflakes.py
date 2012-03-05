""" Can be run as a script or imported as a module.

Module exposes the run_pyflakes method which returns a dictionary.

As a script:

    python run_pyflakes.py <no_warnings>

    prints out all the errors in the library
"""

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
        output, rc = run_shell_command('pyflakes '+filepath)
        if rc not in [0,1]:
            #pyflakes will return 1 if you give it an invalid file or if
            #the file contains errors, so it's not clear how to detect if
            #pyflakes failed
            #however, if pyflakes just plain isn't installed we should get 127
            raise RuntimeError("Couldn't run 'pyflakes "+filepath+"'."\
                    + "Output was: "+output)

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
