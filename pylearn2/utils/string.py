""" Utilities for modifying strings"""

import os

def preprocess(string):
    """
    Preprocesses a string, by replacing ${VARNAME} with
    os.environ['VARNAME']

    Parameters
    ----------
    string: the str object to preprocess

    Returns
    -------
    the preprocessed string
    """

    split = string.split('${')

    rval = [split[0]]

    for candidate in split[1:]:
        subsplit = candidate.split('}')

        if len(subsplit) < 2:
            raise ValueError('Open ${ not followed by } before ' \
                    + 'end of string or next ${ in "' \
                    + string + '"')

        varname = subsplit[0]

        try:
            val = os.environ[varname]
        except KeyError:
            if varname == 'PYLEARN2_DATA_PATH':
                raise EnvironmentVariableError("You need to define your PYLEARN2_DATA_PATH environment variable. If you are using a computer at LISA, this should be set to /data/lisa/data")

            raise

        rval.append(val)

        rval.append('}'.join(subsplit[1:]))

    rval = ''.join(rval)

    return rval

class EnvironmentVariableError(Exception):
    """ An exception raised when a required environment variable is not defined """

    def __init__(self, *args):
        super(EnvironmentVariableError,self).__init__(*args)


