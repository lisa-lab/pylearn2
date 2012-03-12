""" Utilities for modifying strings"""

import os
import warnings
import re

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

        if varname == 'PYLEARN2_TRAIN_FILE_NAME':
            warnings.warn("PYLEARN2_TRAIN_FILE_NAME is deprecated, use PYLEARN2_TRAIN_FILE_FULL_STEM")

        try:
            val = os.environ[varname]
        except KeyError:
            if varname == 'PYLEARN2_DATA_PATH':
                raise EnvironmentVariableError("You need to define your PYLEARN2_DATA_PATH environment variable. If you are using a computer at LISA, this should be set to /data/lisa/data")
            if varname == 'PYLEARN2_VIEWER_COMMAND':
                raise EnvironmentVariableError("""You need to define your PYLEARN2_VIEWER_COMMAND environment variable.
                        ${PYLEARN2_VIEWER_COMMAND} image.png
                        should open an image viewer in a new process and not return until you have closed the image.
                        Acceptable commands include:
                        gwenview
                        eog --new-instance
                        """)

            raise

        rval.append(val)

        rval.append('}'.join(subsplit[1:]))

    rval = ''.join(rval)

    return rval

class EnvironmentVariableError(Exception):
    """ An exception raised when a required environment variable is not defined """

    def __init__(self, *args):
        super(EnvironmentVariableError,self).__init__(*args)



def find_number(s):
    """ s is a string
        returns None if there are no numbers in the string
        otherwise returns the range of characters occupied by the first
        number in the string """

    r = re.search('-?\d+[.e]?\d*',s)
    if r is not None:
        return r.span(0)
    return None

def tokenize_by_number(s):
    """ splits a string into a list of tokens
        each is either a string containing no numbers
        or a float """

    r = find_number(s)

    if r == None:
        return [ s ]
    else:
        tokens = []
        if r[0] > 0:
            tokens.append(s[0:r[0]])
        tokens.append( float(s[r[0]:r[1]]) )
        if r[1] < len(s):
            tokens.extend(tokenize_by_number(s[r[1]:]))
        return tokens
    assert False #line should be unreached
