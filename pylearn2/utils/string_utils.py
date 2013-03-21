""" Utilities for modifying strings"""

import os
import warnings
import re
import functools

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
                raise EnvironmentVariableError(
"""
PYLEARN2_VIEWER_COMMAND not defined. PLEASE READ THE FOLLOWING MESSAGE CAREFULLY
TO SET UP THIS ENVIRONMENT VARIABLE:

pylearn2 uses an external program to display images. Because different systems have different
image programs available, pylearn2 requires the user to specify what image viewer program to
use.

You need to choose an image viewer program that pylearn2 should use. Then tell pylearn2 to use
that image viewer program by defining your PYLEARN2_VIEWER_COMMAND environment variable.

You need to choose PYLEARN_VIEWER_COMMAND such that running

${PYLEARN2_VIEWER_COMMAND} image.png

in a command prompt on your machine will do the following:
    -open an image viewer in a new process.
    -not return until you have closed the image.

Acceptable commands include:
    gwenview
    eog --new-instance

This is assuming that you have gwenview or a version of eog that supports --new-instance
installed on your machine. If you don't, install one of those, or figure out a command
that has the above properties that is available from your setup.

On most linux setups, you can define your environment variable by adding this line to your
~/.bashrc file:

export PYLEARN2_VIEWER_COMMAND="eog --new-instance"

*** YOU MUST INCLUDE THE WORD "export". DO NOT JUST ASSIGN TO THE ENVIRONMENT VARIABLE ***
If you do not include the word "export", the environment variable will be set in your
bash shell, but will not be visible to processes that you launch from it, like the python
interpreter.

Don't forget that changes from your .bashrc file won't apply until you run

source ~/.bashrc

or open a new terminal window. If you're seeing this from an ipython notebook
you'll need to restart the ipython notebook, or maybe modify os.environ from
an ipython cell.
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


def number_aware_alphabetical_cmp(str1, str2):
    """ cmp function for sorting a list of strings by alphabetical order, but with
        numbers sorted numerically.

        i.e., foo1, foo2, foo10, foo11
        instead of foo1, foo10
    """

    def flatten_tokens(tokens):
        l = []
        for token in tokens:
            if isinstance(token, str):
                for char in token:
                    l.append(char)
            else:
                assert isinstance(token, float)
                l.append(token)
        return l

    seq1 = flatten_tokens(tokenize_by_number(str1))
    seq2 = flatten_tokens(tokenize_by_number(str2))

    l = min(len(seq1),len(seq2))

    i = 0

    while i < l:
        if seq1[i] < seq2[i]:
            return -1
        elif seq1[i] > seq2[i]:
            return 1
        i += 1

    if len(seq1) < len(seq2):
        return -1
    elif len(seq1) > len(seq2):
        return 1

    return 0

#key for sorting strings alphabetically with numbers
number_aware_alphabetical_key = functools.cmp_to_key(number_aware_alphabetical_cmp)

def match(wrong, candidates):
    """
        wrong: a mispelling
        candidates: a set of correct words

        returns a guess of which candidate is the right one

        This should be used with a small number of candidates and a high potential
        edit distance.
        ie, use it to correct a wrong filename in a directory, wrong class name
        in a module, etc. Don't use it to correct small typos of freeform natural
        language words.
    """

    assert len(candidates) > 0

    # Current implementation tries all candidates and outputs the one
    # with the min score
    # Could try to do something smarter

    def score(w1,w2):
        # Current implementation returns negative dot product of
        # the two words mapped into a feature space by mapping phi
        # w -> [ phi(w1), .1 phi(first letter of w), .1 phi(last letter of w) ]
        # Could try to do something smarter

        w1 = w1.lower()
        w2 = w2.lower()

        def phi(w):
            # Current feature mapping is to the vector of counts of
            # all letters and two-letter sequences
            # Could try to do something smarter
            rval = {}

            for i in xrange(len(w)):
                l = w[i]
                rval[l] = rval.get(l,0.) + 1.
                if i < len(w)-1:
                    b = w[i:i+2]
                    rval[b] = rval.get(b,0.) + 1.

            return rval

        d1 = phi(w1)
        d2 = phi(w2)

        def mul(d1, d2):
            rval = 0

            for key in set(d1).union(d2):
                rval += d1.get(key,0) * d2.get(key,0)

            return rval

        tot_score = mul(phi(w1),phi(w2)) / float(len(w1)*len(w2)) + \
            0.1 * mul(phi(w1[0:1]), phi(w2[0:1])) + \
            0.1 * mul(phi(w1[-1:]), phi(w2[-1:]))

        return  tot_score

    scored_candidates = [ (-score(wrong, candidate), candidate)
            for candidate in candidates ]

    scored_candidates.sort()

    return scored_candidates[0][1]
