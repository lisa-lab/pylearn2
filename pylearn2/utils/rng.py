"""
Different type of random seed generator, created to prevent redundancy
across the code.

Reference: https://github.com/lisa-lab/pylearn2/issues/165
"""

__author__ = "Abhishek Aggarwal, Xavier Bouthillier"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Abhishek Aggarwal", "Xavier Bouthillier"]
__license__ = "3-clause BSD"
__email__ = "bouthilx@iro"

import numpy

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# more distributions but slower
# from theano.tensor.shared_randomstreams import RandomStreams


def make_rng(rng_or_seed=None, default_seed=None,
             which_method=None, constructor=None):
    """
    Returns a RNG object.

    Parameters
    ----------
    rng_or_seed : RNG object or integer or list of integers
        A random number generator or a valid seed.
    default_seed : integer or list of integers
        Seed used if rng_or_seed is None
    which_method : string or list of strings
        One or more methods that must be defined by the RNG object.
        If one or more specified methods are not defined by it, a
        new one will be constructed with the given constructor.
    constructor : function or class
        Must return a RNG object.
        constructor is called with rng_or_seed, default_seed or 42
        as argument.

    Notes
    -----
    The RNG object is generated using the first of these cases that produces a
    valid result and doesn't use an argument with the value of None:

    1) rng_or_seed itself
    2) constructor(rng_or_seed)
    3) constructor(default_seed)
    4) constructor(42)
    """

    if isinstance(which_method, basestring):
        which_method = [which_method]

    if rng_or_seed is not None and \
       all(hasattr(rng_or_seed, attr) for attr in which_method):
        rng = rng_or_seed
    elif rng_or_seed is not None:
        rng = constructor(rng_or_seed)
    elif default_seed is not None:
        rng = constructor(default_seed)
    else:
        rng = constructor(42)

    return rng


def make_np_rng(rng_or_seed=None, default_seed=None, which_method=None):
    """
    Returns a numpy RandomState.

    Parameters
    ----------
    rng_or_seed : RNG object or integer or list of integers
        A random number generator or a valid seed.
    default_seed : integer or list of integers
        Seed used if rng_or_seed is None
    which_method : string or list of strings
        One or more methods that must be defined by the RNG object.
        If one or more specified methods are not defined by it, a
        new one will be constructed from RandomState.

    Notes
    -----
    The returned RandomState is generated using the first of these cases that
    produces a valid result and doesn't use an argument with the value of None:

    1) rng_or_seed itself
    2) RandomState(rng_or_seed)
    3) RandomState(default_seed)
    4) RandomState(42)
    """
    return make_rng(rng_or_seed, default_seed, which_method,
                    numpy.random.RandomState)


def make_theano_rng(rng_or_seed=None, default_seed=None, which_method=None):
    """
    Returns a theano RandomStreams.

    Parameters
    ----------
    rng_or_seed : RNG object or integer or list of integers
        A random number generator or a valid seed.
    default_seed : integer or list of integers
        Seed used if rng_or_seed is None
    which_method : string or list of strings
        One or more methods that must be defined by the RNG object.
        If one or more specified methods are not defined by it, a
        new one will be constructed from RandomState.

    Notes
    -----
    The returned RandomStreams is generated using the first of these cases that
    produces a valid result and doesn't use an argument with the value of None:

    1) rng_or_seed itself
    2) RandomState(rng_or_seed)
    3) RandomState(default_seed)
    4) RandomState(42)
    """

    return make_rng(rng_or_seed, default_seed, which_method, RandomStreams)
