"""Different type of random seed generator, created to prevent redundancy across the code,
Reference: https://github.com/lisa-lab/pylearn2/issues/165 """

__author__ = "Abhishek Aggarwal"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Abhishek Aggarwal"]
__license__ = "3-clause BSD"
__email__ = "aggarwal@iro"

import numpy

def make_rng(rng_or_seed = None, default_seed=None, typeStr=None):
    """
    Returns a numpy RandomState, using the first of these cases that produces a valid
    result and doesn't use an argument with the value of None:
    
    1) rng_or_seed itself
    2) RandomState(rng_or_seed)
    3) RandomState(default_seed)
    4) RandomState(42)
    
    May raise a TypeError if rng_or_seed or default_seed is bad.
    """
    if rng_or_seed is not None:

        if isinstance(typeStr, (list,tuple)):
            for typestr in typeStr:
                if not hasattr(rng_or_seed, typestr):
                    try:
                        rng = numpy.random.RandomState(rng_or_seed)
                    except ValueError:
                        raise ValueError("user passed seed should be an integer or array_like")
        else:
            for typestr in typeStr:
                if not hasattr(rng_or_seed, typestr):
                    try:
                        rng = numpy.random.RandomState(rng_or_seed)
                    except ValueError:
                        raise ValueError("user passed seed should be an integer or array_like")
                    
    else:
        if default_seed is None:
            try:
                rng = numpy.random.RandomState(default_seed)
            except ValueError:
                raise ValueError("default seed should be an integer or array_like.")
        else:
           rng = numpy.random.RandomState(42)
    return rng


def rng_randn(rng_or_seed = None, default_seed=None):
    return make_rng(rng_or_seed, default_seed, typeStr = 'randn')


def rng_ints(rng_or_seed = None, default_seed=None):
    return make_rng(rng_or_seed, default_seed, typeStr = 'random_integers')


def rng_normal(rng_or_seed = None, default_seed=None):
    return make_rng(rng_or_seed, default_seed, typeStr = 'normal')


def rng_uniform(rng_or_seed =None, default_seed=None):
    return make_rng(rng_or_seed, default_seed, typeStr = 'uniform')
