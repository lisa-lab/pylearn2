"""Different type of random seed generator, created to prevent redundancy
across the code, Reference: https://github.com/lisa-lab/pylearn2/issues/165 """

__author__ = "Abhishek Aggarwal, Xavier Bouthillier"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Abhishek Aggarwal", "Xavier Bouthillier"]
__license__ = "3-clause BSD"
__email__ = "bouthilx@iro"

# Portions cribbed from the standard library logging module,
# Copyright 2001-2010 by Vinay Sajip. All Rights Reserved.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of Vinay Sajip
# not be used in advertising or publicity pertaining to distribution
# of the software without specific, written prior permission.
# VINAY SAJIP DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
# ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# VINAY SAJIP BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
# ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import numpy

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# more distributions but slower
# from theano.tensor.shared_randomstreams import RandomStreams


def make_rng(rng_or_seed=None, default_seed=None,
             which_method=None, constructor=None):

    if not isinstance(which_method, list):
        which_method = [which_method]

    if rng_or_seed is not None and \
       all([hasattr(rng_or_seed, attr) for attr in which_method]):
        rng = rng_or_seed
    elif rng_or_seed is not None:
        rng = constructor(rng_or_seed)
    elif default_seed is not None:
        rng = constructor(default_seed)
    else:
        rng = constructor(42)

    return rng


def make_np_rng(rng_or_seed=None, default_seed=None, which_method=None,
                constructor=numpy.random.RandomState):
    return make_rng(rng_or_seed, default_seed, which_method, constructor)


def make_theano_rng(rng_or_seed=None, default_seed=None, which_method=None,
                    constructor=RandomStreams):
    return make_rng(rng_or_seed, default_seed, which_method, constructor)


#def rng_ints(rng_or_seed=None, default_seed=None):
#    return make_rng(rng_or_seed, default_seed, which_method='random_integers')
#
#
#def rng_normal(rng_or_seed=None, default_seed=None):
#    return make_rng(rng_or_seed, default_seed, which_method='normal')
#
#
#def rng_uniform(rng_or_seed=None, default_seed=None):
#    return make_rng(rng_or_seed, default_seed, which_method='uniform')
