"""
Very simple and basic mathematical expressions used often throughout the library.
"""
__authors__ = "Ian Goodfellow and Razvan Pascanu"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow and Razvan Pascanu"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np
import theano.tensor as T

from pylearn2.base import Block
from pylearn2.utils import as_floatX, constantX

def numpy_norms(W):
    """ returns a vector containing the L2 norm of each
column of W, where W and the return value are
numpy ndarrays """
    return np.sqrt(1e-8+np.square(W).sum(axis=0))

def theano_norms(W):
    """ returns a vector containing the L2 norm of each
column of W, where W and the return value are symbolic
theano variables """
    return T.sqrt(as_floatX(1e-8)+T.sqr(W).sum(axis=0))

def full_min(var):
    """ returns a symbolic expression for the value of the minimal
    element of symbolic tensor. T.min does something else as of
    the time of this writing. """
    return var.min(axis=range(0,len(var.type.broadcastable)))

def full_max(var):
    """ returns a symbolic expression for the value of the maximal
        element of a symbolic tensor. T.max does something else as of the
        time of this writing. """
    return var.max(axis=range(0,len(var.type.broadcastable)))


def multiple_switch(*args):
    """
    Applies a cascade of ifelse. The output will be a Theano expression
    which evaluates:
        if args0:
            then arg1
        elif arg2:
            then arg3
        elif arg4:
            then arg5
        ....
    """
    if len(args) == 3:
        return T.switch(*args)
    else:
        return T.switch(args[0],
                         args[1],
                         multiple_switch(*args[2:]))


def symGivens2(a, b):
    """
    Stable Symmetric Givens rotation plus reflection

    Parameters

        a: (theano scalar) first element of a two-vector  [a; b]
        b: (theano scalar) second element of a two-vector [a; b]
    Returns

        c  cosine(theta), where theta is the implicit angle of
           rotation (counter-clockwise) in a plane-rotation
        s  sine(theta)
        d  two-norm of [a; b]

    Description:
        This method gives c and s such that
            [ c  s ][a] = [d],
            [ s -c ][b]   [0]
      where d = two norm of vector [a, b],
            c = a / sqrt(a^2 + b^2) = a / d,
            s = b / sqrt(a^2 + b^2) = b / d.
      The implementation guards against overflow in computing
         sqrt(a^2 + b^2).

      SEE ALSO:
         (1) Algorithm 4.9, stable *unsymmetric* Givens
         rotations in Golub and van Loan's book Matrix
         Computations, 3rd edition.
         (2) MATLAB's function PLANEROT.

      Observations:
          Implementing this function as a single op in C might improve speed
          considerably ..
    """
    c_branch1 = T.switch(T.eq(a, constantX(0)),
                          constantX(1),
                          T.sgn(a))
    c_branch21 = (a / b) * T.sgn(b) / \
            T.sqrt(constantX(1) + (a / b) ** 2)
    c_branch22 = T.sgn(a) / T.sqrt(constantX(1) + (b / a) ** 2)

    c_branch2 = T.switch(T.eq(a, constantX(0)),
                          constantX(0),
                          T.switch(T.gt(abs(b), abs(a)),
                                    c_branch21,
                                    c_branch22))
    c = T.switch(T.eq(b, constantX(0)),
                  c_branch1,
                  c_branch2)

    s_branch1 = T.sgn(b) / T.sqrt(constantX(1) + (a / b) ** 2)
    s_branch2 = (b / a) * T.sgn(a) / T.sqrt(constantX(1) + (b / a) ** 2)
    s = T.switch(T.eq(b, constantX(0)),
                  constantX(0),
                  T.switch(T.eq(a, constantX(0)),
                            T.sgn(b),
                            T.switch(T.gt(abs(b), abs(a)),
                                      s_branch1,
                                      s_branch2)))

    d_branch1 = b / (T.sgn(b) / T.sqrt(constantX(1) + (a / b) ** 2))
    d_branch2 = a / (T.sgn(a) / T.sqrt(constantX(1) + (b / a) ** 2))
    d = T.switch(T.eq(b, constantX(0)),
                  abs(a),
                  T.switch(T.eq(a, constantX(0)),
                            abs(b),
                            T.switch(T.gt(abs(b), abs(a)),
                                      d_branch1,
                                      d_branch2)))
    return c, s, d


def sqrt_inner_product(xs, ys=None):
    """
        Compute the square root of the inner product between `xs` and `ys`.
        If `ys` is not provided, computes the norm between `xs` and `xs`.
        Since `xs` and `ys` are list of tensor, think of it as the norm
        between the vector obtain by concatenating and flattening all
        tenors in `xs` and the similar vector obtain from `ys`. Note that
        `ys` should match `xs`.

        Parameters:

            xs : list of theano expressions
            ys : None or list of theano expressions
    """
    if ys is None:
        ys = [x for x in xs]
    return T.sqrt(sum((x * y).sum() for x, y in zip(xs, ys)))


def inner_product(xs, ys=None):
    """
        Compute the inner product between `xs` and `ys`. If ys is not provided,
        computes the square norm between `xs` and `xs`.
        Since `xs` and `ys` are list of tensor, think of it as the inner
        product between the vector obtain by concatenating and flattening all
        tenors in `xs` and the similar vector obtain from `ys`. Note that
        `ys` should match `xs`.

        Parameters:

            xs : list of theano expressions
            ys : None or list of theano expressions
    """
    if ys is None:
        ys = [x for x in xs]
    return sum((x * y).sum() for x, y in zip(xs, ys))

def is_binary(x):
    return np.all( (x == 0) + (x == 1))

class Identity(Block):
    """
    A Block that computes the identity transformation. Mostly useful as a
    placeholder.
    """

    def __call__(self, inputs):
        return inputs

