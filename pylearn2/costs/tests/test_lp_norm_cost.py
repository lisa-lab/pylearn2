"""
Test LpNorm cost
"""
import numpy
import theano
from theano import tensor as T
from nose.tools import raises


def test_shared_variables():
    '''
    LpNorm should handle shared variables.
    '''
    assert False


def test_symbolic_expressions_of_shared_variables():
    '''
    LpNorm should handle symbolic expressions of shared variables.
    '''
    assert False


@raises(Exception)
def test_symbolic_variables():
    '''
    LpNorm should not handle symbolic variables
    '''
    assert True


if __name__ == '__main__':
    test_shared_variables()
    test_symbolic_expressions_of_shared_variables()
    test_symbolic_variables()
