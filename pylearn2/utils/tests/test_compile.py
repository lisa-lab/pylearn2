"""Tests for compilation utilities."""
import theano
import pickle

from pylearn2.utils.compile import (
    compiled_theano_function, HasCompiledFunctions
)


class Dummy(HasCompiledFunctions):
    const = 3.14159

    @compiled_theano_function
    def func(self):
        val = theano.tensor.as_tensor_variable(self.const)
        return theano.function([], val)


def test_simple_compilation():
    x = Dummy()
    f = x.func
    g = x.func
    assert f is g
    assert x.func() == Dummy.const


def test_pickling():
    a = Dummy()
    assert a.func() == Dummy.const
    serialized = pickle.dumps(a)
    b = pickle.loads(serialized)
    assert not hasattr(b, '_compiled_functions')
    assert b.func() == Dummy.const
    assert not (a.func is b.func)
