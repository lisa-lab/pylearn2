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
    assert abs(x.func() - Dummy.const) < 1e-6


def test_pickling():
    a = Dummy()
    assert abs(a.func() - Dummy.const) < 1e-6
    serialized = pickle.dumps(a)
    b = pickle.loads(serialized)
    assert not hasattr(b, '_compiled_functions')
    assert abs(b.func() - Dummy.const) < 1e-6
    assert not (a.func is b.func)
