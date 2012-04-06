"""Tests for the InsertAlongAxis op."""
import numpy as np
import theano
from theano import tensor

from pylearn2.utils.insert_along_axis import (
    insert_columns, insert_rows, InsertAlongAxis
)


def test_insert_along_axis():
    x = tensor.matrix()
    y = insert_columns(x, 10, range(0, 10, 2))
    f = theano.function([x], y)
    x_ = np.random.normal(size=(7, 5))
    f_val = f(x_)
    assert f_val.shape == (7, 10)
    assert np.all(f_val[:, range(0, 10, 2)] == x_)
    assert f_val.dtype == x_.dtype

    y = insert_rows(x, 10, range(0, 10, 2))
    f = theano.function([x], y)
    x_ = np.random.normal(size=(5, 6))
    f_val = f(x_)
    assert f_val.shape == (10, 6)
    assert np.all(f_val[range(0, 10, 2)] == x_)
    assert f_val.dtype == x_.dtype

    x = tensor.tensor3()
    y = InsertAlongAxis(3, 1)(x, 10, range(0, 10, 2))
    f = theano.function([x], y)
    x_ = np.random.normal(size=(2, 5, 2))
    f_val = f(x_)
    assert f_val.shape == (2, 10, 2)
    assert np.all(f_val[:, range(0, 10, 2), :] == x_)
    assert f_val.dtype == x_.dtype

    x = tensor.tensor3()
    y = InsertAlongAxis(3, 1, fill=2)(x, 10, range(0, 10, 2))
    f = theano.function([x], y)
    x_ = np.random.normal(size=(2, 5, 2))
    f_val = f(x_)
    assert f_val.shape == (2, 10, 2)
    assert np.all(f_val[:, range(0, 10, 2), :] == x_)
    assert np.all(f_val[:, range(1, 10, 2), :] == 2)
    assert f_val.dtype == x_.dtype


def test_insert_along_axis_gradient():
    x = tensor.matrix()
    y = insert_columns(x, 10, range(0, 10, 2))
    f = theano.function([x], tensor.grad(y.sum(), x))
    f_val = f(np.random.normal(size=(7, 5)))
    assert np.all(f_val == 1)
    assert f_val.shape == (7, 5)

    y = insert_rows(x, 10, range(0, 10, 2))
    f = theano.function([x], tensor.grad(y.sum(), x))
    f_val = f(np.random.normal(size=(5, 6)))
    assert np.all(f_val == 1)
    assert f_val.shape == (5, 6)

    x = tensor.tensor3()
    y = InsertAlongAxis(3, 1)(x, 10, range(0, 10, 2))
    f = theano.function([x], tensor.grad(y.sum(), x))
    f_val = f(np.random.normal(size=(2, 5, 2)))
    assert np.all(f_val == 1)
    assert f_val.shape == (2, 5, 2)
