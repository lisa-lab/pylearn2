import numpy
import theano
from pylearn2.format.target_format import OneHotFormatter
from theano.scalar.basic import all_types


def test_one_hot_formatter_simple():
    def check_one_hot_formatter(seed, max_labels, dtype, ncases):
        rng = numpy.random.RandomState(seed)
        fmt = OneHotFormatter(max_labels=max_labels, dtype=dtype)
        integer_labels = rng.random_integers(0, max_labels - 1, size=ncases)
        one_hot_labels = fmt.format(integer_labels)
        assert len(zip(*one_hot_labels.nonzero())) == ncases
        for case, label in enumerate(integer_labels):
            assert one_hot_labels[case, label] == 1
    rng = numpy.random.RandomState(0)
    for seed, dtype in enumerate(all_types):
        yield (check_one_hot_formatter, seed, rng.random_integers(1, 30), dtype,
               rng.random_integers(1, 100))


def test_one_hot_formatter_symbolic():
    def check_one_hot_formatter_symbolic(seed, max_labels, dtype, ncases):
        rng = numpy.random.RandomState(seed)
        fmt = OneHotFormatter(max_labels=max_labels, dtype=dtype)
        integer_labels = rng.random_integers(0, max_labels - 1, size=ncases)
        x = theano.tensor.vector(dtype='int64')
        y = fmt.theano_expr(x)
        f = theano.function([x], y)
        one_hot_labels = f(integer_labels)
        assert len(zip(*one_hot_labels.nonzero())) == ncases
        for case, label in enumerate(integer_labels):
            assert one_hot_labels[case, label] == 1

    rng = numpy.random.RandomState(0)
    for seed, dtype in enumerate(all_types):
        yield (check_one_hot_formatter_symbolic, seed,
               rng.random_integers(1, 30), dtype,
               rng.random_integers(1, 100))


def test_dtype_errors():
    # Try to call theano_expr with a bad label dtype.
    raised = False
    fmt = OneHotFormatter(max_labels=50)
    try:
        fmt.theano_expr(theano.tensor.vector(dtype=theano.config.floatX))
    except TypeError:
        raised = True
    assert raised

    # Try to call format with a bad label dtype.
    raised = False
    try:
        fmt.format(numpy.zeros(10, dtype='float64'))
    except TypeError:
        raised = True
    assert raised


def test_bad_arguments():
    # Make sure an invalid max_labels raises an error.
    raised = False
    try:
        fmt = OneHotFormatter(max_labels=-10)
    except ValueError:
        raised = True
    assert raised

    raised = False
    try:
        fmt = OneHotFormatter(max_labels='10')
    except ValueError:
        raised = True
    assert raised

    # Make sure an invalid dtype identifier raises an error.
    raised = False
    try:
        fmt = OneHotFormatter(max_labels=10, dtype='invalid')
    except TypeError:
        raised = True
    assert raised

    # Make sure an invalid ndim raises an error for format().
    fmt = OneHotFormatter(max_labels=10)
    raised = False
    try:
        fmt.format(numpy.zeros((2, 3), dtype='int32'))
    except ValueError:
        raised = True
    assert raised

    # Make sure an invalid ndim raises an error for theano_expr().
    raised = False
    try:
        fmt.theano_expr(theano.tensor.imatrix())
    except ValueError:
        raised = True
    assert raised
