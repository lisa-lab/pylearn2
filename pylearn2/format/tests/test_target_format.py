import numpy
import theano
from numpy.testing import assert_equal, assert_, assert_raises
from theano.scalar.basic import all_types
from pylearn2.format.target_format import OneHotFormatter, compressed_one_hot


def test_one_hot_formatter_simple():
    def check_one_hot_formatter(seed, max_labels, dtype, ncases):
        rng = numpy.random.RandomState(seed)
        fmt = OneHotFormatter(max_labels=max_labels, dtype=dtype)
        integer_labels = rng.random_integers(0, max_labels - 1, size=ncases)
        one_hot_labels = fmt.format(integer_labels)
        assert len(list(zip(*one_hot_labels.nonzero()))) == ncases
        assert one_hot_labels.dtype == dtype
        for case, label in enumerate(integer_labels):
            assert one_hot_labels[case, label] == 1
    rng = numpy.random.RandomState(0)
    for seed, dtype in enumerate(all_types):
        yield (check_one_hot_formatter, seed, rng.random_integers(1, 30),
               dtype, rng.random_integers(1, 100))
    fmt = OneHotFormatter(max_labels=10)
    assert fmt.format(numpy.zeros((1, 1), dtype='uint8')).shape == (1, 1, 10)


def test_one_hot_formatter_symbolic():
    def check_one_hot_formatter_symbolic(seed, max_labels, dtype, ncases):
        rng = numpy.random.RandomState(seed)
        fmt = OneHotFormatter(max_labels=max_labels, dtype=dtype)
        integer_labels = rng.random_integers(0, max_labels - 1, size=ncases)
        x = theano.tensor.vector(dtype='int64')
        y = fmt.theano_expr(x)
        f = theano.function([x], y)
        one_hot_labels = f(integer_labels)
        assert len(list(zip(*one_hot_labels.nonzero()))) == ncases
        assert one_hot_labels.dtype == dtype
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
        fmt.format(numpy.zeros((2, 3, 4), dtype='int32'))
    except ValueError:
        raised = True
    assert raised

    # Make sure an invalid ndim raises an error for theano_expr().
    raised = False
    try:
        fmt.theano_expr(theano.tensor.itensor3())
    except ValueError:
        raised = True
    assert raised


def test_one_hot_formatter_merge_simple():
    def check_one_hot_formatter(seed, max_labels, dtype, ncases, nmultis):
        rng = numpy.random.RandomState(seed)
        fmt = OneHotFormatter(max_labels=max_labels, dtype=dtype)
        integer_labels = rng.random_integers(
            0, max_labels - 1, size=ncases*nmultis
        ).reshape(ncases, nmultis)

        one_hot_labels = fmt.format(integer_labels, mode='merge')
        # n_ones was expected to be equal to ncases * nmultis if integer_labels
        # do not contain duplicated tags. (i.e., those labels like
        # [1, 2, 2, 3, 5, 6].) Because that we are not depreciating this kind
        # of duplicated labels, which allows different cases belong to
        # different number of classes, and those duplicated tags will only
        # activate one neuron in the k-hot representation, we need to use
        # numpy.unique() here to eliminate those duplications while counting
        # "1"s in the final k-hot representation.
        n_ones = numpy.concatenate([numpy.unique(l) for l in integer_labels])
        assert len(list(zip(*one_hot_labels.nonzero()))) == len(n_ones)
        for case, label in enumerate(integer_labels):
            assert numpy.sum(one_hot_labels[case, label]) == nmultis

    rng = numpy.random.RandomState(0)
    for seed, dtype in enumerate(all_types):
        yield (check_one_hot_formatter, seed, rng.random_integers(11, 30),
               dtype, rng.random_integers(1, 100),
               rng.random_integers(1, 10))


def test_out_compressed_one_hot():
    out, uniq = compressed_one_hot([2, 5, 3])
    assert_equal(out, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert_equal(uniq, [2, 3, 5])

    out, uniq = compressed_one_hot([2, 5])
    assert_equal(out, [[0], [1]])
    assert_equal(uniq, [2, 5])

    out, uniq = compressed_one_hot([2, 5], simplify_binary=False)
    assert_equal(out, [[1, 0], [0, 1]])
    assert_equal(uniq, [2, 5])
