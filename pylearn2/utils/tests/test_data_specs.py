"""Tests for compilation utilities."""
import theano.tensor as TT
from nose.tools import assert_raises
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.space import VectorSpace, \
        CompositeSpace


def assert_equal(a, b):
    if isinstance(a, tuple) and isinstance(b, tuple):
        msg = 'Length of %s, %d not equal with length of %s, %d' % (
            str(a), len(a), str(b), len(b))
        assert len(a) == len(b), msg
        for elemA, elemB in zip(a, b):
            assert_equal(elemA, elemB)
    else:
        msg = '%s not equal to %s' % (str(a), str(b))
        assert a == b, msg


def test_flatten_specs():
    for space, source, flat_space, flat_source in [
            #(None, None),
            (VectorSpace(dim=5), 'features', VectorSpace(dim=5), 'features'),
            (CompositeSpace([VectorSpace(dim=5), VectorSpace(dim=2)]),
                ('features', 'features'),
                CompositeSpace([VectorSpace(dim=5), VectorSpace(dim=2)]),
                ('features', 'features')),
            (CompositeSpace([VectorSpace(dim=5), VectorSpace(dim=5)]),
                ('features', 'targets'),
                CompositeSpace([VectorSpace(dim=5), VectorSpace(dim=5)]),
                ('features', 'targets')),
            (CompositeSpace([VectorSpace(dim=5), VectorSpace(dim=5)]),
                ('features', 'features'),
                VectorSpace(dim=5),
                'features'),
            (CompositeSpace([VectorSpace(dim=5),
                             CompositeSpace([VectorSpace(dim=9),
                                             VectorSpace(dim=12)])]),
                ('features', ('features', 'targets')),
                CompositeSpace([VectorSpace(dim=5),
                                VectorSpace(dim=9),
                                VectorSpace(dim=12)]),
                ('features', 'features', 'targets')),
            (CompositeSpace([VectorSpace(dim=5),
                             VectorSpace(dim=9),
                             VectorSpace(dim=12)]),
                ('features', 'features', 'targets'),
                CompositeSpace([VectorSpace(dim=5),
                                VectorSpace(dim=9),
                                VectorSpace(dim=12)]),
                ('features', 'features', 'targets'))
            ]:

        mapping = DataSpecsMapping((space, source))
        rval = (mapping.flatten(space), mapping.flatten(source))
        assert_equal((flat_space, flat_source), rval)


def test_nest_specs():
    x1 = TT.matrix('x1')
    x2 = TT.matrix('x2')
    x3 = TT.matrix('x3')
    x4 = TT.matrix('x4')

    for nested_space, nested_source, nested_data in [
            (VectorSpace(dim=10), 'target', x2),
            (CompositeSpace([VectorSpace(dim=3), VectorSpace(dim=9)]),
                ('features', 'features'),
                (x1, x4)),
            (CompositeSpace([VectorSpace(dim=3),
                             CompositeSpace([VectorSpace(dim=10),
                                             VectorSpace(dim=7)])]),
                ('features', ('target', 'features')),
                (x1, (x2, x3))),
            ]:

        mapping = DataSpecsMapping((nested_space, nested_source))
        flat_space = mapping.flatten(nested_space)
        flat_source = mapping.flatten(nested_source)
        flat_data = mapping.flatten(nested_data)

        renested_space = mapping.nest(flat_space)
        renested_source = mapping.nest(flat_source)
        renested_data = mapping.nest(flat_data)

        assert_equal(renested_space, nested_space)
        assert_equal(renested_source, nested_source)
        assert_equal(renested_data, nested_data)

def test_input_validation():
    """
    DataSpecsMapping should raise errors if inputs
    are not formatted as data specs.
    """
    assert_raises(ValueError,
                  DataSpecsMapping,
                  (VectorSpace(dim=10), ('features', 'targets')))
    assert_raises(AssertionError,
                  DataSpecsMapping,
                  (('features', 'targets'), VectorSpace(dim=10)))
