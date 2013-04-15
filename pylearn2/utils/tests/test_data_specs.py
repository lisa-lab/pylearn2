"""Tests for compilation utilities."""
import theano.tensor as TT
from pylearn2.utils.data_specs import flatten_specs, \
        resolve_nested_structure_from_flat
from pylearn2.space import VectorSpace, \
        Conv2DSpace, \
        CompositeSpace

def safe_equal(a, b):
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        msg = 'Length of %s, %d not equal with length of %s, %d' % (
            str(a), len(a), str(b), len(b))
        assert len(a) == len(b), msg
        for elemA, elemB in zip(a,b):
            safe_equal(elemA, elemB)
    else:
        msg = '%s not equal with %s' %(str(a), str(b))
        assert a == b, msg

def test_flatten_specs():
    specs = [None, None ]
    rval = flatten_specs(specs)
    safe_equal(specs, rval)

    specs = [VectorSpace(dim=5), 'features']
    rval = flatten_specs(specs)
    safe_equal(specs, rval)

    specs = [CompositeSpace([
        VectorSpace(dim=5), VectorSpace(dim=2)]),
        ('features', 'features')]
    rval = flatten_specs(specs)
    safe_equal(specs, rval)

    specs = [CompositeSpace([
        VectorSpace(dim=5), VectorSpace(dim=5)]),
        ('features', 'targets')]
    rval = flatten_specs(specs)
    safe_equal(specs, rval)

    specs = [CompositeSpace([
        VectorSpace(dim=5), VectorSpace(dim=5)]),
        ('features', 'features')]
    rval = flatten_specs(specs)
    safe_equal(rval, (VectorSpace(dim=5), 'features'))

    specs = [CompositeSpace([
        VectorSpace(dim=5),
        CompositeSpace([VectorSpace(dim=9),
                       VectorSpace(dim=12)])]),
        ('features', ('features', 'targets'))]
    rval = flatten_specs(specs)
    safe_equal(rval,
              (CompositeSpace([
                   VectorSpace(dim=5),
                   VectorSpace(dim=9),
                   VectorSpace(dim=12)]),
               ('features', 'features', 'targets')))

def test_resolve_nested_structure():
    x1 = TT.matrix('x1')
    x2 = TT.matrix('x2')
    x3 = TT.matrix('x3')
    x4 = TT.matrix('x4')
    data = [x1, x2, x3, x4]
    flat_space = CompositeSpace(
        [VectorSpace(dim=3), VectorSpace(dim=10),
        VectorSpace(dim=7), VectorSpace(dim=9)])

    flat_source = ['features', 'target', 'features', 'features']
    flat = (flat_space, flat_source)
    nested = (VectorSpace(dim=10), 'target')
    rval = resolve_nested_structure_from_flat(data, nested, flat)
    safe_equal(rval, x2)


    nested = (CompositeSpace([
        VectorSpace(dim=3), VectorSpace(dim=9)]), ('features', 'features'))
    rval = resolve_nested_structure_from_flat(data, nested, flat)
    safe_equal(rval, [x1, x4])

    nested = (CompositeSpace([
        VectorSpace(dim=3),
        CompositeSpace([
            VectorSpace(dim=10),
            VectorSpace(dim=7)])]),
        ('features', ('target', 'features')))
    rval = resolve_nested_structure_from_flat(data, nested, flat)
    safe_equal(rval, [x1, [x2,x3]])


