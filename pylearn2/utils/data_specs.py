from pylearn2.space import CompositeSpace
from pylearn2.utils import safe_zip

def resolve_nested_structure_from_flat(data, nested, flat):
    """
    :param data: flat list of theano_like variables
    :param nested: a pair (space, source); the space and source can have a
                    nested structure
    :param flat: a pair (space, source); both have a flat structure
    """
    rval = []
    (targ_space, targ_source) = nested
    (inp_space, inp_source) = flat
    if isinstance(inp_space, CompositeSpace):
        zipped_flat = safe_zip(inp_space.components, inp_source)
    else:
        zipped_flat = [(inp_space, inp_source)]
        data = [data]

    if isinstance(targ_space, CompositeSpace):
        for space, source in safe_zip(targ_space.components,
                                      targ_source):
            rval.append(resolve_nested_structure_from_flat(
                data,
                (space, source),
                flat))
        return rval
    else:
        assert nested in zipped_flat
        idx = zipped_flat.index(nested)
        return data[idx]

def flatten_list(nested_list):
    """ Given a nested list return a flat version of said list
    """
    if not isinstance(nested_list, (list, tuple)):
        return [nested_list]
    rval = []
    for _elem in nested_list:
        elem = flatten_list(elem)
        rval += [x for x in elem if x not in rval]
    return elem


def flatten_specs(data_specs):
    """ Given data specifications ``data_specs``, i.e. a pair (space, sources),
    where space can be a CompositeSpace, re-write it in a flat manner by
    removing any instance of CompositeSpace or nested tuple. The final flat
    list of space, source pairs are unique.
    """
    def recursively_solve(specs):
        space, source = specs
        final_space = []
        final_source = []
        if isinstance(space, CompositeSpace):
            assert type(source) in (list, tuple)
            for (_elem_space, _elem_source) in zip(space.components, source):
                elem_space, elem_source = recursively_solve((_elem_space,
                                                            _elem_source))
                if type(elem_space) in (list, tuple):
                    final_space += elem_space
                    final_source += elem_source
                elif elem_space is not None:
                    final_space.append(elem_space)
                    final_source.append(elem_source)
            pairs = safe_zip(final_space, final_source)
            unique_pairs = []
            for pair in pairs:
                if pair not in unique_pairs:
                    unique_pairs.append(pair)
            return ([x[0] for x in unique_pairs],
                    [x[1] for x in unique_pairs])
        else:
            return space, source

    space, source = recursively_solve(data_specs)
    if isinstance(space, (list, tuple)):
        if len(space) > 1:
            space = CompositeSpace(space)
        else:
            space = space[0]
            source = source[0]
    return (space, source)

def flat_specs_union(A,B):
    """ Compute the union of two flat data specs """
    (a_space, a_source) = A
    (b_space, b_source) = B
    if isinstance(a_space, CompositeSpace):
        zipped_A = safe_zip(a_space.components, a_source)
    else:
        zipped_A = (a_space, a_source)

    if isinstance(b_space, CompositeSapce):
        zipped_B = safe_zip(b_space.components, b_source)
    else:
        zipped_B = (b_space, b_source)
    rval = zipped_A
    rval += [x for x in zipped_B if x not in rval]
    targ_space = [x[0] for x in rval]
    targ_source = [x[1] for x in rval]
    if len(targ_space) > 1:
        return (CompositeSpace(targ_space), targ_source)
    else:
        return (targ_space[0], targ_source[0])
