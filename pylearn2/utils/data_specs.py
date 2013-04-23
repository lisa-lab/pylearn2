from pylearn2.space import CompositeSpace, Space
from pylearn2.utils import safe_zip


class DataSpecsMapping(object):
    """
    Converts between nested tuples and non-redundant flattened ones.

    The mapping is built from data specifications, provided as a
    (space, sources) pair, where space can be a composite space (possibly
    of other composite spaces), and sources is a tuple of string identifiers
    or other sources. Both space and sources must have the same structure.
    """
    def __init__(self, data_specs):
        """Builds the internal mapping"""
        # Maps one elementary (not composite) data_specs pair to its index in
        # the flattened space
        # Not sure if this one should be a member, or passed as a parameter to
        # _fill_mapping. It might be useful to get the index of one data_specs
        # later, but if it is not, then we should remove it.
        self.specs_to_index = {}

        # Size of the flattened space
        self.n_unique_specs = 0

        # Builds the mapping
        space, source = data_specs
        self.spec_mapping = self._fill_mapping(space, source)

    def _fill_mapping(self, space, source):
        """Builds a nested tuple of integers representing the mapping"""
        if not isinstance(space, CompositeSpace):
            # Space is a simple Space, source should be a simple source
            if isinstance(source, tuple):
                source, = source

            # If (space, source) has not already been seen, insert it.
            # We need both the space and the source to match.
            if (space, source) in self.specs_to_index:
                spec_index = self.specs_to_index[(space, source)]
            else:
                spec_index = self.n_unique_specs
                self.specs_to_index[(space, source)] = spec_index
                self.n_unique_specs += 1

            return spec_index

        else:
            # Recursively fill the mapping, and return it
            spec_mapping = tuple(
                    self._fill_mapping(sub_space, sub_source)
                    for sub_space, sub_source in safe_zip(
                        space.components, source))

            return spec_mapping

    def _fill_flat(self, nested, mapping, rval):
        """Auxiliary recursive function used by self.flatten"""
        if isinstance(nested, CompositeSpace):
            nested = tuple(nested.components)
        if isinstance(mapping, int):
            # "nested" should actually be a single element
            idx = mapping
            if isinstance(nested, tuple):
                nested, = nested

            if rval[idx] is None:
                rval[idx] = nested
            else:
                assert rval[idx] == nested, ("This mapping was built "
                        "with the same element occurring more than once "
                        "in the nested representation, but current nested "
                        "sequence has different values (%s and %s) at "
                        "these positions." % (rval[idx], nested))
        else:
            for sub_nested, sub_mapping in safe_zip(nested, mapping):
                self._fill_flat(sub_nested, sub_mapping, rval)

    def flatten(self, nested):
        """
        Iterate jointly through nested and spec_mapping, returns a flat tuple.

        The integer in spec_mapping corresponding to each element in nested
        represents the index of that element in the returned sequence.
        If the original data_specs had duplicate elements at different places,
        then "nested" also have to have equal elements at these positions.
        "nested" can be a nested tuple, or composite space. If it is a
        composite space, a flattened composite space will be returned.
        """
        # Initialize the flatten returned value with Nones
        rval = [None] * self.n_unique_specs

        # Fill rval with the auxiliary function
        self._fill_flat(nested, self.spec_mapping, rval)

        assert None not in rval, ("This mapping is invalid, as it did not "
                "contain all numbers from 0 to %i (or None was in nested), "
                "nested: %s" % (self.n_unique_specs - 1, nested))

        if len(rval) == 1:
            return rval[0]
        if isinstance(nested, tuple):
            return tuple(rval)
        elif isinstance(nested, Space):
            return CompositeSpace(rval)

    def _make_nested_tuple(self, flat, mapping):
        """Auxiliary recursive function used by self.nest"""
        if isinstance(mapping, int):
            # We are at a leaf of the tree
            idx = mapping
            assert 0 <= idx < len(flat)
            return flat[idx]
        else:
            return tuple(
                    self._make_nested_tuple(flat, sub_mapping)
                    for sub_mapping in mapping)

    def _make_nested_space(self, flat, mapping):
        """Auxiliary recursive function used by self.nest"""
        if isinstance(mapping, int):
            # We are at a leaf of the tree
            idx = mapping
            if isinstance(flat, CompositeSpace):
                assert 0 <= idx < len(flat.components)
                return flat.components[idx]
            else:
                assert idx == 0
                return flat
        else:
            return CompositeSpace([
                    self._make_nested_space(flat, sub_mapping)
                    for sub_mapping in mapping])

    def nest(self, flat):
        """
        Iterate through spec_mapping, building a nested tuple from "flat".

        The length of "flat" should be equal to self.n_unique_specs.
        """
        if isinstance(flat, tuple):
            assert len(flat) == self.n_unique_specs
            return self._make_nested_tuple(flat, self.spec_mapping)
        elif isinstance(flat, Space):
            if isinstance(flat, CompositeSpace):
                assert len(flat.components) == self.n_unique_specs
            else:
                assert 1 == self.n_unique_specs
            return self._make_nested_space(flat, self.spec_mapping)
        else:
            # flat is not iterable, this is valid only if spec_mapping
            # is the integer 0.
            if self.spec_mapping == (0,):
                return flat

            raise TypeError("'flat' should be a Space, or tuple. "
                    "It is %s of type %s" % (flat, type(flat)))


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


def is_flat_space(space):
    """Returns True for elementary Spaces and non-nested CompositeSpaces"""
    if isinstance(space, CompositeSpace):
        for sub_space in space.components:
            if isinstance(sub_space, CompositeSpace):
                return False
    elif not isinstance(space, Space):
        raise TypeError("space is not a Space: %s (%s)"
                % (space, type(space)))
    return True


def is_flat_source(source):
    """Returns True for a string or a non-nested tuple of strings"""
    if isinstance(source, tuple):
        for sub_source in source:
            if isinstance(sub_source, tuple):
                return False
    elif not isinstance(source, str):
        raise TypeError("source should be a string or a non-nested tuple "
                "of strings: %s" % source)
    return True


def is_flat_specs(data_specs):
    return is_flat_space(data_specs[0]) and is_flat_source(data_specs[1])


def flatten_list(nested_list):
    """ Given a nested list return a flat version of said list
    """
    if not isinstance(nested_list, (list, tuple)):
        return [nested_list]
    rval = []
    for _elem in nested_list:
        elem = flatten_list(_elem)
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

    if isinstance(b_space, CompositeSpace):
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
