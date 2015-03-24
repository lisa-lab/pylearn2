"""
Utilities for working with data format specifications.

See :ref:`data_specs` for a high level overview of the relevant concepts.
"""
from collections import Sized
from pylearn2.space import CompositeSpace, NullSpace, Space
from pylearn2.utils import safe_zip


class DataSpecsMapping(object):
    """
    Converts between nested tuples and non-redundant flattened ones.

    The mapping is built from data specifications, provided as a
    (space, sources) pair, where space can be a composite space (possibly
    of other composite spaces), and sources is a tuple of string identifiers
    or other sources. Both space and sources must have the same structure.

    Parameters
    ----------
    data_specs : WRITEME
    WRITEME

    Attributes
    ----------
    specs_to_index : dict
    Maps one elementary (not composite) data_specs pair to its
    index in the flattened space.  Not sure if this one should
    be a member, or passed as a parameter to _fill_mapping. It
    might be us
    """
    #might be useful to get the index of one data_specs later
    #but if it is not, then we should remove it.
    def __init__(self, data_specs):
        self.specs_to_index = {}

        # Size of the flattened space
        self.n_unique_specs = 0

        # Builds the mapping
        space, source = data_specs
        assert isinstance(space, Space), 'Given space: ' + str(space) + \
                                         ' was not a instance of Space.'
        self.spec_mapping = self._fill_mapping(space, source)

    def _fill_mapping(self, space, source):
        """
        Builds a nested tuple of integers representing the mapping

        Parameters
        ----------
        space : WRITEME
        source : WRITEME

        Returns
        -------
        WRITEME
        """
        if isinstance(space, NullSpace):
            # This Space does not contain any data, and should not
            # be mapped to anything
            assert source == ''
            return None

        elif not isinstance(space, CompositeSpace):
            # Space is a simple Space, source should be a simple source
            if isinstance(source, (tuple, list)):
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
        """
        Auxiliary recursive function used by self.flatten

        Parameters
        ----------
        nested : WRITEME
        mapping : WRITEME
        rval : WRITEME

        Returns
        -------
        WRITEME
        """
        if isinstance(nested, CompositeSpace):
            nested = tuple(nested.components)

        if mapping is None:
            # The corresponding Space was a NullSpace, which does
            # not correspond to actual data, so nested should evaluate
            # to False, and should not be included in the flattened version
            if not isinstance(nested, NullSpace):
                assert not nested, ("The following element is mapped to "
                    "NullSpace, so it should evaluate to False (for instance, "
                    "None, an empty string or an empty tuple), but is %s"
                    % nested)
            return

        if isinstance(mapping, int):
            # "nested" should actually be a single element
            idx = mapping
            if isinstance(nested, (tuple, list)):
                if len(nested) != 1:
                    raise ValueError("When mapping is an int, we expect "
                            "nested to be a single element. But mapping is "
                            + str(mapping) + " and nested is a tuple of "
                            "length " + str(len(nested)))
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

    def flatten(self, nested, return_tuple=False):
        """
        Iterate jointly through nested and spec_mapping, returns a flat tuple.

        The integer in spec_mapping corresponding to each element in nested
        represents the index of that element in the returned sequence.
        If the original data_specs had duplicate elements at different places,
        then "nested" also have to have equal elements at these positions.
        "nested" can be a nested tuple, or composite space. If it is a
        composite space, a flattened composite space will be returned.

        If `return_tuple` is True, a tuple is always returned (tuple of
        non-composite Spaces if nested is a Space, empty tuple if all
        Spaces are NullSpaces, length-1 tuple if there is only one
        non-composite Space, etc.).

        Parameters
        ----------
        nested : WRITEME
        return_tuple : WRITEME

        Returns
        -------
        WRITEME
        """
        # Initialize the flatten returned value with Nones
        rval = [None] * self.n_unique_specs

        # Fill rval with the auxiliary function
        self._fill_flat(nested, self.spec_mapping, rval)

        assert None not in rval, ("This mapping is invalid, as it did not "
                "contain all numbers from 0 to %i (or None was in nested), "
                "nested: %s" % (self.n_unique_specs - 1, nested))

        if return_tuple:
            return tuple(rval)

        # else, return something close to the type of nested
        if len(rval) == 1:
            return rval[0]
        if isinstance(nested, (tuple, list)):
            return tuple(rval)
        elif isinstance(nested, Space):
            return CompositeSpace(rval)

    def _make_nested_tuple(self, flat, mapping):
        """
        Auxiliary recursive function used by self.nest

        Parameters
        ----------
        flat : WRITEME
        mapping : WRITEME

        Returns
        -------
        WRITEME
        """
        if mapping is None:
            # The corresponding space was a NullSpace,
            # and there is no corresponding value in flat,
            # we use None as a placeholder
            return None
        if isinstance(mapping, int):
            # We are at a leaf of the tree
            idx = mapping
            if isinstance(flat, (tuple, list)):
                assert 0 <= idx < len(flat)
                return flat[idx]
            else:
                assert idx == 0
                return flat
        else:
            return tuple(
                    self._make_nested_tuple(flat, sub_mapping)
                    for sub_mapping in mapping)

    def _make_nested_space(self, flat, mapping):
        """
        Auxiliary recursive function used by self.nest

        Parameters
        ----------
        flat : WRITEME
        mapping : WRITEME

        Returns
        -------
        WRITEME
        """
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

        Parameters
        ----------
        flat : Space or tuple
            WRITEME

        Returns
        -------
        WRITEME
        """
        if isinstance(flat, Space):
            if isinstance(flat, CompositeSpace):
                assert len(flat.components) == self.n_unique_specs
            else:
                assert self.n_unique_specs == 1
            return self._make_nested_space(flat, self.spec_mapping)
        else:
            if isinstance(flat, (list, tuple)):
                assert len(flat) == self.n_unique_specs
            else:
                # flat is not iterable, this is valid only if spec_mapping
                # contains only 0's, that is, when self.n_unique_specs == 1
                assert self.n_unique_specs == 1
            return self._make_nested_tuple(flat, self.spec_mapping)


def is_flat_space(space):
    """
    Returns True for elementary Spaces and non-nested CompositeSpaces

    Parameters
    ----------
    space : WRITEME

    Returns
    -------
    WRITEME
    """
    if isinstance(space, CompositeSpace):
        for sub_space in space.components:
            if isinstance(sub_space, CompositeSpace):
                return False
    elif not isinstance(space, Space):
        raise TypeError("space is not a Space: %s (%s)"
                % (space, type(space)))
    return True


def is_flat_source(source):
    """
    Returns True for a string or a non-nested tuple of strings

    Parameters
    ----------
    source : WRITEME

    Returns
    -------
    WRITEME
    """
    if isinstance(source, (tuple, list)):
        for sub_source in source:
            if isinstance(sub_source, (tuple, list)):
                return False
    elif not isinstance(source, str):
        raise TypeError("source should be a string or a non-nested tuple/list "
                "of strings: %s" % source)
    return True


def is_flat_specs(data_specs):
    """
    .. todo::

        WRITEME
    """
    return is_flat_space(data_specs[0]) and is_flat_source(data_specs[1])
