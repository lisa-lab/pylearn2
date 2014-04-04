"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from pylearn2.datasets.dataset import Dataset
from pylearn2.space import CompositeSpace
from pylearn2.utils.data_specs import is_flat_specs


class TransformerDataset(Dataset):
    """
        A dataset that applies a transformation on the fly
        as examples are requested.
    """

    def __init__(self, raw, transformer, cpu_only = False,
            space_preserving=False):
        """
            .. todo::

                WRITEME properly

            Parameters
            ----------
            raw : pylearn2 Dataset
                Provides raw data
            transformer: pylearn2 Block
                To transform the data
        """
        self.__dict__.update(locals())
        del self.self

    def get_batch_design(self, batch_size, include_labels=False):
        """
        .. todo::

            WRITEME
        """
        raw = self.raw.get_batch_design(batch_size, include_labels)
        if include_labels:
            X, y = raw
        else:
            X = raw
        X = self.transformer.perform(X)
        if include_labels:
            return X, y
        return X

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return TransformerDataset(raw=self.raw.get_test_set(),
                transformer=self.transformer,
                cpu_only=self.cpu_only,
                space_preserving=self.space_preserving)


    def get_batch_topo(self, batch_size):
        """
        If the transformer has changed the space, we don't have a good
        idea of how to do topology in the new space.
        If the transformer just changes the values in the original space,
        we can have the raw dataset provide the topology.
        """
        X = self.get_batch_design(batch_size)
        if self.space_preserving:
            return self.raw.get_topological_view(X)
        return X.reshape(X.shape[0],X.shape[1],1,1)

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):
        """
        .. todo::

            WRITEME
        """
        # Build the right data_specs to query self.raw
        if data_specs is not None:
            assert is_flat_specs(data_specs)
            space, source = data_specs
            if not isinstance(source, tuple):
                source = (source,)
            if isinstance(space, CompositeSpace):
                space = tuple(space.components)
            else:
                space = (space,)

            # Put 'features' first, as this is what TransformerIterator
            # is expecting
            if 'features' not in source:
                # 'features is not needed, get things directly from
                # the original data
                raw_data_specs = data_specs
            else:
                feature_idx = source.index('features')
                if self.space_preserving:
                    # Ask self.raw for the data in the expected space,
                    # and self.transformer will operate in that space
                    feature_input_space = space[feature_idx]
                else:
                    # We need to ask the transformer what its input space is
                    feature_input_space = self.transformer.get_input_space()

                raw_space = CompositeSpace(
                                (feature_input_space,)
                                + space[:feature_idx]
                                + space[feature_idx + 1:])
                raw_source = (('features',)
                              + source[:feature_idx]
                              + source[feature_idx + 1:])
                raw_data_specs = (raw_space, raw_source)
        else:
            raw_data_specs = None

        raw_iterator = self.raw.iterator(mode=mode, batch_size=batch_size,
                num_batches=num_batches, topo=topo, targets=targets, rng=rng,
                data_specs=raw_data_specs, return_tuple=return_tuple)

        final_iterator = TransformerIterator(raw_iterator, self,
                                             data_specs=data_specs)

        return final_iterator

    def has_targets(self):
        """
        .. todo::

            WRITEME
        """
        return self.raw.has_targets()

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        if self.space_preserving:
            return self.raw.adjust_for_viewer(X)
        return X

    def get_weights_view(self, *args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        if self.space_preserving:
            return self.raw.get_weights_view(*args, **kwargs)
        raise NotImplementedError()

    def get_topological_view(self, *args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        if self.space_preserving:
            return self.raw.get_weights_view(*args, **kwargs)
        raise NotImplementedError()

    def adjust_to_be_viewed_with(self, *args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        return self.raw.adjust_to_be_viewed_with(*args, **kwargs)


class TransformerIterator(object):
    """
    .. todo::

        WRITEME
    """

    def __init__(self, raw_iterator, transformer_dataset, data_specs):
        """
        .. todo::

            WRITEME
        """
        self.raw_iterator = raw_iterator
        self.transformer_dataset = transformer_dataset
        self.stochastic = raw_iterator.stochastic
        self.uneven = raw_iterator.uneven
        self.data_specs = data_specs

    def __iter__(self):
        """
        .. todo::

            WRITEME
        """
        return self

    def next(self):
        """
        .. todo::

            WRITEME
        """
        raw_batch = self.raw_iterator.next()

        # Apply transformation on raw_batch, and format it
        # in the requested Space
        transformer = self.transformer_dataset.transformer
        out_space = self.data_specs[0]
        if isinstance(out_space, CompositeSpace):
            out_space = out_space.components[0]

        if self.transformer_dataset.space_preserving:
            # If the space is preserved, then raw_batch is already provided
            # in the requested space
            rval_space = out_space
        else:
            rval_space = transformer.get_output_space()

        def transform(X_batch):
            rval = transformer.perform(X_batch)
            if rval_space != out_space:
                rval = rval_space.np_format_as(rval, out_space)
            return rval

        if not isinstance(raw_batch, tuple):
            # Only one source, return_tuple is False
            rval = transform(raw_batch)
        else:
            # Apply the transformer only on the first element
            rval = (transform(raw_batch[0]),) + raw_batch[1:]

        return rval

    @property
    def num_examples(self):
        """
        .. todo::

            WRITEME
        """
        return self.raw_iterator.num_examples
