__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
from pylearn2.datasets.dataset import Dataset

class TransformerDataset(Dataset):
    """
        A dataset that applies a transformation on the fly
        as examples are requested.
    """
    def __init__(self, raw, transformer, cpu_only = False,
            space_preserving=False):
        """
            raw: a pylearn2 Dataset that provides raw data
            transformer: a pylearn2 Block to transform the data
        """
        self.__dict__.update(locals())
        del self.self

    def get_batch_design(self, batch_size):
        X = self.raw.get_batch_design(batch_size)
        X = self.transformer.perform(X)
        return X

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
                 topo=None, targets=None, rng=None):

        raw_iterator = self.raw.iterator(mode, batch_size, num_batches, topo, targets, rng)

        final_iterator = TransformerIterator(raw_iterator, self)

        return final_iterator

    def has_targets(self):
        return self.raw.y is not None

    def adjust_for_viewer(self, X):
        if self.space_preserving:
            return self.raw.adjust_for_viewer(X)
        return X

    def get_weights_view(self, *args, **kwargs):
        if self.space_preserving:
            return self.raw.get_weights_view(*args, **kwargs)
        raise NotImplementedError()


class TransformerIterator(object):

    def __init__(self, raw_iterator, transformer_dataset):
        self.raw_iterator = raw_iterator
        self.transformer_dataset = transformer_dataset
        self.stochastic = raw_iterator.stochastic
        self.uneven = raw_iterator.uneven

    def __iter__(self):
        return self

    def next(self):

        raw_batch = self.raw_iterator.next()

        if self.raw_iterator._targets:
            rval = (self.transformer_dataset.transformer.perform(raw_batch[0]), raw_batch[1])
        else:
            rval = self.transformer_dataset.transformer.perform(raw_batch)

        return rval

    @property
    def num_examples(self):
        return self.raw_iterator.num_examples
