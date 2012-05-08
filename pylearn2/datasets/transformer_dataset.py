from pylearn2.datasets.dataset import Dataset

class TransformerDataset(Dataset):
    """
        A dataset that applies a transformation on the fly
        as examples are requested.
    """
    def __init__(self, raw, transformer, cpu_only = False):
        """
            raw: a pylearn2 Dataset that provides raw data
            transformer: a pylearn2 Block to transform the data
        """
        if raw is None:
            raise ValueError("raw data can't be None.")
        if transformer is None:
            raise ValueError("pylearn2 transformer block must not be None.")

        self.raw = raw
        self.transformer = transformer
        self.transformer.cpu_only = cpu_only

    def get_batch_design(self, batch_size):
        X = self.raw.get_batch_design(batch_size)
        X = self.transformer.perform(X)
        return X

    def get_batch_topo(self, batch_size):
        """ there's no concept of a topology-aware
        transformation right now so we just treat the
        dataset as consisting of big 1D images
        this is kind of a hack, long term solution is
        to make topo pipeline support having 0 topological
        dimensions (right now I believe it only supports 2,
        it should support N >= 0)"""
        X = self.get_batch_design(batch_size)
        return X.reshape(X.shape[0],X.shape[1],1,1)

    def set_iteration_scheme(self, mode=None, batch_size=None,
                             num_batches=None, topo=False, targets=False):
        self.raw.set_iteration_scheme(mode, batch_size, num_batches, topo, targets)


    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None):

        raw_iterator = self.raw.iterator(mode, batch_size, num_batches, topo, targets, rng)

        final_iterator = TransformerIterator(raw_iterator, self)
        return final_iterator

class TransformerIterator(object):

    def __init__(self, raw_iterator, transformer_dataset):
        self.raw_iterator = raw_iterator
        self.transformer_dataset = transformer_dataset

    def __iter__(self):
        return self

    def next(self):
        raw_batch = self.raw_iterator.next()
        
        if self.raw_iterator._targets:
            rval = (self.transformer_dataset.transformer.perform(raw_batch[0]), raw_batch[1])
        else:
            rval = self.transformer_dataset.transformer.perform(raw_batch)

        return rval
