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
