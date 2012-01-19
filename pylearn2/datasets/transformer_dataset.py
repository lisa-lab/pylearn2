from pylearn2.datasets.dataset import Dataset

class TransformerDataset(Dataset):
    """
        A dataset that applies a transformation on the fly
        as examples are requested.
    """
    def __init__(self, raw, transformer):
        """
            raw: a pylearn2 Dataset that provides raw data
            transformer: a pylearn2 Block to transform the data
        """
        self.raw = raw
        self.transformer = transformer

    def get_batch_design(self, batch_size):
        X = self.raw.get_batch_design(batch_size)
        X = self.transformer.perform(X)
        return X
