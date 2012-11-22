__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.expr.sampling import SampleBernoulli

class Binarizer(TransformerDataset):
    """
        A TransformerDataset that takes examples with features in the interval
        [0,1], and uses these as Bernoulli parameters to sample examples
        with features in {0,1}.
    """
    def __init__(self, raw):
        """
            raw: a pylearn2 Dataset that provides examples with features
                in the interval [0, 1]
        """

        transformer = SampleBernoulli()

        super(Binarizer, self).__init__(raw, transformer, space_preserving=True)

