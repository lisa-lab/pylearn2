"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.expr.sampling import SampleBernoulli


class Binarizer(TransformerDataset):

    """
    A TransformerDataset that takes examples with features in the interval
    [0,1], and uses these as Bernoulli parameters to sample examples
    with features in {0,1}.

    Parameters
    ----------
    raw : pylearn2 Dataset
        It must provide examples with features in the interval [0,1].
    seed : integer or list of integers, optional
        The seed passed to MRG_RandomStreams to make the Bernoulli
        samples. If not specified, all class instances default to
        the same seed so two instances can be run synchronized side
        by side.
    """

    def __init__(self, raw, seed=None):
        transformer = SampleBernoulli(seed=seed)

        super(Binarizer, self).__init__(
            raw, transformer, space_preserving=True)

    def get_design_matrix(self, topo=None):
        """
        .. todo::

            WRITEME
        """
        if topo is not None:
            return self.raw.get_design_matrix(topo)
        X = self.raw.get_design_matrix()
        return self.transformer.perform(X)
