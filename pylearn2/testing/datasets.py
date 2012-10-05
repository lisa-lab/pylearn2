""" Simple datasets to be used for unit tests. """
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class ArangeDataset(DenseDesignMatrix):
    """ A dataset where example i is just the number i.
    Makes it easy to track which sets of examples are visited."""

    def __init__(self, num_examples):
        X = np.zeros((num_examples,1))
        X[:,0] = np.arange(num_examples)
        super(ArangeDataset, self).__init__(X)
