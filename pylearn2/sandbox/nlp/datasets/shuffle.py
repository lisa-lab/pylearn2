import os
import warnings

import numpy as np
import tables

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils.iteration import resolve_iterator_class


class HDF5Shuffle(DenseDesignMatrix):
    """
    Loads n-grams from a matrix, shuffles them and creates targets

    Parameters
    ----------
    path : string
        The location of the HDF5 to load
    node : string, optional
        The node to load data from, if not given, node is assumed
        to have the same name as the file
    n : int, optional
        Size of the n-grams, should be less or equal than the HDF5 data
    start : int
        First n-gram (0-indexed)
    stop : int
        Last n-gram (0-index)
    shuffle : bool, optional
        Whether to shuffle the data
    permutation : int
        Number of permutations to perform, defaults to 1
    """
    def __init__(self, path, node=None, n=None, shuffle=True, permutation=1,
                 start=0, stop=None):
        if node is None:
            head, tail = os.path.split(path)
            root, ext = os.path.splitext(tail)
            node = root
        assert permutation == 1, "No support for more than 1 permutation"

        cache_size = 100000
        totalInputSize = stop-start
        # TEST

        for i in range(totalInputSize/cache_size):
            s = start+i*cache_size
            e = s+cache_size
            with tables.open_file(path) as f:
                print "Loading n-grams..."
                node = f.get_node('/' + node)

                X = node[s:e]
                if n is None:
                    n = X.shape[1]
            print "Loaded %d n-grams" % len(X)
            X = X[X.all(1), :n]
            print "After filtering: %d n-grams" % len(X)

            print "Creating targets"
            swaps = np.random.randint(0, n - 1, len(X))
            y = np.zeros((len(X), n - 1))
            y[np.arange(len(X)), swaps] = 1
            print "Performing permutations...",
            for sample, swap in enumerate(swaps):
                X[sample, swap], X[sample, swap + 1] = \
                                                  X[sample, swap + 1], X[sample, swap]
            print "Done"
            super(HDF5Shuffle, self).__init__(
                X=X, y=y, X_labels=15000
            )

            if shuffle:
                warnings.warn("Note that the samples are only "
                              "shuffled when the iterator method is used to "
                              "retrieve them.")
                self._iter_subset_class = resolve_iterator_class(
                    'shuffled_sequential'
                )
