"""
Several tilities to evaluate an ALC on the dataset, to iterate over
minibatches from a dataset, or to merge three data with given proportions
"""
# Standard library imports
import os
from itertools import repeat

# Third-party imports
import numpy
from matplotlib import pyplot

# Local imports
from utils.viz3d import do_3d_scatter
from .utlc import get_constant, sharedX

##################################################
# 3D Visualization
##################################################

def save_plot(repr, path, name="figure.pdf", title="features"):
    # TODO : Maybe run a PCA if shape[1] > 3
    assert repr.get_value(borrow=True).shape[1] == 3

    # Take the first 3 columns
    x, y, z = repr.get_value(borrow=True).T
    do_3d_scatter(x, y, z)

    # Save the produces figure
    filename = os.path.join(path, name)
    pyplot.savefig(filename, format="pdf")
    print '... figure saved: %s' % filename

##################################################
# Features or examples filtering
##################################################

def filter_labels(train, label):
    """ Filter examples of train for which we have labels """
    # Examples for which any label is set
    condition = label.any(axis=1)

    # Compress train and label arrays according to condition
    def aux(var):
        return var.get_value(borrow=True).compress(condition, axis=0)

    return (aux(train), aux(label))

##################################################
# Iterator object for minibatches of datasets
##################################################

class BatchIterator(object):
    """
    Builds an iterator object that can be used to go through the minibatches
    of a dataset, with respect to the given proportions in conf
    """
    def __init__(self, dataset, set_proba, batch_size, seed=300):
        # Local shortcuts for array operations
        flo = numpy.floor
        sub = numpy.subtract
        mul = numpy.multiply
        div = numpy.divide
        mod = numpy.mod

        # Record external parameters
        self.batch_size = batch_size
        self.dataset = dataset

        # Compute maximum number of samples for one loop
        set_sizes = [get_constant(data.shape[0]) for data in dataset]
        set_batch = [float(self.batch_size) for i in xrange(3)]
        set_range = div(mul(set_proba, set_sizes), set_batch)
        set_range = map(int, numpy.ceil(set_range))

        # Upper bounds for each minibatch indexes
        set_limit = numpy.ceil(numpy.divide(set_sizes, set_batch))
        self.limit = map(int, set_limit)

        # Number of rows in the resulting union
        set_tsign = sub(set_limit, flo(div(set_sizes, set_batch)))
        set_tsize = mul(set_tsign, flo(div(set_range, set_limit)))

        l_trun = mul(flo(div(set_range, set_limit)), mod(set_sizes, set_batch))
        l_full = mul(sub(set_range, set_tsize), set_batch)

        self.length = sum(l_full) + sum(l_trun)

        # Random number generation using a permutation
        index_tab = []
        for i in xrange(3):
            index_tab.extend(repeat(i, set_range[i]))

        # Use a deterministic seed
        self.seed = seed
        rng = numpy.random.RandomState(seed=self.seed)
        self.permut = rng.permutation(index_tab)

    def __iter__(self):
        """ Generator function to iterate through all minibatches """
        counter = [0, 0, 0]
        for chosen in self.permut:
            # Retrieve minibatch from chosen set
            index = counter[chosen]
            minibatch = self.dataset[chosen].get_value(borrow=True)[
                index * self.batch_size:(index + 1) * self.batch_size
            ]
            # Increment the related counter
            counter[chosen] = (counter[chosen] + 1) % self.limit[chosen]
            # Return the computed minibatch
            yield minibatch

    def __len__(self):
        """ Return length of the weighted union """
        return self.length

    def by_index(self):
        """ Same generator as __iter__, but yield only the chosen indexes """
        counter = [0, 0, 0]
        for chosen in self.permut:
            index = counter[chosen]
            counter[chosen] = (counter[chosen] + 1) % self.limit[chosen]
            yield chosen, index

    def by_subtensor(self):
        """ Generator function to iterate through all minibatches subtensors """
        counter = [0, 0, 0]
        for chosen in self.permut:
            # Retrieve minibatch from chosen set
            index = counter[chosen]
            minibatch = self.dataset[chosen][
                index * self.batch_size:(index + 1) * self.batch_size
            ]
            # Increment the related counter
            counter[chosen] = (counter[chosen] + 1) % self.limit[chosen]
            # Return the computed minibatch
            yield minibatch


##################################################
# Miscellaneous
##################################################

def nonzero_features(data, all_subsets=True):
    """
    Get features for which there are nonzero entries in the data.

    Note: I would return a mask (bool array) here, but scipy.sparse doesn't
    appear to fully support advanced indexing.

    Parameters
    ----------
    data : list of 3 ndarray objects
        List of data matrices, each with the same number of features.
    all_subsets : bool
        If true, discard features not found in any subset; if false, discard
        features found in neither valid nor test subsets.

    Returns
    -------
    indices : ndarray object
        Indices of nonzero features.
    """

    if not all_subsets:
        data = data[1:]

    # Assumes all values are >0, which is the case for all sparse datasets.
    masks = numpy.array([subset.sum(axis=0) for subset in data]).squeeze()
    nz_feats = masks.prod(axis=0).nonzero()[0]

    return nz_feats

def blend(dataset, set_proba, **kwargs):
    """
    Randomized blending of datasets in data according to parameters in conf
    """
    iterator = BatchIterator(dataset, set_proba, 1, **kwargs)
    nrow = len(iterator)
    ncol = get_constant(dataset[0].shape[1])
    array = numpy.empty((nrow, ncol), dataset[0].dtype)
    row = 0
    for chosen, index in iterator.by_index():
        array[row] = dataset[chosen].get_value(borrow=True)[index]
        row += 1

    return sharedX(array, borrow=True)
