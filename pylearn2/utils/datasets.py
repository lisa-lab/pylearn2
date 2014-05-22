"""
Several utilities to evaluate an ALC on the dataset, to iterate over
minibatches from a dataset, or to merge three data with given proportions
"""
# Standard library imports
import logging
import os
import functools
from itertools import repeat
import warnings

# Third-party imports
import numpy
import scipy
import theano
try:
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    warnings.warn("Could not import some dependencies.")

# Local imports
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_np_rng


logger = logging.getLogger(__name__)


##################################################
# 3D Visualization
##################################################

def do_3d_scatter(x, y, z, figno=None, title=None):
    """
    Generate a 3D scatterplot figure and optionally give it a title.

    Parameters
    ----------
    x : WRITEME
    y : WRITEME
    z : WRITEME
    figno : WRITEME
    title : WRITEME
    """
    fig = pyplot.figure(figno)
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    pyplot.suptitle(title)

def save_plot(repr, path, name="figure.pdf", title="features"):
    """
    .. todo::

        WRITEME
    """
    # TODO : Maybe run a PCA if shape[1] > 3
    assert repr.get_value(borrow=True).shape[1] == 3

    # Take the first 3 columns
    x, y, z = repr.get_value(borrow=True).T
    do_3d_scatter(x, y, z)

    # Save the produces figure
    filename = os.path.join(path, name)
    pyplot.savefig(filename, format="pdf")
    logger.info('... figure saved: {0}'.format(filename))

##################################################
# Features or examples filtering
##################################################

def filter_labels(train, label, classes=None):
    """
    Filter examples of train for which we have labels

    Parameters
    ----------
    train : WRITEME
    label : WRITEME
    classes : WRITEME

    Returns
    -------
    WRITEME
    """
    if isinstance(train, theano.tensor.sharedvar.SharedVariable):
        train = train.get_value(borrow=True)
    if isinstance(label, theano.tensor.sharedvar.SharedVariable):
        label = label.get_value(borrow=True)

    if not (isinstance(train, numpy.ndarray) or scipy.sparse.issparse(train)):
        raise TypeError('train must be a numpy array, a scipy sparse matrix,'
                        ' or a theano shared array')

    # Examples for which any label is set
    if classes is not None:
        label = label[:, classes]

    # Special case for sparse matrices
    if scipy.sparse.issparse(train):
        idx = label.sum(axis=1).nonzero()[0]
        return (train[idx], label[idx])

    # Compress train and label arrays according to condition
    condition = label.any(axis=1)
    return tuple(var.compress(condition, axis=0) for var in (train, label))

def nonzero_features(data, combine=None):
    """
    Get features for which there are nonzero entries in the data.

    Parameters
    ----------
    data : list of matrices
        List of data matrices, either in sparse format or not.
        They must have the same number of features (column number).
    combine : function, optional
        A function to combine elementwise which features to keep.
        Default keeps the intersection of each non-zero columns.

    Returns
    -------
    indices : ndarray object
        Indices of the nonzero features.

    Notes
    -----
    I would return a mask (bool array) here, but scipy.sparse doesn't appear to
    fully support advanced indexing.
    """

    if combine is None:
        combine = functools.partial(reduce, numpy.logical_and)

    # Assumes all values are >0, which is the case for all sparse datasets.
    masks = numpy.asarray([subset.sum(axis=0) for subset in data]).squeeze()
    nz_feats = combine(masks).nonzero()[0]

    return nz_feats


# TODO: Is this a duplicate?
def filter_nonzero(data, combine=None):
    """
    Filter non-zero features of data according to a certain combining function

    Parameters
    ----------
    data : list of matrices
        List of data matrices, either in sparse format or not.
        They must have the same number of features (column number).
    combine : function
        A function to combine elementwise which features to keep.
        Default keeps the intersection of each non-zero columns.

    Returns
    -------
    indices : ndarray object
        Indices of the nonzero features.
    """

    nz_feats = nonzero_features(data, combine)

    return [set[:, nz_feats] for set in data]

##################################################
# Iterator object for minibatches of datasets
##################################################

class BatchIterator(object):
    """
    Builds an iterator object that can be used to go through the minibatches
    of a dataset, with respect to the given proportions in conf

    Parameters
    ----------
    dataset : WRITEME
    set_proba : WRITEME
    batch_size : WRITEME
    seed : WRITEME
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
        if (isinstance(dataset[0], theano.Variable)):
            self.dataset = [set.get_value(borrow=True) for set in dataset]
        else:
            self.dataset = dataset

        # Compute maximum number of samples for one loop
        set_sizes = [set.shape[0] for set in self.dataset]
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
        rng = make_np_rng(seed, which_method="permutation")
        self.permut = rng.permutation(index_tab)

    def __iter__(self):
        """Generator function to iterate through all minibatches"""
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

    def __len__(self):
        """Return length of the weighted union"""
        return self.length

    def by_index(self):
        """Same generator as __iter__, but yield only the chosen indexes"""
        counter = [0, 0, 0]
        for chosen in self.permut:
            index = counter[chosen]
            counter[chosen] = (counter[chosen] + 1) % self.limit[chosen]
            yield chosen, index

##################################################
# Miscellaneous
##################################################


def blend(dataset, set_proba, **kwargs):
    """
    Randomized blending of datasets in data according to parameters in conf

    .. note:: pylearn2.utils.datasets.blend is deprecated and will be
              removed on or after 13 August 2014.

    Parameters
    ----------
    set_proba : WRITEME
    kwargs : WRITEME

    Returns
    -------
    WRITEME
    """
    warnings.warn("pylearn2.utils.datasets.blend is deprecated"
                  "and will be removed on or after 13 August 2014.",
                  stacklevel=2)
    iterator = BatchIterator(dataset, set_proba, 1, **kwargs)
    nrow = len(iterator)
    if (isinstance(dataset[0], theano.Variable)):
        ncol = dataset[0].get_value().shape[1]
    else:
        ncol = dataset[0].shape[1]
    if (scipy.sparse.issparse(dataset[0])):
        # Special case: the dataset is sparse
        blocks = [[batch] for batch in iterator]
        return scipy.sparse.bmat(blocks, 'csr')

    else:
        # Normal case: the dataset is dense
        row = 0
        array = numpy.empty((nrow, ncol), dataset[0].dtype)
        for batch in iterator:
            array[row] = batch
            row += 1

        return sharedX(array, borrow=True)

def minibatch_map(fn, batch_size, input_data, output_data=None,
                  output_width=None):
    """
    Apply a function on input_data, one minibatch at a time.

    Storage for the output can be provided. If it is the case,
    it should have appropriate size.

    If output_data is not provided, then output_width should be specified.

    Parameters
    ----------
    fn : WRITEME
    batch_size : WRITEME
    input_data : WRITEME
    output_data : WRITEME
    output_width : WRITEME

    Returns
    -------
    WRITEME
    """

    if output_width is None:
        if output_data is None:
            raise ValueError('output_data or output_width should be provided')

        output_width = output_data.shape[1]

    output_length = input_data.shape[0]
    if output_data is None:
        output_data = numpy.empty((output_length, output_width))
    else:
        assert output_data.shape[0] == input_data.shape[0], ('output_data '
                'should have the same length as input_data',
                output_data.shape[0], input_data.shape[0])

    for i in xrange(0, output_length, batch_size):
        output_data[i:i+batch_size] = fn(input_data[i:i+batch_size])

    return output_data
