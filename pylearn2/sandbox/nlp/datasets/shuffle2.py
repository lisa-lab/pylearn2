"""
Pylearn2 wrapper for h5-format datasets of sentences. Dataset generates
ngrams and swaps 2 adjacent words. Targets are n-1 vectors indicating where 
swap happened. 
"""
__authors__ = ["Coline Devin"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Coline Devin", "Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Coline Devin"
__email__ = "devincol@iro"


import os.path
import functools
import numpy
import tables
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dataset import Dataset
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace, Conv2DSpace
from pylearn2.utils import serial
from pylearn2.utils import safe_zip
from pylearn2.utils.iteration import FiniteDatasetIterator

def index_from_one_hot(one_hot):
    return numpy.where(one_hot == 1.0)[0][0]

class H5Shuffle(Dataset):
    """
    Frame-based WMT14 dataset
    """
    _default_seed = (17, 2, 946)

    def __init__(self, path, node, which_set, frame_length,
                 start=0, stop=None, X_labels=None,
                 rng=_default_seed):
        """
        Parameters
        ----------
        path : str
            The base path to the data
        node: str
            The node in the h5 file
        which_set : str
            Either "train", "valid" or "test"
        frame_length : int
            Number of words contained in a frame
        start : int, optional
            Starting index of the sequences to use. Defaults to 0.
        stop : int, optional
            Ending index of the sequences to use. Defaults to `None`, meaning
            sequences are selected all the way to the end of the array.
        rng : object, optional
            A random number generator used for picking random indices into the
            design matrix when choosing minibatches.
        """
        self.base_path = path
        self.node_name = node
        self.frame_length = frame_length
        self.X_labels = X_labels
        #self.y_labels = y_labels

        # RNG initialization
        if hasattr(rng, 'random_integers'):
            self.rng = rng
        else:
            self.rng = numpy.random.RandomState(rng)

        # Load data from disk
        self._load_data(which_set, start, stop)
        self.samples_sequences = numpy.asarray(self.raw_data)
        shorts = []
        for i in range(len(self.samples_sequences)):
            if len(self.samples_sequences[i]) < self.frame_length:
                shorts.append(i)

        self.samples_sequences = numpy.delete(self.samples_sequences, shorts)
        self.num_examples = len(self.samples_sequences)
  
        self.cumulative_sequence_indexes = numpy.cumsum(len(s) for s in self.raw_data)
  
        # DataSpecs
        features_space = IndexSpace(
            dim=self.frame_length,
            max_labels=self.X_labels
        )
        features_source = 'features'

        targets_space = VectorSpace(dim=self.frame_length-1)
        targets_source = 'targets'
        # def targets_map_fn(indexes):

        space = CompositeSpace([features_space, targets_space])
        source = (features_source, targets_source)

        self.data_specs = (space, source)

        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('shuffled_sequential')
        self._iter_data_specs = (CompositeSpace((features_space,
                                                 targets_space)),
                                 (features_source, targets_source))

        def getFeatures(indexes):
            """
            .. todo::
                Write me
            """
            sequences = self.samples_sequences[indexes]

            # Remove sequences that are shorter than frame length to avoid padding
            shorts = []
            for i in range(len(sequences)):
                if len(sequences[i]) < self.frame_length:
                    shorts.append(i)

            sequences = numpy.delete(sequences, shorts)

            # Get random start point for ngram
            wis = [numpy.random.randint(0, len(s)-self.frame_length+1, 1)[0] for s in sequences]
            # end = min(len(s), self.frame_length+wi)
            # diff = max(self.frame_length +wi - len(s), 0)
            # x = s[wi:end] + [0]*diff

            # X = numpy.asarray([numpy.concatenate((s[wi:(min(len(s), self.frame_length+wi))],
            #      [0]*(max(self.frame_length +wi - len(s), 0)))) for s, wi in 
            #      zip(sequences, wis)])

            X = numpy.asarray([s[wi:self.frame_length+wi] for s, wi in zip(sequences, wis)])

            # Words mapped to integers greater than input max are set to 1 (unknown)
            X[X>=self.X_labels] = 1

            swaps = numpy.random.randint(0, self.frame_length - 1, len(X))
            y = numpy.zeros((len(X), self.frame_length - 1))
            y[numpy.arange(len(X)), swaps] = 1
            # print "Performing permutations...",
            for sample, swap in enumerate(swaps):
                X[sample, swap], X[sample, swap + 1] = \
                                                  X[sample, swap + 1], X[sample, swap]

            # Store the targets generated by these indices.
            self.lastY = (y, indexes)
            return X

        def getTarget(indexes):
            if numpy.array_equal(indexes, self.lastY[1]):
                return self.lastY[0]
            else:
                print "You can only ask for targets immediately after asking for those features"
                return None

        self.sourceFNs = {'features': getFeatures, 'targets': getTarget}

    def _load_data(self, which_set, start, stop):
        """
        Load the WMT14 data from disk.

        Parameters
        ----------
        which_set : str
            Subset of the dataset to use (either "train", "valid" or "test")
        """
        # TODO: Make files work with this terminology

        # Check which_set
        #if which_set not in ['train', 'valid', 'test']:
        #    raise ValueError(which_set + " is not a recognized value. " +
        #                     "Valid values are ['train', 'valid', 'test'].")
            
        # Load Data
        with tables.open_file(self.base_path) as f:
            print "Loading n-grams..."
            node = f.get_node(self.node_name)
            if stop is not None:
                self.raw_data = node[start:stop]
            else:
                self.raw_data = node[start:]
 
    def _validate_source(self, source):
        """
        Verify that all sources in the source tuple are provided by the
        dataset. Raise an error if some requested source is not available.

        Parameters
        ----------
        source : `tuple` of `str`
            Requested sources
        """
        for s in source:
            try:
                self.data_specs[1].index(s)
            except ValueError:
                raise ValueError("the requested source named '" + s + "' " +
                                 "is not provided by the dataset")

    

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.

        .. note::

            Once again, this is very hacky, as the data is not stored that way
            internally. However, the data that's returned by `.get()`
            _does_ respect those data specs.
        """
        return self.data_specs

    def get(self, source, indexes):
        """
        .. todo::

            WRITEME
        """
        if type(indexes) is slice:
            indexes = numpy.arange(indexes.start, indexes.stop)
        self._validate_source(source)
        rval = []
        for so in source:
            batch = self.sourceFNs[so](indexes)
            rval.append(batch)
        return tuple(rval)

    def get_num_examples(self):
        return self.num_examples

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        """
        .. todo::

            WRITEME
        """
        if data_specs is None:
            data_specs = self._iter_data_specs

        # If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator
        # can return.
        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        convert = []
        for sp, src in safe_zip(sub_spaces, sub_sources):
            convert.append(None)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIterator(self,
                                     mode(self.num_examples, batch_size,
                                          num_batches, rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)