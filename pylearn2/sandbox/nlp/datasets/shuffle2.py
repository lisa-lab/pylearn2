"""
Pylearn2 wrapper for h5-format datasets of sentences
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
# from research.code.pylearn2.space import (
#     VectorSequenceSpace,
#     IndexSequenceSpace,
# )
from pylearn2.utils import serial
from pylearn2.utils import safe_zip
# from research.code.scripts.segmentaxis import segment_axis
from pylearn2.utils.iteration import FiniteDatasetIterator
# import scipy.stats

def index_from_one_hot(one_hot):
    return numpy.where(one_hot == 1.0)[0][0]

class H5Shuffle(Dataset):
    """
    Frame-based WMT14 dataset
    """
    _default_seed = (17, 2, 946)

    def __init__(self, path, node, which_set, frame_length, overlap=0,
                 start=0, stop=None, X_labels=None,# y_labels
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
        overlap : int, optional
            Number of overlapping acoustic samples for two consecutive frames.
            Defaults to 0, meaning frames don't overlap.
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
        self.overlap = overlap
        self.offset = self.frame_length - self.overlap
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
        self.num_examples = len(self.samples_sequences)
        #examples_per_sequence = [0]

        # for sequence_id, samples_sequence in enumerate(self.raw_wav):
          
            # TODO: look at this, does it force copying the data?
            # Sequence segmentation
            # samples_segmented_sequence = segment_axis(samples_sequence,
            #                                           frame_length,
            #                                           overlap)
            # self.raw_wav[sequence_id] = samples_segmented_sequence

            # # TODO: change me
            # # Generate features/targets/phones/phonemes/words map
            # num_frames = samples_segmented_sequence.shape[0]
            # num_examples = num_frames - self.frames_per_example
            # examples_per_sequence.append(num_examples)

        self.cumulative_sequence_indexes = numpy.cumsum(len(s) for s in self.raw_data)
        # self.samples_sequences = self.raw_wav
        # if not self.audio_only:
        #     self.phones_sequences = self.phones
        #     self.phonemes_sequences = self.phonemes
        #     self.words_sequences = self.words
        # self.num_examples = self.cumulative_example_indexes[-1]

        # DataSpecs
        features_space = IndexSpace(
            dim=self.frame_length,
            max_labels=self.X_labels
        )
        features_source = 'features'

        # # Maps from iterator index in a word index.
        # def features_map_fn(indexes):
        #     rval = []
        #     for sequence_index, example_index in self._fetch_index(indexes):
        #         rval.append(self.samples_sequences[sequence_index][example_index:example_index
        #             + self.frames_per_example].ravel())
        #     return rval

        targets_space = VectorSpace(dim=self.frame_length-1)
        targets_source = 'targets'
        # def targets_map_fn(indexes):
        #     rval = []
        #     for sequence_index, example_index in self._fetch_index(indexes):
        #         rval.append(self.samples_sequences[sequence_index][example_index
        #             + self.frames_per_example].ravel())
        #     return rval

        space_components = [features_space, targets_space]
        source_components = [features_source, targets_source]
        # map_fn_components = [features_map_fn, targets_map_fn] 
        batch_components = [None, None]

        space = CompositeSpace(space_components)
        source = tuple(source_components)
        self.data_specs = (space, source)
        # self.map_functions = tuple(map_fn_components)
        self.batch_buffers = batch_components

        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('shuffled_sequential')
        self._iter_data_specs = (CompositeSpace((features_space,
                                                 targets_space)),
                                 (features_source, targets_source))

        def getExample(indexes):
            """
            .. todo::
                Write me
            """
            sequences = self.samples_sequences[indexes]
            shorts = []
            for i in range(len(sequences)):
                if len(sequences[i]) < self.frame_length:
                    shorts.append(i)

            sequences = numpy.delete(sequences, shorts)
            wis = [numpy.random.randint(0, len(s)-self.frame_length+1, 1)[0] for s in sequences]
            # end = min(len(s), self.frame_length+wi)
            # diff = max(self.frame_length +wi - len(s), 0)
            # x = s[wi:end] + [0]*diff

            # X = numpy.asarray([numpy.concatenate((s[wi:(min(len(s), self.frame_length+wi))],
            #      [0]*(max(self.frame_length +wi - len(s), 0)))) for s, wi in 
            #      zip(sequences, wis)])
            X = numpy.asarray([s[wi:self.frame_length+wi] for s, wi in zip(sequences, wis)])
            X[X>=self.X_labels] = 1

            swaps = numpy.random.randint(0, self.frame_length - 1, len(X))
            y = numpy.zeros((len(X), self.frame_length - 1))
            y[numpy.arange(len(X)), swaps] = 1
            print "Performing permutations...",
            for sample, swap in enumerate(swaps):
                X[sample, swap], X[sample, swap + 1] = \
                                                  X[sample, swap + 1], X[sample, swap]
            self.lastY = (y, indexes)
            return X

        def getTarget(indexes):
            if numpy.array_equal(indexes, self.lastY[1]):
                return self.lastY[0]
            else:
                print "You can only ask for targets immediately after asking for those features"
                return None

        self.sourceFNs = {'features': getExample, 'targets': getTarget}

    def _fetch_index(self, indexes):
        digit = numpy.digitize(indexes, self.cumulative_sequence_indexes) - 1
        return zip(digit,
                   numpy.array(indexes) - self.cumulative_sequence_indexes[digit])

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