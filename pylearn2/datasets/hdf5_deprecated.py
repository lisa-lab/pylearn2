"""
Objects for datasets serialized in HDF5 format (.h5).
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

try:
    import h5py
except ImportError:
    h5py = None
import numpy as np
from theano.compat.six.moves import xrange
import warnings

from pylearn2.datasets.dense_design_matrix import (DenseDesignMatrix,
                                                   DefaultViewConverter)
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace
from pylearn2.utils.iteration import FiniteDatasetIterator, safe_izip
from pylearn2.utils import contains_nan


class HDF5DatasetDeprecated(DenseDesignMatrix):

    """
    Dense dataset loaded from an HDF5 file.

    Parameters
    ----------
    filename : str
        HDF5 file name.
    X : str, optional
        Key into HDF5 file for dataset design matrix.
    topo_view: str, optional
        Key into HDF5 file for topological view of dataset.
    y : str, optional
        Key into HDF5 file for dataset targets.
    load_all : bool, optional (default False)
        If true, datasets are loaded into memory instead of being left
        on disk.
    cache_size: int, optionally specify the size in bytes for the chunk
        cache of the HDF5 library. Useful when the HDF5 files has large
        chunks and when using a sequantial iterator. The chunk cache allows
        to only access the disk for the chunks and then copy the batches to
        the GPU from memory, which can result in a significant speed up.
        Sensible default values depend on the size of your data and the
        batch size you wish to use. A rule of thumb is to make a chunk
        contain 100 - 1000 batches and make sure they encompass complete
        samples.
    kwargs : dict, optional
        Keyword arguments passed to `DenseDesignMatrix`.
    """

    def __init__(self, filename, X=None, topo_view=None, y=None,
                 load_all=False, cache_size=None, **kwargs):
        self.load_all = load_all
        if h5py is None:
            raise RuntimeError("Could not import h5py.")
        if cache_size:
            propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
            settings = list(propfaid.get_cache())
            settings[2] = cache_size
            propfaid.set_cache(*settings)
            fid = h5py.h5f.open(filename, fapl=propfaid)
            self._file = h5py.File(fid)
        else:
            self._file = h5py.File(filename)
        if X is not None:
            X = self.get_dataset(X, load_all)
        if topo_view is not None:
            topo_view = self.get_dataset(topo_view, load_all)
        if y is not None:
            y = self.get_dataset(y, load_all)

        super(HDF5DatasetDeprecated, self).__init__(X=X, topo_view=topo_view,
                                                    y=y, **kwargs)

    def _check_labels(self):
        """
        Sanity checks for X_labels and y_labels.

        Since the np.all test used for these labels does not work with HDF5
        datasets, we issue a warning that those values are not checked.
        """
        if self.X_labels is not None:
            assert self.X is not None
            assert self.view_converter is None
            assert self.X.ndim <= 2
            if self.load_all:
                assert np.all(self.X < self.X_labels)
            else:
                warnings.warn("HDF5Dataset cannot perform test np.all(X < " +
                              "X_labels). Use X_labels at your own risk.")

        if self.y_labels is not None:
            assert self.y is not None
            assert self.y.ndim <= 2
            if self.load_all:
                assert np.all(self.y < self.y_labels)
            else:
                warnings.warn("HDF5Dataset cannot perform test np.all(y < " +
                              "y_labels). Use y_labels at your own risk.")

    def get_dataset(self, dataset, load_all=False):
        """
        Get a handle for an HDF5 dataset, or load the entire dataset into
        memory.

        Parameters
        ----------
        dataset : str
            Name or path of HDF5 dataset.
        load_all : bool, optional (default False)
            If true, load dataset into memory.
        """
        if load_all:
            data = self._file[dataset][:]
        else:
            data = self._file[dataset]
            data.ndim = len(data.shape)  # hdf5 handle has no ndim
        return data

    def iterator(self, *args, **kwargs):
        """
        Get an iterator for this dataset.

        The FiniteDatasetIterator uses indexing that is not supported by
        HDF5 datasets, so we change the class to HDF5DatasetIterator to
        override the iterator.next method used in dataset iteration.

        Parameters
        ----------
        WRITEME
        """
        iterator = super(HDF5DatasetDeprecated, self).iterator(*args, **kwargs)
        iterator.__class__ = HDF5DatasetIterator
        return iterator

    def set_topological_view(self, V, axes=('b', 0, 1, 'c')):
        """
        Set up dataset topological view, without building an in-memory
        design matrix.

        This is mostly copied from DenseDesignMatrix, except:
        * HDF5ViewConverter is used instead of DefaultViewConverter
        * Data specs are derived from topo_view, not X
        * NaN checks have been moved to HDF5DatasetIterator.next

        Note that y may be loaded into memory for reshaping if y.ndim != 2.

        Parameters
        ----------
        V : ndarray
            Topological view.
        axes : tuple, optional (default ('b', 0, 1, 'c'))
            Order of axes in topological view.
        """
        shape = [V.shape[axes.index('b')],
                 V.shape[axes.index(0)],
                 V.shape[axes.index(1)],
                 V.shape[axes.index('c')]]
        self.view_converter = HDF5ViewConverter(shape[1:], axes=axes)
        self.X = self.view_converter.topo_view_to_design_mat(V)
        # self.X_topo_space stores a "default" topological space that
        # will be used only when self.iterator is called without a
        # data_specs, and with "topo=True", which is deprecated.
        self.X_topo_space = self.view_converter.topo_space

        # Update data specs
        X_space = VectorSpace(dim=V.shape[axes.index('b')])
        X_source = 'features'
        if self.y is None:
            space = X_space
            source = X_source
        else:
            if self.y.ndim == 1:
                dim = 1
            else:
                dim = self.y.shape[-1]

            # check if y_labels has been specified
            if getattr(self, 'y_labels', None) is not None:
                y_space = IndexSpace(dim=dim, max_labels=self.y_labels)
            elif getattr(self, 'max_labels', None) is not None:
                y_space = IndexSpace(dim=dim, max_labels=self.max_labels)
            else:
                y_space = VectorSpace(dim=dim)
            y_source = 'targets'
            space = CompositeSpace((X_space, y_space))
            source = (X_source, y_source)

        self.data_specs = (space, source)
        self.X_space = X_space
        self._iter_data_specs = (X_space, X_source)


class HDF5DatasetIterator(FiniteDatasetIterator):

    """
    Dataset iterator for HDF5 datasets.

    FiniteDatasetIterator expects a design matrix to be available, but this
    will not always be the case when using HDF5 datasets with topological
    views.

    Parameters
    ----------
    dataset : Dataset
        Dataset over which to iterate.
    subset_iterator : object
        Iterator that returns slices of the dataset.
    data_specs : tuple, optional
        A (space, source) tuple.
    return_tuple : bool, optional (default False)
        Whether to return a tuple even if only one source is used.
    convert : list, optional
        A list of callables (in the same order as the sources in
        data_specs) that will be applied to each slice of the dataset.
    """

    def next(self):
        """
        Get the next subset of the dataset during dataset iteration.

        Converts index selections for batches to boolean selections that
        are supported by HDF5 datasets.
        """
        next_index = self._subset_iterator.next()

        # convert to boolean selection
        sel = np.zeros(self.num_examples, dtype=bool)
        sel[next_index] = True
        next_index = sel

        rval = []
        for data, fn in safe_izip(self._raw_data, self._convert):
            try:
                this_data = data[next_index]
            except TypeError:
                # FB: Why this try..except is there? I think this is useless.
                # Do not hide the original if we can't fall back.
                # FV: This is triggered if the shape of next_index is
                # incompatible with the shape of the dataset. See for an
                # example test_hdf5_topo_view(), where where i
                # next.index.shape = (10,) and data is 'data': <HDF5
                # dataset "y": shape (10, 3), type "<f8">
                # I think it would be better to explicitly check if
                # next_index.shape is incompatible with data.shape, for
                # instance checking if next_index.ndim == data.ndim
                if data.ndim > 1:
                    this_data = data[next_index, :]
                else:
                    raise
            # Check if the dataset data is a vector and transform it into a
            # one-column matrix. This is needed to automatically convert the
            # shape of the data later (in the format_as method of the
            # Space.)
            if fn:
                this_data = fn(this_data)
            assert not contains_nan(this_data)
            rval.append(this_data)
        rval = tuple(rval)
        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval


class HDF5ViewConverter(DefaultViewConverter):

    """
    View converter that doesn't have to transpose the data.

    In order to keep data on disk, does not generate a full design matrix.
    Instead, an instance of HDF5TopoViewConverter is returned, which
    transforms data from the topological view into the design view for each
    batch.

    Parameters
    ----------
    shape : tuple
        Shape of this view.
    axes : tuple, optional (default ('b', 0, 1, 'c'))
        Order of axes in topological view.
    """

    def topo_view_to_design_mat(self, V):
        """
        Generate a design matrix from the topological view.

        This override of DefaultViewConverter.topo_view_to_design_mat does
        not attempt to transpose the topological view, since transposition
        is not supported by HDF5 datasets.

        Parameters
        ----------
        WRITEME
        """
        v_shape = (V.shape[self.axes.index('b')],
                   V.shape[self.axes.index(0)],
                   V.shape[self.axes.index(1)],
                   V.shape[self.axes.index('c')])

        if np.any(np.asarray(self.shape) != np.asarray(v_shape[1:])):
            raise ValueError('View converter for views of shape batch size '
                             'followed by ' + str(self.shape) +
                             ' given tensor of shape ' + str(v_shape))

        rval = HDF5TopoViewConverter(V, self.axes)
        return rval


class HDF5TopoViewConverter(object):

    """
    Class for transforming batches from the topological view to the design
    matrix view.

    Parameters
    ----------
    topo_view : HDF5 dataset
        On-disk topological view.
    axes : tuple, optional (default ('b', 0, 1, 'c'))
        Order of axes in topological view.
    """

    def __init__(self, topo_view, axes=('b', 0, 1, 'c')):
        self.topo_view = topo_view
        self.axes = axes
        self.topo_view_shape = (topo_view.shape[axes.index('b')],
                                topo_view.shape[axes.index(0)],
                                topo_view.shape[axes.index(1)],
                                topo_view.shape[axes.index('c')])
        self.pixels_per_channel = (self.topo_view_shape[1] *
                                   self.topo_view_shape[2])
        self.n_channels = self.topo_view_shape[3]
        self.shape = (self.topo_view_shape[0],
                      np.product(self.topo_view_shape[1:]))
        self.ndim = len(self.shape)

    def __getitem__(self, item):
        """
        Indexes the design matrix and transforms the requested batch from
        the topological view.

        Parameters
        ----------
        item : slice or ndarray
            Batch selection. Either a slice or a boolean mask.
        """
        sel = [slice(None)] * len(self.topo_view_shape)
        sel[self.axes.index('b')] = item
        sel = tuple(sel)
        V = self.topo_view[sel]
        batch_size = V.shape[self.axes.index('b')]
        rval = np.zeros((batch_size,
                         self.pixels_per_channel * self.n_channels),
                        dtype=V.dtype)
        for i in xrange(self.n_channels):
            ppc = self.pixels_per_channel
            sel = [slice(None)] * len(V.shape)
            sel[self.axes.index('c')] = i
            sel = tuple(sel)
            rval[:, i * ppc:(i + 1) * ppc] = V[sel].reshape(batch_size, ppc)
        return rval
