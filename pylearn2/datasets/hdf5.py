"""
Objects for datasets serialized in HDF5 format (.h5).
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

try:
    import h5py
except ImportError:
    h5py = None
try:
    import tables
except ImportError:
    tables = None
import numpy as np
from theano.compat.six.moves import xrange
import warnings

from pylearn2.datasets.dense_design_matrix import (DenseDesignMatrix,
                                                   DefaultViewConverter)
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.utils.iteration import FiniteDatasetIterator, safe_izip
from pylearn2.utils import contains_nan
from pylearn2.utils import safe_zip


class HDF5Dataset(DenseDesignMatrix):

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
        chunks and when using a sequential iterator. The chunk cache allows
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
                 load_all=False, cache_size=None, use_h5py=True, **kwargs):
        self.load_all = load_all
        if use_h5py:
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
        else:
            if tables is None:
                raise RuntimeError("Could not import tables.")
            self._file = tables.openFile(filename, mode='r')
            if X is not None:
                X = self._file.getNode('/', X)
            if topo_view is not None:
                topo_view = self._file.getNode('/', topo_view)
            if y is not None:
                y = self._file.getNode('/', y)
        super(HDF5Dataset, self).__init__(X=X, topo_view=topo_view, y=y,
                                          **kwargs)

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
        iterator = super(HDF5Dataset, self).iterator(*args, **kwargs)
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
        * Support for "old pickled models" is dropped.

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
                # Why this try..except is there? FB: I think this is useless.
                # Do not hide the original if we can't fall back.
                if data.ndim > 1:
                    this_data = data[next_index, :]
                else:
                    raise
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

    Parameters
    ----------
    shape: list
        Shape of this view, [num_rows, num_cols, channels].
    axes: tuple, optional (default ('b', 0, 1, 'c'))
        The axis ordering to use in topological views of the data. Must be some
        permutation of ('b', 0, 1, 'c'). Default: ('b', 0, 1, 'c')

    Attributes
    ----------
    axes
    pixels_per_channel
    shape
    topo_space
    view_shape
    weights_view_shape
    """

    def design_mat_to_topo_view(self, design_matrix):
        """
        Returns a topological view/copy of design matrix.

        Parameters
        ----------
        design_matrix: numpy.ndarray
            A design matrix with data in rows. Data is assumed to be laid out
            in memory according to the axis order ('b', 'c', 0, 1)

        Returns
        -------
        rval: HDF5DesignMatConverter
            This override of DefaultViewConverter.design_mat_to_topo_view does
            not attempt to transpose the design matrix, since transposition
            is not supported by HDF5 datasets. Instead an instance of
            HDF5DesignMatConverter is returned, which transforms the data from
            the design matrix view into the topological view for each batch.
            The axis order is given by self.axes and the batch shape by
            self.shape (in case you reordered self.shape to match self.axes,
            as self.shape is always in 'c', 0, 1 order).
        """
        if len(design_matrix.shape) != 2:
            raise ValueError("design_matrix must have 2 dimensions, but shape "
                             "was %s." % str(design_matrix.shape))

        expected_row_size = np.prod(self.shape)
        if design_matrix.shape[1] != expected_row_size:
            raise ValueError("This HDF5ViewConverter's self.shape is %s, "
                             "for a total size of %d, but the design_matrix's "
                             "row size was different (%d)." %
                             (str(self.shape),
                              expected_row_size,
                              design_matrix.shape[1]))

        bc01_shape = tuple([design_matrix.shape[0], ] +  # num. batches
                           # Maps the (0, 1, 'c') of self.shape to ('c', 0, 1)
                           [self.shape[i] for i in (2, 0, 1)])
        axes_order = [('b', 'c', 0, 1).index(axis) for axis in self.axes]

        return HDF5VirtualTopoView(design_matrix, bc01_shape, axes_order)

    def design_mat_to_weights_view(self, X):
        raise NotImplemented

    def topo_view_to_design_mat(self, topo_array):
        """
        Generate a design matrix from the topological view.

        Parameters
        ----------
        topo_array: numpy.ndarray
            An N-D array with axis order given by self.axes. Non-batch axes'
            dimension sizes must agree with corresponding sizes in self.shape.

        Returns
        -------
        rval: HDF5TopoViewConverter
            This override of DefaultViewConverter.topo_view_to_design_mat does
            not attempt to transpose the topological view, since transposition
            is not supported by HDF5 datasets. In order to keep data on disk,
            does not generate a full design matrix. Instead, an instance of
            HDF5TopoViewConverter is returned, which transforms data from the
            topological view into the design view for each batch, with data in
            rows according to the default axis order ('b', 'c', 0, 1).

        """
        for shape_elem, axis in safe_zip(self.shape, (0, 1, 'c')):
            if topo_array.shape[self.axes.index(axis)] != shape_elem:
                raise ValueError(
                    "topo_array's %s axis has a different size "
                    "(%d) from the corresponding size (%d) in "
                    "self.shape.\n"
                    "  self.shape:       %s (uses standard axis order: 0, 1, "
                    "'c')\n"
                    "  self.axes:        %s\n"
                    "  topo_array.shape: %s (should be in self.axes' order)" %
                    (axis, topo_array.shape[self.axes.index(axis)], shape_elem,
                     self.shape, self.axes, topo_array.shape))

        return HDF5VirtualDesignMat(topo_array, self.axes)


class HDF5VirtualTopoView(object):

    """
    Class to simulate a topological view. Reads data stored in the disk as a
    dense design matrix and transforms the batches in a topological view.

    Attributes
    ----------
    design_matrix: HDF5 dataset
        The hdf5 dataset object
    bc01_shape: list
        The shape in the 'bc01' order
    shape: list
        The shape in the 'axes_order' order (can change after 'virtual'
        transpositions and reshapes)
    axes_order: list of ints
        The order of the axes

    """
    @property
    def shape(self):
        return [self.bc01_shape[idx] for idx in self.axes_order]

    def __init__(self, design_matrix, bc01_shape, axes_order=[0, 1, 2, 3]):
        """
        Parameters
        ----------
        design_matrix: HDF5 dataset
            On-disk design matrix.
        bc01_shape: list
            The batch shape, in 'bc01' order
        axes_order: list, optional (default [0, 1, 2, 3])
            Order of axes in topological view.
        """
        self.design_matrix = design_matrix
        self.bc01_shape = bc01_shape
        self.axes_order = axes_order

    def __getitem__(self, item):
        """
        Receives indexes in topological view format, transforms them in design
        matrix format, retrieves the requested data on disk and returns them
        in topological view format.

        Parameters
        ----------
        item : tuple
            Slice selection for each axis of the topological view. Each element
            of item is either a slice or an int.
        """
        assert isinstance(item, tuple), 'The argument should be a tuple'
        assert len(item) == 4, (
            'Expecting a 4D tuple, but item is ' + str(len(item)) + 'D')
        assert np.all([isinstance(el, (int, slice)) for el in item]), (
            'The elements of item should be either int or slice')

        def slice_to_range(this_slice, end):
            start = this_slice.start
            stop = this_slice.stop
            step = this_slice.step

            if start is None:
                start = 1
            if stop is None:
                stop = end
            if step is None:
                step = 1
            return [start, stop, step]

        # Convert all of the elements of item to slices
        item = [i if isinstance(i, slice) else slice(i, i+1) for i in item]

        # Get the start, stop and step value of each slice
        range_b = slice_to_range(item[0],
                                 self.bc01_shape[0])
        range_c = slice_to_range(item[self.axes_order.index(1)],
                                 self.bc01_shape[1])
        range_0 = slice_to_range(item[self.axes_order.index(2)],
                                 self.bc01_shape[2])
        range_1 = slice_to_range(item[self.axes_order.index(3)],
                                 self.bc01_shape[3])

        # Populate rval iterating through the bc01 intervals
        # TODO refactor to aggregate indexes and reduce the access to the disk
        # to a minimum
        rval = []

        for b in range(*range_b):
            for c in range(*range_c):
                for x in range(*range_0):
                    # self.design_matrix is indexed as [b, c01]
                    # Let's compute the 'base' index (based on c, x and the
                    # 'start' value of range_1) and the offset (based on the
                    # 'start' and 'stop' values of range_1.
                    base = np.ravel_multi_index([c, x, range_1[0]],
                                                self.bc01_shape[1:])
                    offset = range_1[1] - range_1[0]
                    rval.append(
                        self.design_matrix[b, base:base+offset:range_1[2]])

        raise NotImplementedError(
            'Reshape of rdata is missing: rdata should have the shape it '
            'would have if it were a topoview selection.')

        return rval

    def transpose(self, axes=None):
        """
        Permute the dimensions of an array.

        Parameters
        ----------
        axes : list of ints, optional
            By default, reverse the dimensions, otherwise permute the axes
            according to the values given.

        Returns
        -------
        rval: HDF5DesignMatConverter
            'self' with its axes permuted
        """
        if axes is None:
            axes = self.axes_order
            axes.reverse()
        return HDF5DesignMat2TopoViewConverter(self.design_matrix, self.bc01_shape, axes)


class HDF5VirtualDesignMat(object):
    """
    Class to simulate a dense design matrix. Reads data stored in the disk as a
    topological view and transforms the batches in a dense design matrix.

    Attributes
    ----------
    axes : tuple
        Order of axes in topological view.
    ndim: int
        The number of dimensions of the dense design matrix (i.e. 2).
    shape: tuple
        The shape of the dense design matrix.
    topo_view : HDF5 dataset
        On-disk topological view.
    """

    def __init__(self, topo_view, axes=('b', 0, 1, 'c')):
        """
        Parameters
        ----------
        topo_view : HDF5 dataset
            On-disk topological view.
        axes : tuple, optional (defaults to 'b', 0, 1, 'c')
            Order of axes in topological view.
        """
        self.topo_view = topo_view
        self.axes = axes

        self.shape = (
            topo_view.shape[self.axes.index('b')],
            np.product([topo_view.shape[self.axes.index(0)],
                        topo_view.shape[self.axes.index(1)],
                        topo_view.shape[self.axes.index('c')]]))
        self.ndim = len(self.shape)

    def __getitem__(self, item):
        """
        Receives indexes in design matrix view format, transforms them in
        topologica view format, retrieves the requested data on disk and
        returns them in design matrix view format.

        Parameters
        ----------
        item : slice or ndarray or tuple
            Batch selection.        Either a slice or a boolean mask (ndarray).
            Design matrix index.    Tuple
        """
        if isinstance(item, tuple):
            # item's element can be either an int or a slice. Convert all of
            # them to slices
            # item = [i if isinstance(i, slice) else slice(i, i+1) for i in item]

            def num_el(it):
                """
                Compute the number of elements in a slice

                Parameters

                it: slice
                """
                b = it.start
                e = it.stop
                s = it.step
                return (e-b)/s if (e-b) % s == 0 else (e-b)/s + 1

            rval_shape = tuple([num_el(it) if isinstance(it, slice) else 1
                               for it in item])

            # Convert the indexes from the (b, c01) format of the design matrix
            # to the (b,c,0,1) format of the topological view
            topo_bc01_slices = self.design_mat2topo_view_idx(
                self.topo_view.shape, item)

            # Transpose the axes of the indices, according to the shape of the
            # topological view, and get the data from the dataset
            rval = np.array([])
            axes_order = [('b', 'c', 0, 1).index(ax) for ax in self.axes]

            for el in topo_bc01_slices:
                data = self.topo_view[tuple([el[i] for i in axes_order])]
                rval = np.append(rval, data.flatten())

            rval = rval.reshape(rval_shape)
        else:
            # FV: this code should be probably checked and modified. It
            # bases on assumptions that are not verified, s.a. the shape of the
            # input. It is working though, so I am keeping it as is for now.
            # The boolean mask is given as argument at least in one case: in
            # HDF5DatasetIterator.next() by 'is_data = data[next_index]'.
            # I couldn't find a call with an ndarray argument.
            pixels_per_channel = (self.topo_view.shape[self.axes.index(0)] *
                                  self.topo_view.shape[self.axes.index(1)])
            n_channels = self.topo_view.shape[self.axes.index('c')]

            sel = [slice(None)] * len(self.topo_view.shape)
            sel[self.axes.index('b')] = item
            sel = tuple(sel)
            V = self.topo_view[sel]
            batch_size = V.shape[self.axes.index('b')]
            rval = np.zeros(
                (batch_size, pixels_per_channel * n_channels),
                dtype=V.dtype)
            for i in xrange(n_channels):
                ppc = pixels_per_channel
                sel = [slice(None)] * len(V.shape)
                sel[self.axes.index('c')] = i
                sel = tuple(sel)
                rval[:, i * ppc:(i + 1) * ppc] = V[sel].reshape(batch_size,
                                                                ppc)
        return rval

    def design_mat2topo_view_idx(topo_view_shape, idx):
        """
        Converts a design matrix index into a topo view index.
        Note that this method is "axes agnostic" as far as you provide it with
        the shape of the topo_view.

        Parameters
        ----------
        topo_view_shape: tuple
            The shape of the topological view.
        idx: 2D tuple
            A 2D tuple representing the matrix design index to be converted.
            Each element of idx can be either an int or a slice.

        Returns
        -------
        A list of elements. Each element being a list of slices, one for each
        axis of the topological view.
        """
        import itertools

        assert len(idx) == 2, ('A design matrix index should be 2D, but a '
                               + len(idx) + 'D index has been received')
        assert all(isinstance(i, (slice, int)) for i in idx), (
            'idx should contain either int or slices. %s is not a valid '
            'type.' % str(type(idx[0])))

        # Convert slices to lists of indexes
        b_list = range(idx[0].start, idx[0].stop, idx[0].step) if (
            isinstance(idx[0], slice)) else [idx[0]]
        c01_list = range(idx[1].start, idx[1].stop, idx[1].step) if (
            isinstance(idx[1], slice)) else [idx[1]]

        # Convert to a list of 4D indices
        # TODO: can be probably replaced by a single line code with np.unravel
        # but I don't have time to dig into it. Something similar to
        # np.unravel_index(el[1:], tvs[1:]) but on the whole lists
        topo_idx_list = []
        for el in itertools.product(b_list, c01_list):
            topo_idx = []
            tvs = topo_view_shape
            topo_idx.append(el[0])
            topo_idx.append(el[1] / (tvs[2] * tvs[3]))
            topo_idx.append(el[1] % (tvs[2] * tvs[3]) / tvs[2])
            topo_idx.append(el[1] % (tvs[2] * tvs[3]) % tvs[2])
            # FIXME: convert this into a proper test
            assert (topo_idx[1]*tvs[2]*tvs[3] + topo_idx[2]*tvs[2] +
                    topo_idx[3] == el[1]), 'Something went terribly wrong!'
            topo_idx_list.append(tuple(topo_idx))

        def isConsecutive(curr, prev):
            """
            Returns true if two slices index consecutive elements
            """
            assert None not in (curr.start, curr.stop, prev.start, prev.stop)
            return prev.stop == curr.start

        # Compact the list where possible using slices
        for dim in range(3, -1, -1):
            it = iter(topo_idx_list)
            slices = []
            start = next(it)
            last = topo_idx_list[-1]
            if not all(isinstance(el, slice) for el in start):
                start = [slice(j, j+1) for j in start]
            if not all(isinstance(el, slice) for el in last):
                last = [slice(j, j+1) for j in last]

            for i, x in enumerate(it):
                # Get current and previous element as 4D slice
                prev = topo_idx_list[i]
                curr = x
                if not all(isinstance(el, slice) for el in prev):
                    prev = [slice(j, j+1) for j in topo_idx_list[i]]
                else:
                    prev = topo_idx_list[i]
                if not all(isinstance(el, slice) for el in x):
                    curr = [slice(j, j+1) for j in x]

                # if the 'dim' axis is not consecutive or at least
                # one of the other axes' value varied, append a new element
                bool_diff = np.equal(prev, curr)
                if (sum(bool_diff) != 3 or
                        dim != np.where(bool_diff == 0)[0][0] or
                        not isConsecutive(curr[dim], prev[dim])):
                    end = prev
                    if start == end:
                        slices.append(start)
                    else:
                        slices.append(
                            start[:dim] +
                            [slice(start[dim].start, end[dim].stop)] +
                            end[dim+1:])
                    start = curr

            # Manage the end of the for loop
            if last == start:
                slices.append(start)
            else:
                slices.append(start[:dim] +
                              [slice(start[dim].start, last[dim].stop)] +
                              last[dim+1:])
            topo_idx_list = slices
        return slices


