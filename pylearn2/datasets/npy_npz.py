"""Objects for datasets serialized in the NumPy native format (.npy/.npz)."""
import functools
import numpy
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


class NpyDataset(DenseDesignMatrix):

    """A dense dataset based on a single array stored as a .npy file."""

    def __init__(self, file, mmap_mode=None):
        """
        Creates an NpyDataset object.

        Parameters
        ----------
        file : file-like object or str
            A file-like object or string indicating a filename. Passed
            directly to `numpy.load`.
        mmap_mode : str, optional
            Memory mapping options for memory-mapping an array on disk,
            rather than loading it into memory. See the `numpy.load`
            docstring for details.
        """
        self._path = file
        self._loaded = False

    def _deferred_load(self):
        """
        .. todo::

            WRITEME
        """
        self._loaded = True
        loaded = numpy.load(self._path)
        assert isinstance(loaded, numpy.ndarray), (
            "single arrays (.npy) only"
        )
        if len(loaded.shape) == 2:
            super(NpyDataset, self).__init__(X=loaded)
        else:
            super(NpyDataset, self).__init__(topo_view=loaded)

    @functools.wraps(DenseDesignMatrix.get_design_matrix)
    def get_design_matrix(self, topo=None):
        if not self._loaded:
            self._deferred_load()
        return super(NpyDataset, self).get_design_matrix(topo)

    @functools.wraps(DenseDesignMatrix.get_topological_view)
    def get_topological_view(self, mat=None):
        if not self._loaded:
            self._deferred_load()
        return super(NpyDataset, self).get_topological_view(mat)

    @functools.wraps(DenseDesignMatrix.iterator)
    def iterator(self, *args, **kwargs):
        # TODO: Factor this out of iterator() and into something that
        # can be called by multiple methods. Maybe self.prepare().
        if not self._loaded:
            self._deferred_load()
        return super(NpyDataset, self).iterator(*args, **kwargs)


class NpzDataset(DenseDesignMatrix):

    """A dense dataset based on a .npz archive."""

    def __init__(self, file, key, target_key=None):
        """
        Creates an NpzDataset object.

        Parameters
        ----------
        file : file-like object or str
            A file-like object or string indicating a filename. Passed
            directly to `numpy.load`.
        key : str
            A string indicating which key name to use to pull out the
            input data.
        target_key : str, optional
            A string indicating which key name to use to pull out the
            output data.
        """
        loaded = numpy.load(file)
        assert not isinstance(loaded, numpy.ndarray), (
            "zipped groups of arrays (.npz) only"
        )
        assert key in loaded, "%s not found in loaded NPZFile" % key

        if target_key is not None:
            assert target_key in loaded, \
                "%s not found in loaded NPZFile" % target_key
            y = loaded[target_key]
        else:
            y = None

        if len(loaded[key].shape) == 2:
            super(NpzDataset, self).__init__(X=loaded[key], y=y)
        else:
            super(NpzDataset, self).__init__(topo_view=loaded[key], y=y)

        loaded.close()
