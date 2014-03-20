"""Objects for datasets serialized in HDF5 format (.h5)."""
import h5py
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class HDF5Dataset(DenseDesignMatrix):
    """Dense dataset loaded from an HDF5 file."""
    def __init__(self, filename, key):
        with h5py.File(filename) as f:
            data = f[key][:]
        if data.ndim == 2:
            super(HDF5Dataset, self).__init__(X=data)
        else:
            super(HDF5Dataset, self).__init__(topo_view=data)
