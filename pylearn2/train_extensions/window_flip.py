"""
TrainExtensions for doing random spatial windowing and flipping of an
image dataset on every epoch.
"""
import numpy
from . import TrainExtension

try:
    from ..utils._window_flip import random_window_and_flip_c01b
except ImportError:
    raise ValueError("You should run setup.py build_ext --inplace in the "
                     "utils directory.")

__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "David Warde-Farley"
__email__ = "wardefar@iro"


class WindowAndFlipC01B(TrainExtension):
    axes = ['c', 0, 1, 'b']

    def __init__(self, window_shape, other_datasets=None, rng=(2013, 02, 20)):
        self._window_shape = tuple(window_shape)
        self._original = None
        self._other_datasets = (other_datasets
                                if other_datasets is not None else [])
        if not hasattr(rng, 'random_integers'):
            self._rng = numpy.random.RandomState(rng)
        else:
            self._rng = rng

    def setup(self, model, dataset, algorithm):
        # Central windowing of auxiliary datasets (e.g. validation sets)
        w_rows, w_cols = self._window_shape
        for data in self._other_datasets:
            arr = data.get_topological_view()
            r_off = (arr.shape[1] - w_rows) // 2
            c_off = (arr.shape[2] - w_cols) // 2
            new_arr = arr[:, r_off:r_off + w_rows, c_off:c_off + w_cols, :]
            data.set_topological_view(new_arr, axes=self.axes)

        # Do the initial random windowing of the training set.
        self._original = dataset.get_topological_view()
        self.on_monitor(model, dataset, algorithm)

    def on_monitor(self, model, dataset, algorithm):
        arr = random_window_and_flip_c01b(self._original,
                                          self._window_shape,
                                          rng=self._rng)
        dataset.set_topological_view(arr, axes=self.axes)
