"""
TrainExtensions for doing random spatial windowing and flipping of an
image dataset on every epoch.
"""
import numpy
from . import TrainExtension
from pylearn2.datasets.preprocessing import CentralWindow

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
    # This variable is shared among all instances of this class.
    # Don't modify it, unless you really want it to get changed
    # for all other instances.
    axes = ('c', 0, 1, 'b')

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
        preprocessor = CentralWindow(self._window_shape)
        for data in self._other_datasets:
            assert data.view_converter.axes == self.axes
            preprocessor.apply(data)

        # Do the initial random windowing of the training set.
        self._original = dataset.get_topological_view()
        self.on_monitor(model, dataset, algorithm)

    def on_monitor(self, model, dataset, algorithm):
        assert dataset.view_converter.axes == self.axes
        arr = random_window_and_flip_c01b(self._original,
                                          self._window_shape,
                                          rng=self._rng)
        dataset.set_topological_view(arr, axes=self.axes)
