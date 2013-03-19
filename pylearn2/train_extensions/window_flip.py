"""
TrainExtensions for doing random spatial windowing and flipping of an
image dataset on every epoch.
"""
import warnings
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


def _zero_pad(array, amount, axes=(1, 2)):
    if amount == 0:
        return array
    new_shape = []
    slices = []
    for i, s in enumerate(array.shape):
        if i in axes:
            new_shape.append(s + 2 * amount)
            slices.append(slice(amount, -amount))
        else:
            new_shape.append(s)
            slices.append(slice(None))
    new_shape = tuple(new_shape)
    slices = tuple(slices)
    new_array = numpy.zeros(new_shape, dtype=array.dtype)
    new_array[slices] = array
    return new_array


class WindowAndFlipC01B(TrainExtension):
    # Immutable class-level attribute. This should not be a list, as then
    # mutating self.axes will cause the class-level attribute to change.
    # self.axes can be safely assigned to, however.
    axes = ('c', 0, 1, 'b')

    def __init__(self, window_shape, randomize=None, randomize_once=None,
            center=None, rng=(2013, 02, 20), pad_randomized=0):
        """
        An extension that allows an image dataset to be flipped
        and windowed after each epoch of training.

        Parameters
        ----------
        randomize : list, optional
            If specified, a list of Datasets to randomly window and
            flip at each epoch.

        randomize_once : list, optional
            If specified, a list of Dataasets to randomly window and
            flip once at the start of training.

        center : list, optional
            If specified, a list of Datasets to centrally window
            once at the start of training.

        rng : numpy.random.RandomState object or seed, optional
            A random number generator or seed used to create one.
            Seeded deterministically by default.

        pad_randomized : int, optional
            Amount of padding to add to each side of the images
            in `randomize` and `randomize_once`. Useful if you
            want to do zero-padded windowing with `window_shape`
            the actual size of the dataset, and validate/test on
            full-size images instead of central patches. Default
            is 0.
        """
        self._window_shape = tuple(window_shape)
        self._original = None

        self._randomize = randomize if randomize else []
        self._randomize_once = randomize_once if randomize_once else []
        self._center = center if center else []
        self._pad_randomized = pad_randomized

        if randomize is None and randomize_once is None and center is None:
            warnings.warn(self.__class__.__name__ + " instantiated without "
                          "any dataset arguments, and therefore does nothing",
                          stacklevel=2)

        if not hasattr(rng, 'random_integers'):
            self._rng = numpy.random.RandomState(rng)
        else:
            self._rng = rng

    def setup(self, model, dataset, algorithm):
        """
        Note: dataset argument is ignored.
        """
        dataset = None

        # Central windowing of auxiliary datasets (e.g. validation sets)
        preprocessor = CentralWindow(self._window_shape)
        for data in self._center:
            if not (tuple(data.view_converter.axes) == self.axes):
                raise ValueError("Expected axes: %s Actual axes: %s" % (str(data.view_converter.axes), str(self.axes)))
            preprocessor.apply(data)

        # Do the initial random windowing
        randomize_now = self._randomize + self._randomize_once
        self._original = dict((data, _zero_pad(data.get_topological_view(),
                                               self._pad_randomized))
                for data in randomize_now)
        self.randomize_datasets(randomize_now)

    def randomize_datasets(self, datasets):
        """
        Applies random translations and flips to the selected datasets.
        """
        for dataset in datasets:
            assert tuple(dataset.view_converter.axes) == self.axes
            arr = random_window_and_flip_c01b(self._original[dataset],
                                              self._window_shape,
                                              rng=self._rng)
            dataset.set_topological_view(arr, axes=self.axes)

    def on_monitor(self, model, dataset, algorithm):
        """
        Note: all arguments are ignored.
        """
        model = None
        dataset = None
        algorithm = None

        self.randomize_datasets(self._randomize)
