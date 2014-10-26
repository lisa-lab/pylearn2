"""
TrainExtensions for doing random spatial windowing and flipping of an
image dataset on every epoch.
"""
import warnings
import numpy
from . import TrainExtension
from pylearn2.datasets.preprocessing import CentralWindow
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.rng import make_np_rng

try:
    from ..utils._window_flip import random_window_and_flip_c01b
    from ..utils._window_flip import random_window_and_flip_b01c
except ImportError:
    reraise_as(ImportError("Import of Cython module failed. Please make sure "
                           "you have run 'python setup.py develop' in the "
                           "pylearn2 directory"))

__authors__ = "David Warde-Farley"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "David Warde-Farley"
__email__ = "wardefar@iro"


def _zero_pad(array, amount, axes=(1, 2)):
    """
    Returns a copy of <array> with zero-filled padding around the margins.

    The new array has the same dimensions as the input array, except for
    the dimensions given by <axes>, which are increased by 2*<amount>.

    Parameters
    ----------
    array: numpy.ndarray
      The array to zero-pad.

    amount: int
      The number of zeros to append to the beginning and end of each dimension
      in <axes>. (That axis will grow by 2*<amount>).

    axes: tuple
      The dimensions to pad. These are indices, not axis names like the 0, 1
      in ('b', 0, 1, 'c').
    """
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


class WindowAndFlip(TrainExtension):
    """
    An extension that allows an image dataset to be flipped and
    windowed after each epoch of training.

    Parameters
    ----------
    window_shape : WRITEME
    randomize : list, optional
        If specified, a list of Datasets to randomly window and
        flip at each epoch.
    randomize_once : list, optional
        If specified, a list of Datasets to randomly window and
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
    flip : bool, optional
        Reflect images on the horizontal axis with probability
        0.5. `True` by default.
    """
    def __init__(self,
                 window_shape,
                 randomize=None,
                 randomize_once=None,
                 center=None,
                 rng=(2013, 02, 20),
                 pad_randomized=0,
                 flip=True):
        self._window_shape = tuple(window_shape)

        # Defined in setup(). A dict that maps Datasets in self._randomize and
        # self._randomize_once to zero-padded versions of their topological
        # views.
        self._original = None

        self._randomize = randomize if randomize else []
        self._randomize_once = randomize_once if randomize_once else []
        self._center = center if center else []
        self._pad_randomized = pad_randomized
        self._flip = flip

        if randomize is None and randomize_once is None and center is None:
            warnings.warn(self.__class__.__name__ + " instantiated without "
                          "any dataset arguments, and therefore does nothing",
                          stacklevel=2)

        self._rng = make_np_rng(rng, which_method="random_integers")

    def setup(self, model, dataset, algorithm):
        """
        .. todo::

            WRITEME

        Notes
        -----
        `dataset` argument is ignored
        """
        dataset = None

        # Central windowing of auxiliary datasets (e.g. validation sets)
        preprocessor = CentralWindow(self._window_shape)
        for data in self._center:
            preprocessor.apply(data)

        #
        # Do the initial random windowing
        #

        randomize_now = self._randomize + self._randomize_once

        # maps each dataset in randomize_now to a zero-padded topological view
        # of its data.
        self._original = dict((data,
                               _zero_pad(data.get_topological_view().astype('float32'),
                                         self._pad_randomized))
                              for data in randomize_now)

        # For each dataset, for each image, extract a randomly positioned and
        # potentially horizontal-flipped window
        self.randomize_datasets(randomize_now)

    def randomize_datasets(self, datasets):
        """
        Applies random translations and flips to the selected datasets.

        Parameters
        ----------
        datasets : WRITEME
        """
        for dataset in datasets:
            if tuple(dataset.view_converter.axes) == ('c', 0, 1, 'b'):
                wf_func = random_window_and_flip_c01b
            elif tuple(dataset.view_converter.axes) == ('b', 0, 1, 'c'):
                wf_func = random_window_and_flip_b01c
            else:
                raise ValueError("Axes of dataset is not supported: %s" %
                                 (str(dataset.view_converter.axes)))
            arr = wf_func(self._original[dataset],
                          self._window_shape,
                          rng=self._rng, flip=self._flip)
            dataset.set_topological_view(arr, axes=dataset.view_converter.axes)

    def on_monitor(self, model, dataset, algorithm):
        """
        .. todo::

            WRITEME

        Notes
        -----
        All arguments are ignored.
        """
        model = None
        dataset = None
        algorithm = None

        self.randomize_datasets(self._randomize)


class WindowAndFlipC01B(WindowAndFlip):
    """
    WindowAndFlipC01B is deprecated, use WindowAndFlip.
    WindowAndFlipC01B will be removed on or after August 25, 2014.

    A specialized version of WindowAndFlip accepting datasets with axes C01B.
    It exists due to backward compatibility.

    Parameters
    ----------
    window_shape : WRITEME
    randomize : list, optional
        If specified, a list of Datasets to randomly window and
        flip at each epoch.
    randomize_once : list, optional
        If specified, a list of Datasets to randomly window and
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
    flip : bool, optional
        Reflect images on the horizontal axis with probability
        0.5. `True` by default.
    """

    def __init__(self,
                 window_shape,
                 randomize=None,
                 randomize_once=None,
                 center=None,
                 rng=(2013, 02, 20),
                 pad_randomized=0,
                 flip=True):

        _randomize = randomize if randomize else []
        _randomize_once = randomize_once if randomize_once else []

        for data in _randomize + _randomize_once:
            if tuple(data.view_converter.axes) != ('c', 0, 1, 'b'):
                raise ValueError("Expected axes: ('c', 0, 1, 'b') "
                                 "Actual axes: %s" %
                                 str(tuple(data.view_converter.axes)))

        warnings.warn("WindowAndFlipC01B is deprecated, use WindowAndFlip. " +
                      "WindowAndFlipC01B will be removed on or " +
                      "after August 25, 2014.", stacklevel=2)

        super(WindowAndFlipC01B, self).__init__(window_shape,
                                                randomize=randomize,
                                                randomize_once=randomize_once,
                                                center=center,
                                                rng=rng,
                                                pad_randomized=pad_randomized,
                                                flip=flip)
