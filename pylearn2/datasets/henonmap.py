"""
Class for creating Henon map datasets.
"""
__authors__ = ["Dustin Webb"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

import numpy as np
from pylearn2.datasets import DenseDesignMatrix
import pylearn2.utils.serial
import theano


class HenonMap(DenseDesignMatrix):
    """
    Generates data for Henon map, i.e.

       x_{n+1} = 1 - \alpha*x_n^2 + y_n
       y_{n+1} = \beta*x_n

    Parameters
    ----------
    alpha : double
        Alpha parameter in equations above.
    beta : double
        Beta parameter in equations above.
    init_state : ndarray
        The initial state of the system of size 2.
    samples : int
        Number of desired samples. Must be an integer multiple of
        frame_length.
    frame_length : int
        Number of samples contained in a frame. Must divide samples.
    rng : int
        Seed for random number generator.
    load_path : string
        Path from which to load data.
    save_path : string
        Path to which the data should be saved.
    """
    _default_seed = 1

    def __init__(
        self, alpha=1.4, beta=0.3, init_state=np.array([0, 0]),
        samples=1000, frame_length=10, rng=None,
        load_path=None, save_path=None
    ):
        # Validate parameters and set member variables
        self.alpha = alpha
        self.beta = beta

        assert(samples % frame_length == 0)

        assert(frame_length > 0)
        self.frame_length = frame_length

        assert(samples > 0)
        self.samples = samples

        assert(init_state.shape in [(2,), (2)])
        self.init_state = init_state

        # Initialize RNG
        if rng is None:
            self.rng = np.random.RandomState(self._default_seed)
        else:
            self.rng = np.random.RandomState(rng)

        # Load or save as specified
        if (load_path is None):
            (X, y) = self._generate_data()
        else:
            (X, y) = serial.load(load_path)

        if (save_path is not None):
            serial.save(save_path, (X, y))

        # Setup the parent class
        super(HenonMap, self).__init__(X=X, y=y)

    def _generate_data(self):
        """
        Generates X matrix for DenseDesignMatrix initialization
        function.
        """
        # Create space for the data
        X = np.zeros((self.samples+1, 2), dtype=theano.config.floatX)
        X[0, :] = self.init_state
        y = np.zeros(self.samples, dtype=theano.config.floatX)

        # Generate data
        for i in range(1, X.shape[0]):
            X[i, 0] = 1 - self.alpha*X[i-1, 0]**2 + X[i-1, 1]
            X[i, 1] = self.beta*X[i-1, 0]

        # Capture the last state for the last target and then remove it
        # from the training data.
        last_target = X[-1, :]
        X = X[:-1, :]

        # Reshape data in to dense matrix
        # TODO This should be refactored to be more memory efficient by
        # eliminating redudancy.
        final_rows = X.shape[0] - self.frame_length/2
        Z = np.zeros(
            (final_rows, self.frame_length),
            dtype=theano.config.floatX
        )
        for i in range(final_rows):
            record = X[i:i+self.frame_length/2].reshape(1, self.frame_length)
            Z[i, :] = record

        # Grab the targets from the training data. They happen to be the last
        # two columns of every row except the first. Also add the last target.
        y = np.zeros((Z.shape[0], 2), dtype=theano.config.floatX)
        y[:-1, :] = Z[1:, self.frame_length-2:self.frame_length]
        y[-1, :] = last_target

        return (Z, y)
