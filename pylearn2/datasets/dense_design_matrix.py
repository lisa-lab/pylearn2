"""TODO: module-level docstring."""
import numpy as N
import copy
from pylearn2.datasets.dataset import Dataset


class DenseDesignMatrix(Dataset):
    """A class for representing datasets that can be stored
       as a dense design matrix, such as MNIST or CIFAR10.
       """
    def __init__(self, X=None, topo_view=None, y=None,
                 view_converter=None, rng=None):
        """
            Parameters
            ----------

            X:  Should be supplied if topo_view is not
                A design matrix of shape (number examples, number features)
                that defines the dataset

            topo_view:  Should be supplied if X is not.
                        A tensor whose first dimension is of length number
                        examples. The remaining tensor dimensions are examples
                        with topological significance, e.g. for images
                        the remaining axes are rows, columns, and channels.
            y:  Labels for the examples. Optional.
            view_converter: An object for converting between design matrices
                            and topological views. Currently DefaultViewConverter
                            is the only type available but later we may want
                            to add one that uses the retina encoding that the
                            U of T group uses.
            rng:    A random number generator used for picking random indices
                    into the design matrix when choosing minibatches
        """


        if X is not None:
            self.X = X
            self.view_converter = view_converter
        else:
            assert topo_view is not None
            self.set_topological_view(topo_view)
        #

        self.y = y
        if rng is None:
            rng = N.random.RandomState([17, 2, 946])
        #
        self.default_rng = copy.copy(rng)
        self.rng = rng
        self.compress = False
        self.design_loc = None

    def use_design_loc(self, path):
        """ When pickling, save the design matrix to path as a .npy file
            rather than pickling the design matrix along with the rest
            of the dataset object. This avoids pickle's unfortunate
            behavior of using 2X the RAM when unpickling. """
        self.design_loc = path

    def enable_compression(self):
        """ If called, when pickled the dataset will be saved using only
            8 bits per element """
        self.compress = True

    def __getstate__(self):
        rval = copy.copy(self.__dict__)

        if self.compress:
            rval['compress_min'] = rval['X'].min(axis=0)
            rval['X'] -= rval['compress_min']
            rval['compress_max'] = rval['X'].max(axis=0)
            rval['compress_max'][rval['compress_max'] == 0] = 1
            rval['X'] *= 255. / rval['compress_max']
            rval['X'] = N.cast['uint8'](rval['X'])

        if self.design_loc is not None:
            N.save(self.design_loc, rval['X'])
            del rval['X']

        return rval

    def __setstate__(self, d):

        if d['design_loc'] is not None:
            d['X'] = N.load(d['design_loc'])

        if d['compress']:
            X = d['X']
            mx = d['compress_max']
            mn = d['compress_min']
            del d['compress_max']
            del d['compress_min']
            d['X'] = 0
            self.__dict__.update(d)
            self.X = N.cast['float32'](X) * mx / 255. + mn
        else:
            self.__dict__.update(d)

    def get_stream_position(self):
        """ If we view the dataset as providing a stream of random examples to read,
            the object returned uniquely identifies our current position in that stream. """
        return copy.copy(self.rng)

    def set_stream_position(self, pos):
        """ Return to a state specified by an object returned from get_stream_position """
        self.rng = copy.copy(pos)

    def restart_stream(self):
        """ Return to the default initial state of the random example stream """
        self.reset_RNG()

    def reset_RNG(self):
        """ Restore the default seed of the rng used for choosing random examples """

        if 'default_rng' not in dir(self):
            self.default_rng = N.random.RandomState([17, 2, 946])
        self.rng = copy.copy(self.default_rng)

    def apply_preprocessor(self, preprocessor, can_fit=False):
        preprocessor.apply(self, can_fit)

    def get_topological_view(self, mat=None):
        """ Return mat, in a topology preserving format
            If mat is None, uses the entire dataset as mat"""
        if self.view_converter is None:
            raise Exception("Tried to call get_topological_view on a dataset "
                            "that has no view converter")
        if mat is None:
            mat = self.X
        return self.view_converter.design_mat_to_topo_view(mat)

    def get_weights_view(self, mat):
        """ Return a view of mat in the topology preserving format.
            Currently the same as get_topological_view """

        if self.view_converter is None:
            raise Exception("Tried to call get_weights_view on a dataset "
                            "that has no view converter")

        return self.view_converter.design_mat_to_weights_view(mat)

    def set_topological_view(self, V):
        """ Sets the dataset to represent V, where V is a batch
            of topological views of examples """
        assert not N.any(N.isnan(V))
        self.view_converter = DefaultViewConverter(V.shape[1:])
        self.X = self.view_converter.topo_view_to_design_mat(V)
        assert not N.any(N.isnan(self.X))

    def get_design_matrix(self, topo=None):
        """ Return topo (a batch of examples in topology preserving format),
        in design matrix format

        If topo is None, uses the entire dataset as topo"""
        if topo is not None:
            if self.view_converter is None:
                raise Exception("Tried to convert from topological_view to design matrix "
                        "using a dataset that has no view converter")
            return self.view_converter.topo_view_to_design_mat(topo)

        return self.X

    def set_design_matrix(self, X):
        assert len(X.shape) == 2
        assert not N.any(N.isnan(X))
        self.X = X

    def get_batch_design(self, batch_size, include_labels=False):
        idx = self.rng.randint(self.X.shape[0] - batch_size + 1)
        rx = self.X[idx:idx + batch_size, :]
        if include_labels:
            ry = self.y[idx:idx + batch_size]
            return rx, ry
        return rx

    def get_batch_topo(self, batch_size):

        batch_design  = self.get_batch_design(batch_size)

        rval = self.view_converter.design_mat_to_topo_view(batch_design)

        return rval


    def view_shape(self):
        return self.view_converter.view_shape()


class DefaultViewConverter(object):
    def __init__(self, shape):
        self.shape = shape
        self.pixels_per_channel = 1
        for dim in self.shape[:-1]:
            self.pixels_per_channel *= dim

    def view_shape(self):
        return self.shape

    def design_mat_to_topo_view(self, X):
        assert len(X.shape) == 2
        batch_size = X.shape[0]
        channel_shape = [batch_size]
        for dim in self.shape[:-1]:
            channel_shape.append(dim)
        channel_shape.append(1)
        if self.shape[-1] * self.pixels_per_channel != X.shape[1]:
            raise ValueError('View converter with '+str(self.shape[-1]) + \
                    ' channels and '+str(self.pixels_per_channel)+' pixels '
                    'per channel asked to convert design matrix with'
                    ' '+str(X.shape[1])+' columns.')
        start = lambda i: self.pixels_per_channel * i
        stop = lambda i: self.pixels_per_channel * (i + 1)
        channels = [X[:, start(i):stop(i)].reshape(*channel_shape)
                    for i in xrange(self.shape[-1])]

        rval = N.concatenate(channels, axis=len(self.shape))
        assert rval.shape[0] == X.shape[0]
        assert len(rval.shape) == len(self.shape) + 1
        return rval

    def design_mat_to_weights_view(self, X):
        return self.design_mat_to_topo_view(X)

    def topo_view_to_design_mat(self, V):
        num_channels = self.shape[-1]
        if N.any(N.asarray(self.shape) != N.asarray(V.shape[1:])):
            raise ValueError('View converter for views of shape batch size '
                             'followed by '
                             + str(self.shape) +
                             ' given tensor of shape ' + str(V.shape))
        batch_size = V.shape[0]
        channels = [
                    V[:, :, :, i].reshape(batch_size, self.pixels_per_channel)
                    for i in xrange(num_channels)
                    ]

        return N.concatenate(channels, axis=1)


def from_dataset(dataset, num_examples):
    V = dataset.get_batch_topo(num_examples)
    return DenseDesignMatrix(topo_view=V)
