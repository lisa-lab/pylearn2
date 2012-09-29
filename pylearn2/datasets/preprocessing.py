"""
Functionality for preprocessing Datasets.
"""

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import warnings
import copy
import numpy as np
try:
    from scipy import linalg
except ImportError:
    warnings.warn("Could not import scipy.linalg")
from theano import function
import theano.tensor as T

from pylearn2.base import Block
from pylearn2.utils.insert_along_axis import insert_columns

class Preprocessor(object):
    """
        Abstract class.

        An object that can preprocess a dataset.

        Preprocessing a dataset implies changing the data that
        a dataset actually stores. This can be useful to save
        memory--if you know you are always going to access only
        the same processed version of the dataset, it is better
        to process it once and discard the original.

        Preprocessors are capable of modifying many aspects of
        a dataset. For example, they can change the way that it
        converts between different formats of data. They can
        change the number of examples that a dataset stores.
        In other words, preprocessors can do a lot more than
        just example-wise transformations of the examples stored
        in the dataset.
    """

    def apply(self, dataset, can_fit=False):
        """
            dataset: The dataset to act on.
            can_fit: If True, the Preprocessor can adapt internal parameters
                     based on the contents of dataset. Otherwise it must not
                     fit any parameters, or must re-use old ones.

            Typical usage:
                # Learn PCA preprocessing and apply it to the training set
                my_pca_preprocessor.apply(training_set, can_fit = True)
                # Now apply the same transformation to the test set
                my_pca_preprocessor.apply(test_set, can_fit = False)

            Note: this method must take a dataset, rather than a numpy ndarray,
                  for a variety of reasons:
                      1) Preprocessors should work on any dataset, and not all
                         datasets will store their data as ndarrays.
                      2) Preprocessors often need to change a dataset's metadata.
                         For example, suppose you have a DenseDesignMatrix dataset
                         of images. If you implement a fovea Preprocessor that
                         reduces the dimensionality of images by sampling them finely
                         near the center and coarsely with blurring at the edges,
                         then your preprocessor will need to change the way that the
                         dataset converts example vectors to images for visualization.
        """

        raise NotImplementedError(str(type(self))+" does not implement an apply method.")

class ExamplewisePreprocessor(Preprocessor):
    """
        Abstract class.

        A Preprocessor that restricts the actions it can do in its
        apply method so that it could be implemented as a Block's
        perform method.

        In other words, this Preprocessor can't modify the Dataset's
        metadata, etc.

        TODO: can these things fit themselves in their apply method?
        That seems like a difference from Block.
    """

    def as_block(self):
        raise NotImplementedError(str(type(self))+" does not implement as_block.")

class BlockPreprocessor(ExamplewisePreprocessor):
    """
        An ExamplewisePreprocessor implemented by a Block.
    """

    def __init__(self, block):
        self.block = block

    def apply(self, dataset, can_fit = False):
        assert not can_fit
        dataset.X = self.block.perform(dataset.X)



class Pipeline(Preprocessor):
    """
        A Preprocessor that sequentially applies a list
        of other Preprocessors.
    """
    def __init__(self):
        self.items = []

    def apply(self, dataset, can_fit=False):
        for item in self.items:
            item.apply(dataset, can_fit)


class ExtractGridPatches(Preprocessor):
    """
    Converts a dataset of images into a dataset of patches extracted along a
    regular grid from each image.  The order of the images is
    preserved.
    """
    def __init__(self, patch_shape, patch_stride):
        self.patch_shape = patch_shape
        self.patch_stride = patch_stride

    def apply(self, dataset, can_fit=False):
        X = dataset.get_topological_view()
        num_topological_dimensions = len(X.shape) - 2
        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ExtractGridPatches with "
                             + str(len(self.patch_shape))
                             + " topological dimensions called on"
                             + " dataset with " +
                             str(num_topological_dimensions) + ".")
        num_patches = X.shape[0]
        max_strides = [X.shape[0] - 1]
        for i in xrange(num_topological_dimensions):
            patch_width = self.patch_shape[i]
            data_width = X.shape[i + 1]
            last_valid_coord = data_width - patch_width
            if last_valid_coord < 0:
                raise ValueError('On topological dimension ' + str(i) +
                                 ', the data has width ' + str(data_width) +
                                 ' but the requested patch width is ' +
                                 str(patch_width))
            stride = self.patch_stride[i]
            if stride == 0:
                max_stride_this_axis = 0
            else:
                max_stride_this_axis = last_valid_coord / stride
            num_strides_this_axis = max_stride_this_axis + 1
            max_strides.append(max_stride_this_axis)
            num_patches *= num_strides_this_axis
        # batch size
        output_shape = [num_patches]
        # topological dimensions
        for dim in self.patch_shape:
            output_shape.append(dim)
        # number of channels
        output_shape.append(X.shape[-1])
        output = np.zeros(output_shape, dtype=X.dtype)
        channel_slice = slice(0, X.shape[-1])
        coords = [0] * (num_topological_dimensions + 1)
        keep_going = True
        i = 0
        while keep_going:
            args = [coords[0]]
            for j in xrange(num_topological_dimensions):
                coord = coords[j + 1] * self.patch_stride[j]
                args.append(slice(coord, coord + self.patch_shape[j]))
            args.append(channel_slice)
            patch = X[args]
            output[i, :] = patch
            i += 1
            # increment coordinates
            j = 0
            keep_going = False
            while not keep_going:
                if coords[-(j + 1)] < max_strides[-(j + 1)]:
                    coords[-(j + 1)] += 1
                    keep_going = True
                else:
                    coords[-(j + 1)] = 0
                    if j == num_topological_dimensions:
                        break
                    j = j + 1
        dataset.set_topological_view(output)


class ReassembleGridPatches(Preprocessor):
    """ Converts a dataset of patches into a dataset of full examples
        This is the inverse of ExtractGridPatches for patch_stride=patch_shape
    """
    def __init__(self, orig_shape, patch_shape):
        self.patch_shape = patch_shape
        self.orig_shape = orig_shape

    def apply(self, dataset, can_fit=False):

        patches = dataset.get_topological_view()

        num_topological_dimensions = len(patches.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ReassembleGridPatches with " +
                             str(len(self.patch_shape)) +
                             " topological dimensions called on dataset " +
                             " with " +
                             str(num_topological_dimensions) + ".")
        num_patches = patches.shape[0]
        num_examples = num_patches
        for im_dim, patch_dim in zip(self.orig_shape, self.patch_shape):
            if im_dim % patch_dim != 0:
                raise Exception('Trying to assemble patches of shape ' +
                                str(self.patch_shape) + ' into images of ' +
                                'shape ' + str(self.orig_shape))
            patches_this_dim = im_dim / patch_dim
            if num_examples % patches_this_dim != 0:
                raise Exception('Trying to re-assemble ' + str(num_patches) +
                                ' patches of shape ' + str(self.patch_shape) +
                                ' into images of shape ' + str(self.orig_shape)
                               )
            num_examples /= patches_this_dim

        # batch size
        reassembled_shape = [num_examples]
        # topological dimensions
        for dim in self.orig_shape:
            reassembled_shape.append(dim)
        # number of channels
        reassembled_shape.append(patches.shape[-1])
        reassembled = np.zeros(reassembled_shape, dtype=patches.dtype)
        channel_slice = slice(0, patches.shape[-1])
        coords = [0] * (num_topological_dimensions + 1)
        max_strides = [num_examples - 1]
        for dim, pd in zip(self.orig_shape, self.patch_shape):
            assert dim % pd == 0
            max_strides.append(dim / pd - 1)
        keep_going = True
        i = 0
        while keep_going:
            args = [coords[0]]
            for j in xrange(num_topological_dimensions):
                coord = coords[j + 1]
                args.append(slice(coord * self.patch_shape[j],
                                  (coord + 1) * self.patch_shape[j]))
                next_shape_coord = reassembled.shape[j + 1]
                assert (coord + 1) * self.patch_shape[j] <= next_shape_coord

            args.append(channel_slice)

            try:
                patch = patches[i, :]
            except IndexError:
                raise IndexError('Gave index of ' + str(i) +
                                 ', : into thing of shape ' +
                                 str(patches.shape))
            reassembled[args] = patch
            i += 1
            j = 0
            keep_going = False
            while not keep_going:
                if coords[-(j + 1)] < max_strides[-(j + 1)]:
                    coords[-(j + 1)] += 1
                    keep_going = True
                else:
                    coords[-(j + 1)] = 0
                    if j == num_topological_dimensions:
                        break
                    j = j + 1

        dataset.set_topological_view(reassembled)


class ExtractPatches(Preprocessor):
    """ Converts an image dataset into a dataset of patches
        extracted at random from the original dataset. """
    def __init__(self, patch_shape, num_patches, rng=None):
        self.patch_shape = patch_shape
        self.num_patches = num_patches

        if rng != None:
            self.start_rng = copy.copy(rng)
        else:
            self.start_rng = np.random.RandomState([1, 2, 3])

    def apply(self, dataset, can_fit=False):
        rng = copy.copy(self.start_rng)

        X = dataset.get_topological_view()

        num_topological_dimensions = len(X.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ExtractPatches with "
                             + str(len(self.patch_shape))
                             + " topological dimensions called on "
                             + "dataset with "
                             + str(num_topological_dimensions) + ".")

        # batch size
        output_shape = [self.num_patches]
        # topological dimensions
        for dim in self.patch_shape:
            output_shape.append(dim)
        # number of channels
        output_shape.append(X.shape[-1])
        output = np.zeros(output_shape, dtype=X.dtype)
        channel_slice = slice(0, X.shape[-1])
        for i in xrange(self.num_patches):
            args = []
            args.append(rng.randint(X.shape[0]))

            for j in xrange(num_topological_dimensions):
                max_coord = X.shape[j + 1] - self.patch_shape[j]
                coord = rng.randint(max_coord + 1)
                args.append(slice(coord, coord + self.patch_shape[j]))
            args.append(channel_slice)
            output[i, :] = X[args]
        dataset.set_topological_view(output)


class ExamplewiseUnitNormBlock(Block):
    """
    A block that takes n-tensors, with training examples indexed along
    the first axis, and normalizes each example to lie on the unit
    sphere.
    """
    def __call__(self, batch):
        squared_batch = batch ** 2
        for dim in range(batch.ndim - 1, 0, -1):
            squared_batch = squared_batch.sum(axis=dim)
        return T.sqrt(squared_batch)


class MakeUnitNorm(ExamplewisePreprocessor):
    def apply(self, dataset, can_fit):
        X = dataset.get_design_matrix()
        X_norm = np.sqrt(np.sum(X ** 2, axis=1))
        X /= X_norm[:, None]
        dataset.set_design_matrix(X)

    def as_block(self):
        return ExamplewiseUnitNormBlock()


class ExamplewiseAddScaleTransform(Block):
    """
    A block that encodes an per-feature addition/scaling transform.
    The addition/scaling can be done in either order.
    """
    def __init__(self, add=None, multiply=None, multiply_first=False):
        """
        Initialize an ExamplewiseAddScaleTransform instance.

        Parameters
        ----------
        add : array_like or scalar, optional
            Array or array-like object or scalar, to be added to each
            training example by this Block.

        multiply : array_like, optional
            Array or array-like object or scalar, to be element-wise
            multiplied with each training example by this Block.

        multiply_first : boolean, optional
            Whether to perform the multiplication before the addition.
            (default is False).
        """
        self._add = np.asarray(add)
        self._multiply = np.asarray(multiply)
        # TODO: put the constant somewhere sensible.
        if multiply is not None:
            self._has_zeros = np.any(abs(multiply) < 1e-14)
        else:
            self._has_zeros = False
        self._multiply_first = multiply_first

    def _multiply(self, batch):
        if self._multiply is not None:
            batch *= self._multiply
        return batch

    def _add(self, batch):
        if self._add is not None:
            batch += self._add
        return batch

    def __call__(self, batch):
        cur = batch
        if self._multiply_first:
            batch = self._add(self._multiply(batch))
        else:
            batch = self._multiply(self._add(batch))
        return batch

    def inverse(self):
        if self._multiply is not None and self._has_zeros:
            raise ZeroDivisionError("%s transformation not invertible "
                                    "due to (near-) zeros in multiplicand" %
                                    self.__class__.__name__)
        else:
            mult_inverse = self._multiply ** -1.
            return self.__class__(add=-self._add, multiply=mult_inverse,
                                  multiply_first=not self._multiply_first)


class RemoveMean(ExamplewisePreprocessor):
    """
    Subtracts the mean along a given axis, or from every element
    if `axis=None`.
    """
    def __init__(self, axis=0):
        """
        Initialize a RemoveMean preprocessor.

        Parameters
        ----------
        axis : int or None
            Axis over which to take the mean, with the exact same
            semantics as the `axis` parameter of `numpy.mean`.
        """
        self._axis = axis
        self._mean = None

    def apply(self, dataset, can_fit=True):
        X = dataset.get_design_matrix()
        if can_fit:
            self._mean = X.mean(axis=self._axis)
        else:
            if self._mean is None:
                raise ValueError("can_fit is False, but Standardize object "
                                 "has no stored mean or standard deviation")
        X -= self._mean
        dataset.set_design_matrix(X)

    def as_block(self):
        if self._mean is None:
            raise  ValueError("can't convert %s to block without fitting"
                              % self.__class__.__name__)
        return ExamplewiseAddScaleTransform(add=-self._mean)


class Standardize(ExamplewisePreprocessor):
    """Subtracts the mean and divides by the standard deviation."""
    def __init__(self, global_mean=False, global_std=False, std_eps=1e-4):
        """
        Initialize a Standardize preprocessor.

        Parameters
        ----------
        global_mean : bool
            If `True`, subtract the (scalar) mean over every element
            in the design matrix. If `False`, subtract the mean from
            each column (feature) separately. Default is `False`.
        global_std : bool
            If `True`, after centering, divide by the (scalar) standard
            deviation of every element in the design matrix. If `False`,
            divide by the column-wise (per-feature) standard deviation.
            Default is `False`.
        std_eps : float
            Stabilization factor added to the standard deviations before
            dividing, to prevent standard deviations very close to zero
            from causing the feature values to blow up too much.
            Default is `1e-4`.
        """
        self._global_mean = global_mean
        self._global_std = global_std
        self._std_eps = std_eps
        self._mean = None
        self._std = None

    def apply(self, dataset, can_fit=True):
        X = dataset.get_design_matrix()
        if can_fit:
            self._mean = X.mean() if self._global_mean else X.mean(axis=0)
            self._std = X.std() if self._global_std else X.std(axis=0)
        else:
            if self._mean is None or self._std is None:
                raise ValueError("can_fit is False, but Standardize object "
                                 "has no stored mean or standard deviation")
        new = (X - self._mean) / (self._std_eps + self._std)
        dataset.set_design_matrix(new)

    def as_block(self):
        if self._mean is None or self._std is None:
            raise  ValueError("can't convert %s to block without fitting"
                              % self.__class__.__name__)
        return ExamplewiseAddScaleTransform(add=-self._mean,
                                            multiply=self._std ** -1)


class ColumnSubsetBlock(Block):
    def __init__(self, columns, total):
        self._columns = columns
        self._total = total

    def __call__(self, batch):
        if batch.ndim != 2:
            raise ValueError("Only two-dimensional tensors are supported")
        return batch.dimshuffle(1, 0)[self._columns].dimshuffle(1, 0)

    def inverse(self):
        return ZeroColumnInsertBlock(self._columns, self._total)


class ZeroColumnInsertBlock(Block):
    def __init__(self, columns, total):
        self._columns = columns
        self._total = total

    def __call__(self, batch):
        if batch.ndim != 2:
            raise ValueError("Only two-dimensional tensors are supported")
        return insert_columns(batch, self._total, self._columns)

    def inverse(self):
        return ColumnSubsetBlock(self._columns, self._total)


class RemoveZeroColumns(ExamplewisePreprocessor):
    _eps = 1e-8

    def __init__(self):
        self._block = None

    def apply(self, dataset, can_fit=True):
        design_matrix = dataset.get_design_matrix()
        mean = design_matrix.mean(axis=0)
        var = design_matrix.var(axis=0)
        columns, = np.where((var < self._eps) & (mean < self._eps))
        self._block = ColumnSubsetBlock

    def as_block(self):
        if self._block is None:
            raise  ValueError("can't convert %s to block without fitting"
                              % self.__class__.__name__)
        return self._block


class RemapInterval(ExamplewisePreprocessor):
    # TODO: Implement as_block
    def __init__(self, map_from, map_to):
        assert map_from[0] < map_from[1] and len(map_from) == 2
        assert map_to[0] < map_to[1] and len(map_to) == 2
        self.map_from = [np.float(x) for x in map_from]
        self.map_to = [np.float(x) for x in map_to]

    def apply(self, dataset, can_fit):
        X = dataset.get_design_matrix()
        X = (X - self.map_from[0]) / np.diff(self.map_from)
        X = X * np.diff(self.map_to) + self.map_to[0]
        dataset.set_design_matrix(X)


class PCA_ViewConverter(object):
    def __init__(self, to_pca, to_input, to_weights, orig_view_converter):
        self.to_pca = to_pca
        self.to_input = to_input
        self.to_weights = to_weights
        if orig_view_converter is None:
            raise ValueError("It doesn't make any sense to make a PCA view "
                             "converter when there's no original view "
                             "converter to define a topology in the first "
                             "place.")
        self.orig_view_converter = orig_view_converter

    def view_shape(self):
        return self.orig_view_converter.shape

    def design_mat_to_topo_view(self, X):
        to_input = self.to_input(X)
        return self.orig_view_converter.design_mat_to_topo_view(to_input)

    def design_mat_to_weights_view(self, X):
        to_weights = self.to_weights(X)
        return self.orig_view_converter.design_mat_to_weights_view(to_weights)

    def topo_view_to_design_mat(self, V):
        return self.to_pca(self.orig_view_converter.topo_view_to_design_mat(V))


class PCA(object):
    def __init__(self, num_components):
        self._num_components = num_components
        self._pca = None
        # TODO: Is storing these really necessary? This computation
        # can't really be merged since we're basically creating the
        # functions in apply(); I see no reason to keep these around.
        self._input = T.matrix()
        self._output = T.matrix()

    def apply(self, dataset, can_fit=False):
        if self._pca is None:
            if not can_fit:
                raise ValueError("can_fit is False, but PCA preprocessor "
                                 "object has no fitted model stored")
            from pylearn2 import pca
            self._pca = pca.CovEigPCA(self._num_components)
            self._pca.train(dataset.get_design_matrix())
            self._transform_func = function([self._input],
                                            self._pca(self._input))
            self._invert_func = function([self._output],
                                         self._pca.reconstruct(self._output))
            self._convert_weights_func = function(
                [self._output],
                self._pca.reconstruct(self._output, add_mean=False)
            )

        orig_data = dataset.get_design_matrix()
        dataset.set_design_matrix(
            self._transform_func(dataset.get_design_matrix())
        )
        proc_data = dataset.get_design_matrix()
        orig_var = orig_data.var(axis=0)
        proc_var = proc_data.var(axis=0)
        assert proc_var[0] > orig_var.max()
        # TODO: logging
        print 'original variance: ' + str(orig_var.sum())
        print 'processed variance: ' + str(proc_var.sum())
        if dataset.view_converter is not None:
            new_converter = PCA_ViewConverter(self._transform_func,
                                              self._invert_func,
                                              self._convert_weights_func,
                                              dataset.view_converter)
            dataset.view_converter = new_converter


class Downsample(object):
    def __init__(self, sampling_factor):
        """
            downsamples the topological view

            parameters
            ----------
            sampling_factor: a list or array with one element for
                            each topological dimension of the data
        """

        self.sampling_factor = sampling_factor

    def apply(self, dataset, can_fit=False):
        X = dataset.get_topological_view()
        d = len(X.shape) - 2
        assert d in [2, 3]
        assert X.dtype == 'float32' or X.dtype == 'float64'
        if d == 2:
            X = X.reshape([X.shape[0], X.shape[1], X.shape[2], 1, X.shape[3]])
        kernel_size = 1
        kernel_shape = [X.shape[-1]]
        for factor in self.sampling_factor:
            kernel_size *= factor
            kernel_shape.append(factor)
        if d == 2:
            kernel_shape.append(1)
        kernel_shape.append(X.shape[-1])
        kernel_value = 1. / float(kernel_size)
        kernel = np.zeros(kernel_shape, dtype=X.dtype)
        for i in xrange(X.shape[-1]):
            kernel[i, :, :, :, i] = kernel_value
        from theano.tensor.nnet.Conv3D import conv3D
        X_var = T.TensorType(broadcastable=[s == 1 for s in X.shape],
                             dtype=X.dtype)()
        downsampled = conv3D(X_var, kernel, np.zeros(X.shape[-1], X.dtype),
                             kernel_shape[1:-1])
        f = function([X_var], downsampled)
        X = f(X)
        if d == 2:
            X = X.reshape([X.shape[0], X.shape[1], X.shape[2], X.shape[4]])
        dataset.set_topological_view(X)


class GlobalContrastNormalization(object):
    def __init__(self, subtract_mean=True, std_bias=10.0, use_norm=False):
        """

        Optionally subtracts the mean of each example
        Then divides each example either by the standard deviation of the
        pixels contained in that example or by the norm of that example

        Parameters:

            subtract_mean: boolean, if True subtract the mean of each example
            std_bias: Add this amount inside the square root when computing
                      the standard deviation or the norm
            use_norm: If True uses the norm instead of the standard deviation


            The default parameters of subtract_mean = True, std_bias = 10.0,
            use_norm = False are used in replicating one step of the
            preprocessing used by Coates, Lee and Ng on CIFAR10 in their paper
            "An Analysis of Single Layer Networks in Unsupervised Feature
            Learning"
        """

        self.subtract_mean = subtract_mean
        self.std_bias = std_bias
        self.use_norm = use_norm

    def apply(self, dataset, can_fit=False):
        X = dataset.get_design_matrix()

        assert X.dtype == 'float32' or X.dtype == 'float64'

        if self.subtract_mean:
            X -= X.mean(axis=1)[:, None]

        if self.use_norm:
            scale = np.sqrt(np.square(X).sum(axis=1) + self.std_bias)
        else:
            # use standard deviation
            scale = np.sqrt(np.square(X).mean(axis=1) + self.std_bias)
        eps = 1e-8
        scale[scale < eps] = 1.
        X /= scale[:, None]
        dataset.set_design_matrix(X)


class ZCA(Preprocessor):
    def __init__(self, n_components=None, n_drop_components=None,
                 filter_bias=0.1):
        warnings.warn("This ZCA preprocessor class is known to yield very "
                      "different results on different platforms. If you plan "
                      "to conduct experiments with this preprocessing on "
                      "multiple machines, it is probably a good idea to do "
                      "the preprocessing on a single machine and copy the "
                      "preprocessed datasets to the others, rather than "
                      "preprocessing the data independently in each "
                      "location.")
        # TODO: test to see if differences across platforms
        # e.g., preprocessing STL-10 patches in LISA lab versus on
        # Ian's Ubuntu 11.04 machine
        # are due to the problem having a bad condition number or due to
        # different version numbers of scipy or something
        self.n_components = n_components
        self.n_drop_components = n_drop_components
        self.copy = True
        self.filter_bias = filter_bias
        self.has_fit_ = False

    def fit(self, X):
        assert X.dtype in ['float32', 'float64']
        assert not np.any(np.isnan(X))
        assert len(X.shape) == 2
        n_samples = X.shape[0]
        if self.copy:
            X = X.copy()
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        # TODO: logging
        print 'computing zca'
        eigs, eigv = linalg.eigh(np.dot(X.T, X) / X.shape[0])
        assert not np.any(np.isnan(eigs))
        assert not np.any(np.isnan(eigv))
        if self.n_components:
            eigs = eigs[:self.n_components]
            eigv = eigv[:, :self.n_components]
        if self.n_drop_components:
            eigs = eigs[self.n_drop_components:]
            eigv = eigv[:, self.n_drop_components:]
        self.P_ = np.dot(eigv * np.sqrt(1.0 / (eigs + self.filter_bias)),
                         eigv.T)
        # print 'zca components'
        # print np.square(self.P_).sum(axis=0)
        assert not np.any(np.isnan(self.P_))
        self.has_fit_ = True

    def apply(self, dataset, can_fit=False):
        X = dataset.get_design_matrix()
        assert X.dtype in ['float32', 'float64']
        if not self.has_fit_:
            assert can_fit
            self.fit(X)
        new_X = np.dot(X - self.mean_, self.P_)
        dataset.set_design_matrix(new_X)

    def invert(self):
        self.inv_P_ = np.linalg.inv(self.P_)

    def inverse(self, X):
        assert X.ndim == 2
        return np.dot(X, self.inv_P_) + self.mean_

