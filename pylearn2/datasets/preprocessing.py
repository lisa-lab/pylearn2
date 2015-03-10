"""
Functionality for preprocessing Datasets.
"""

__authors__ = "Ian Goodfellow, David Warde-Farley, Guillaume Desjardins, " \
              "and Mehdi Mirza"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley", "Guillaume Desjardins",
               "Mehdi Mirza"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"


import copy
import logging
import time
import warnings
import os
import numpy
from theano.compat.six.moves import xrange
import scipy
try:
    from scipy import linalg
except ImportError:
    warnings.warn("Could not import scipy.linalg")
import theano
from theano import function, tensor

from pylearn2.blocks import Block
from pylearn2.linear.conv2d import Conv2D
from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils.insert_along_axis import insert_columns
from pylearn2.utils import sharedX
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import contains_nan


log = logging.getLogger(__name__)


convert_axes = Conv2DSpace.convert_numpy


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
        .. todo::

            WRITEME

        Parameters
        ----------
        dataset : Dataset
            The dataset to act on.
        can_fit : bool
            If True, the Preprocessor can adapt internal parameters
            based on the contents of dataset. Otherwise it must not
            fit any parameters, or must re-use old ones.
            Subclasses should still have this default to False, so
            that the behavior of the preprocessors is uniform.

        Notes
        -----
        Typical usage:

        .. code-block::  python

            # Learn PCA preprocessing and apply it to the training set
            my_pca_preprocessor.apply(training_set, can_fit = True)
            # Now apply the same transformation to the test set
            my_pca_preprocessor.apply(test_set, can_fit = False)

        This method must take a dataset, rather than a numpy ndarray, for a
        variety of reasons:

        - Preprocessors should work on any dataset, and not all
          datasets will store their data as ndarrays.
        - Preprocessors often need to change a dataset's
          metadata.  For example, suppose you have a
          DenseDesignMatrix dataset of images. If you implement
          a fovea Preprocessor that reduces the dimensionality
          of images by sampling them finely near the center and
          coarsely with blurring at the edges, then your
          preprocessor will need to change the way that the
          dataset converts example vectors to images for
          visualization.
        """

        raise NotImplementedError(str(type(self)) +
                                  " does not implement an apply method.")

    def invert(self):
        """
        Do any necessary prep work to be able to support the "inverse" method
        later. Default implementation is no-op.
        """
        pass


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
        raise NotImplementedError(str(type(self)) +
                                  " does not implement as_block.")


class BlockPreprocessor(ExamplewisePreprocessor):

    """
    An ExamplewisePreprocessor implemented by a Block.

    Parameters
    ----------
    block : WRITEME
    """

    def __init__(self, block):
        self.block = block

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        assert not can_fit
        dataset.X = self.block.perform(dataset.X)


class Pipeline(Preprocessor):

    """
    A Preprocessor that sequentially applies a list
    of other Preprocessors.

    Parameters
    ----------
    items : WRITEME
    """

    def __init__(self, items=None):
        self.items = items if items is not None else []

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        for item in self.items:
            item.apply(dataset, can_fit)


class ExtractGridPatches(Preprocessor):

    """
    Converts a dataset of images into a dataset of patches extracted along a
    regular grid from each image.  The order of the images is
    preserved.

    Parameters
    ----------
    patch_shape : WRITEME
    patch_stride : WRITEME
    """

    def __init__(self, patch_shape, patch_stride):
        self.patch_shape = patch_shape
        self.patch_stride = patch_stride

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
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
        output = numpy.zeros(output_shape, dtype=X.dtype)
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

        # fix lables
        if dataset.y is not None:
            dataset.y = numpy.repeat(dataset.y, num_patches / X.shape[0])


class ReassembleGridPatches(Preprocessor):

    """
    Converts a dataset of patches into a dataset of full examples.

    This is the inverse of ExtractGridPatches for patch_stride=patch_shape.

    Parameters
    ----------
    orig_shape : WRITEME
    patch_shape : WRITEME
    """

    def __init__(self, orig_shape, patch_shape):
        self.patch_shape = patch_shape
        self.orig_shape = orig_shape

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
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
        reassembled = numpy.zeros(reassembled_shape, dtype=patches.dtype)
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
                reraise_as(IndexError('Gave index of ' + str(i) +
                                      ', : into thing of shape ' +
                                      str(patches.shape)))
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

        # fix labels
        if dataset.y is not None:
            dataset.y = dataset.y[::patches.shape[0] / reassembled_shape[0]]


class ExtractPatches(Preprocessor):

    """
    Converts an image dataset into a dataset of patches
    extracted at random from the original dataset.

    Parameters
    ----------
    patch_shape : WRITEME
    num_patches : WRITEME
    rng : WRITEME
    """

    def __init__(self, patch_shape, num_patches, rng=None):
        self.patch_shape = patch_shape
        self.num_patches = num_patches
        self.start_rng = make_np_rng(copy.copy(rng),
                                     [1, 2, 3],
                                     which_method="randint")

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
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
        output = numpy.zeros(output_shape, dtype=X.dtype)
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
        dataset.y = None


class ExamplewiseUnitNormBlock(Block):

    """
    A block that takes n-tensors, with training examples indexed along
    the first axis, and normalizes each example to lie on the unit
    sphere.

    Parameters
    ----------
    input_space : WRITEME
    """

    def __init__(self, input_space=None):
        super(ExamplewiseUnitNormBlock, self).__init__()
        self.input_space = input_space

    def __call__(self, batch):
        """
        .. todo::

            WRITEME
        """
        if self.input_space:
            self.input_space.validate(batch)
        squared_batch = batch ** 2
        squared_norm = squared_batch.sum(axis=1)
        norm = tensor.sqrt(squared_norm)
        return batch / norm

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME
        """
        self.input_space = space

    def get_input_space(self):
        """
        .. todo::

            WRITEME
        """
        if self.input_space is not None:
            return self.input_space
        raise ValueError("No input space was specified for this Block (%s). "
                         "You can call set_input_space to correct that." %
                         str(self))

    def get_output_space(self):
        """
        .. todo::

            WRITEME
        """
        return self.get_input_space()


class MakeUnitNorm(ExamplewisePreprocessor):

    """
    .. todo::

        WRITEME
    """

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        X = dataset.get_design_matrix()
        X_norm = numpy.sqrt(numpy.sum(X ** 2, axis=1))
        X /= X_norm[:, None]
        dataset.set_design_matrix(X)

    def as_block(self):
        """
        .. todo::

            WRITEME
        """
        return ExamplewiseUnitNormBlock()


class ExamplewiseAddScaleTransform(Block):

    """
    A block that encodes an per-feature addition/scaling transform.
    The addition/scaling can be done in either order.

    Parameters
    ----------
    add : array_like or scalar, optional
        Array or array-like object or scalar, to be added to each
        training example by this Block.
    multiply : array_like, optional
        Array or array-like object or scalar, to be element-wise
        multiplied with each training example by this Block.
    multiply_first : bool, optional
        Whether to perform the multiplication before the addition.
        (default is False).
    input_space : Space, optional
        The input space describing the data
    """

    def __init__(self, add=None, multiply=None, multiply_first=False,
                 input_space=None):
        self.add = numpy.asarray(add)
        self.multiply = numpy.asarray(multiply)
        # TODO: put the constant somewhere sensible.
        if multiply is not None:
            self._has_zeros = numpy.any(abs(multiply) < 1e-14)
        else:
            self._has_zeros = False
        self._multiply_first = multiply_first
        self.input_space = input_space

    def _multiply(self, batch):
        """
        .. todo::

            WRITEME
        """
        if self.multiply is not None:
            batch *= self.multiply
        return batch

    def _add(self, batch):
        """
        .. todo::

            WRITEME
        """
        if self.add is not None:
            batch += self.add
        return batch

    def __call__(self, batch):
        """
        .. todo::

            WRITEME
        """
        if self.input_space:
            self.input_space.validate(batch)
        cur = batch
        if self._multiply_first:
            batch = self._add(self._multiply(batch))
        else:
            batch = self._multiply(self._add(batch))
        return batch

    def inverse(self):
        """
        .. todo::

            WRITEME
        """
        if self._multiply is not None and self._has_zeros:
            raise ZeroDivisionError("%s transformation not invertible "
                                    "due to (near-) zeros in multiplicand" %
                                    self.__class__.__name__)
        else:
            mult_inverse = self._multiply ** -1.
            return self.__class__(add=-self._add, multiply=mult_inverse,
                                  multiply_first=not self._multiply_first)

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME
        """
        self.input_space = space

    def get_input_space(self):
        """
        .. todo::

            WRITEME
        """
        if self.input_space is not None:
            return self.input_space
        raise ValueError("No input space was specified for this Block (%s). "
                         "You can call set_input_space to correct that." %
                         str(self))

    def get_output_space(self):
        """
        .. todo::

            WRITEME
        """
        return self.get_input_space()


class RemoveMean(ExamplewisePreprocessor):

    """
    Subtracts the mean along a given axis, or from every element
    if `axis=None`.

    Parameters
    ----------
    axis : int or None, optional
        Axis over which to take the mean, with the exact same
        semantics as the `axis` parameter of `numpy.mean`.
    """

    def __init__(self, axis=0):
        self._axis = axis
        self._mean = None

    def apply(self, dataset, can_fit=True):
        """
        .. todo::

            WRITEME
        """
        X = dataset.get_design_matrix()
        if can_fit:
            self._mean = X.mean(axis=self._axis)
        else:
            if self._mean is None:
                raise ValueError("can_fit is False, but RemoveMean object "
                                 "has no stored mean or standard deviation")
        X -= self._mean
        dataset.set_design_matrix(X)

    def as_block(self):
        """
        .. todo::

            WRITEME
        """
        if self._mean is None:
            raise ValueError("can't convert %s to block without fitting"
                             % self.__class__.__name__)
        return ExamplewiseAddScaleTransform(add=-self._mean)


class Standardize(ExamplewisePreprocessor):

    """
    Subtracts the mean and divides by the standard deviation.

    Parameters
    ----------
    global_mean : bool, optional
        If `True`, subtract the (scalar) mean over every element
        in the design matrix. If `False`, subtract the mean from
        each column (feature) separately. Default is `False`.
    global_std : bool, optional
        If `True`, after centering, divide by the (scalar) standard
        deviation of every element in the design matrix. If `False`,
        divide by the column-wise (per-feature) standard deviation.
        Default is `False`.
    std_eps : float, optional
        Stabilization factor added to the standard deviations before
        dividing, to prevent standard deviations very close to zero
        from causing the feature values to blow up too much.
        Default is `1e-4`.
    """

    def __init__(self, global_mean=False, global_std=False, std_eps=1e-4):
        self._global_mean = global_mean
        self._global_std = global_std
        self._std_eps = std_eps
        self._mean = None
        self._std = None

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
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
        """
        .. todo::

            WRITEME
        """
        if self._mean is None or self._std is None:
            raise ValueError("can't convert %s to block without fitting"
                             % self.__class__.__name__)
        return ExamplewiseAddScaleTransform(add=-self._mean,
                                            multiply=self._std ** -1)


class ColumnSubsetBlock(Block):

    """
    .. todo::

        WRITEME
    """

    def __init__(self, columns, total):
        self._columns = columns
        self._total = total

    def __call__(self, batch):
        """
        .. todo::

            WRITEME
        """
        if batch.ndim != 2:
            raise ValueError("Only two-dimensional tensors are supported")
        return batch.dimshuffle(1, 0)[self._columns].dimshuffle(1, 0)

    def inverse(self):
        """
        .. todo::

            WRITEME
        """
        return ZeroColumnInsertBlock(self._columns, self._total)

    def get_input_space(self):
        """
        .. todo::

            WRITEME
        """
        return VectorSpace(dim=self._total)

    def get_output_space(self):
        """
        .. todo::

            WRITEME
        """
        return VectorSpace(dim=self._columns)


class ZeroColumnInsertBlock(Block):

    def __init__(self, columns, total):
        """
        .. todo::

            WRITEME
        """
        self._columns = columns
        self._total = total

    def __call__(self, batch):
        """
        .. todo::

            WRITEME
        """
        if batch.ndim != 2:
            raise ValueError("Only two-dimensional tensors are supported")
        return insert_columns(batch, self._total, self._columns)

    def inverse(self):
        """
        .. todo::

            WRITEME
        """
        return ColumnSubsetBlock(self._columns, self._total)

    def get_input_space(self):
        """
        .. todo::

            WRITEME
        """
        return VectorSpace(dim=self._columns)

    def get_output_space(self):
        """
        .. todo::

            WRITEME
        """
        return VectorSpace(dim=self._total)


class RemoveZeroColumns(ExamplewisePreprocessor):

    """
    .. todo::

        WRITEME
    """
    _eps = 1e-8

    def __init__(self):
        self._block = None

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        design_matrix = dataset.get_design_matrix()
        mean = design_matrix.mean(axis=0)
        var = design_matrix.var(axis=0)
        columns, = numpy.where((var < self._eps) & (mean < self._eps))
        self._block = ColumnSubsetBlock

    def as_block(self):
        """
        .. todo::

            WRITEME
        """
        if self._block is None:
            raise ValueError("can't convert %s to block without fitting"
                             % self.__class__.__name__)
        return self._block


class RemapInterval(ExamplewisePreprocessor):

    """
    .. todo::

        WRITEME
    """
    # TODO: Implement as_block

    def __init__(self, map_from, map_to):
        assert map_from[0] < map_from[1] and len(map_from) == 2
        assert map_to[0] < map_to[1] and len(map_to) == 2
        self.map_from = [numpy.float(x) for x in map_from]
        self.map_to = [numpy.float(x) for x in map_to]

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        X = dataset.get_design_matrix()
        X = (X - self.map_from[0]) / numpy.diff(self.map_from)
        X = X * numpy.diff(self.map_to) + self.map_to[0]
        dataset.set_design_matrix(X)


class PCA_ViewConverter(object):

    """
    .. todo::

        WRITEME

    Parameters
    ----------
    to_pca : WRITEME
    to_input : WRITEME
    to_weights : WRITEME
    orig_view_converter : WRITEME
    """

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
        """
        .. todo::

            WRITEME
        """
        return self.orig_view_converter.shape

    def design_mat_to_topo_view(self, X):
        """
        .. todo::

            WRITEME
        """
        to_input = self.to_input(X)
        return self.orig_view_converter.design_mat_to_topo_view(to_input)

    def design_mat_to_weights_view(self, X):
        """
        .. todo::

            WRITEME
        """
        to_weights = self.to_weights(X)
        return self.orig_view_converter.design_mat_to_weights_view(to_weights)

    def topo_view_to_design_mat(self, V):
        """
        .. todo::

            WRITEME
        """
        return self.to_pca(self.orig_view_converter.topo_view_to_design_mat(V))

    def get_formatted_batch(self, batch, dspace):
        """
        .. todo::

            WRITEME
        """
        if isinstance(dspace, VectorSpace):
            # Return the batch in the original storage space
            dspace.np_validate(batch)
            return batch
        else:
            # Uncompress and go through the original view converter
            to_input = self.to_input(batch)
            return self.orig_view_converter.get_formatted_batch(to_input,
                                                                dspace)


class PCA(object):

    """
    .. todo::

        WRITEME

    Parameters
    ----------
    num_components : WRITEME
    whiten : bool, optional
        If False, whitening (or sphering) will not be performed (default).
        If True, the preprocessed data will have zero mean and unit covariance.
    """

    def __init__(self, num_components, whiten=False):
        self._num_components = num_components
        self._whiten = whiten
        self._pca = None
        # TODO: Is storing these really necessary? This computation
        # can't really be merged since we're basically creating the
        # functions in apply(); I see no reason to keep these around.
        self._input = tensor.matrix()
        self._output = tensor.matrix()

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        if self._pca is None:
            if not can_fit:
                raise ValueError("can_fit is False, but PCA preprocessor "
                                 "object has no fitted model stored")
            from pylearn2.models import pca
            self._pca = pca.CovEigPCA(num_components=self._num_components,
                                      whiten=self._whiten)
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
        # assert below fails when 'whiten' is True or sometimes on test
        # or validation set when the preprocessor was fit on train set
        if not self._whiten and can_fit:
            assert proc_var[0] > orig_var.max()

        log.info('original variance: {0}'.format(orig_var.sum()))
        log.info('processed variance: {0}'.format(proc_var.sum()))
        if hasattr(dataset, 'view_converter'):
            if dataset.view_converter is not None:
                new_converter = PCA_ViewConverter(self._transform_func,
                                                  self._invert_func,
                                                  self._convert_weights_func,
                                                  dataset.view_converter)
                dataset.view_converter = new_converter


class Downsample(object):

    """
    Downsamples the topological view

    Parameters
    ----------
    sampling_factor : list or array
        One element for each topological
        dimension of the data
    """

    def __init__(self, sampling_factor):
        self.sampling_factor = sampling_factor

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
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
        kernel = numpy.zeros(kernel_shape, dtype=X.dtype)
        for i in xrange(X.shape[-1]):
            kernel[i, :, :, :, i] = kernel_value
        from theano.tensor.nnet.Conv3D import conv3D
        X_var = tensor.TensorType(broadcastable=[s == 1 for s in X.shape],
                                  dtype=X.dtype)()
        downsampled = conv3D(X_var, kernel, numpy.zeros(X.shape[-1], X.dtype),
                             kernel_shape[1:-1])
        f = function([X_var], downsampled)
        X = f(X)
        if d == 2:
            X = X.reshape([X.shape[0], X.shape[1], X.shape[2], X.shape[4]])
        dataset.set_topological_view(X)


class GlobalContrastNormalization(Preprocessor):

    """
    .. todo::

        WRITEME properly

    See the docstring for `global_contrast_normalize` in
    `pylearn2.expr.preprocessing`.

    Parameters
    ----------
    batch_size : int or None, optional
        If specified, read, apply and write the transformed data
        in batches no larger than `batch_size`.
    sqrt_bias : float, optional
        Defaults to 0 if nothing is specified
    use_std : bool, optional
        Defaults to False if nothing is specified
    """

    def __init__(self, subtract_mean=True,
                 scale=1., sqrt_bias=0., use_std=False, min_divisor=1e-8,
                 batch_size=None):
        self._subtract_mean = subtract_mean
        self._use_std = use_std
        self._sqrt_bias = sqrt_bias
        self._scale = scale
        self._min_divisor = min_divisor
        if batch_size is not None:
            batch_size = int(batch_size)
            assert batch_size > 0, "batch_size must be positive"
        self._batch_size = batch_size

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        if self._batch_size is None:
            X = global_contrast_normalize(dataset.get_design_matrix(),
                                          scale=self._scale,
                                          subtract_mean=self._subtract_mean,
                                          use_std=self._use_std,
                                          sqrt_bias=self._sqrt_bias,
                                          min_divisor=self._min_divisor)
            dataset.set_design_matrix(X)
        else:
            data = dataset.get_design_matrix()
            data_size = data.shape[0]
            last = (numpy.floor(data_size / float(self._batch_size)) *
                    self._batch_size)
            for i in xrange(0, data_size, self._batch_size):
                stop = i + self._batch_size
                log.info("GCN processing data from %d to %d" % (i, stop))
                X = data[i:stop]
                X = global_contrast_normalize(
                    X,
                    scale=self._scale,
                    subtract_mean=self._subtract_mean,
                    use_std=self._use_std,
                    sqrt_bias=self._sqrt_bias,
                    min_divisor=self._min_divisor)
                dataset.set_design_matrix(X, start=i)


class ZCA(Preprocessor):

    """
    Performs ZCA whitening.

    .. TODO::

        WRITEME properly
        add reference

    Parameters
    ----------
    n_components : integer, optional
        Keeps the n_components biggest eigenvalues and corresponding
        eigenvectors of covariance matrix.
    n_drop_components : integer, optional
        Drops the n_drop_components smallest eigenvalues and corresponding
        eigenvectors of covariance matrix. Will only drop components
        when n_components is not set i.e. n_components has preference over
        n_drop_components.
    filter_bias : float, optional
        TODO: verify that default of 0.1 is what was used in the
        Coates and Ng paper, add reference
    store_inverse : bool, optional
        When self.apply(dataset, can_fit=True) store not just the
        preprocessing matrix, but its inverse. This is necessary when
        using this preprocessor to instantiate a ZCA_Dataset.
    """

    def __init__(self, n_components=None, n_drop_components=None,
                 filter_bias=0.1, store_inverse=True):
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
        self.filter_bias = numpy.cast[theano.config.floatX](filter_bias)
        self.has_fit_ = False
        self.store_inverse = store_inverse
        self.P_ = None  # set by fit()
        self.inv_P_ = None  # set by fit(), if self.store_inverse is True

        # Analogous to DenseDesignMatrix.design_loc. If not None, the
        # matrices P_ and inv_P_ will be saved together in <save_path>
        # (or <save_path>.npz, if the suffix is omitted).
        self.matrices_save_path = None

    @staticmethod
    def _gpu_matrix_dot(matrix_a, matrix_b, matrix_c=None):
        """
        Performs matrix multiplication.

        Attempts to use the GPU if it's available. If the matrix multiplication
        is too big to fit on the GPU, this falls back to the CPU after throwing
        a warning.

        Parameters
        ----------
        matrix_a : WRITEME
        matrix_b : WRITEME
        matrix_c : WRITEME
        """
        if not hasattr(ZCA._gpu_matrix_dot, 'theano_func'):
            ma, mb = theano.tensor.matrices('A', 'B')
            mc = theano.tensor.dot(ma, mb)
            ZCA._gpu_matrix_dot.theano_func = \
                theano.function([ma, mb], mc, allow_input_downcast=True)

        theano_func = ZCA._gpu_matrix_dot.theano_func

        try:
            if matrix_c is None:
                return theano_func(matrix_a, matrix_b)
            else:
                matrix_c[...] = theano_func(matrix_a, matrix_b)
                return matrix_c
        except MemoryError:
            warnings.warn('Matrix multiplication too big to fit on GPU. '
                          'Re-doing with CPU. Consider using '
                          'THEANO_FLAGS="device=cpu" for your next '
                          'preprocessor run')
            return numpy.dot(matrix_a, matrix_b, matrix_c)

    @staticmethod
    def _gpu_mdmt(mat, diags):
        """
        Performs the matrix multiplication M * D * M^T.

        First tries to do this on the GPU. If this throws a MemoryError, it
        falls back to the CPU, with a warning message.

        Parameters
        ----------
        mat : WRITEME
        diags : WRITEME
        """

        floatX = theano.config.floatX

        # compile theano function
        if not hasattr(ZCA._gpu_mdmt, 'theano_func'):
            t_mat = theano.tensor.matrix('M')
            t_diags = theano.tensor.vector('D')
            result = theano.tensor.dot(t_mat * t_diags, t_mat.T)
            ZCA._gpu_mdmt.theano_func = theano.function(
                [t_mat, t_diags],
                result,
                allow_input_downcast=True)

        try:
            # function()-call above had to downcast the data. Emit warnings.
            if str(mat.dtype) != floatX:
                warnings.warn('Implicitly converting mat from dtype=%s to '
                              '%s for gpu' % (mat.dtype, floatX))
            if str(diags.dtype) != floatX:
                warnings.warn('Implicitly converting diag from dtype=%s to '
                              '%s for gpu' % (diags.dtype, floatX))

            return ZCA._gpu_mdmt.theano_func(mat, diags)

        except MemoryError:
            # fall back to cpu
            warnings.warn('M * D * M^T was too big to fit on GPU. '
                          'Re-doing with CPU. Consider using '
                          'THEANO_FLAGS="device=cpu" for your next '
                          'preprocessor run')
            return numpy.dot(mat * diags, mat.T)

    def set_matrices_save_path(self, matrices_save_path):
        """
        Analogous to DenseDesignMatrix.use_design_loc().

        If a matrices_save_path is set, when this ZCA is pickled, the internal
        parameter matrices will be saved separately to `matrices_save_path`, as
        a numpy .npz archive. This uses half the memory that a normal pickling
        does.

        Parameters
        ----------
        matrices_save_path : WRITEME
        """
        if matrices_save_path is not None:
            assert isinstance(matrices_save_path, str)
            matrices_save_path = os.path.abspath(matrices_save_path)

            if os.path.isdir(matrices_save_path):
                raise IOError('Matrix save path "%s" must not be an existing '
                              'directory.')

            assert matrices_save_path[-1] not in ('/', '\\')
            if not os.path.isdir(os.path.split(matrices_save_path)[0]):
                raise IOError('Couldn\'t find parent directory:\n'
                              '\t"%s"\n'
                              '\t of matrix path\n'
                              '\t"%s"')

        self.matrices_save_path = matrices_save_path

    def __getstate__(self):
        """
        Used by pickle.  Returns a dictionary to pickle in place of
        self.__dict__.

        If self.matrices_save_path is set, this saves the matrices P_ and
        inv_P_ separately in matrices_save_path as a .npz archive, which uses
        much less space & memory than letting pickle handle them.
        """
        result = copy.copy(self.__dict__)  # shallow copy
        if self.matrices_save_path is not None:
            matrices = {'P_': self.P_}
            if self.inv_P_ is not None:
                matrices['inv_P_'] = self.inv_P_

            numpy.savez(self.matrices_save_path, **matrices)

            # Removes the matrices from the dictionary to be pickled.
            for key, matrix in matrices.items():
                del result[key]

        return result

    def __setstate__(self, state):
        """
        Used to unpickle.

        Parameters
        ----------
        state : dict
            The dictionary created by __setstate__, presumably unpickled
            from disk.
        """

        # Patch old pickle files
        if 'matrices_save_path' not in state:
            state['matrices_save_path'] = None

        if state['matrices_save_path'] is not None:
            matrices = numpy.load(state['matrices_save_path'])

            # puts matrices' items into state, overriding any colliding keys in
            # state.
            state = dict(state.items() + matrices.items())
            del matrices

        self.__dict__.update(state)

        if not hasattr(self, "inv_P_"):
            self.inv_P_ = None

    def fit(self, X):
        """
        Fits this `ZCA` instance to a design matrix `X`.

        Parameters
        ----------
        X : ndarray
            A matrix where each row is a datum.

        Notes
        -----
        Implementation details:
        Stores result as `self.P_`.
        If self.store_inverse is true, this also computes `self.inv_P_`.
        """

        assert X.dtype in ['float32', 'float64']
        assert not contains_nan(X)
        assert len(X.shape) == 2
        n_samples = X.shape[0]
        if self.copy:
            X = X.copy()
        # Center data
        self.mean_ = numpy.mean(X, axis=0)
        X -= self.mean_

        log.info('computing zca of a {0} matrix'.format(X.shape))
        t1 = time.time()

        bias = self.filter_bias * scipy.sparse.identity(X.shape[1],
                                                        theano.config.floatX)

        covariance = ZCA._gpu_matrix_dot(X.T, X) / X.shape[0] + bias
        t2 = time.time()
        log.info("cov estimate took {0} seconds".format(t2 - t1))

        t1 = time.time()
        eigs, eigv = linalg.eigh(covariance)
        t2 = time.time()

        log.info("eigh() took {0} seconds".format(t2 - t1))
        assert not contains_nan(eigs)
        assert not contains_nan(eigv)
        assert eigs.min() > 0

        if self.n_components and self.n_drop_components:
            raise ValueError('Either n_components or n_drop_components'
                             'should be specified')

        if self.n_components:
            eigs = eigs[-self.n_components:]
            eigv = eigv[:, -self.n_components:]

        if self.n_drop_components:
            eigs = eigs[self.n_drop_components:]
            eigv = eigv[:, self.n_drop_components:]

        t1 = time.time()

        sqrt_eigs = numpy.sqrt(eigs)
        try:
            self.P_ = ZCA._gpu_mdmt(eigv, 1.0 / sqrt_eigs)
        except MemoryError:
            warnings.warn()
            self.P_ = numpy.dot(eigv * (1.0 / sqrt_eigs), eigv.T)

        t2 = time.time()
        assert not contains_nan(self.P_)
        self.has_fit_ = True

        if self.store_inverse:
            self.inv_P_ = ZCA._gpu_mdmt(eigv, sqrt_eigs)
        else:
            self.inv_P_ = None

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        # Compiles apply.x_minus_mean_times_p(), a numeric Theano function that
        # evauates dot(X - mean, P)
        if not hasattr(ZCA, '_x_minus_mean_times_p'):
            x_symbol = tensor.matrix('X')
            mean_symbol = tensor.vector('mean')
            p_symbol = tensor.matrix('P_')
            new_x_symbol = tensor.dot(x_symbol - mean_symbol, p_symbol)
            ZCA._x_minus_mean_times_p = theano.function([x_symbol,
                                                         mean_symbol,
                                                         p_symbol],
                                                        new_x_symbol)

        X = dataset.get_design_matrix()
        assert X.dtype in ['float32', 'float64']
        if not self.has_fit_:
            assert can_fit
            self.fit(X)

        new_X = ZCA._gpu_matrix_dot(X - self.mean_, self.P_)
        dataset.set_design_matrix(new_X)

    def inverse(self, X):
        """
        .. todo::

            WRITEME
        """
        assert X.ndim == 2

        if self.inv_P_ is None:
            warnings.warn("inv_P_ was None. Computing "
                          "inverse of P_ now. This will take "
                          "some time. For efficiency, it is recommended that "
                          "in the future you compute the inverse in ZCA.fit() "
                          "instead, by passing it store_inverse=True.")
            log.info('inverting...')
            self.inv_P_ = numpy.linalg.inv(self.P_)
            log.info('...done inverting')

        return self._gpu_matrix_dot(X, self.inv_P_) + self.mean_


class LeCunLCN(ExamplewisePreprocessor):

    """
    Yann LeCun local contrast normalization

    .. todo::

        WRITEME properly

    Parameters
    ----------
    img_shape : WRITEME
    kernel_size : int, optional
        local contrast kernel size
    batch_size: int, optional
        If dataset is based on PyTables use a batch size smaller than
        10000. Otherwise any batch size diffrent than datasize is not
        supported yet.
    threshold : float
        Threshold for denominator
    channels : list or None, optional
        List of channels to normalize.
        If none, will apply it on all channels.
    """

    def __init__(self, img_shape, kernel_size=7, batch_size=5000,
                 threshold=1e-4, channels=None):
        self._img_shape = img_shape
        self._kernel_size = kernel_size
        self._batch_size = batch_size
        self._threshold = threshold
        if channels is None:
            self._channels = range(3)
        else:
            if isinstance(channels, list) or isinstance(channels, tuple):
                self._channels = channels
            elif isinstance(channels, int):
                self._channels = [channels]
            else:
                raise ValueError("channels should be either a list or int")

    def transform(self, x):
        """
        .. todo::

            WRITEME properly

        Parameters
        ----------
        X : WRITEME
            data with axis [b, 0, 1, c]
        """
        for i in self._channels:
            assert isinstance(i, int)
            assert i >= 0 and i <= x.shape[3]

            x[:, :, :, i] = lecun_lcn(x[:, :, :, i],
                                      self._img_shape,
                                      self._kernel_size,
                                      self._threshold)
            return x

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        axes = ['b', 0, 1, 'c']
        data_size = dataset.X.shape[0]

        if self._channels is None:
            self._channels

        last = (numpy.floor(data_size / float(self._batch_size)) *
                self._batch_size)
        for i in xrange(0, data_size, self._batch_size):
            stop = (i + numpy.mod(data_size, self._batch_size)
                    if i >= last else
                    i + self._batch_size)
            log.info("LCN processing data from {0} to {1}".format(i, stop))
            transformed = self.transform(convert_axes(
                dataset.get_topological_view(dataset.X[i:stop, :]),
                dataset.view_converter.axes, axes))
            transformed = convert_axes(transformed,
                                       axes,
                                       dataset.view_converter.axes)
            if self._batch_size != data_size:
                if isinstance(dataset.X, numpy.ndarray):
                    # TODO have a separate class for non pytables datasets
                    transformed = convert_axes(transformed,
                                               dataset.view_converter.axes,
                                               ['b', 0, 1, 'c'])
                    transformed = transformed.reshape(transformed.shape[0],
                                                      transformed.shape[1] *
                                                      transformed.shape[2] *
                                                      transformed.shape[3])
                    dataset.X[i:stop] = transformed
                else:
                    dataset.set_topological_view(transformed,
                                                 dataset.view_converter.axes,
                                                 start=i)

        if self._batch_size == data_size:
            dataset.set_topological_view(transformed,
                                         dataset.view_converter.axes)


class RGB_YUV(ExamplewisePreprocessor):

    """
    Converts image color channels from rgb to yuv and vice versa

    Parameters
    ----------
    rgb_yuv : bool, optional
        If true converts from rgb to yuv,
        if false converts from yuv to rgb
    batch_size : int, optional
        Batch_size to make conversions in batches
    """

    def __init__(self, rgb_yuv=True, batch_size=5000):

        self._batch_size = batch_size
        self._rgb_yuv = rgb_yuv

    def yuv_rgb(self, x):
        """
        .. todo::

            WRITEME
        """
        y = x[:, :, :, 0]
        u = x[:, :, :, 1]
        v = x[:, :, :, 2]

        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u

        x[:, :, :, 0] = r
        x[:, :, :, 1] = g
        x[:, :, :, 2] = b

        return x

    def rgb_yuv(self, x):
        """
        .. todo::

            WRITEME
        """
        r = x[:, :, :, 0]
        g = x[:, :, :, 1]
        b = x[:, :, :, 2]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b

        x[:, :, :, 0] = y
        x[:, :, :, 1] = u
        x[:, :, :, 2] = v

        return x

    def transform(self, x, dataset_axes):
        """
        .. todo::

            WRITEME
        """
        axes = ['b', 0, 1, 'c']
        x = convert_axes(x, dataset_axes, axes)
        if self._rgb_yuv:
            x = self.rgb_yuv(x)
        else:
            x = self.yuv_rgb(x)
        x = convert_axes(x, axes, dataset_axes)
        return x

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        X = dataset.X
        data_size = X.shape[0]
        last = (numpy.floor(data_size / float(self._batch_size)) *
                self._batch_size)
        for i in xrange(0, data_size, self._batch_size):
            stop = (i + numpy.mod(data_size, self._batch_size)
                    if i >= last else
                    i + self._batch_size)
            log.info("RGB_YUV processing data from {0} to {1}".format(i, stop))
            data = dataset.get_topological_view(X[i:stop])
            transformed = self.transform(data, dataset.view_converter.axes)

            # TODO have a separate class for non pytables datasets
            # or add start option to dense_design_matrix
            if isinstance(dataset.X, numpy.ndarray):
                transformed = convert_axes(transformed,
                                           dataset.view_converter.axes,
                                           ['b', 0, 1, 'c'])
                transformed = transformed.reshape(transformed.shape[0],
                                                  transformed.shape[1] *
                                                  transformed.shape[2] *
                                                  transformed.shape[3])
                dataset.X[i:stop] = transformed
            else:
                dataset.set_topological_view(transformed,
                                             dataset.view_converter.axes,
                                             start=i)


class CentralWindow(Preprocessor):

    """
    Preprocesses an image dataset to contain only the central window.

    Parameters
    ----------
    window_shape : WRITEME
    """

    def __init__(self, window_shape):
        self.__dict__.update(locals())
        del self.self

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        w_rows, w_cols = self.window_shape

        arr = dataset.get_topological_view()

        try:
            axes = dataset.view_converter.axes
        except AttributeError:
            reraise_as(NotImplementedError("I don't know how to tell what the "
                                           "axes of this kind of dataset "
                                           "are."))

        needs_transpose = not axes[1:3] == (0, 1)

        if needs_transpose:
            arr = numpy.transpose(arr,
                                  (axes.index('c'),
                                   axes.index(0),
                                   axes.index(1),
                                   axes.index('b')))

        r_off = (arr.shape[1] - w_rows) // 2
        c_off = (arr.shape[2] - w_cols) // 2
        new_arr = arr[:, r_off:r_off + w_rows, c_off:c_off + w_cols, :]

        if needs_transpose:
            index_map = tuple(('c', 0, 1, 'b').index(axis) for axis in axes)
            new_arr = numpy.transpose(new_arr, index_map)

        dataset.set_topological_view(new_arr, axes=axes)


def lecun_lcn(input, img_shape, kernel_shape, threshold=1e-4):
    """
    Yann LeCun's local contrast normalization

    Original code in Theano by: Guillaume Desjardins

    Parameters
    ----------
    input : WRITEME
    img_shape : WRITEME
    kernel_shape : WRITEME
    threshold : WRITEME
    """
    input = input.reshape((input.shape[0], input.shape[1], input.shape[2], 1))
    X = tensor.matrix(dtype=input.dtype)
    X = X.reshape((len(input), img_shape[0], img_shape[1], 1))

    filter_shape = (1, 1, kernel_shape, kernel_shape)
    filters = sharedX(gaussian_filter(kernel_shape).reshape(filter_shape))

    input_space = Conv2DSpace(shape=img_shape, num_channels=1)
    transformer = Conv2D(filters=filters, batch_size=len(input),
                         input_space=input_space,
                         border_mode='full')
    convout = transformer.lmul(X)

    # For each pixel, remove mean of 9x9 neighborhood
    mid = int(numpy.floor(kernel_shape / 2.))
    centered_X = X - convout[:, mid:-mid, mid:-mid, :]

    # Scale down norm of 9x9 patch if norm is bigger than 1
    transformer = Conv2D(filters=filters,
                         batch_size=len(input),
                         input_space=input_space,
                         border_mode='full')
    sum_sqr_XX = transformer.lmul(X ** 2)

    denom = tensor.sqrt(sum_sqr_XX[:, mid:-mid, mid:-mid, :])
    per_img_mean = denom.mean(axis=[1, 2])
    divisor = tensor.largest(per_img_mean.dimshuffle(0, 'x', 'x', 1), denom)
    divisor = tensor.maximum(divisor, threshold)

    new_X = centered_X / divisor
    new_X = tensor.flatten(new_X, outdim=3)

    f = function([X], new_X)
    return f(input)


def gaussian_filter(kernel_shape):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    kernel_shape : WRITEME
    """
    x = numpy.zeros((kernel_shape, kernel_shape),
                    dtype=theano.config.floatX)

    def gauss(x, y, sigma=2.0):
        Z = 2 * numpy.pi * sigma ** 2
        return 1. / Z * numpy.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = numpy.floor(kernel_shape / 2.)
    for i in xrange(0, kernel_shape):
        for j in xrange(0, kernel_shape):
            x[i, j] = gauss(i - mid, j - mid)

    return x / numpy.sum(x)


class ShuffleAndSplit(Preprocessor):

    """
    .. todo::

        WRITEME properly

    Allocates a numpy rng with the specified seed.
    Note: this must be a seed, not a RandomState. A new RandomState is
    re-created with the same seed every time the preprocessor is called.
    This way if you save the preprocessor and re-use it later it will give
    the same dataset regardless of whether you save the preprocessor before
    or after applying it.
    Shuffles the data, then takes examples in range (start, stop)

    Parameters
    ----------
    seed : WRITEME
    start : int
        WRITEME
    stop : int
        WRITEME
    """

    def __init__(self, seed, start, stop):
        self.__dict__.update(locals())
        del self.self

    def apply(self, dataset, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        start = self.start
        stop = self.stop
        rng = make_np_rng(self.seed, which_method="randint")
        X = dataset.X
        y = dataset.y

        if y is not None:
            assert X.shape[0] == y.shape[0]

        for i in xrange(X.shape[0]):
            j = rng.randint(X.shape[0])
            tmp = X[i, :].copy()
            X[i, :] = X[j, :].copy()
            X[j, :] = tmp

            if y is not None:
                tmp = y[i, :].copy()
                y[i, :] = y[j, :].copy()
                y[j, :] = tmp
        assert start >= 0
        assert stop > start
        assert stop <= X.shape[0]

        dataset.X = X[start:stop, :]
        if y is not None:
            dataset.y = y[start:stop, :]
