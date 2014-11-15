"""
The ZCA Dataset class.

This is basically a prototype for a more general idea of being
able to invert preprocessors and view data in more than one
format. This should be expected to change, but had to go in
pylearn2 to support pylearn2/scripts/papers/maxout
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

from functools import wraps
import logging
import warnings
import numpy as np
from theano.compat.six.moves import xrange
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.config import yaml_parse
from pylearn2.datasets import control


logger = logging.getLogger(__name__)


class ZCA_Dataset(DenseDesignMatrix):
    """
    A Dataset that was created by ZCA whitening a DenseDesignMatrix.
    Supports viewing the data both in the new ZCA whitened space and
    mapping examples back to the original space.

    Parameters
    ----------
    preprocessed_dataset : Dataset
        The underlying raw dataset
    preprocessor : ZCA
        The ZCA preprocessor.
    start : int
        Start reading examples from this index (inclusive)
    stop : int
        Stop reading examples at this index (exclusive)
    """

    def get_test_set(self):
        """
        Returns the test set.
        """
        yaml = self.preprocessed_dataset.yaml_src
        yaml = yaml.replace('train', 'test')
        args = {}
        args.update(self.args)
        del args['self']
        args['start'] = None
        args['stop'] = None
        args['preprocessed_dataset'] = yaml_parse.load(yaml)
        return ZCA_Dataset(**args)

    def __init__(self,
                 preprocessed_dataset,
                 preprocessor,
                 start=None,
                 stop=None,
                 axes=None):

        if axes is not None:
            warnings.warn("The axes argument to ZCA_Dataset no longer has "
                          "any effect. Its role is now carried out by the "
                          "Space you pass to Dataset.iterator. You should "
                          "remove 'axes' arguments from calls to "
                          "ZCA_Dataset. This argument may be removed from "
                          "the library after 2015-05-05.")
        self.args = locals()

        self.preprocessed_dataset = preprocessed_dataset
        self.preprocessor = preprocessor
        self.rng = self.preprocessed_dataset.rng
        self.data_specs = preprocessed_dataset.data_specs
        self.X_space = preprocessed_dataset.X_space
        self.X_topo_space = preprocessed_dataset.X_topo_space
        self.view_converter = preprocessed_dataset.view_converter

        self.y = preprocessed_dataset.y
        self.y_labels = preprocessed_dataset.y_labels

        # Defined up here because PEP8 requires excessive indenting if defined
        # where it is used.
        msg = ("Expected self.y to have dim 2, but it has %d. Maybe you are "
               "loading from an outdated pickle file?")
        if control.get_load_data():
            if start is not None:
                self.X = preprocessed_dataset.X[start:stop, :]
                if self.y is not None:
                    if self.y.ndim != 2:
                        raise ValueError(msg % self.y.ndim)
                    self.y = self.y[start:stop, :]
                assert self.X.shape[0] == stop - start
            else:
                self.X = preprocessed_dataset.X
        else:
            self.X = None
        if self.X is not None:
            if self.y is not None:
                assert self.y.shape[0] == self.X.shape[0]

        if getattr(preprocessor, "inv_P_", None) is None:
            warnings.warn("ZCA preprocessor.inv_P_ was none. Computing "
                          "inverse of preprocessor.P_ now. This will take "
                          "some time. For efficiency, it is recommended that "
                          "in the future you compute the inverse in ZCA.fit() "
                          "instead, by passing it compute_inverse=True.")
            logger.info('inverting...')
            preprocessor.inv_P_ = np.linalg.inv(preprocessor.P_)
            logger.info('...done inverting')

    @wraps(DenseDesignMatrix.has_targets)
    def has_targets(self):
        return self.preprocessed_dataset.has_targets()

    def adjust_for_viewer(self, X):
        """
        Formats examples for use with PatchViewer

        Parameters
        ----------
        X : 2d numpy array
            One example per row

        Returns
        -------
        output : 2d numpy array
            One example per row, rescaled so the maximum absolute value
            within each row is (almost) 1.
        """
        rval = X.copy()

        for i in xrange(rval.shape[0]):
            rval[i, :] /= np.abs(rval[i, :]).max() + 1e-12

        return rval

    def adjust_to_be_viewed_with(self, X, other, per_example=False):
        """
        Adjusts `X` using the same transformation that would
        be applied to `other` if `other` were passed to
        `adjust_for_viewer`. This is useful for visualizing `X`
        alongside `other`.

        Parameters
        ----------
        X : 2d ndarray
            Examples to be adjusted
        other : 2d ndarray
            Examples that define the scale
        per_example : bool
            Default: False. If True, compute the scale separately
            for each example. If False, compute one scale for the
            whole batch.
        """

        assert X.shape == other.shape, (X.shape, other.shape)

        rval = X.copy()

        if per_example:
            for i in xrange(rval.shape[0]):
                rval[i, :] /= np.abs(other[i, :]).max()
        else:
            rval /= np.abs(other).max()

        rval = np.clip(rval, -1., 1.)

        return rval

    def mapback_for_viewer(self, X):
        """
        Map `X` back to the original space (before ZCA preprocessing)
        and adjust it for display with PatchViewer.

        Parameters
        ----------
        X : 2d numpy array
            The examples to be mapped back and adjusted

        Returns
        -------
        output : 2d numpy array
            The examples in the original space, adjusted for display
        """
        assert X.ndim == 2
        rval = self.preprocessor.inverse(X)
        rval = self.preprocessed_dataset.adjust_for_viewer(rval)

        return rval

    def mapback(self, X):
        """
        Map `X` back to the original space (before ZCA preprocessing)

        Parameters
        ----------
        X : 2d numpy array
            The examples to be mapped back

        Returns
        -------
        output : 2d numpy array
            The examples in the original space
        """
        return self.preprocessor.inverse(X)
