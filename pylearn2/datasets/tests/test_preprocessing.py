"""
Unit tests for ./preprocessing.py
"""

import numpy as np

from theano import config
import theano

from pylearn2.utils import as_floatX
from pylearn2.utils import isfinite
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.preprocessing import (GlobalContrastNormalization,
                                             ExtractGridPatches,
                                             ReassembleGridPatches,
                                             LeCunLCN,
                                             RGB_YUV,
                                             ZCA)


class testGlobalContrastNormalization:

    """Tests for the GlobalContrastNormalization class """

    def test_zero_vector(self):
        """ Test that passing in the zero vector does not result in
            a divide by 0 """

        dataset = DenseDesignMatrix(X=as_floatX(np.zeros((1, 1))))

        # the settings of subtract_mean and use_norm are not relevant to
        # the test
        # std_bias = 0.0 is the only value for which there should be a risk
        # of failure occurring
        preprocessor = GlobalContrastNormalization(subtract_mean=True,
                                                   sqrt_bias=0.0,
                                                   use_std=True)

        dataset.apply_preprocessor(preprocessor)

        result = dataset.get_design_matrix()

        assert isfinite(result)

    def test_unit_norm(self):
        """ Test that using std_bias = 0.0 and use_norm = True
            results in vectors having unit norm """

        tol = 1e-5

        num_examples = 5
        num_features = 10

        rng = np.random.RandomState([1, 2, 3])

        X = as_floatX(rng.randn(num_examples, num_features))

        dataset = DenseDesignMatrix(X=X)

        # the setting of subtract_mean is not relevant to the test
        # the test only applies when std_bias = 0.0 and use_std = False
        preprocessor = GlobalContrastNormalization(subtract_mean=False,
                                                   sqrt_bias=0.0,
                                                   use_std=False)

        dataset.apply_preprocessor(preprocessor)

        result = dataset.get_design_matrix()

        norms = np.sqrt(np.square(result).sum(axis=1))

        max_norm_error = np.abs(norms - 1.).max()

        tol = 3e-5

        assert max_norm_error < tol


def test_extract_reassemble():
    """ Tests that ExtractGridPatches and ReassembleGridPatches are
    inverse of each other """

    rng = np.random.RandomState([1, 3, 7])

    topo = rng.randn(4, 3 * 5, 3 * 7, 2)

    dataset = DenseDesignMatrix(topo_view=topo)

    patch_shape = (3, 7)

    extractor = ExtractGridPatches(patch_shape, patch_shape)
    reassemblor = ReassembleGridPatches(patch_shape=patch_shape,
                                        orig_shape=topo.shape[1:3])

    dataset.apply_preprocessor(extractor)
    dataset.apply_preprocessor(reassemblor)

    new_topo = dataset.get_topological_view()

    assert new_topo.shape == topo.shape

    if not np.all(new_topo == topo):
        assert False


class testLeCunLCN:

    """
    Test LeCunLCN
    """

    def test_random_image(self):
        """
        Test on a random image if the per-processor loads and works without
        anyerror and doesn't result in any nan or inf values

        """

        rng = np.random.RandomState([1, 2, 3])
        X = as_floatX(rng.randn(5, 32 * 32 * 3))

        axes = ['b', 0, 1, 'c']
        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3),
                                                                  axes)
        dataset = DenseDesignMatrix(X=X, view_converter=view_converter)
        dataset.axes = axes
        preprocessor = LeCunLCN(img_shape=[32, 32])
        dataset.apply_preprocessor(preprocessor)
        result = dataset.get_design_matrix()

        assert isfinite(result)

    def test_zero_image(self):
        """
        Test on zero-value image if cause any division by zero
        """

        X = as_floatX(np.zeros((5, 32 * 32 * 3)))

        axes = ['b', 0, 1, 'c']
        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3),
                                                                  axes)
        dataset = DenseDesignMatrix(X=X, view_converter=view_converter)
        dataset.axes = axes
        preprocessor = LeCunLCN(img_shape=[32, 32])
        dataset.apply_preprocessor(preprocessor)
        result = dataset.get_design_matrix()

        assert isfinite(result)

    def test_channel(self):
        """
        Test if works fine withe different number of channel as argument
        """

        rng = np.random.RandomState([1, 2, 3])
        X = as_floatX(rng.randn(5, 32 * 32 * 3))

        axes = ['b', 0, 1, 'c']
        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3),
                                                                  axes)
        dataset = DenseDesignMatrix(X=X, view_converter=view_converter)
        dataset.axes = axes
        preprocessor = LeCunLCN(img_shape=[32, 32], channels=[1, 2])
        dataset.apply_preprocessor(preprocessor)
        result = dataset.get_design_matrix()

        assert isfinite(result)


def test_rgb_yuv():
    """
    Test on a random image if the per-processor loads and works without
    anyerror and doesn't result in any nan or inf values

    """

    rng = np.random.RandomState([1, 2, 3])
    X = as_floatX(rng.randn(5, 32 * 32 * 3))

    axes = ['b', 0, 1, 'c']
    view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3),
                                                              axes)
    dataset = DenseDesignMatrix(X=X, view_converter=view_converter)
    dataset.axes = axes
    preprocessor = RGB_YUV()
    dataset.apply_preprocessor(preprocessor)
    result = dataset.get_design_matrix()

    assert isfinite(result)


def test_zca():
    """
    Confirm that ZCA.inv_P_ is the correct inverse of ZCA.P_.
    There's a lot else about the ZCA class that could be tested here.
    """

    rng = np.random.RandomState([1, 2, 3])
    X = as_floatX(rng.randn(15, 10))
    preprocessor = ZCA()
    preprocessor.fit(X)

    def is_identity(matrix):
        identity = np.identity(matrix.shape[0], theano.config.floatX)
        abs_difference = np.abs(identity - matrix)
        return (abs_difference < .0001).all()

    assert preprocessor.P_.shape == (X.shape[1], X.shape[1])
    assert not is_identity(preprocessor.P_)
    assert is_identity(np.dot(preprocessor.P_, preprocessor.inv_P_))


def test_zca_dtypes():
    """
    Confirm that ZCA.fit works regardless of dtype of data and config.floatX
    """

    orig_floatX = config.floatX

    try:
        for floatX in ['float32', 'float64']:
            for dtype in ['float32', 'float64']:
                rng = np.random.RandomState([1, 2, 3])
                X = rng.randn(15, 10).astype(dtype)
                preprocessor = ZCA()
                preprocessor.fit(X)
    finally:
        config.floatX = orig_floatX
