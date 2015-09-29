"""
Unit tests for ./preprocessing.py
"""

import copy
import numpy as np

from theano import config
import theano
from theano.tests.unittest_tools import assert_allclose

from pylearn2.utils import as_floatX
from pylearn2.utils import isfinite
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.preprocessing import (GlobalContrastNormalization,
                                             ExtractGridPatches,
                                             ReassembleGridPatches,
                                             LeCunLCN,
                                             RGB_YUV,
                                             ZCA,
                                             PCA)


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


class testZCA:

    def setup(self):
        """
        We use a small predefined 8x5 matrix for
        which we know the ZCA transform.
        """
        self.X = np.array([[-10.0, 3.0, 19.0, 9.0, -15.0],
                          [7.0, 26.0, 26.0, 26.0, -3.0],
                          [17.0, -17.0, -37.0, -36.0, -11.0],
                          [19.0, 15.0, -2.0, 5.0, 9.0],
                          [-3.0, -8.0, -35.0, -25.0, -8.0],
                          [-18.0, 3.0, 4.0, 15.0, 14.0],
                          [5.0, -4.0, -5.0, -7.0, -11.0],
                          [23.0, 22.0, 15.0, 20.0, 12.0]])
        self.dataset = DenseDesignMatrix(X=as_floatX(self.X),
                                         y=as_floatX(np.ones((8, 1))))
        self.num_components = self.dataset.get_design_matrix().shape[1] - 1

    def get_preprocessed_data(self, preprocessor):
        X = copy.copy(self.X)
        dataset = DenseDesignMatrix(X=X,
                                    preprocessor=preprocessor,
                                    fit_preprocessor=True)
        return dataset.get_design_matrix()

    def test_zca(self):
        """
        Confirm that ZCA.inv_P_ is the correct inverse of ZCA.P_.
        There's a lot else about the ZCA class that could be tested here.
        """
        preprocessor = ZCA()
        preprocessor.fit(self.X)

        identity = np.identity(self.X.shape[1], theano.config.floatX)
        # Check some basics of transformation matrix
        assert preprocessor.P_.shape == (self.X.shape[1], self.X.shape[1])
        assert_allclose(np.dot(preprocessor.P_,
                               preprocessor.inv_P_), identity, rtol=1e-4)

        preprocessor = ZCA(filter_bias=0.0)
        preprocessed_X = self.get_preprocessed_data(preprocessor)

        # Check if preprocessed data matrix is white
        assert_allclose(np.cov(preprocessed_X.transpose(),
                               bias=1), identity, rtol=1e-4, atol=1e-4)

        # Check if we obtain correct solution
        zca_transformed_X = np.array(
            [[-1.0199, -0.1832, 1.9528, -0.9603, -0.8162],
             [0.0729, 1.4142, 0.2529, 1.1861, -1.0876],
             [0.9575, -1.1173, -0.5435, -1.4372, -0.1057],
             [0.6348, 1.1258, 0.2692, -0.8893, 1.1669],
             [-0.9769, 0.8297, -1.8676, -0.6055, -0.5096],
             [-1.5700, -0.8389, -0.0931, 0.8877, 1.6089],
             [0.4993, -1.4219, -0.3443, 0.9664, -1.1022],
             [1.4022, 0.1917, 0.3736, 0.8520, 0.8456]]
        )
        assert_allclose(preprocessed_X, zca_transformed_X, rtol=1e-3)

    def test_num_components(self):
        # Keep 3 components
        preprocessor = ZCA(filter_bias=0.0, n_components=3)
        preprocessed_X = self.get_preprocessed_data(preprocessor)

        zca_truncated_X = np.array(
            [[-0.8938, -0.3084, 1.1105, 0.1587, -1.4073],
             [0.3346, 0.5193, 1.1371, 0.6545, -0.4199],
             [0.7613, -0.4823, -1.0578, -1.1997, -0.4993],
             [0.9250, 0.5012, -0.2743, 0.1735, 0.8105],
             [-0.4928, -0.6319, -1.0359, -0.7173, 0.1469],
             [-1.8060, -0.1758, -0.2943, 0.7208, 1.4359],
             [0.0079, -0.2582, 0.1368, -0.3571, -0.8147],
             [1.1636, 0.8362, 0.2777, 0.5666, 0.7480]]
        )
        assert_allclose(zca_truncated_X, preprocessed_X, rtol=1e-3)

        # Drop 2 components: result should be similar
        preprocessor = ZCA(filter_bias=0.0, n_drop_components=2)
        preprocessed_X = self.get_preprocessed_data(preprocessor)
        assert_allclose(zca_truncated_X, preprocessed_X, rtol=1e-3)

    def test_zca_inverse(self):
        """
        Calculates the inverse of X with numpy.linalg.inv
        if inv_P_ is not stored.
        """
        def test(store_inverse):
            preprocessed_X = copy.copy(self.X)
            preprocessor = ZCA(store_inverse=store_inverse)

            dataset = DenseDesignMatrix(X=preprocessed_X,
                                        preprocessor=preprocessor,
                                        fit_preprocessor=True)

            preprocessed_X = dataset.get_design_matrix()
            assert_allclose(self.X, preprocessor.inverse(preprocessed_X),
                            atol=5e-5, rtol=1e-5)

        test(store_inverse=True)
        test(store_inverse=False)

    def test_zca_dtypes(self):
        """
        Confirm that ZCA.fit works regardless of dtype of
        data and config.floatX
        """

        orig_floatX = config.floatX

        try:
            for floatX in ['float32', 'float64']:
                for dtype in ['float32', 'float64']:
                    preprocessor = ZCA()
                    preprocessor.fit(self.X)
        finally:
            config.floatX = orig_floatX


class testPCA:
    """
    Tests for PCA preprocessor
    """
    def setup(self):
        rng = np.random.RandomState([1, 2, 3])
        self.dataset = DenseDesignMatrix(X=as_floatX(rng.randn(15, 10)),
                                         y=as_floatX(rng.randn(15, 1)))
        self.num_components = self.dataset.get_design_matrix().shape[1] - 1

    def test_apply_no_whiten(self):
        """
        Confirms that PCA has decorrelated the input dataset and
        principal components are arranged in decreasing order by variance
        """
        # sut is an abbreviation for System Under Test
        sut = PCA(self.num_components)
        sut.apply(self.dataset, True)
        cm = np.cov(self.dataset.get_design_matrix().T)  # covariance matrix

        # testing whether the covariance matrix is a diagonal one
        np.testing.assert_almost_equal(cm * (np.ones(cm.shape[0]) -
                                       np.eye(cm.shape[0])),
                                       np.zeros((cm.shape[0], cm.shape[0])))

        # testing whether the eigenvalues are in decreasing order
        assert (np.diag(cm)[:-1] > np.diag(cm)[1:]).all()

    def test_apply_whiten(self):
        """
        Confirms that PCA has decorrelated the input dataset and
        variance is the same along all principal components and equal to one
         """
        sut = PCA(self.num_components, whiten=True)
        sut.apply(self.dataset, True)
        cm = np.cov(self.dataset.get_design_matrix().T)  # covariance matrix

        # testing whether the covariance matrix is a diagonal one
        np.testing.assert_almost_equal(cm * (np.ones(cm.shape[0]) -
                                       np.eye(cm.shape[0])),
                                       np.zeros((cm.shape[0], cm.shape[0])))

        # testing whether the eigenvalues are all ones
        np.testing.assert_almost_equal(np.diag(cm), np.ones(cm.shape[0]))

    def test_apply_reduce_num_components(self):
        """
        Checks whether PCA performs dimensionality reduction
        """
        sut = PCA(self.num_components - 1, whiten=True)
        sut.apply(self.dataset, True)

        assert self.dataset.get_design_matrix().shape[1] ==\
            self.num_components - 1
