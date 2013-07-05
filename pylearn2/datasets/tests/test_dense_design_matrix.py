import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrixPyTables
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.utils import serial
import unittest


class TestDenseDesignMatrix(unittest.TestCase):
    def setUp(self):
        pass

    def test_init_with_X_or_topo(self):
        #tests that constructing with topo_view works
        #tests that construction with design matrix works
        #tests that conversion from topo_view to design matrix and back works
        #tests that conversion the other way works too
        rng = np.random.RandomState([1,2,3])
        topo_view = rng.randn(5,2,2,3)
        d1 = DenseDesignMatrix(topo_view = topo_view)
        X = d1.get_design_matrix()
        d2 = DenseDesignMatrix(X = X, view_converter = d1.view_converter)
        topo_view_2 = d2.get_topological_view()
        self.assertTrue(np.allclose(topo_view,topo_view_2))
        X = rng.randn(*X.shape)
        topo_view_3 = d2.get_topological_view(X)
        X2 = d2.get_design_matrix(topo_view_3)
        self.assertTrue(np.allclose(X,X2))


    def test_init_with_vc(self):
        rng = np.random.RandomState([4,5,6])
        d = DenseDesignMatrix(
                X=rng.randn(12, 5),
                view_converter = DefaultViewConverter([1,2,3]))

    def get_rnd_design_matrix(self):
        rng = np.random.RandomState([1,2,3])
        topo_view = rng.randn(10,2,2,3)
        d1 = DenseDesignMatrix(topo_view = topo_view)
        return d1

    def test_split_datasets(self):
        #Test the split dataset function.
        ddm = self.get_rnd_design_matrix()
        (train, valid) = ddm.split_dataset_holdout(train_prop=0.5)
        self.assertEquals(valid.shape[0], np.ceil(ddm.num_examples * 0.5))
        self.assertEquals(train.shape[0], (ddm.num_examples - valid.shape[0]))

    def test_split_nfold_datasets(self):
        #Load and create ddm from cifar100
        ddm = self.get_rnd_design_matrix()
        folds = ddm.split_dataset_nfolds(10)
        self.assertEquals(folds[0].shape[0], np.ceil(ddm.num_examples / 10))

    def test_pytables(self):
        """
        tests wether DenseDesignMatrixPyTables can be loaded and
        initialize iterator
        """
        # TODO more through test

        x = np.ones((2, 3))
        y = np.ones(2)
        ds = DenseDesignMatrixPyTables(X = x, y = y)

        it = ds.iterator(mode = 'sequential', batch_size = 1)
        it.next()
