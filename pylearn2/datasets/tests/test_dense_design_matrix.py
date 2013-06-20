import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.utils import serial


def test_init_with_X_or_topo():
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
    assert np.allclose(topo_view,topo_view_2)
    X = rng.randn(*X.shape)
    topo_view_3 = d2.get_topological_view(X)
    X2 = d2.get_design_matrix(topo_view_3)
    assert np.allclose(X,X2)


def test_init_with_vc():
    d = DenseDesignMatrix(view_converter = DefaultViewConverter([1,2,3]))

def get_rnd_design_matrix():
    rng = np.random.RandomState([1,2,3])
    topo_view = rng.randn(10,2,2,3)
    d1 = DenseDesignMatrix(topo_view = topo_view)
    return d1

def test_split_datasets():
    #Test the split dataset function.
    ddm = get_rnd_design_matrix()
    (train, valid) = ddm.split_dataset_holdout(train_prop=0.5)
    assert valid.shape[0] == np.ceil(ddm.num_examples * 0.5)
    assert train.shape[0] == (ddm.num_examples - valid.shape[0])

def test_split_nfold_datasets():
    #Load and create ddm from cifar100
    ddm = get_rnd_design_matrix()
    folds = ddm.split_dataset_nfolds(10)
    assert folds[0].shape[0] == np.ceil(ddm.num_examples / 10)

#test_split_datasets()
#test_split_nfold_datasets()
