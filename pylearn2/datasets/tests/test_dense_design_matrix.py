import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrixPyTables
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import from_dataset
from pylearn2.utils import serial


def test_init_with_X_or_topo():
    # tests that constructing with topo_view works
    # tests that construction with design matrix works
    # tests that conversion from topo_view to design matrix and back works
    # tests that conversion the other way works too
    rng = np.random.RandomState([1, 2, 3])
    topo_view = rng.randn(5, 2, 2, 3)
    d1 = DenseDesignMatrix(topo_view=topo_view)
    X = d1.get_design_matrix()
    d2 = DenseDesignMatrix(X=X, view_converter=d1.view_converter)
    topo_view_2 = d2.get_topological_view()
    assert np.allclose(topo_view, topo_view_2)
    X = rng.randn(*X.shape)
    topo_view_3 = d2.get_topological_view(X)
    X2 = d2.get_design_matrix(topo_view_3)
    assert np.allclose(X, X2)


def test_convert_to_one_hot():
    rng = np.random.RandomState([2013, 11, 14])
    m = 11
    d = DenseDesignMatrix(
        X=rng.randn(m, 4),
        y=rng.randint(low=0, high=10, size=(m,)))
    d.convert_to_one_hot()


def test_init_with_vc():
    rng = np.random.RandomState([4, 5, 6])
    d = DenseDesignMatrix(
        X=rng.randn(12, 5),
        view_converter=DefaultViewConverter([1, 2, 3]))


def get_rnd_design_matrix():
    rng = np.random.RandomState([1, 2, 3])
    topo_view = rng.randn(10, 2, 2, 3)
    d1 = DenseDesignMatrix(topo_view=topo_view)
    return d1


def test_split_datasets():
    # Test the split dataset function.
    ddm = get_rnd_design_matrix()
    (train, valid) = ddm.split_dataset_holdout(train_prop=0.5)
    assert valid.shape[0] == np.ceil(ddm.get_num_examples() * 0.5)
    assert train.shape[0] == (ddm.get_num_examples() - valid.shape[0])


def test_split_nfold_datasets():
    # Load and create ddm from cifar100
    ddm = get_rnd_design_matrix()
    folds = ddm.split_dataset_nfolds(10)
    assert folds[0].shape[0] == np.ceil(ddm.get_num_examples() / 10)


def test_pytables():
    """
    tests wether DenseDesignMatrixPyTables can be loaded and
    initialize iterator
    """
    # TODO more through test

    x = np.ones((2, 3))
    y = np.ones(2)
    ds = DenseDesignMatrixPyTables(X=x, y=y)

    it = ds.iterator(mode='sequential', batch_size=1)
    it.next()


def test_init_pytables_with_labels():
    """
    Test whether DenseDesignMatrixPytables can be constructed with X_labels
    and y_labels.
    """

    rng = np.random.RandomState([34, 22, 89])
    X = rng.randn(2, 3)
    y = rng.randint(low=0, high=5, size=(2,))
    ds = DenseDesignMatrixPyTables(
        X=X,
        y=y,
        X_labels=len(np.unique(X).flat),
        y_labels=np.max(y)+1
    )


def test_from_dataset():
    """
    Tests whether it supports integer labels.
    """
    rng = np.random.RandomState([1, 2, 3])
    topo_view = rng.randn(12, 2, 3, 3)
    y = rng.randint(0, 5, (12, 1))

    # without y:
    d1 = DenseDesignMatrix(topo_view=topo_view)
    slice_d = from_dataset(d1, 5)
    assert slice_d.X.shape[1] == d1.X.shape[1]
    assert slice_d.X.shape[0] == 5

    # with y:
    d2 = DenseDesignMatrix(topo_view=topo_view, y=y)
    slice_d = from_dataset(d2, 5)
    assert slice_d.X.shape[1] == d2.X.shape[1]
    assert slice_d.X.shape[0] == 5
    assert slice_d.y.shape[0] == 5

    # without topo_view:
    x = topo_view.reshape(12, 18)
    d3 = DenseDesignMatrix(X=x, y=y)
    slice_d = from_dataset(d3, 5)
    assert slice_d.X.shape[1] == d3.X.shape[1]
    assert slice_d.X.shape[0] == 5
    assert slice_d.y.shape[0] == 5
