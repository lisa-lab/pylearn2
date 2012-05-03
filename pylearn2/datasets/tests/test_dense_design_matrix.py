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

def test_split_datasets():
    #Load and create ddm from cifar100
    path = "/data/lisa/data/cifar100/cifar-100-python/train"
    obj = serial.load(path)
    X = obj['data']

    assert X.max() == 255.
    assert X.min() == 0.

    X = np.cast['float32'](X)
    y = None #not implemented yet

    view_converter = DefaultViewConverter((32,32,3))

    ddm = DenseDesignMatrix(X = X, y =y, view_converter = view_converter)

    assert not np.any(np.isnan(ddm.X))
    ddm.y_fine = np.asarray(obj['fine_labels'])
    ddm.y_coarse = np.asarray(obj['coarse_labels'])
    ddm.set_iteration_scheme("sequential", batch_size=1000, num_batches=2, targets=True)
    (train, valid) = ddm.split_dataset_holdout(train_prop=0.5)
    assert valid.shape[0] == np.ceil(ddm.num_examples * 0.5)
    assert train.shape[0] == (ddm.num_examples - valid.shape[0])

def test_split_nfold_datasets():
    #Load and create ddm from cifar100
    path = "/data/lisa/data/cifar100/cifar-100-python/train"
    obj = serial.load(path)
    X = obj['data']

    assert X.max() == 255.
    assert X.min() == 0.

    X = np.cast['float32'](X)
    y = None #not implemented
    view_converter = DefaultViewConverter((32,32,3))

    ddm = DenseDesignMatrix(X=X, y=y, view_converter = view_converter)

    assert not np.any(np.isnan(ddm.X))
    ddm.y_fine = np.asarray(obj['fine_labels'])
    ddm.y_coarse = np.asarray(obj['coarse_labels'])
    folds = ddm.split_dataset_nfolds(10)
    assert folds[0].shape[0] == np.ceil(ddm.num_examples / 10)

def test_split_on_labeled_mnist():
    which_set = "train"
    path = "/data/lisa/data/mnist/mnist-python/" + which_set
    obj = serial.load(path)
    X = obj['data']
    X = np.cast['float32'](X)
    y = np.asarray(obj['labels'])

    view_converter = DefaultViewConverter((28,28,1))

    ddm = DenseDesignMatrix(X = X, y = y, view_converter = view_converter)

    (train, valid) = ddm.split_dataset_holdout(train_prop=0.5)
    import pprint
    pprint.pprint(train)
    assert len(train) == 2
    assert train[1].shape[0] == (ddm.num_examples * 0.5)
    assert not np.any(np.isnan(X))

#test_split_datasets()
#test_split_nfold_datasets()
#test_split_on_labeled_mnist()
