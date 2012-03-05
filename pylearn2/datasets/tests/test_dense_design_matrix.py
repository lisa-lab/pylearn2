from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
import numpy as np

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



