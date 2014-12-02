"""
This file tests the NORBSmall and the FoveatedNORB class. majorly
concerning the X and y member of the dataset and their corresponding
sizes, data scales and topological views.
"""

import numpy
from pylearn2.testing.skip import skip_if_no_data
from pylearn2.datasets.norb_small import NORBSmall, FoveatedNORB


def test_NORBSmall():
    """
    This function tests the NORBSmall class. Checks the shape and datatype of
    X, and shape of y in the returned object. The topological value is also
    checked.
    """
    skip_if_no_data()
    data = NORBSmall('train')
    assert data.X.shape == (24300, 18432)
    assert data.X.dtype == 'float32'
    assert data.y.shape == (24300, )
    assert data.y_labels == 5
    assert data.get_topological_view().shape == (24300, 96, 96, 2)

    data = NORBSmall('test')
    assert data.X.shape == (24300, 18432)
    assert data.X.dtype == 'float32'
    assert data.y.shape == (24300, )
    assert data.y_labels == 5
    assert data.get_topological_view().shape == (24300, 96, 96, 2)


def test_FoveatedNORB():
    """
    This function tests the FoveatedNORB class. In addition to the shape and
    datatype of X and y member of the returned object, it also checks the
    scale of data while passing different parameters to the constructor.
    """
    skip_if_no_data()
    data = FoveatedNORB('train')
    datamin = data.X.min()
    datamax = data.X.max()
    assert data.X.shape == (24300, 8976)
    assert data.X.dtype == 'float32'
    assert data.y.shape == (24300, )
    assert data.y_labels == 5
    assert data.get_topological_view().shape == (24300, 96, 96, 2)

    data = FoveatedNORB('train', center=True)
    assert data.X.min() == datamin - 127.5
    assert data.X.max() == datamax - 127.5

    data = FoveatedNORB('train', center=True, scale=True)
    assert numpy.all(data.X <= 1.)
    assert numpy.all(data.X >= -1.)

    data = FoveatedNORB('train', scale=True)
    assert numpy.all(data.X <= 1.)
    assert numpy.all(data.X >= 0.)

    data = FoveatedNORB('test')
    assert data.X.shape == (24300, 8976)
    assert data.X.dtype == 'float32'
    assert data.y.shape == (24300, )
    assert data.y_labels == 5
    assert data.get_topological_view().shape == (24300, 96, 96, 2)
