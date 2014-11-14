from __future__ import print_function

import unittest

import numpy
import scipy.sparse

from pylearn2.testing.skip import skip_if_no_data
import pylearn2.datasets.utlc as utlc


def test_ule():
    skip_if_no_data()
    # Test loading of transfer data
    train, valid, test, transfer = utlc.load_ndarray_dataset("ule",
                                                             normalize=True,
                                                             transfer=True)
    assert train.shape[0] == transfer.shape[0]


# @unittest.skip("Slow and needs >8 GB of RAM")
def test_all_utlc():
    skip_if_no_data()
    # not testing rita, because it requires a lot of memorz and is slow
    for name in ['avicenna', 'harry', 'ule']:
        print("Loading ", name)
        train, valid, test = utlc.load_ndarray_dataset(name, normalize=True)
        print("dtype, max, min, mean, std")
        print(train.dtype, train.max(), train.min(), train.mean(), train.std())
        assert isinstance(train, numpy.ndarray)
        assert isinstance(valid, numpy.ndarray)
        assert isinstance(test, numpy.ndarray)
        assert train.shape[1] == test.shape[1] == valid.shape[1]


def test_sparse_ule():
    skip_if_no_data()
    # Test loading of transfer data
    train, valid, test, transfer = utlc.load_sparse_dataset("ule",
                                                            normalize=True,
                                                            transfer=True)
    assert train.shape[0] == transfer.shape[0]


def test_all_sparse_utlc():
    skip_if_no_data()
    for name in ['harry', 'terry', 'ule']:
        print("Loading sparse ", name)
        train, valid, test = utlc.load_sparse_dataset(name, normalize=True)
        nb_elem = numpy.prod(train.shape)
        mi = train.data.min()
        ma = train.data.max()
        mi = min(0, mi)
        ma = max(0, ma)
        su = train.data.sum()
        mean = float(su) / nb_elem
        print(name, "dtype, max, min, mean, nb non-zero, nb element, %sparse")
        print(train.dtype, ma, mi, mean, train.nnz, end='')
        print(nb_elem, (nb_elem - float(train.nnz)) / nb_elem)
        print(name, "max, min, mean, std (all stats on non-zero element)")
        print(train.data.max(), train.data.min(), end='')
        print(train.data.mean(), train.data.std())
        assert scipy.sparse.issparse(train)
        assert scipy.sparse.issparse(valid)
        assert scipy.sparse.issparse(test)
        assert train.shape[1] == test.shape[1] == valid.shape[1]
