"""
Utility functions to load data from the UTLC challenge (Unsupervised Transfer
Learning).

The user should use the load_ndarray_dataset or load_sparse_dataset function
See the file ${PYLEARN2_DATA_PATH}/UTLC/README for details on the datasets.
"""

import cPickle
import gzip
import os

import numpy
import theano

import pylearn2.datasets.filetensor as ft
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.rng import make_np_rng


def load_ndarray_dataset(name, normalize=True, transfer=False,
                         normalize_on_the_fly=False, randomize_valid=False,
                         randomize_test=False):
    """
    Load the train,valid,test data for the dataset `name` and return it in
    ndarray format.

    We suppose the data was created with ift6266h11/pretraitement/to_npy.py
    that shuffle the train. So the train should already be shuffled.

    Parameters
    ----------
    name : 'avicenna', 'harry', 'rita', 'sylvester' or 'ule'
        Which dataset to load

    normalize : bool
        If True, we normalize the train dataset before returning it

    transfer : bool
        If True also return the transfer labels

    normalize_on_the_fly : bool
        If True, we return a Theano Variable that will give as output the
        normalized value. If the user only take a subtensor of that variable,
        Theano optimization should make that we will only have in memory the
        subtensor portion that is computed in normalized form. We store the
        original data in shared memory in its original dtype. This is usefull
        to have the original data in its original dtype in memory to same
        memory. Especialy usefull to be able to use rita and harry with 1G per
        jobs.

    randomize_valid : bool
        Do we randomize the order of the valid set?  We always use the same
        random order If False, return in the same order as downloaded on the
        web

    randomize_test : bool
        Do we randomize the order of the test set?  We always use the same
        random order If False, return in the same order as downloaded on the
        web

    Returns
    -------
    train, valid, test : ndarrays
        Datasets returned if transfer = False
    train, valid, test, transfer : ndarrays
        Datasets returned if transfer = False

    """
    assert not (normalize and normalize_on_the_fly), \
        "Can't normalize in 2 way at the same time!"

    assert name in ['avicenna', 'harry', 'rita', 'sylvester', 'ule']
    common = os.path.join(
        preprocess('${PYLEARN2_DATA_PATH}'), 'UTLC', 'filetensor', name + '_')
    trname, vname, tename = [
        common + subset + '.ft' for subset in ['train', 'valid', 'test']]

    train = load_filetensor(trname)
    valid = load_filetensor(vname)
    test = load_filetensor(tename)
    if randomize_valid:
        rng = make_np_rng(None, [1, 2, 3, 4], which_method='permutation')
        perm = rng.permutation(valid.shape[0])
        valid = valid[perm]
    if randomize_test:
        rng = make_np_rng(None, [1, 2, 3, 4], which_method='permutation')
        perm = rng.permutation(test.shape[0])
        test = test[perm]

    if normalize or normalize_on_the_fly:
        if normalize_on_the_fly:
            # Shared variables of the original type
            train = theano.shared(train, borrow=True, name=name + "_train")
            valid = theano.shared(valid, borrow=True, name=name + "_valid")
            test = theano.shared(test, borrow=True, name=name + "_test")
            # Symbolic variables cast into floatX
            train = theano.tensor.cast(train, theano.config.floatX)
            valid = theano.tensor.cast(valid, theano.config.floatX)
            test = theano.tensor.cast(test, theano.config.floatX)
        else:
            train = numpy.asarray(train, theano.config.floatX)
            valid = numpy.asarray(valid, theano.config.floatX)
            test = numpy.asarray(test, theano.config.floatX)

        if name == "ule":
            train /= 255
            valid /= 255
            test /= 255
        elif name in ["avicenna", "sylvester"]:
            if name == "avicenna":
                train_mean = 514.62154022835455
                train_std = 6.829096494224145
            else:
                train_mean = 403.81889927027686
                train_std = 96.43841050784053
            train -= train_mean
            valid -= train_mean
            test -= train_mean
            train /= train_std
            valid /= train_std
            test /= train_std
        elif name == "harry":
            std = 0.69336046033925791  # train.std()slow to compute
            train /= std
            valid /= std
            test /= std
        elif name == "rita":
            v = numpy.asarray(230, dtype=theano.config.floatX)
            train /= v
            valid /= v
            test /= v
        else:
            raise Exception(
                "This dataset don't have its normalization defined")
    if transfer:
        transfer = load_ndarray_transfer(name)
        return train, valid, test, transfer
    else:
        return train, valid, test


def load_sparse_dataset(name, normalize=True, transfer=False,
                        randomize_valid=False,
                        randomize_test=False):
    """
    Load the train,valid,test data for the dataset `name` and return it in
    sparse format.

    We suppose the data was created with ift6266h11/pretraitement/to_npy.py
    that shuffle the train. So the train should already be shuffled.

    name : 'avicenna', 'harry', 'rita', 'sylvester' or 'ule'
        Which dataset to load

    normalize : bool
        If True, we normalize the train dataset before returning it

    transfer :
        If True also return the transfer label

    randomize_valid : bool
        Do we randomize the order of the valid set?  We always use the same
        random order If False, return in the same order as downloaded on the
        web

    randomize_test : bool
        Do we randomize the order of the test set?  We always use the same
        random order If False, return in the same order as downloaded on the
        web

    Returns
    -------
    train, valid, test : ndarrays
        Datasets returned if transfer = False

    train, valid, test, transfer : ndarrays
        Datasets returned if transfer = False

    """
    assert name in ['harry', 'terry', 'ule']
    common = os.path.join(
        preprocess('${PYLEARN2_DATA_PATH}'), 'UTLC', 'sparse', name + '_')
    trname, vname, tename = [
        common + subset + '.npy' for subset in ['train', 'valid', 'test']]

    train = load_sparse(trname)
    valid = load_sparse(vname)
    test = load_sparse(tename)

    # Data should already be in csr format that support
    # this type of indexing.
    if randomize_valid:
        rng = make_np_rng(None, [1, 2, 3, 4], which_method='permutation')
        perm = rng.permutation(valid.shape[0])
        valid = valid[perm]
    if randomize_test:
        rng = make_np_rng(None, [1, 2, 3, 4], which_method='permutation')
        perm = rng.permutation(test.shape[0])
        test = test[perm]

    if normalize:
        if name == "ule":
            train = train.astype(theano.config.floatX) / 255
            valid = valid.astype(theano.config.floatX) / 255
            test = test.astype(theano.config.floatX) / 255
        elif name == "harry":
            train = train.astype(theano.config.floatX)
            valid = valid.astype(theano.config.floatX)
            test = test.astype(theano.config.floatX)
            std = 0.69336046033925791  # train.std()slow to compute
            train = (train) / std
            valid = (valid) / std
            test = (test) / std
        elif name == "terry":
            train = train.astype(theano.config.floatX)
            valid = valid.astype(theano.config.floatX)
            test = test.astype(theano.config.floatX)
            train = (train) / 300
            valid = (valid) / 300
            test = (test) / 300
        else:
            raise Exception(
                "This dataset don't have its normalization defined")
    if transfer:
        fname = os.path.join(preprocess("${PYLEARN2_DATA_PATH}"),
                             "UTLC",
                             "filetensor",
                             name + "_transfer.ft")
        transfer = load_filetensor(fname)
        return train, valid, test, transfer
    else:
        return train, valid, test


def load_ndarray_transfer(name):
    """
    Load the transfer labels for the training set of data set `name`.

    Parameters
    ----------
    name : 'avicenna', 'harry', 'rita', 'sylvester' or 'ule'
        Which dataset to load

    Returns
    -------
    transfer : ndarray
        Transfer dataset loaded
    """
    assert name in ['avicenna', 'harry', 'rita', 'sylvester', 'terry', 'ule']

    fname = os.path.join(preprocess('${PYLEARN2_DATA_PATH}'),
                         'UTLC',
                         'filetensor', name + '_transfer.ft')
    transfer = load_filetensor(fname)
    return transfer


def load_ndarray_label(name):
    """
    Load the train,valid,test label data for the dataset `name` and return it
    in ndarray format.  This is only available for the toy dataset ule.

    Parameters
    ----------
    name : 'ule'
        Must be 'ule'

    Returns
    -------
    train_l. valid_l, test_l : ndarray
        Label data loaded

    """
    assert name in ['ule']

    common_path = os.path.join(
        preprocess('${PYLEARN2_DATA_PATH}'), 'UTLC', 'filetensor', name + '_')
    trname, vname, tename = [common_path + subset + '.tf'
                             for subset in ['trainl', 'validl', 'testl']]

    trainl = load_filetensor(trname)
    validl = load_filetensor(vname)
    testl = load_filetensor(tename)
    return trainl, validl, testl


def load_filetensor(fname):
    """
    .. todo::

        WRITEME
    """
    f = None
    try:
        if not os.path.exists(fname):
            fname = fname + '.gz'
            f = gzip.open(fname)
        elif fname.endswith('.gz'):
            f = gzip.open(fname)
        else:
            f = open(fname)
        d = ft.read(f)
    finally:
        if f:
            f.close()

    return d


def load_sparse(fname):
    """
    .. todo::

        WRITEME
    """
    f = None
    try:
        if not os.path.exists(fname):
            fname = fname + '.gz'
            f = gzip.open(fname)
        elif fname.endswith('.gz'):
            f = gzip.open(fname)
        else:
            f = open(fname)
        d = cPickle.load(f)
    finally:
        if f:
            f.close()
    return d
