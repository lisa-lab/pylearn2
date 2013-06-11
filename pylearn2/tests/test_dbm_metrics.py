"""
Test dbm_metrics script
"""
from pylearn2.scripts.dbm import dbm_metrics
from pylearn2.datasets.mnist import MNIST


def test_ais():
    """
    Test ais computation
    """
    w_list = [None]
    b_list = []
    # Add parameters import

    trainset = MNIST(which_set='train')
    testset = MNIST(which_set='test')

    train_ll, test_ll, log_z = dbm_metrics.estimate_likelihood(w_list,
                                                               b_list,
                                                               trainset,
                                                               testset,
                                                               pos_mf_steps=5)

    # Add log_z, test_ll import
    russ_log_z = 100.
    russ_train_ll = -100.
    russ_test_ll = -100.

    assert log_z == russ_log_z
    assert train_ll == russ_train_ll
    assert test_ll == russ_test_ll
