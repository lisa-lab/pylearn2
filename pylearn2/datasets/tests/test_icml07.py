import numpy as np
import unittest

from pylearn2.testing.skip import skip_if_no_data
import pylearn2.datasets.icml07 as icml07


# Basic tests to see if data is loadable
def test_MNIST_rot_back():
    skip_if_no_data()
    data = icml07.MNIST_rotated_background(which_set='train')
    data = icml07.MNIST_rotated_background(which_set='valid')
    data = icml07.MNIST_rotated_background(which_set='test')


def test_Convex():
    skip_if_no_data()
    data = icml07.Convex(which_set='train')
    data = icml07.Convex(which_set='valid')
    data = icml07.Convex(which_set='test')


def test_Rectangles():
    skip_if_no_data()
    data = icml07.Rectangles(which_set='train')
    data = icml07.Rectangles(which_set='valid')
    data = icml07.Rectangles(which_set='test')


def test_RectanglesImage():
    skip_if_no_data()
    data = icml07.RectanglesImage(which_set='train')
    data = icml07.RectanglesImage(which_set='valid')
    data = icml07.RectanglesImage(which_set='test')


# Test features
def test_split():
    skip_if_no_data()
    n_train = 100
    n_valid = 200
    n_test = 300

    data = icml07.MNIST_rotated_background(which_set='train',
                                           split=(n_train, n_valid, n_test))
    assert data.X.shape[0] == n_train, "Unexpected size of train set"
    assert data.y.shape[0] == n_train, "Unexpected size of train set"

    data = icml07.MNIST_rotated_background(which_set='valid',
                                           split=(n_train, n_valid, n_test))
    assert data.X.shape[0] == n_valid, "Unexpected size of validation set"
    assert data.y.shape[0] == n_valid, "Unexpected size of validation set"

    data = icml07.MNIST_rotated_background(which_set='test',
                                           split=(n_train, n_valid, n_test))
    assert data.X.shape[0] == n_test, "Unexpected size of test set"
    assert data.y.shape[0] == n_test, "Unexpected size of test set"
