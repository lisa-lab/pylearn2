"""Tests for cross validation module."""
__author__ = 'Steven Kearnes'

from pylearn2.cross_validation import DatasetKFold
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import unittest
import numpy as np

class TestCrossValidation(unittest.TestCase):
    def setUp(self):
