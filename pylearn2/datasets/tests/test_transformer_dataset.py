"""
Unit tests for ../transformer_dataset.py
"""

import os

import pylearn2
from pylearn2.blocks import Block
from pylearn2.datasets.csv_dataset import CSVDataset
from pylearn2.datasets.transformer_dataset import TransformerDataset


def test_transformer_iterator():
    """
    Tests whether TransformerIterator is iterable
    """

    test_path = os.path.join(pylearn2.__path__[0],
                             'datasets', 'tests', 'test.csv')
    raw = CSVDataset(path=test_path, expect_headers=False)
    block = Block()
    dataset = TransformerDataset(raw, block)
    iterator = dataset.iterator('shuffled_sequential', 3)
    try:
        iter(iterator)
    except TypeError:
        assert False, "TransformerIterator isn't iterable"
