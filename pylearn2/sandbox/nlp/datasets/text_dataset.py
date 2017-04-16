"""
This module provides abstract datasets that deals with text data.
"""
__authors__     = "Trung Huynh"
__copyrights__  = "Copyright 2010-2012, Universite de Montreal"
__license__     = "3-clause BSD license"
__contact__     = "trunghlt@gmail.com"

from pylearn2.datasets.sparse_dataset import SparseDataset


class TextDataset(SparseDataset):
    """
    TextDataset is a abstract class that contains interface for representing
    datasets that can store text values.
    """
    @property
    def vocabulary(self):
        """ return vocabulary used to transform data """
        raise NotImplementedError()

    @property
    def docs(self):
        """  return original documents """
        raise NotImplementedError()

