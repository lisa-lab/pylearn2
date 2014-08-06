"""
This module contains a serie of tests for the datasets in testing.datasets
"""

from pylearn2.datasets.preprocessing import RemoveMean
from pylearn2.testing.datasets import ArangeDataset


def test_arangedataset():
    """
    This test will verify if ArangeDataset can be used with preprocessors
    """
    preprocessor = RemoveMean()
    dataset = ArangeDataset(1000, preprocessor=preprocessor,
                            fit_preprocessor=True)
    dataset_no_preprocessing = ArangeDataset(1000)
    assert (dataset.get_data() !=
            dataset_no_preprocessing.get_data()).any()
