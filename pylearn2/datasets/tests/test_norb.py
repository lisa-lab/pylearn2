"""
Unit tests for ./norb.py
"""

import unittest
import numpy
from pylearn2.datasets.norb import SmallNORB
from pylearn2.utils import safe_zip
from pylearn2.testing.skip import skip_if_no_data

class TestCIFAR10(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()

    def test_get_topological_view(self):
        norb = SmallNORB('train')

        # Get a topological view as a single "(b, s, 0 1, c)" tensor.
        topo_tensor = norb.get_topological_view(single_tensor=True)
        shape = (norb.X.shape[0], 2) + SmallNORB.original_image_shape + (1, )
        expected_topo_tensor = norb.X.reshape(shape)
        assert numpy.all(topo_tensor == expected_topo_tensor)

        # Get a topological view as two "(b, 0, 1, c)" tensors
        topo_tensors = norb.get_topological_view(single_tensor=False)
        expected_topo_tensors = tuple(expected_topo_tensor[:, i, ...]
                                      for i in range(2))

        for topo_tensor, expected_topo_tensor in \
            safe_zip(topo_tensors, expected_topo_tensors):
            assert numpy.all(topo_tensor == expected_topo_tensor)
