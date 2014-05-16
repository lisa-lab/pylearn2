"""
Unit tests for ./norb.py
"""

import unittest
import numpy
from pylearn2.datasets.norb import SmallNORB
from pylearn2.datasets.norb_small import FoveatedNORB
from pylearn2.utils import safe_zip
from pylearn2.testing.skip import skip_if_no_data


class TestNORB(unittest.TestCase):
    def setUp(self):
        skip_if_no_data()

    def test_foveated_norb(self):

        # Test that the FoveatedNORB class can be instantiated
        norb_train = FoveatedNORB(which_set="train",
                                  scale=1, restrict_instances=[4, 6, 7, 8],
                                  one_hot=1)

    def test_get_topological_view(self):
        #This is just to lower the memory usage. Otherwise, the
        #buildbot use close to 10G of ram.
        norb = SmallNORB('train', stop=1000)

        # Get a topological view as a single "(b, s, 0 1, c)" tensor.
        topo_tensor = norb.get_topological_view(single_tensor=True)
        shape = (norb.X.shape[0], 2) + SmallNORB.original_image_shape + (1, )
        expected_topo_tensor = norb.X.reshape(shape)
        #We loop to lower the peak memory usage
        for i in range(topo_tensor.shape[0]):
            assert numpy.all(topo_tensor[i] == expected_topo_tensor[i])

        # Get a topological view as two "(b, 0, 1, c)" tensors
        topo_tensors = norb.get_topological_view(single_tensor=False)
        expected_topo_tensors = tuple(expected_topo_tensor[:, i, ...]
                                      for i in range(2))

        for topo_tensor, expected_topo_tensor in \
            safe_zip(topo_tensors, expected_topo_tensors):
            assert numpy.all(topo_tensor == expected_topo_tensor)
