"""
Unit tests for ./norb.py
"""

import unittest
import numpy
from theano.compat import six
from pylearn2.datasets.norb import SmallNORB
from pylearn2.datasets.norb_small import FoveatedNORB
from pylearn2.datasets.new_norb import NORB
from pylearn2.utils import safe_zip
from pylearn2.testing.skip import skip_if_no_data


class TestNORB(unittest.TestCase):

    def setUp(self):
        skip_if_no_data()

    def test_foveated_norb(self):

        # Test that the FoveatedNORB class can be instantiated
        norb_train = FoveatedNORB(which_set="train",
                                  scale=1, restrict_instances=[4, 6, 7, 8])

    def test_get_topological_view(self):

        def test_impl(norb):
            # Get a topological view as a single "(b, s, 0 1, c)" tensor.
            topo_tensor = norb.get_topological_view(single_tensor=True)
            shape = ((norb.X.shape[0], 2) +
                     SmallNORB.original_image_shape +
                     (1, ))
            expected_topo_tensor = norb.X.reshape(shape)
            # We loop to lower the peak memory usage
            for i in range(topo_tensor.shape[0]):
                assert numpy.all(topo_tensor[i] == expected_topo_tensor[i])

            # Get a topological view as two "(b, 0, 1, c)" tensors
            topo_tensors = norb.get_topological_view(single_tensor=False)
            expected_topo_tensors = tuple(expected_topo_tensor[:, i, ...]
                                          for i in range(2))

            for topo_tensor, expected_topo_tensor in safe_zip(
                    topo_tensors, expected_topo_tensors):
                assert numpy.all(topo_tensor == expected_topo_tensor)

        # Use stop parameter for SmallNORB; otherwise the buildbot uses close
        # to 10G of RAM.
        for norb in (SmallNORB('train', stop=1000),
                     NORB(which_norb='small', which_set='train')):
            test_impl(norb)

    def test_label_to_value_funcs(self):
        def test_impl(norb):
            label_to_value_maps = (
                # category
                {0: 'animal',
                 1: 'human',
                 2: 'airplane',
                 3: 'truck',
                 4: 'car',
                 5: 'blank'},

                # instance
                dict(safe_zip(range(10), range(10))),

                # elevation
                dict(safe_zip(range(9), numpy.arange(9) * 5 + 30)),

                # azimuth
                dict(safe_zip(range(0, 36, 2), numpy.arange(0, 360, 20))),

                # lighting
                dict(safe_zip(range(5), range(5))),

                # horizontal shift
                dict(safe_zip(range(-5, 6), range(-5, 6))),

                # vertical shift
                dict(safe_zip(range(-5, 6), range(-5, 6))),

                # lumination change
                dict(safe_zip(range(-19, 20), range(-19, 20))),

                # contrast change
                dict(safe_zip(range(2), (0.8, 1.3))))

            # Use of zip rather than safe_zip intentional;
            # norb.label_to_value_funcs will be shorter than
            # label_to_value_maps if norb is small NORB.
            for (label_to_value_map,
                 label_to_value_func) in zip(label_to_value_maps,
                                             norb.label_to_value_funcs):
                for label, expected_value in six.iteritems(label_to_value_map):
                    actual_value = label_to_value_func(label)
                    assert expected_value == actual_value

        test_impl(NORB(which_set='test', which_norb='small'))
        test_impl(NORB(which_set='test', which_norb='big'))

    def test_image_dtype(self):
        expected_dtypes = ('uint8', 'float32')
        norbs = (NORB(which_set='train',
                      which_norb='small'),
                 NORB(which_set='train',
                      which_norb='small',
                      image_dtype='float32'))

        for norb, expected_dtype in safe_zip(norbs, expected_dtypes):
            assert str(norb.X.dtype) == expected_dtype
