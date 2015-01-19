"""
Tests for ../retina.py
"""
import numpy as np

from pylearn2.utils import as_floatX
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.retina import (foveate_channel,
                                      defoveate_channel,
                                      get_encoded_size,
                                      encode,
                                      decode,
                                      RetinaEncodingBlock,
                                      RetinaDecodingBlock,
                                      RetinaCodingViewConverter
                                      )


class test_retina():
    """
    Parameters
    ----------
    None

    Notes
    -----
    Testing class that checks retina preprocessing functions
    """

    def test_transform(self):
        """
        This runs tests on the foveate/defoveate channel
        process for different sizes of input
        """
        # square input
        image = np.random.randn(1, 20, 20)
        rings = [5, 2]
        encoded_size = get_encoded_size(image.shape[1], image.shape[2], rings)
        assert encoded_size == 64, "Different from  previously computed value"
        self.foveate_defoveate_channel(image, rings, encoded_size)

        # non-square input
        rec_image = np.random.randn(1, 70, 42)
        rec_rings = [7, 4, 2]
        rec_encoded_size = get_encoded_size(rec_image.shape[1],
                                            rec_image.shape[2], rec_rings)
        assert encoded_size == 64, "Different from  previously computed value"
        self.foveate_defoveate_channel(rec_image, rec_rings,
                                       rec_encoded_size)

    def foveate_defoveate_channel(self, image, rings, encoded_size, bs=1):
        """
        Helper function

        Here we foveate the image, which reduces the quality of the rings
        Defoveate it back into the original image, and see if we
        once again refoveate whether it returns to the original image
        """
        output = np.zeros((bs, encoded_size))
        foveate_channel(image, rings, output, 0)
        defoveate_channel(image, rings, output, 0)
        refoveated_output = np.zeros((bs, encoded_size))
        foveate_channel(image, rings, refoveated_output, 0)
        np.testing.assert_allclose(refoveated_output, output)

    def test_encode_decode(self):
        """
        Try to encode and decode to get back the same value
        """
        topo_X = np.random.randn(1, 20, 25, 5)
        rings = [5]
        output = encode(topo_X, rings)
        reconstruct = decode(output, (20, 25, 5), [5])
        np.testing.assert_allclose(encode(reconstruct, rings), output)

    def test_RetinaCodingViewConverter(self):
        """
        Test to see if the RetinaCodingViewConverter loads using the
        typically shape in ../norb_small.py
        """
        view_converter = RetinaCodingViewConverter((96, 96, 2), (8, 4, 2, 2))
