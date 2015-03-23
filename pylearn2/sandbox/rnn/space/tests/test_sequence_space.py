"""
Unit tests for the SequenceSpace
"""

import numpy as np
import theano
from pylearn2.space import VectorSpace, Conv2DSpace
from pylearn2.sandbox.rnn.space import SequenceSpace

floatX = theano.config.floatX


def test_sequence_vector_space_make_theano_batch():
    """
    SequenceSpate(VectorSpace).make_theano_batch should create 3D tensor
    """
    sequence_vector_space = SequenceSpace(VectorSpace(dim=10, dtype=floatX))
    theano_batch = sequence_vector_space.make_theano_batch()
    assert isinstance(theano_batch, tuple)
    data, mask = theano_batch
    assert data.ndim == 3  # (seq_len, batch_size, feature_dim)
    assert mask.ndim == 2  # (seq_len, batch_size)


def test_sequence_conv2d_space_make_theano_batch():
    """
    SequenceSpate(Conv2DSpace).make_theano_batch should create 5D tensor
    """
    sequence_conv2d_space = SequenceSpace(
        Conv2DSpace(shape=(10, 10), channels=1, dtype=floatX))
    theano_batch = sequence_conv2d_space.make_theano_batch()
    assert isinstance(theano_batch, tuple)
    data, mask = theano_batch
    assert data.ndim == 5  # (seq_len, batch_size, rows, cols, channels)
    assert mask.ndim == 2  # (seq_len, batch_size)


def test_np_format_as_sequencevector2sequencevector():
    """
    format from SequenceSpate(VectorSpace) to SequenceSpate(VectorSpace)
    """
    seq_len, batch_size, dim = (5, 2, 10)
    sequence_vector_space = SequenceSpace(VectorSpace(dim=dim, dtype=floatX))

    data_val = np.arange(seq_len * batch_size * dim, dtype=floatX).reshape(
        seq_len, batch_size, dim)
    mask_val = np.ones((seq_len, batch_size))
    seq_val = (data_val, mask_val)

    rval = sequence_vector_space.np_format_as(seq_val, sequence_vector_space)
    data_rval, mask_rval = rval
    assert np.all(data_rval == data_val)
    assert np.all(mask_rval == mask_val)
    assert data_rval.shape == (seq_len, batch_size, dim)
    assert mask_rval.shape == (seq_len, batch_size)


def test_np_format_as_sequenceconv2d2sequenceconv2d():
    """
    format from SequenceSpate(Conv2DSpace) to SequenceSpate(Conv2DSpace)
    """
    seq_len, batch_size, rows, cols, channels = (7, 5, 3, 2, 1)
    shape = (rows, cols)
    sequence_conv2d_space = SequenceSpace(
        Conv2DSpace(shape=shape, channels=channels, dtype=floatX))

    data_val = np.arange(
        seq_len * batch_size * rows * cols * channels, dtype=floatX
    ).reshape(seq_len, batch_size, rows, cols, channels)
    mask_val = np.ones((seq_len, batch_size))
    seq_val = (data_val, mask_val)

    rval = sequence_conv2d_space.np_format_as(seq_val, sequence_conv2d_space)
    data_rval, mask_rval = rval
    assert np.all(data_rval == data_val)
    assert np.all(mask_rval == mask_val)
    assert data_rval.shape == (seq_len, batch_size, rows, cols, channels)
    assert mask_rval.shape == (seq_len, batch_size)

    # from a sequence conv2d to a sequence conv2d which has different axes
    sequence_conv2d_space2 = SequenceSpace(
        Conv2DSpace(shape=shape, channels=channels,
                    axes=('b', 'c', 0, 1), dtype=floatX))

    rval2 = sequence_conv2d_space.np_format_as(seq_val, sequence_conv2d_space2)
    data_rval2, mask_rval2 = rval2
    assert np.all(data_rval2 != data_val)
    assert np.all(data_rval2.ravel() == data_val.ravel())
    assert np.all(mask_rval2 == mask_val)
    assert data_rval2.shape == (seq_len, batch_size, channels, rows, cols)
    assert mask_rval2.shape == (seq_len, batch_size)


def test_np_format_as_sequenceconv2d2sequencevector():
    """
    format from SequenceSpate(VectorSpace) to SequenceSpate(Conv2DSpace)
    """
    seq_len, batch_size, rows, cols, channels = (7, 5, 3, 2, 1)
    shape = (rows, cols)
    sequence_conv2d_space = SequenceSpace(
        Conv2DSpace(shape=shape, channels=channels, dtype=floatX))
    dim = rows * cols * channels
    sequence_vector_space = SequenceSpace(VectorSpace(dim=dim, dtype=floatX))

    data_val = np.arange(
        seq_len * batch_size * dim, dtype=floatX
    ).reshape(seq_len, batch_size, rows, cols, channels)
    mask_val = np.ones((seq_len, batch_size))
    seq_val = (data_val, mask_val)

    seq_rval = sequence_conv2d_space.np_format_as(
        seq_val, sequence_vector_space)
    data_rval, mask_rval = seq_rval

    assert data_rval.ndim == 3
    expected_shape = (seq_len, batch_size, dim)
    assert data_rval.shape == expected_shape

    expected_val = np.arange(
        seq_len * batch_size * dim, dtype=floatX
    ).reshape(expected_shape)
    assert np.all(data_rval == expected_val)


def test_np_format_as_sequencevector2sequenceconv2d():
    """
    format from SequenceSpate(Conv2DSpace) to SequenceSpate(VectorSpace)
    """
    seq_len, batch_size, rows, cols, channels = (7, 5, 3, 2, 1)
    shape = (rows, cols)
    dim = rows * cols * channels
    sequence_vector_space = SequenceSpace(VectorSpace(dim=dim, dtype=floatX))
    sequence_conv2d_space = SequenceSpace(
        Conv2DSpace(shape=shape, channels=channels, dtype=floatX))

    data_val = np.arange(
        seq_len * batch_size * dim, dtype=floatX
    ).reshape(seq_len, batch_size, dim)
    mask_val = np.ones((seq_len, batch_size))
    seq_val = (data_val, mask_val)

    seq_rval = sequence_vector_space.np_format_as(
        seq_val, sequence_conv2d_space)
    data_rval, mask_rval = seq_rval

    assert data_rval.ndim == 5
    expected_shape = (seq_len, batch_size, rows, cols, channels)
    assert data_rval.shape == expected_shape

    expected_val = np.arange(
        seq_len * batch_size * dim, dtype=floatX
    ).reshape(expected_shape)
    assert np.all(data_rval == expected_val)


if __name__ == '__main__':
    test_sequence_vector_space_make_theano_batch()
    test_sequence_conv2d_space_make_theano_batch()
    test_np_format_as_sequencevector2sequencevector()
    test_np_format_as_sequenceconv2d2sequenceconv2d()
    test_np_format_as_sequenceconv2d2sequencevector()
    test_np_format_as_sequencevector2sequenceconv2d()
