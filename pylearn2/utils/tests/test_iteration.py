"""Tests for iterators."""
import numpy as np
from pylearn2.utils.iteration import (
    SubsetIterator,
    SequentialSubsetIterator,
    ShuffledSequentialSubsetIterator,
    RandomSliceSubsetIterator,
    RandomUniformSubsetIterator
)


def test_misc_exceptions():
    raised = False
    try:
        SubsetIterator.__new__(SubsetIterator).next()
    except NotImplementedError:
        raised = True
    assert raised
    raised = False
    try:
        SubsetIterator(1, 2, 3)
    except NotImplementedError:
        raised = True
    assert raised
    raised = False
    try:
        SequentialSubsetIterator(10, 3, 3, rng=0)
    except ValueError:
        raised = True
    assert raised


def test_correct_sequential_slices():
    iterator = SequentialSubsetIterator(10, 3, 4)
    sl = iterator.next()
    assert sl.start == 0
    assert sl.stop == 3
    assert sl.step is None
    sl = iterator.next()
    assert sl.start == 3
    assert sl.stop == 6
    assert sl.step is None
    sl = iterator.next()
    assert sl.start == 6
    assert sl.stop == 9
    assert sl.step is None
    sl = iterator.next()
    assert sl.start == 9
    assert sl.stop == 10
    assert sl.step is None

def test_correct_shuffled_sequential_slices():

    dataset_size = 13
    batch_size = 3

    iterator = ShuffledSequentialSubsetIterator(
            dataset_size = dataset_size, batch_size = batch_size,
            num_batches = None, rng = 2)
    visited = [ False ] * dataset_size

    num_batches = 0

    for idxs in iterator:
        for idx in idxs:
            assert not visited[idx]
            visited[idx] = True
        num_batches += 1

    print visited
    assert all(visited)

    assert num_batches == np.ceil(float(dataset_size)/float(batch_size))


def test_sequential_num_batches_and_batch_size():
    try:
        # This should be fine, we have enough examples for 4 batches
        # (with one under-sized batch).
        iterator = SequentialSubsetIterator(10, 3, 4)
        for i in range(4):
            iterator.next()
    except Exception as e:
        assert False
    raised = False
    try:
        iterator.next()
    except StopIteration:
        raised = True
    assert raised
    try:
        # This should be fine, we have enough examples for 4 batches
        # (with one to spare).
        iterator = SequentialSubsetIterator(10, 3, 3)
        for i in range(3):
            iterator.next()
    except Exception:
        assert False
    raised = False
    try:
        iterator.next()
    except StopIteration:
        raised = True
    assert raised
    try:
        # This should fail, since you can't make 5 batches of 3 from 10.
        iterator = SequentialSubsetIterator(10, 3, 5)
    except ValueError:
        return
    assert False


def test_random_slice():
    iterator = RandomSliceSubsetIterator(50, num_batches=10, batch_size=5)
    num = 0
    for iter_slice in iterator:
        assert iter_slice.start >= 0
        assert iter_slice.step is None or iter_slice.step == 1
        assert iter_slice.stop <= 50
        assert iter_slice.stop - iter_slice.start == 5
        num += 1
    assert num == 10


def test_random_uniform():
    iterator = RandomUniformSubsetIterator(50, num_batches=10, batch_size=5)
    num = 0
    for iter_slice in iterator:
        assert len(iter_slice) == 5
        arr = np.array(iter_slice)
        assert np.all(arr < 50)
        assert np.all(arr >= 0)
        num += 1
    assert num == 10
