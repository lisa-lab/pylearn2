from pylearn2.utils.iteration import SequentialSubsetIterator

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

