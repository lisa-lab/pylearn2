from pylearn2.utils.mem import TypicalMemoryError


def test_typical_memory_error():
    try:
        raise TypicalMemoryError("test")
    except TypicalMemoryError as e:
        print e
