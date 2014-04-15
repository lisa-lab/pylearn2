from pylearn2.utils.mem import TypicalMemoryError


def test_typical_memory_error():
    """
    A dummy test that instantiates a TypicalMemoryError
    to see if there is no bugs.
    """
    try:
        raise TypicalMemoryError("test")
    except TypicalMemoryError as e:
        pass
